"""Performance Tracker — per-regime outcome tracking and MetaWeigher auto-tuning.

Maintains a rolling window of trade outcomes bucketed by regime, agent source,
and signal type.  Uses this data to:

1. Track per-regime win rates and Sharpe ratios.
2. Identify which agents (technical, sentiment, macro) are contributing
   most to profitable trades in each regime.
3. Propose and (optionally) apply updated MetaWeigher weights based on
   realized agent performance — closing the feedback loop between
   trade outcomes and signal generation.

The tracker persists state to SQLite so it survives restarts and can
be queried by the dashboard/MCP server.

Usage::

    from hogan_bot.performance_tracker import PerformanceTracker

    tracker = PerformanceTracker(db_path="data/hogan.db")

    # After each trade closes
    tracker.record_trade_outcome(
        symbol="BTC/USD",
        regime="trending_up",
        tech_action="buy", tech_confidence=0.8,
        sent_bias="bullish", sent_strength=0.6,
        macro_regime="risk_on",
        realized_pnl=125.50,
    )

    # Periodically evaluate and propose weight updates
    proposal = tracker.propose_weight_update(symbol="BTC/USD")
    if proposal and proposal["should_update"]:
        meta_weigher.update_weights(proposal["weights"])
"""
from __future__ import annotations

import json
import logging
import time
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)

_MIN_TRADES_PER_REGIME = 15
_MIN_TOTAL_TRADES = 40
_WEIGHT_BOUNDS = {
    "technical": (0.30, 0.75),
    "sentiment": (0.08, 0.40),
    "macro":     (0.08, 0.35),
}
_LEARNING_RATE = 0.15
_DEFAULT_WEIGHTS = {"technical": 0.55, "sentiment": 0.25, "macro": 0.20}


@dataclass
class RegimePerformance:
    """Aggregated performance stats for one regime."""
    regime: str
    trade_count: int = 0
    win_count: int = 0
    total_pnl: float = 0.0
    pnl_values: list[float] = field(default_factory=list)
    tech_alignment: float = 0.0
    sent_alignment: float = 0.0
    macro_alignment: float = 0.0

    @property
    def win_rate(self) -> float:
        return self.win_count / self.trade_count if self.trade_count > 0 else 0.0

    @property
    def avg_pnl(self) -> float:
        return self.total_pnl / self.trade_count if self.trade_count > 0 else 0.0

    @property
    def sharpe(self) -> float:
        """Per-trade Sharpe ratio (mean / std, not annualized).

        Previous implementation used sqrt(252) which assumes daily
        trade frequency.  Since pnl_values are per-trade with no
        timestamp, annualization is misleading.  Callers that need
        an annualized figure should multiply by sqrt(trades_per_year).
        """
        if len(self.pnl_values) < 3:
            return 0.0
        arr = np.array(self.pnl_values)
        std = float(np.std(arr))
        if std < 1e-9:
            return 0.0
        return float(np.mean(arr) / std)

    def to_dict(self) -> dict:
        return {
            "regime": self.regime,
            "trade_count": self.trade_count,
            "win_rate": round(self.win_rate, 4),
            "avg_pnl": round(self.avg_pnl, 6),
            "sharpe": round(self.sharpe, 3),
            "total_pnl": round(self.total_pnl, 4),
            "tech_alignment": round(self.tech_alignment / max(1, self.trade_count), 4),
            "sent_alignment": round(self.sent_alignment / max(1, self.trade_count), 4),
            "macro_alignment": round(self.macro_alignment / max(1, self.trade_count), 4),
        }


@dataclass
class WeightProposal:
    """Proposed MetaWeigher weight update with reasoning."""
    weights: dict[str, float]
    regime_weights: dict[str, dict[str, float]]
    should_update: bool
    confidence: float
    reasoning: str
    regime_stats: dict[str, dict]


class PerformanceTracker:
    """Tracks per-regime trading outcomes and proposes weight adjustments."""

    def __init__(
        self,
        db_path: str = "data/hogan.db",
        max_window: int = 500,
    ) -> None:
        self.db_path = db_path
        self.max_window = max_window
        self._regime_data: dict[str, RegimePerformance] = {}

    def record_trade_outcome(
        self,
        symbol: str,
        regime: str,
        tech_action: str,
        tech_confidence: float,
        sent_bias: str,
        sent_strength: float,
        macro_regime: str,
        realized_pnl: float,
    ) -> None:
        """Record a completed trade outcome for performance tracking."""
        if regime not in self._regime_data:
            self._regime_data[regime] = RegimePerformance(regime=regime)

        rp = self._regime_data[regime]
        rp.trade_count += 1
        rp.total_pnl += realized_pnl
        rp.pnl_values.append(realized_pnl)
        if realized_pnl > 0:
            rp.win_count += 1

        if len(rp.pnl_values) > self.max_window:
            removed = rp.pnl_values.pop(0)
            rp.total_pnl -= removed
            rp.trade_count -= 1
            if removed > 0:
                rp.win_count -= 1

        outcome_sign = 1.0 if realized_pnl > 0 else -1.0

        tech_vote = {"buy": 1.0, "sell": -1.0}.get(tech_action, 0.0)
        rp.tech_alignment += tech_vote * tech_confidence * outcome_sign

        sent_vote = {"bullish": 1.0, "bearish": -1.0}.get(sent_bias, 0.0)
        rp.sent_alignment += sent_vote * sent_strength * outcome_sign

        macro_vote = {"risk_on": 0.5, "risk_off": -0.5}.get(macro_regime, 0.0)
        rp.macro_alignment += macro_vote * outcome_sign

    def load_from_db(self, symbol: str = "BTC/USD", lookback: int = 500) -> int:
        """Bootstrap tracker state from the decision_log table."""
        from hogan_bot.storage import get_connection

        conn = get_connection(self.db_path)
        try:
            rows = conn.execute(
                """
                SELECT regime, tech_action, tech_confidence,
                       sent_bias, sent_strength,
                       macro_regime, macro_risk_on,
                       realized_pnl
                FROM decision_log
                WHERE realized_pnl IS NOT NULL AND symbol = ?
                ORDER BY ts_ms DESC LIMIT ?
                """,
                (symbol, lookback),
            ).fetchall()
        finally:
            conn.close()

        self._regime_data.clear()
        for row in reversed(rows):
            regime, tech_action, tech_conf, sent_bias, sent_str, macro_regime, _, pnl = row
            self.record_trade_outcome(
                symbol=symbol,
                regime=regime or "unknown",
                tech_action=tech_action or "hold",
                tech_confidence=float(tech_conf or 0.5),
                sent_bias=sent_bias or "neutral",
                sent_strength=float(sent_str or 0.0),
                macro_regime=macro_regime or "neutral",
                realized_pnl=float(pnl or 0),
            )

        total = sum(rp.trade_count for rp in self._regime_data.values())
        logger.info(
            "PerformanceTracker loaded %d trades across %d regimes from DB",
            total, len(self._regime_data),
        )
        return total

    # ------ Persist / restore in-memory state for restart survival ------

    def save_to_db(self) -> None:
        """Persist current regime performance state to SQLite."""
        try:
            from hogan_bot.storage import get_connection
            conn = get_connection(self.db_path)
            conn.execute(
                """CREATE TABLE IF NOT EXISTS perf_tracker_state (
                       id   INTEGER PRIMARY KEY AUTOINCREMENT,
                       ts_ms INTEGER NOT NULL,
                       state_json TEXT NOT NULL
                   )"""
            )
            payload = {}
            for regime, rp in self._regime_data.items():
                payload[regime] = {
                    "trade_count": rp.trade_count,
                    "win_count": rp.win_count,
                    "total_pnl": rp.total_pnl,
                    "pnl_values": rp.pnl_values[-self.max_window:],
                    "tech_alignment": rp.tech_alignment,
                    "sent_alignment": rp.sent_alignment,
                    "macro_alignment": rp.macro_alignment,
                }
            conn.execute(
                "INSERT INTO perf_tracker_state (ts_ms, state_json) VALUES (?, ?)",
                (int(time.time() * 1000), json.dumps(payload)),
            )
            conn.commit()
            conn.close()
            logger.debug("PerformanceTracker state saved (%d regimes)", len(payload))
        except Exception as exc:
            logger.debug("PerformanceTracker save_to_db failed: %s", exc)

    def restore_from_db(self) -> bool:
        """Restore in-memory state from the most recent DB snapshot.

        Returns True if state was successfully restored.
        """
        try:
            from hogan_bot.storage import get_connection
            conn = get_connection(self.db_path)
            row = conn.execute(
                "SELECT state_json FROM perf_tracker_state ORDER BY ts_ms DESC LIMIT 1"
            ).fetchone()
            conn.close()
            if not row:
                return False
            payload = json.loads(row[0])
            self._regime_data.clear()
            for regime, data in payload.items():
                rp = RegimePerformance(
                    regime=regime,
                    trade_count=data["trade_count"],
                    win_count=data["win_count"],
                    total_pnl=data["total_pnl"],
                    pnl_values=data["pnl_values"],
                    tech_alignment=data.get("tech_alignment", 0.0),
                    sent_alignment=data.get("sent_alignment", 0.0),
                    macro_alignment=data.get("macro_alignment", 0.0),
                )
                self._regime_data[regime] = rp
            total = sum(rp.trade_count for rp in self._regime_data.values())
            logger.info(
                "PerformanceTracker restored %d trades across %d regimes from DB snapshot",
                total, len(self._regime_data),
            )
            return True
        except Exception as exc:
            logger.debug("PerformanceTracker restore_from_db failed: %s", exc)
            return False

    def propose_weight_update(
        self,
        symbol: str = "BTC/USD",
        current_weights: dict[str, float] | None = None,
    ) -> WeightProposal | None:
        """Analyze regime performance and propose MetaWeigher weight updates.

        Uses agent-outcome alignment scores to determine which agents
        are contributing most to profitable trades, then shifts weights
        toward better-performing agents while respecting bounds.
        """
        if not self._regime_data:
            self.load_from_db(symbol)

        total_trades = sum(rp.trade_count for rp in self._regime_data.values())
        if total_trades < _MIN_TOTAL_TRADES:
            logger.info(
                "PerformanceTracker: only %d trades, need %d for weight proposal",
                total_trades, _MIN_TOTAL_TRADES,
            )
            return None

        cw = current_weights or _DEFAULT_WEIGHTS.copy()

        global_alignment = {"technical": 0.0, "sentiment": 0.0, "macro": 0.0}
        regime_proposals: dict[str, dict[str, float]] = {}
        regime_stats: dict[str, dict] = {}

        for regime, rp in self._regime_data.items():
            regime_stats[regime] = rp.to_dict()
            if rp.trade_count < _MIN_TRADES_PER_REGIME:
                continue

            n = rp.trade_count
            tech_align = rp.tech_alignment / n
            sent_align = rp.sent_alignment / n
            macro_align = rp.macro_alignment / n

            global_alignment["technical"] += tech_align * n
            global_alignment["sentiment"] += sent_align * n
            global_alignment["macro"] += macro_align * n

            regime_proposals[regime] = self._alignment_to_weights(
                tech_align, sent_align, macro_align, cw,
            )

        for k in global_alignment:
            global_alignment[k] /= max(1, total_trades)

        proposed = self._alignment_to_weights(
            global_alignment["technical"],
            global_alignment["sentiment"],
            global_alignment["macro"],
            cw,
        )

        max_shift = max(abs(proposed[k] - cw[k]) for k in proposed)
        should_update = max_shift > 0.03 and total_trades >= _MIN_TOTAL_TRADES

        overall_wr = sum(rp.win_count for rp in self._regime_data.values()) / max(1, total_trades)
        confidence = min(1.0, total_trades / 200.0) * min(1.0, overall_wr * 2.0)

        reasoning_parts = []
        for k in ("technical", "sentiment", "macro"):
            delta = proposed[k] - cw[k]
            if abs(delta) > 0.02:
                direction = "increase" if delta > 0 else "decrease"
                reasoning_parts.append(
                    f"{k}: {direction} {abs(delta):.1%} (alignment={global_alignment[k]:+.3f})"
                )

        reasoning = "; ".join(reasoning_parts) if reasoning_parts else "No significant changes"

        proposal = WeightProposal(
            weights=proposed,
            regime_weights=regime_proposals,
            should_update=should_update,
            confidence=confidence,
            reasoning=reasoning,
            regime_stats=regime_stats,
        )

        logger.info(
            "Weight proposal: tech=%.3f sent=%.3f macro=%.3f (update=%s, conf=%.2f) | %s",
            proposed["technical"], proposed["sentiment"], proposed["macro"],
            should_update, confidence, reasoning,
        )

        self._persist_proposal(proposal, symbol)
        return proposal

    def get_regime_summary(self) -> dict[str, dict]:
        """Return per-regime performance stats for monitoring."""
        return {r: rp.to_dict() for r, rp in self._regime_data.items()}

    def _alignment_to_weights(
        self,
        tech_align: float,
        sent_align: float,
        macro_align: float,
        current: dict[str, float],
    ) -> dict[str, float]:
        """Convert agent-outcome alignment scores to weight adjustments."""
        adjustments = {
            "technical": tech_align * _LEARNING_RATE,
            "sentiment": sent_align * _LEARNING_RATE,
            "macro": macro_align * _LEARNING_RATE,
        }

        raw = {k: current[k] + adjustments[k] for k in current}

        clamped = {}
        for k, v in raw.items():
            lo, hi = _WEIGHT_BOUNDS[k]
            clamped[k] = max(lo, min(hi, v))

        total = sum(clamped.values())
        return {k: round(v / total, 4) for k, v in clamped.items()}

    def _persist_proposal(self, proposal: WeightProposal, symbol: str) -> None:
        """Write the weight proposal to the DB for audit trail."""
        try:
            from hogan_bot.storage import get_connection
            conn = get_connection(self.db_path)
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS weight_proposals (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_ms       INTEGER NOT NULL,
                    symbol      TEXT NOT NULL,
                    proposal_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "INSERT INTO weight_proposals (ts_ms, symbol, proposal_json) VALUES (?, ?, ?)",
                (
                    int(time.time() * 1000),
                    symbol,
                    json.dumps({
                        "weights": proposal.weights,
                        "regime_weights": proposal.regime_weights,
                        "should_update": proposal.should_update,
                        "confidence": proposal.confidence,
                        "reasoning": proposal.reasoning,
                        "regime_stats": proposal.regime_stats,
                    }),
                ),
            )
            conn.commit()
            conn.close()
        except Exception as exc:
            logger.debug("Failed to persist weight proposal: %s", exc)
