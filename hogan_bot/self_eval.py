"""Self-Evaluation Loop — Phase 4.3.

Periodically evaluates Hogan's recent trading performance, detects
strategy degradation, and auto-triggers retraining when metrics fall
below acceptable thresholds.

The evaluator computes rolling performance windows and compares them
against historical baselines.  When degradation is detected it:

1. Logs a structured alert to the ``strategy_evals`` table.
2. Emits Prometheus metrics (if configured).
3. Optionally triggers a retrain via ``retrain.retrain_once()``.

Usage::

    from hogan_bot.self_eval import StrategyEvaluator

    evaluator = StrategyEvaluator(db_path="data/hogan.db")
    report = evaluator.evaluate(symbol="BTC/USD")

    if report.degraded:
        evaluator.trigger_retrain(symbol="BTC/USD", timeframe="1h")

Run as a daemon::

    python -m hogan_bot.self_eval --db data/hogan.db --interval 6
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

_FAST_WINDOW = 20
_SLOW_WINDOW = 100
_MIN_TRADES_FAST = 8
_MIN_TRADES_SLOW = 30


@dataclass
class EvalReport:
    """Snapshot of strategy health at evaluation time."""
    symbol: str
    timestamp: str
    fast_win_rate: float = 0.0
    slow_win_rate: float = 0.0
    fast_sharpe: float = 0.0
    slow_sharpe: float = 0.0
    fast_avg_pnl: float = 0.0
    slow_avg_pnl: float = 0.0
    fast_max_drawdown: float = 0.0
    current_drawdown: float = 0.0
    trade_count_fast: int = 0
    trade_count_slow: int = 0
    win_rate_delta: float = 0.0
    sharpe_delta: float = 0.0
    degraded: bool = False
    degradation_reasons: list[str] = field(default_factory=list)
    regime: str = "unknown"
    confidence_score: float = 1.0

    def to_dict(self) -> dict:
        return {k: v for k, v in self.__dict__.items()}

    def summary(self) -> str:
        status = "DEGRADED" if self.degraded else "HEALTHY"
        parts = [
            f"[{status}] {self.symbol}",
            f"fast_wr={self.fast_win_rate:.1%} slow_wr={self.slow_win_rate:.1%} (Δ={self.win_rate_delta:+.1%})",
            f"fast_sharpe={self.fast_sharpe:.2f} slow_sharpe={self.slow_sharpe:.2f} (Δ={self.sharpe_delta:+.2f})",
            f"dd={self.current_drawdown:.1%} trades={self.trade_count_fast}/{self.trade_count_slow}",
        ]
        if self.degradation_reasons:
            parts.append(f"reasons: {', '.join(self.degradation_reasons)}")
        return " | ".join(parts)


@dataclass
class DegradationThresholds:
    """Configurable thresholds for detecting strategy degradation."""
    min_win_rate: float = 0.35
    win_rate_decline: float = 0.10
    min_sharpe: float = -0.5
    sharpe_decline: float = 0.5
    max_drawdown: float = 0.15
    max_consecutive_losses: int = 8
    min_avg_pnl: float = -0.005
    confidence_floor: float = 0.4


class StrategyEvaluator:
    """Monitors recent trade performance and detects degradation.

    Parameters
    ----------
    db_path
        Path to the Hogan SQLite database.
    thresholds
        Degradation detection thresholds. Uses sensible defaults.
    fast_window / slow_window
        Number of recent trades for fast (reactive) and slow (baseline)
        evaluation windows.
    auto_retrain
        When True, automatically triggers retrain on degradation.
    """

    def __init__(
        self,
        db_path: str = "data/hogan.db",
        thresholds: DegradationThresholds | None = None,
        fast_window: int = _FAST_WINDOW,
        slow_window: int = _SLOW_WINDOW,
        auto_retrain: bool = False,
    ) -> None:
        self.db_path = db_path
        self.thresholds = thresholds or DegradationThresholds()
        self.fast_window = fast_window
        self.slow_window = slow_window
        self.auto_retrain = auto_retrain

    def evaluate(self, symbol: str = "BTC/USD") -> EvalReport:
        """Run a full evaluation of recent trading performance.

        Loads closed decisions from the ``decision_log`` table, computes
        rolling performance metrics, and checks for degradation signals.
        """
        from hogan_bot.storage import get_connection

        conn = get_connection(self.db_path)
        try:
            decisions = self._load_closed_decisions(conn, symbol)
            report = self._compute_report(decisions, symbol)
            self._check_degradation(report)
            self._persist_eval(conn, report)
        finally:
            conn.close()

        logger.info("Self-eval: %s", report.summary())

        if report.degraded:
            self._emit_degradation_alert(report)
            if self.auto_retrain:
                logger.info("Auto-retrain triggered for %s", symbol)
                self.trigger_retrain(symbol=symbol)

        return report

    def trigger_retrain(
        self,
        symbol: str = "BTC/USD",
        timeframe: str = "1h",
        model_type: str = "xgboost",
    ) -> dict | None:
        """Trigger a walk-forward retrain cycle."""
        try:
            from hogan_bot.retrain import _build_parser, retrain_once

            argv = [
                "--symbol", symbol,
                "--timeframe", timeframe,
                "--model-type", model_type,
                "--from-db",
                "--db", self.db_path,
                "--model-path", f"models/hogan_{model_type}.pkl",
            ]
            args = _build_parser().parse_args(argv)

            from hogan_bot.timeframe_utils import default_horizon_bars
            args.horizon_bars = default_horizon_bars(timeframe, target_hours=6)

            result = retrain_once(args)
            logger.info(
                "Auto-retrain complete: promoted=%s score=%.4f",
                result.get("promoted"), result.get("new_score", 0),
            )
            return result
        except Exception as exc:
            logger.error("Auto-retrain failed: %s", exc)
            return None

    def _load_closed_decisions(
        self, conn, symbol: str, max_rows: int = 500,
    ) -> pd.DataFrame:
        """Load decisions that have realized P&L outcomes."""
        query = """
            SELECT id, ts_ms, symbol, regime,
                   tech_action, tech_confidence,
                   sent_bias, sent_strength,
                   macro_regime, macro_risk_on,
                   final_action, final_confidence,
                   position_size, ml_up_prob,
                   realized_pnl, outcome_ts_ms
            FROM decision_log
            WHERE realized_pnl IS NOT NULL
              AND symbol = ?
            ORDER BY ts_ms DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(symbol, max_rows))
        return df.sort_values("ts_ms").reset_index(drop=True)

    def _compute_report(self, decisions: pd.DataFrame, symbol: str) -> EvalReport:
        """Compute fast/slow window metrics from closed decisions."""
        now_iso = datetime.now(tz=timezone.utc).isoformat()
        report = EvalReport(symbol=symbol, timestamp=now_iso)

        if decisions.empty:
            return report

        pnls = decisions["realized_pnl"].values
        report.trade_count_slow = min(len(pnls), self.slow_window)
        report.trade_count_fast = min(len(pnls), self.fast_window)

        fast_pnls = pnls[-self.fast_window:] if len(pnls) >= _MIN_TRADES_FAST else pnls
        slow_pnls = pnls[-self.slow_window:] if len(pnls) >= _MIN_TRADES_SLOW else pnls

        report.fast_win_rate = float(np.mean(fast_pnls > 0)) if len(fast_pnls) > 0 else 0
        report.slow_win_rate = float(np.mean(slow_pnls > 0)) if len(slow_pnls) > 0 else 0
        report.win_rate_delta = report.fast_win_rate - report.slow_win_rate

        report.fast_avg_pnl = float(np.mean(fast_pnls)) if len(fast_pnls) > 0 else 0
        report.slow_avg_pnl = float(np.mean(slow_pnls)) if len(slow_pnls) > 0 else 0

        report.fast_sharpe = self._rolling_sharpe(fast_pnls)
        report.slow_sharpe = self._rolling_sharpe(slow_pnls)
        report.sharpe_delta = report.fast_sharpe - report.slow_sharpe

        report.fast_max_drawdown = self._max_drawdown(fast_pnls)
        report.current_drawdown = self._trailing_drawdown(pnls)

        if len(decisions) > 0 and "regime" in decisions.columns:
            last_regime = decisions["regime"].iloc[-1]
            report.regime = str(last_regime) if last_regime else "unknown"

        report.confidence_score = self._compute_confidence(report)

        return report

    def _check_degradation(self, report: EvalReport) -> None:
        """Check all degradation signals and populate the report."""
        t = self.thresholds
        reasons: list[str] = []

        if report.trade_count_fast < _MIN_TRADES_FAST:
            return

        if report.fast_win_rate < t.min_win_rate:
            reasons.append(f"win_rate={report.fast_win_rate:.1%} < {t.min_win_rate:.1%}")

        if report.win_rate_delta < -t.win_rate_decline and report.trade_count_slow >= _MIN_TRADES_SLOW:
            reasons.append(f"win_rate_decline={report.win_rate_delta:+.1%}")

        if report.fast_sharpe < t.min_sharpe:
            reasons.append(f"sharpe={report.fast_sharpe:.2f} < {t.min_sharpe:.2f}")

        if report.sharpe_delta < -t.sharpe_decline and report.trade_count_slow >= _MIN_TRADES_SLOW:
            reasons.append(f"sharpe_decline={report.sharpe_delta:+.2f}")

        if report.current_drawdown > t.max_drawdown:
            reasons.append(f"drawdown={report.current_drawdown:.1%} > {t.max_drawdown:.1%}")

        if report.fast_avg_pnl < t.min_avg_pnl:
            reasons.append(f"avg_pnl={report.fast_avg_pnl:.4f} < {t.min_avg_pnl:.4f}")

        if report.confidence_score < t.confidence_floor:
            reasons.append(f"confidence={report.confidence_score:.2f} < {t.confidence_floor:.2f}")

        report.degradation_reasons = reasons
        report.degraded = len(reasons) >= 2

    def _persist_eval(self, conn, report: EvalReport) -> None:
        """Write evaluation results to the database for audit trail."""
        try:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS strategy_evals (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    ts_ms       INTEGER NOT NULL,
                    symbol      TEXT NOT NULL,
                    report_json TEXT NOT NULL
                )
                """
            )
            conn.execute(
                "INSERT INTO strategy_evals (ts_ms, symbol, report_json) VALUES (?, ?, ?)",
                (int(time.time() * 1000), report.symbol, json.dumps(report.to_dict())),
            )
            conn.commit()
        except Exception as exc:
            logger.debug("Failed to persist eval: %s", exc)

    def _emit_degradation_alert(self, report: EvalReport) -> None:
        """Log and optionally emit Prometheus counter."""
        logger.warning(
            "STRATEGY DEGRADATION: %s — %s",
            report.symbol, ", ".join(report.degradation_reasons),
        )
        try:
            from hogan_bot.metrics import STRATEGY_DEGRADATION
            STRATEGY_DEGRADATION.labels(symbol=report.symbol).inc()
        except Exception:
            pass

    @staticmethod
    def _rolling_sharpe(pnls: np.ndarray, annualize: float = np.sqrt(252)) -> float:
        if len(pnls) < 3:
            return 0.0
        std = float(np.std(pnls))
        if std < 1e-9:
            return 0.0
        return float(np.mean(pnls) / std * annualize)

    @staticmethod
    def _max_drawdown(pnls: np.ndarray) -> float:
        if len(pnls) == 0:
            return 0.0
        cumulative = np.cumsum(pnls)
        peak = np.maximum.accumulate(cumulative)
        dd = peak - cumulative
        max_dd = float(np.max(dd)) if len(dd) > 0 else 0.0
        peak_val = float(np.max(peak)) if float(np.max(peak)) > 0 else 1.0
        return max_dd / peak_val

    @staticmethod
    def _trailing_drawdown(pnls: np.ndarray) -> float:
        if len(pnls) == 0:
            return 0.0
        cumulative = np.cumsum(pnls)
        peak = float(np.max(cumulative))
        current = float(cumulative[-1])
        if peak <= 0:
            return 0.0
        return max(0.0, (peak - current) / peak)

    @staticmethod
    def _compute_confidence(report: EvalReport) -> float:
        """Composite confidence score in [0, 1] reflecting overall strategy health."""
        scores = []

        wr = min(1.0, max(0.0, report.fast_win_rate))
        scores.append(wr)

        sharpe_norm = min(1.0, max(0.0, (report.fast_sharpe + 1.0) / 3.0))
        scores.append(sharpe_norm)

        dd_score = 1.0 - min(1.0, report.current_drawdown / 0.20)
        scores.append(dd_score)

        if report.trade_count_slow >= _MIN_TRADES_SLOW:
            consistency = 1.0 - min(1.0, max(0.0, -report.win_rate_delta / 0.15))
            scores.append(consistency)

        return float(np.mean(scores)) if scores else 0.5


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Strategy self-evaluation loop")
    p.add_argument("--db", default="data/hogan.db")
    p.add_argument("--symbol", default="BTC/USD")
    p.add_argument("--interval", type=float, default=6.0,
                   help="Evaluation interval in hours (default 6)")
    p.add_argument("--auto-retrain", action="store_true",
                   help="Automatically trigger retrain on degradation")
    p.add_argument("--once", action="store_true", help="Run once and exit")
    p.add_argument("--fast-window", type=int, default=_FAST_WINDOW)
    p.add_argument("--slow-window", type=int, default=_SLOW_WINDOW)
    args = p.parse_args()

    evaluator = StrategyEvaluator(
        db_path=args.db,
        fast_window=args.fast_window,
        slow_window=args.slow_window,
        auto_retrain=args.auto_retrain,
    )

    while True:
        report = evaluator.evaluate(symbol=args.symbol)
        print(report.summary())
        if args.once:
            break
        interval_secs = args.interval * 3600.0
        logger.info("Next evaluation in %.1f hours.", args.interval)
        try:
            time.sleep(interval_secs)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    _main()
