"""Expectancy tracking — per-regime, per-symbol edge and risk metrics.

A model can be "accurate" and still be a financial clown. This module tracks
the metrics that actually matter for profitability.

Usage::

    tracker = ExpectancyTracker()
    tracker.record_trade(
        symbol="BTC/USD", regime="trending_up",
        gross_pnl_pct=0.015, net_pnl_pct=0.010,
        mae_pct=0.005, mfe_pct=0.022,
        hold_bars=8, close_reason="signal",
    )
    report = tracker.summary()
"""
from __future__ import annotations

import logging
from collections import defaultdict
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class TradeRecord:
    symbol: str
    regime: str
    gross_pnl_pct: float
    net_pnl_pct: float
    mae_pct: float
    mfe_pct: float
    hold_bars: int
    close_reason: str
    is_win: bool


class ExpectancyTracker:
    """Tracks per-regime, per-symbol expectancy metrics."""

    def __init__(self, max_history: int = 500):
        self._max_history = max_history
        self._trades: list[TradeRecord] = []
        self._by_regime: dict[str, list[TradeRecord]] = defaultdict(list)
        self._by_symbol: dict[str, list[TradeRecord]] = defaultdict(list)

    def record_trade(
        self,
        symbol: str,
        regime: str = "unknown",
        gross_pnl_pct: float = 0.0,
        net_pnl_pct: float = 0.0,
        mae_pct: float = 0.0,
        mfe_pct: float = 0.0,
        hold_bars: int = 0,
        close_reason: str = "unknown",
    ) -> None:
        """Record a completed trade for expectancy tracking."""
        rec = TradeRecord(
            symbol=symbol, regime=regime,
            gross_pnl_pct=gross_pnl_pct, net_pnl_pct=net_pnl_pct,
            mae_pct=mae_pct, mfe_pct=mfe_pct,
            hold_bars=hold_bars, close_reason=close_reason,
            is_win=net_pnl_pct > 0,
        )
        self._trades.append(rec)
        self._by_regime[regime].append(rec)
        self._by_symbol[symbol].append(rec)

        if len(self._trades) > self._max_history:
            old = self._trades.pop(0)
            regime_list = self._by_regime.get(old.regime, [])
            if regime_list and regime_list[0] is old:
                regime_list.pop(0)
            sym_list = self._by_symbol.get(old.symbol, [])
            if sym_list and sym_list[0] is old:
                sym_list.pop(0)

    def summary(self) -> dict:
        """Return a comprehensive expectancy report."""
        if not self._trades:
            return {"total_trades": 0}

        result = {
            "total_trades": len(self._trades),
            "overall": self._compute_stats(self._trades),
            "by_regime": {},
            "by_symbol": {},
        }

        for regime, trades in self._by_regime.items():
            if trades:
                result["by_regime"][regime] = self._compute_stats(trades)

        for symbol, trades in self._by_symbol.items():
            if trades:
                result["by_symbol"][symbol] = self._compute_stats(trades)

        return result

    def signal_exit_loss_rate(self) -> float:
        """Fraction of signal-exit trades that were losers."""
        signal_exits = [t for t in self._trades if t.close_reason == "signal"]
        if not signal_exits:
            return 0.0
        losers = sum(1 for t in signal_exits if not t.is_win)
        return losers / len(signal_exits)

    @staticmethod
    def _compute_stats(trades: list[TradeRecord]) -> dict:
        n = len(trades)
        if n == 0:
            return {}

        wins = [t for t in trades if t.is_win]
        losses = [t for t in trades if not t.is_win]

        win_rate = len(wins) / n
        avg_gross = sum(t.gross_pnl_pct for t in trades) / n
        avg_net = sum(t.net_pnl_pct for t in trades) / n
        avg_mae = sum(t.mae_pct for t in trades) / n
        avg_mfe = sum(t.mfe_pct for t in trades) / n

        avg_win = sum(t.net_pnl_pct for t in wins) / len(wins) if wins else 0
        avg_loss = sum(t.net_pnl_pct for t in losses) / len(losses) if losses else 0
        payoff_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float("inf")

        expectancy = win_rate * avg_win + (1 - win_rate) * avg_loss

        avg_hold_winners = (
            sum(t.hold_bars for t in wins) / len(wins) if wins else 0
        )
        avg_hold_losers = (
            sum(t.hold_bars for t in losses) / len(losses) if losses else 0
        )

        signal_exits = [t for t in trades if t.close_reason == "signal"]
        signal_exit_loss_rate = (
            sum(1 for t in signal_exits if not t.is_win) / len(signal_exits)
            if signal_exits else 0
        )

        return {
            "n": n,
            "win_rate": round(win_rate, 3),
            "avg_gross_edge_pct": round(avg_gross, 4),
            "avg_net_edge_pct": round(avg_net, 4),
            "payoff_ratio": round(payoff_ratio, 2),
            "expectancy_pct": round(expectancy, 4),
            "avg_mae_pct": round(avg_mae, 4),
            "avg_mfe_pct": round(avg_mfe, 4),
            "avg_hold_bars_winners": round(avg_hold_winners, 1),
            "avg_hold_bars_losers": round(avg_hold_losers, 1),
            "signal_exit_loss_rate": round(signal_exit_loss_rate, 3),
        }
