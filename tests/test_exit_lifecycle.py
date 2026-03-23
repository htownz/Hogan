"""Tests for hogan_bot.exit_lifecycle analytics."""
from __future__ import annotations

from hogan_bot.exit_lifecycle import (
    compute_exit_quality_by_regime,
    compute_hold_duration_vs_pnl,
    compute_tail_loss_metrics,
    compute_time_in_trade_by_regime,
    summarize_exit_lifecycle,
)


def _make_trades() -> list[dict]:
    return [
        {"pnl_pct": 2.5, "bars_held": 5, "entry_regime": "trending_up", "close_reason": "take_profit", "side": "long"},
        {"pnl_pct": -1.2, "bars_held": 8, "entry_regime": "trending_up", "close_reason": "trailing_stop", "side": "long"},
        {"pnl_pct": 1.0, "bars_held": 3, "entry_regime": "ranging", "close_reason": "take_profit", "side": "long"},
        {"pnl_pct": -3.5, "bars_held": 20, "entry_regime": "volatile", "close_reason": "max_hold", "side": "long"},
        {"pnl_pct": -0.5, "bars_held": 2, "entry_regime": "volatile", "close_reason": "trailing_stop", "side": "short"},
        {"pnl_pct": 0.8, "bars_held": 14, "entry_regime": "ranging", "close_reason": "signal_exit", "side": "long"},
        {"pnl_pct": -5.0, "bars_held": 24, "entry_regime": "volatile", "close_reason": "max_hold", "side": "long"},
    ]


class TestTailLossMetrics:
    def test_empty(self) -> None:
        r = compute_tail_loss_metrics([])
        assert r["n_trades"] == 0

    def test_correct_counts(self) -> None:
        trades = _make_trades()
        r = compute_tail_loss_metrics(trades)
        assert r["n_trades"] == 7
        assert r["n_losses"] == 4
        assert r["worst_loss_pct"] == -5.0
        assert r["losses_beyond_3.0pct"] == 2  # -3.5 and -5.0
        assert r["losses_beyond_1.0pct"] >= 3  # -1.2, -3.5, -5.0

    def test_percentiles_present(self) -> None:
        trades = _make_trades()
        r = compute_tail_loss_metrics(trades)
        assert "p5_loss_pct" in r
        assert "p10_loss_pct" in r


class TestTimeInTradeByRegime:
    def test_empty(self) -> None:
        assert compute_time_in_trade_by_regime([]) == {}

    def test_groups_correctly(self) -> None:
        trades = _make_trades()
        r = compute_time_in_trade_by_regime(trades)
        assert "trending_up" in r
        assert "volatile" in r
        assert "ranging" in r
        assert r["trending_up"]["n_trades"] == 2
        assert r["volatile"]["n_trades"] == 3
        assert r["volatile"]["worst_loss_pct"] == -5.0

    def test_avg_bars(self) -> None:
        trades = _make_trades()
        r = compute_time_in_trade_by_regime(trades)
        assert r["trending_up"]["avg_bars_held"] > 0


class TestExitQualityByRegime:
    def test_empty(self) -> None:
        assert compute_exit_quality_by_regime([]) == {}

    def test_nested_structure(self) -> None:
        trades = _make_trades()
        r = compute_exit_quality_by_regime(trades)
        assert "volatile" in r
        assert "max_hold" in r["volatile"]
        assert r["volatile"]["max_hold"]["count"] == 2


class TestHoldDurationVsPnl:
    def test_empty(self) -> None:
        assert compute_hold_duration_vs_pnl([]) == {}

    def test_buckets(self) -> None:
        trades = _make_trades()
        r = compute_hold_duration_vs_pnl(trades)
        assert len(r) > 0
        assert any("0-" in k for k in r)


class TestSummarize:
    def test_all_keys_present(self) -> None:
        trades = _make_trades()
        s = summarize_exit_lifecycle(trades)
        assert "tail_losses" in s
        assert "time_in_trade_by_regime" in s
        assert "exit_quality_by_regime" in s
        assert "hold_duration_vs_pnl" in s
