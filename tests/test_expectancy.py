"""Tests for ExpectancyTracker — per-regime, per-symbol expectancy metrics."""
from __future__ import annotations

import pytest

from hogan_bot.expectancy import ExpectancyTracker, TradeRecord


class TestTradeRecord:
    def test_win_classification(self):
        t = TradeRecord("S", "r", 0.01, 0.005, 0.002, 0.015, 5, "tp", True)
        assert t.is_win is True
        t2 = TradeRecord("S", "r", -0.01, -0.015, 0.01, 0.0, 3, "signal", False)
        assert t2.is_win is False

    def test_zero_pnl_is_loss(self):
        t = TradeRecord("S", "r", 0.0, 0.0, 0.0, 0.0, 1, "tp", False)
        assert t.is_win is False


class TestExpectancyTracker:
    def test_empty_tracker_summary(self):
        et = ExpectancyTracker()
        s = et.summary()
        assert s["total_trades"] == 0

    def test_record_and_summary(self):
        et = ExpectancyTracker()
        et.record_trade("BTC/USD", "trending_up", 0.02, 0.015, hold_bars=5, close_reason="trailing_stop")
        et.record_trade("BTC/USD", "trending_up", -0.01, -0.015, hold_bars=3, close_reason="signal")
        et.record_trade("ETH/USD", "ranging", 0.005, 0.002, hold_bars=8, close_reason="take_profit")
        s = et.summary()
        assert s["total_trades"] == 3
        assert s["overall"]["win_rate"] > 0
        assert "by_regime" in s
        assert "trending_up" in s["by_regime"]
        assert "by_symbol" in s
        assert "BTC/USD" in s["by_symbol"]

    def test_payoff_ratio_no_losses(self):
        et = ExpectancyTracker()
        et.record_trade("BTC/USD", "bull", 0.05, 0.04, hold_bars=10, close_reason="tp")
        et.record_trade("BTC/USD", "bull", 0.03, 0.02, hold_bars=8, close_reason="tp")
        s = et.summary()
        assert s["overall"]["payoff_ratio"] == 99.99

    def test_signal_exit_loss_rate_none_when_no_signal_exits(self):
        et = ExpectancyTracker()
        et.record_trade("BTC/USD", "r", 0.01, 0.005, hold_bars=5, close_reason="trailing_stop")
        assert et.signal_exit_loss_rate() is None

    def test_signal_exit_loss_rate_with_signal_exits(self):
        et = ExpectancyTracker()
        et.record_trade("BTC/USD", "r", -0.01, -0.015, hold_bars=5, close_reason="signal")
        et.record_trade("BTC/USD", "r", 0.02, 0.015, hold_bars=3, close_reason="signal")
        rate = et.signal_exit_loss_rate()
        assert rate == 0.5

    def test_mae_mfe_tracking(self):
        et = ExpectancyTracker()
        et.record_trade("BTC/USD", "r", 0.02, 0.015, hold_bars=5,
                         close_reason="tp", mae_pct=0.005, mfe_pct=0.03)
        s = et.summary()
        assert s["overall"]["avg_mae_pct"] == pytest.approx(0.005)
        assert s["overall"]["avg_mfe_pct"] == pytest.approx(0.03)

    def test_eviction_keeps_max_history(self):
        et = ExpectancyTracker(max_history=5)
        for i in range(10):
            et.record_trade("S", "r", 0.01 * (i + 1), 0.005 * (i + 1),
                             hold_bars=1, close_reason="tp")
        assert len(et._trades) == 5

    def test_per_regime_breakdown(self):
        et = ExpectancyTracker()
        et.record_trade("S", "trending_up", 0.03, 0.025, hold_bars=5, close_reason="tp")
        et.record_trade("S", "ranging", -0.01, -0.015, hold_bars=3, close_reason="signal")
        s = et.summary()
        assert s["by_regime"]["trending_up"]["win_rate"] == pytest.approx(1.0)
        assert s["by_regime"]["ranging"]["win_rate"] == pytest.approx(0.0)
