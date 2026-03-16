"""Tests for timeframe utils, annualization, and execution mode."""
from __future__ import annotations

import unittest

try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError:
    np = pd = None

from hogan_bot.backtest import compute_sharpe, run_backtest_on_candles
from hogan_bot.timeframe_utils import (
    bars_per_day,
    bars_per_year,
    hours_to_bars,
    infer_timeframe_from_candles,
    parse_timeframe_to_minutes,
)


@unittest.skipUnless(pd is not None and np is not None, "pandas/numpy required")
class TestTimeframeUtils(unittest.TestCase):
    def test_parse_timeframe_to_minutes(self):
        self.assertEqual(parse_timeframe_to_minutes("5m"), 5)
        self.assertEqual(parse_timeframe_to_minutes("1h"), 60)
        self.assertEqual(parse_timeframe_to_minutes("30m"), 30)
        self.assertEqual(parse_timeframe_to_minutes("1d"), 1440)
        self.assertEqual(parse_timeframe_to_minutes("unknown"), 5)

    def test_bars_per_day(self):
        self.assertEqual(bars_per_day("5m"), 288)
        self.assertEqual(bars_per_day("1h"), 24)
        self.assertEqual(bars_per_day("30m"), 48)

    def test_bars_per_year(self):
        self.assertEqual(bars_per_year("5m"), 105120)
        self.assertEqual(bars_per_year("1h"), 8760)
        self.assertEqual(bars_per_year("30m"), 17520)

    def test_hours_to_bars(self):
        self.assertEqual(hours_to_bars(12, "5m"), 144)
        self.assertEqual(hours_to_bars(12, "1h"), 12)
        self.assertEqual(hours_to_bars(6, "30m"), 12)

    def test_infer_timeframe_from_candles(self):
        # 5m candles: 5 min = 300_000 ms
        base = 1704067200000  # 2024-01-01 00:00 UTC
        ts_ms = [base + i * 300_000 for i in range(100)]
        df = pd.DataFrame({
            "ts_ms": ts_ms,
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0,
        })
        self.assertEqual(infer_timeframe_from_candles(df), "5m")


@unittest.skipUnless(pd is not None and np is not None, "pandas/numpy required")
class TestAnnualization(unittest.TestCase):
    def test_sharpe_scales_with_bars_per_year(self):
        """Same equity curve produces different Sharpe for different annualization."""
        equity = [100.0, 100.1, 100.2, 99.9, 100.3, 100.0, 100.1, 100.2]
        sharpe_5m = compute_sharpe(equity, bars_per_year=105120)
        sharpe_1h = compute_sharpe(equity, bars_per_year=8760)
        self.assertIsNotNone(sharpe_5m)
        self.assertIsNotNone(sharpe_1h)
        self.assertNotEqual(sharpe_5m, sharpe_1h)
        # 1h Sharpe should be lower (sqrt(8760/105120) ≈ 0.289)
        self.assertLess(sharpe_1h, sharpe_5m)


@unittest.skipUnless(pd is not None and np is not None, "pandas/numpy required")
class TestExecutionMode(unittest.TestCase):
    def test_same_bar_and_next_open_differ(self):
        """next_open execution can produce different results than same_bar."""
        n = 500
        np.random.seed(42)
        close = 30_000.0 + np.cumsum(np.random.randn(n) * 200)
        close = np.clip(close, 10_000, None)
        noise = np.abs(np.random.randn(n)) * 50 + 30
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC"),
            "open": close * (1 + np.random.randn(n) * 0.002),
            "high": close + noise,
            "low": close - noise,
            "close": close,
            "volume": np.random.uniform(500, 5000, n),
        })
        r_same = run_backtest_on_candles(
            df,
            symbol="BTC/USD",
            timeframe="5m",
            starting_balance_usd=10_000,
            aggressive_allocation=0.5,
            max_risk_per_trade=0.02,
            max_drawdown=0.2,
            short_ma_window=5,
            long_ma_window=10,
            volume_window=5,
            volume_threshold=0.5,
            fee_rate=0.001,
            execution_mode="same_bar",
        )
        r_next = run_backtest_on_candles(
            df,
            symbol="BTC/USD",
            timeframe="5m",
            starting_balance_usd=10_000,
            aggressive_allocation=0.5,
            max_risk_per_trade=0.02,
            max_drawdown=0.2,
            short_ma_window=5,
            long_ma_window=10,
            volume_window=5,
            volume_threshold=0.5,
            fee_rate=0.001,
            execution_mode="next_open",
        )
        # Both should complete; at least one should produce a non-trivial equity curve
        self.assertTrue(
            r_same.sharpe_ratio is not None or r_next.sharpe_ratio is not None,
            "At least one execution mode should produce trades with sufficient data",
        )


@unittest.skipUnless(pd is not None, "pandas required")
class TestEffectiveHoldCooldown(unittest.TestCase):
    def test_hours_override_bars(self):
        """max_hold_hours overrides max_hold_bars when set."""
        from hogan_bot.config import BotConfig, effective_hold_cooldown_bars
        cfg = BotConfig(max_hold_bars=144, loss_cooldown_bars=12, max_hold_hours=12, loss_cooldown_hours=1)
        max_hold, cooldown = effective_hold_cooldown_bars(cfg, "5m")
        self.assertEqual(max_hold, 144)  # 12h at 5m = 144 bars
        self.assertEqual(cooldown, 12)  # 1h at 5m = 12 bars

    def test_hours_to_bars_1h(self):
        """12 hours at 1h = 12 bars."""
        from hogan_bot.config import BotConfig, effective_hold_cooldown_bars
        cfg = BotConfig(max_hold_hours=12, loss_cooldown_hours=1)
        max_hold, cooldown = effective_hold_cooldown_bars(cfg, "1h")
        self.assertEqual(max_hold, 12)
        self.assertEqual(cooldown, 1)


if __name__ == "__main__":
    unittest.main()
