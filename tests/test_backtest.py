import unittest

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover
    pd = None

from hogan_bot.backtest import run_backtest_on_candles


@unittest.skipUnless(pd is not None, "pandas is not installed in this environment")
class BacktestTests(unittest.TestCase):
    def test_empty_candles_returns_flat_result(self):
        df = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        result = run_backtest_on_candles(
            candles=df,
            symbol="BTC/USD",
            timeframe="5m",
            starting_balance_usd=1800,
            aggressive_allocation=0.75,
            max_risk_per_trade=0.03,
            max_drawdown=0.15,
            short_ma_window=5,
            long_ma_window=10,
            volume_window=5,
            volume_threshold=1.2,
            fee_rate=0.001,
        )
        self.assertEqual(result.start_equity, 1800)
        self.assertEqual(result.end_equity, 1800)
        self.assertEqual(result.trades, 0)


if __name__ == "__main__":
    unittest.main()
