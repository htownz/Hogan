import unittest
from unittest.mock import patch

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    pd = None

from hogan_bot.strategy import generate_signal


@unittest.skipUnless(pd is not None, "pandas is not installed in this environment")
class StrategyTests(unittest.TestCase):
    def test_hold_when_not_enough_data(self):
        df = pd.DataFrame(
            {
                "close": [1, 2, 3],
                "high": [1, 2, 3],
                "low": [1, 2, 3],
                "volume": [10, 12, 11],
            }
        )
        sig = generate_signal(df, short_window=3, long_window=5, volume_window=4, volume_threshold=1.2)
        self.assertEqual(sig.action, "hold")

    def test_buy_signal_with_volume_confirmation(self):
        # Prices dip then spike so the short MA crosses above the long MA exactly
        # on the last bar (fresh bullish crossover with volume confirmation).
        close = [100] * 50 + [95, 90, 90, 90, 90, 140]
        volume = [100] * 54 + [180, 220]
        df = pd.DataFrame(
            {
                "close": close,
                "high": [c * 1.001 for c in close],
                "low": [c * 0.999 for c in close],
                "volume": volume,
            }
        )
        sig = generate_signal(df, short_window=5, long_window=10, volume_window=5, volume_threshold=1.2)
        self.assertEqual(sig.action, "buy")
        self.assertGreater(sig.volume_ratio, 1.2)


    def test_any_mode_holds_on_conflicting_votes(self):
        close = list(range(100, 180))
        volume = [100] * 79 + [220]
        df = pd.DataFrame(
            {
                "close": close,
                "high": [c * 1.001 for c in close],
                "low": [c * 0.999 for c in close],
                "volume": volume,
            }
        )

        with patch("hogan_bot.strategy.cloud_signal") as cloud, patch("hogan_bot.strategy.detect_fvgs") as detect, patch("hogan_bot.strategy.fvg_entry_signal") as fvg_entry:
            cloud.return_value = pd.Series(["bearish"])
            detect.return_value = []
            fvg_entry.return_value = "hold"
            sig = generate_signal(
                df,
                short_window=5,
                long_window=10,
                volume_window=5,
                volume_threshold=1.2,
                use_ema_clouds=True,
                use_fvg=True,
                signal_mode="any",
                min_vote_margin=1,
            )
        self.assertEqual(sig.action, "hold")

    def test_any_mode_requires_vote_margin(self):
        close = list(range(100, 180))
        volume = [100] * 79 + [220]
        df = pd.DataFrame(
            {
                "close": close,
                "high": [c * 1.001 for c in close],
                "low": [c * 0.999 for c in close],
                "volume": volume,
            }
        )

        with patch("hogan_bot.strategy.cloud_signal") as cloud, patch("hogan_bot.strategy.detect_fvgs") as detect, patch("hogan_bot.strategy.fvg_entry_signal") as fvg_entry:
            cloud.return_value = pd.Series(["hold"])
            detect.return_value = []
            fvg_entry.return_value = "buy"
            sig = generate_signal(
                df,
                short_window=5,
                long_window=10,
                volume_window=5,
                volume_threshold=1.2,
                use_ema_clouds=True,
                use_fvg=True,
                signal_mode="any",
                min_vote_margin=3,
            )
        self.assertEqual(sig.action, "hold")


if __name__ == "__main__":
    unittest.main()
