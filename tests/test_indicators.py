import unittest

try:
    import numpy as np
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - environment dependent
    pd = None
    np = None

from hogan_bot.indicators import (
    active_fvgs,
    cloud_signal,
    detect_fvgs,
    fvg_entry_signal,
    fvg_features_frame,
    ripster_ema_clouds,
)


def _make_flat_df(price: float = 100.0, n: int = 60) -> "pd.DataFrame":
    """Minimal OHLCV DataFrame with constant price, useful as a baseline."""
    return pd.DataFrame(
        {
            "open": [price] * n,
            "high": [price * 1.001] * n,
            "low": [price * 0.999] * n,
            "close": [price] * n,
            "volume": [1000.0] * n,
        }
    )


@unittest.skipUnless(pd is not None, "pandas is not installed in this environment")
class EMACloudTests(unittest.TestCase):
    def test_ema_cloud_columns_added(self):
        df = _make_flat_df()
        out = ripster_ema_clouds(df)
        for col in ("ema_fast_short", "ema_fast_long", "ema_slow_short", "ema_slow_long"):
            self.assertIn(col, out.columns, f"Missing column: {col}")

    def test_original_df_not_mutated(self):
        df = _make_flat_df()
        ripster_ema_clouds(df)
        self.assertNotIn("ema_fast_short", df.columns)

    def test_cloud_signal_bullish(self):
        # Build a DataFrame where close rises sharply so fast EMAs >> slow EMAs.
        low_prices = [100.0] * 50
        high_prices = [300.0] * 20
        close = low_prices + high_prices
        n = len(close)
        df = pd.DataFrame(
            {
                "open": close,
                "high": [c * 1.001 for c in close],
                "low": [c * 0.999 for c in close],
                "close": close,
                "volume": [1000.0] * n,
            }
        )
        enriched = ripster_ema_clouds(df)
        signal = cloud_signal(enriched)
        self.assertEqual(signal.iloc[-1], "bullish")

    def test_cloud_signal_bearish(self):
        # Build a DataFrame where close falls sharply so fast EMAs << slow EMAs.
        high_prices = [300.0] * 50
        low_prices = [100.0] * 20
        close = high_prices + low_prices
        n = len(close)
        df = pd.DataFrame(
            {
                "open": close,
                "high": [c * 1.001 for c in close],
                "low": [c * 0.999 for c in close],
                "close": close,
                "volume": [1000.0] * n,
            }
        )
        enriched = ripster_ema_clouds(df)
        signal = cloud_signal(enriched)
        self.assertEqual(signal.iloc[-1], "bearish")

    def test_cloud_signal_neutral_on_flat_prices(self):
        df = _make_flat_df(price=100.0, n=60)
        enriched = ripster_ema_clouds(df)
        signal = cloud_signal(enriched)
        # Flat prices → all EMAs converge to the same value → neutral
        self.assertEqual(signal.iloc[-1], "neutral")


@unittest.skipUnless(pd is not None, "pandas is not installed in this environment")
class FVGDetectionTests(unittest.TestCase):
    def _gap_up_df(self):
        """Three candles with a clear upward gap between bar 0 and bar 2."""
        # bar 0: high=100, bar 1: middle candle, bar 2: low=110 → bullish FVG (110-100 gap)
        return pd.DataFrame(
            {
                "open": [99.0, 104.0, 109.0],
                "high": [100.0, 106.0, 115.0],
                "low": [98.0, 103.0, 110.0],
                "close": [99.5, 105.0, 112.0],
                "volume": [1000.0, 1000.0, 1000.0],
            }
        )

    def _gap_down_df(self):
        """Three candles with a clear downward gap between bar 0 and bar 2."""
        # bar 0: low=110, bar 1: middle candle, bar 2: high=100 → bearish FVG (110-100 gap)
        return pd.DataFrame(
            {
                "open": [112.0, 106.0, 101.0],
                "high": [115.0, 107.0, 100.0],
                "low": [110.0, 104.0, 95.0],
                "close": [111.0, 105.0, 97.0],
                "volume": [1000.0, 1000.0, 1000.0],
            }
        )

    def test_fvg_detect_bullish(self):
        df = self._gap_up_df()
        fvgs = detect_fvgs(df, min_gap_pct=0.0)
        bull = [g for g in fvgs if g["direction"] == "bull"]
        self.assertEqual(len(bull), 1)
        self.assertAlmostEqual(bull[0]["bottom"], 100.0)
        self.assertAlmostEqual(bull[0]["top"], 110.0)
        self.assertEqual(bull[0]["formed_at"], 2)

    def test_fvg_detect_bearish(self):
        df = self._gap_down_df()
        fvgs = detect_fvgs(df, min_gap_pct=0.0)
        bear = [g for g in fvgs if g["direction"] == "bear"]
        self.assertEqual(len(bear), 1)
        self.assertAlmostEqual(bear[0]["bottom"], 100.0)
        self.assertAlmostEqual(bear[0]["top"], 110.0)
        self.assertEqual(bear[0]["formed_at"], 2)

    def test_fvg_no_gap_on_overlapping_candles(self):
        # Candles whose ranges overlap → no FVG should be detected.
        df = pd.DataFrame(
            {
                "open": [100.0, 101.0, 100.5],
                "high": [102.0, 103.0, 102.5],
                "low": [99.0, 100.0, 99.5],
                "close": [101.0, 102.0, 101.0],
                "volume": [1000.0, 1000.0, 1000.0],
            }
        )
        fvgs = detect_fvgs(df, min_gap_pct=0.0)
        self.assertEqual(len(fvgs), 0)

    def test_fvg_entry_signal_buy_inside_bullish_zone(self):
        df = self._gap_up_df()
        fvgs = detect_fvgs(df, min_gap_pct=0.0)
        # Price exactly in the middle of the bullish gap (100–110)
        sig = fvg_entry_signal(fvgs, close_price=105.0)
        self.assertEqual(sig, "buy")

    def test_fvg_entry_signal_hold_outside_zone(self):
        df = self._gap_up_df()
        fvgs = detect_fvgs(df, min_gap_pct=0.0)
        sig = fvg_entry_signal(fvgs, close_price=50.0)
        self.assertEqual(sig, "hold")

    def test_fvg_filled_when_close_enters_zone(self):
        """A bullish FVG formed at bar 2 should be marked filled if a later
        close enters the gap zone."""
        df = pd.DataFrame(
            {
                "open": [99.0, 104.0, 109.0, 112.0, 108.0],
                "high": [100.0, 106.0, 115.0, 114.0, 112.0],
                "low": [98.0, 103.0, 110.0, 108.0, 104.0],
                # close[4] = 105.0 is inside the gap zone (100–110) → fills the FVG
                "close": [99.5, 105.0, 112.0, 113.0, 105.0],
                "volume": [1000.0] * 5,
            }
        )
        fvgs = detect_fvgs(df, min_gap_pct=0.0)
        bull = [g for g in fvgs if g["direction"] == "bull"]
        self.assertTrue(len(bull) > 0)
        self.assertTrue(bull[0]["filled"])

    def test_active_fvgs_excludes_filled(self):
        """active_fvgs() should return only unfilled gaps."""
        fvg_list = [
            {"direction": "bull", "top": 110.0, "bottom": 100.0, "formed_at": 2, "filled": True},
            {"direction": "bull", "top": 120.0, "bottom": 115.0, "formed_at": 5, "filled": False},
        ]
        live = active_fvgs(fvg_list)
        self.assertEqual(len(live), 1)
        self.assertAlmostEqual(live[0]["top"], 120.0)


@unittest.skipUnless(pd is not None, "pandas is not installed in this environment")
class FVGFeaturesFrameTests(unittest.TestCase):
    def test_features_frame_shape_and_columns(self):
        df = _make_flat_df(n=30)
        features = fvg_features_frame(df)
        self.assertEqual(len(features), len(df))
        for col in ("fvg_bull_active", "fvg_bear_active", "in_bull_fvg", "in_bear_fvg"):
            self.assertIn(col, features.columns)

    def test_no_fvg_on_flat_prices(self):
        df = _make_flat_df(n=30)
        features = fvg_features_frame(df)
        self.assertEqual(features["fvg_bull_active"].sum(), 0)
        self.assertEqual(features["fvg_bear_active"].sum(), 0)


if __name__ == "__main__":
    unittest.main()
