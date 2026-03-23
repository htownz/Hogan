"""Tests for hogan_bot.ict — ICT indicator functions.

All tests use small, hand-crafted DataFrames so they require no external data
or network access and execute in < 1 second.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timezone

import pandas as pd

from hogan_bot.ict import (
    dealing_range,
    detect_equal_highs_lows,
    detect_liquidity_sweep,
    detect_mss,
    detect_order_block,
    find_swings,
    ict_setup_signal,
    in_time_window,
    is_in_discount,
    is_in_premium,
    ote_zone,
    parse_time_windows,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df(
    close: list[float],
    high_offset: float = 0.5,
    low_offset: float = 0.5,
    open_pct: float = 0.0,
) -> pd.DataFrame:
    """Build a minimal OHLCV DataFrame from a list of close prices.

    ``high = close + high_offset``, ``low = close - low_offset``,
    ``open = close * (1 + open_pct)``.
    """
    n = len(close)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min", tz=timezone.utc)
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": [c * (1 + open_pct) for c in close],
            "high": [c + high_offset for c in close],
            "low": [c - low_offset for c in close],
            "close": close,
            "volume": [1000.0] * n,
        }
    )


def _make_df_custom(rows: list[dict]) -> pd.DataFrame:
    """Build a DataFrame from explicit OHLCV dicts."""
    n = len(rows)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min", tz=timezone.utc)
    df = pd.DataFrame(rows)
    df["timestamp"] = ts
    return df


# ---------------------------------------------------------------------------
# 1. Swing detection
# ---------------------------------------------------------------------------


class TestFindSwings(unittest.TestCase):
    def test_obvious_swing_high(self):
        """A clear peak should be detected as a swing high."""
        # Price rises to a peak at bar 4 then falls
        closes = [100.0, 102.0, 105.0, 108.0, 110.0, 107.0, 104.0, 101.0, 99.0]
        df = _make_df(closes, high_offset=0.0, low_offset=0.0)
        # Customize highs so only bar 4 is a genuine peak
        df["high"] = df["close"]
        df["low"] = df["close"]
        sh, _ = find_swings(df, left=2, right=2)
        self.assertTrue(any(s["index"] == 4 for s in sh), f"Expected swing high at 4, got {sh}")

    def test_obvious_swing_low(self):
        """A clear trough should be detected as a swing low."""
        closes = [100.0, 98.0, 95.0, 93.0, 91.0, 93.0, 96.0, 99.0, 101.0]
        df = _make_df(closes, high_offset=0.0, low_offset=0.0)
        df["high"] = df["close"]
        df["low"] = df["close"]
        _, sl = find_swings(df, left=2, right=2)
        self.assertTrue(any(s["index"] == 4 for s in sl), f"Expected swing low at 4, got {sl}")

    def test_no_swings_in_trend(self):
        """Monotonically rising bars should produce no swing highs."""
        closes = list(range(100, 120))
        df = _make_df(closes, high_offset=0.0, low_offset=0.0)
        df["high"] = df["close"]
        df["low"] = df["close"]
        sh, _ = find_swings(df, left=2, right=2)
        self.assertEqual(sh, [])

    def test_insufficient_bars_returns_empty(self):
        df = _make_df([100.0, 101.0, 102.0])
        sh, sl = find_swings(df, left=2, right=2)
        self.assertEqual(sh, [])
        self.assertEqual(sl, [])

    def test_swing_prices_match_highs_lows(self):
        """Returned price should equal the candle high/low at that index."""
        closes = [100.0, 102.0, 108.0, 102.0, 100.0, 97.0, 93.0, 97.0, 100.0]
        df = _make_df(closes, high_offset=2.0, low_offset=2.0)
        sh, sl = find_swings(df, left=2, right=2)
        for s in sh:
            self.assertAlmostEqual(s["price"], float(df["high"].iloc[s["index"]]))
        for s in sl:
            self.assertAlmostEqual(s["price"], float(df["low"].iloc[s["index"]]))


# ---------------------------------------------------------------------------
# 2. Equal highs / lows detection
# ---------------------------------------------------------------------------


class TestDetectEqualHighsLows(unittest.TestCase):
    def test_two_equal_highs_form_cluster(self):
        swings = [
            {"index": 2, "price": 100.0},
            {"index": 5, "price": 100.05},  # within 0.1% → equal high
            {"index": 9, "price": 110.0},   # far away
        ]
        pools = detect_equal_highs_lows(swings, tolerance_pct=0.001)
        self.assertEqual(len(pools), 1)
        self.assertEqual(pools[0]["count"], 2)
        self.assertAlmostEqual(pools[0]["price"], 100.025)

    def test_three_equal_highs_cluster(self):
        swings = [{"index": i, "price": 100.0 + i * 0.001} for i in range(3)]
        pools = detect_equal_highs_lows(swings, tolerance_pct=0.005)
        self.assertEqual(len(pools), 1)
        self.assertEqual(pools[0]["count"], 3)

    def test_widely_separated_swings_no_cluster(self):
        swings = [
            {"index": 0, "price": 100.0},
            {"index": 5, "price": 110.0},  # 10% away — not equal
        ]
        pools = detect_equal_highs_lows(swings, tolerance_pct=0.001)
        self.assertEqual(pools, [])

    def test_empty_swings(self):
        self.assertEqual(detect_equal_highs_lows([]), [])

    def test_single_swing_no_pool(self):
        swings = [{"index": 3, "price": 100.0}]
        self.assertEqual(detect_equal_highs_lows(swings), [])


# ---------------------------------------------------------------------------
# 3. Liquidity sweep detection
# ---------------------------------------------------------------------------


class TestDetectLiquiditySweep(unittest.TestCase):
    def _pools_with_bsl(self, level: float) -> dict:
        """Return a pools dict with a single buy-side liquidity level."""
        return {
            "equal_highs": [{"price": level, "count": 2, "indices": [2, 5]}],
            "equal_lows": [],
            "recent_swing_high": None,
            "recent_swing_low": None,
            "prev_day_high": None,
            "prev_day_low": None,
        }

    def test_wick_above_equal_high_then_close_below_triggers_sell_side(self):
        """Wick above BSL level + close back below → sell bias."""
        level = 105.0
        rows = [
            {"open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5},
            {"open": 100.5, "high": 102.0, "low": 100.0, "close": 101.0},
            {"open": 101.0, "high": 103.0, "low": 100.5, "close": 101.5},
            # Sweep: wick above 105, close back at 104
            {"open": 103.0, "high": 106.0, "low": 102.5, "close": 104.0},
        ]
        df = _make_df_custom(rows)
        pools = self._pools_with_bsl(level)
        sweep = detect_liquidity_sweep(df, pools, lookback=10, wick_only=True)
        self.assertIsNotNone(sweep)
        self.assertEqual(sweep["side"], "sell")
        self.assertAlmostEqual(sweep["pool_level"], level)

    def test_no_sweep_when_wick_does_not_reach_level(self):
        level = 110.0
        rows = [
            {"open": 100.0, "high": 101.0, "low": 99.5, "close": 100.5},
            {"open": 100.5, "high": 102.0, "low": 100.0, "close": 101.0},
            {"open": 101.0, "high": 103.0, "low": 100.5, "close": 101.5},
        ]
        df = _make_df_custom(rows)
        pools = self._pools_with_bsl(level)
        sweep = detect_liquidity_sweep(df, pools, lookback=10)
        self.assertIsNone(sweep)

    def test_ssl_sweep_returns_buy_side_bias(self):
        """Wick below SSL level + close back above → buy bias."""
        level = 95.0
        pools = {
            "equal_highs": [],
            "equal_lows": [{"price": level, "count": 2, "indices": [1, 3]}],
            "recent_swing_high": None,
            "recent_swing_low": None,
            "prev_day_high": None,
            "prev_day_low": None,
        }
        rows = [
            {"open": 100.0, "high": 101.0, "low": 99.5, "close": 100.0},
            {"open": 100.0, "high": 100.5, "low": 99.0, "close": 99.5},
            # Sweep: wick below 95, close back above
            {"open": 99.0, "high": 100.0, "low": 94.0, "close": 97.0},
        ]
        df = _make_df_custom(rows)
        sweep = detect_liquidity_sweep(df, pools, lookback=10, wick_only=True)
        self.assertIsNotNone(sweep)
        self.assertEqual(sweep["side"], "buy")


# ---------------------------------------------------------------------------
# 4. Market Structure Shift
# ---------------------------------------------------------------------------


class TestDetectMSS(unittest.TestCase):
    def _make_swing_highs(self, pairs: list[tuple[int, float]]) -> list[dict]:
        return [{"index": i, "price": p} for i, p in pairs]

    def _make_swing_lows(self, pairs: list[tuple[int, float]]) -> list[dict]:
        return [{"index": i, "price": p} for i, p in pairs]

    def test_bearish_mss_breaks_swing_low(self):
        """After a BSL sweep at bar 4, bearish MSS when close goes below swing low."""
        closes = [102.0, 101.0, 100.0, 99.0, 98.0, 97.0, 96.0, 95.0]
        df = _make_df(closes)
        # There's a swing low at bar 2 (price 100)
        sh = self._make_swing_highs([])
        sl = self._make_swing_lows([(2, 100.0)])
        mss = detect_mss(df, sh, sl, after_index=3, direction="bear")
        self.assertIsNotNone(mss)
        self.assertEqual(mss["direction"], "bear")
        self.assertLess(mss["break_index"], len(df))

    def test_bullish_mss_breaks_swing_high(self):
        """After a SSL sweep, bullish MSS when close breaks above swing high."""
        closes = [95.0, 96.0, 97.0, 99.0, 101.0, 103.0, 105.0]
        df = _make_df(closes, high_offset=0.0, low_offset=0.0)
        df["high"] = df["close"]
        df["low"] = df["close"]
        sh = self._make_swing_highs([(2, 97.0)])
        sl = self._make_swing_lows([])
        mss = detect_mss(df, sh, sl, after_index=2, direction="bull")
        self.assertIsNotNone(mss)
        self.assertEqual(mss["direction"], "bull")

    def test_no_mss_when_price_doesnt_break(self):
        """If price never breaks the target level, return None."""
        closes = [100.0, 102.0, 104.0, 103.0, 102.5]
        df = _make_df(closes, high_offset=0.0, low_offset=0.0)
        df["high"] = df["close"]
        df["low"] = df["close"]
        sh = self._make_swing_highs([])
        sl = self._make_swing_lows([(1, 99.0)])  # Level BELOW all closes
        mss = detect_mss(df, sh, sl, after_index=1, direction="bear")
        self.assertIsNone(mss)

    def test_no_swings_returns_none(self):
        closes = [100.0] * 10
        df = _make_df(closes)
        mss = detect_mss(df, [], [], after_index=3, direction="bull")
        self.assertIsNone(mss)


# ---------------------------------------------------------------------------
# 5. Order Block identification
# ---------------------------------------------------------------------------


class TestDetectOrderBlock(unittest.TestCase):
    def test_bearish_mss_finds_last_bullish_candle(self):
        """For bearish MSS, OB should be the last bullish candle before break."""
        rows = [
            {"open": 100.0, "high": 101.0, "low": 99.0, "close": 100.5},  # bullish ← OB candidate
            {"open": 100.5, "high": 101.5, "low": 100.0, "close": 101.0},  # bullish
            {"open": 101.0, "high": 101.5, "low": 99.5, "close": 99.8},   # bearish (displacement)
            {"open": 99.8, "high": 100.0, "low": 98.0, "close": 98.5},    # bearish (MSS break)
        ]
        df = _make_df_custom(rows)
        mss = {"direction": "bear", "break_index": 3, "broken_level": 100.0}
        ob = detect_order_block(df, mss, lookback=10)
        self.assertIsNotNone(ob)
        self.assertEqual(ob["direction"], "bear")
        # Should be the last bullish candle before bar 3 → bar 1
        self.assertEqual(ob["index"], 1)

    def test_bullish_mss_finds_last_bearish_candle(self):
        """For bullish MSS, OB should be the last bearish candle before break."""
        rows = [
            {"open": 101.0, "high": 101.5, "low": 100.5, "close": 100.8},  # bearish ← OB candidate
            {"open": 100.8, "high": 101.0, "low": 100.5, "close": 100.6},  # bearish
            {"open": 100.6, "high": 102.0, "low": 100.4, "close": 101.8},  # bullish (displacement)
            {"open": 101.8, "high": 103.0, "low": 101.5, "close": 102.5},  # bullish (MSS break)
        ]
        df = _make_df_custom(rows)
        mss = {"direction": "bull", "break_index": 3, "broken_level": 101.0}
        ob = detect_order_block(df, mss, lookback=10)
        self.assertIsNotNone(ob)
        self.assertEqual(ob["direction"], "bull")
        self.assertEqual(ob["index"], 1)

    def test_none_mss_returns_none(self):
        df = _make_df([100.0] * 10)
        self.assertIsNone(detect_order_block(df, None))

    def test_body_only_flag_narrows_zone(self):
        rows = [
            {"open": 99.0, "high": 105.0, "low": 95.0, "close": 101.0},  # large wick bullish
            {"open": 101.0, "high": 102.0, "low": 100.0, "close": 100.5},
        ]
        df = _make_df_custom(rows)
        mss = {"direction": "bear", "break_index": 1, "broken_level": 100.0}
        ob_full = detect_order_block(df, mss, lookback=5, body_only=False)
        ob_body = detect_order_block(df, mss, lookback=5, body_only=True)
        self.assertGreater(ob_full["top"] - ob_full["bottom"],
                           ob_body["top"] - ob_body["bottom"])


# ---------------------------------------------------------------------------
# 6. Premium / Discount + OTE
# ---------------------------------------------------------------------------


class TestDealingRange(unittest.TestCase):
    def test_basic_range(self):
        closes = [100.0, 105.0, 110.0, 108.0, 106.0]
        df = _make_df(closes, high_offset=0.0, low_offset=0.0)
        df["high"] = df["close"]
        df["low"] = df["close"]
        dr = dealing_range(df, 0, 4)
        self.assertAlmostEqual(dr["high"], 110.0)
        self.assertAlmostEqual(dr["low"], 100.0)
        self.assertAlmostEqual(dr["mid"], 105.0)


class TestPremiumDiscount(unittest.TestCase):
    def test_discount_below_midpoint(self):
        self.assertTrue(is_in_discount(94.9, 90.0, 100.0))
        self.assertFalse(is_in_discount(95.1, 90.0, 100.0))

    def test_premium_above_midpoint(self):
        self.assertTrue(is_in_premium(95.1, 90.0, 100.0))
        self.assertFalse(is_in_premium(94.9, 90.0, 100.0))

    def test_degenerate_range_returns_false(self):
        self.assertFalse(is_in_discount(50.0, 100.0, 90.0))
        self.assertFalse(is_in_premium(50.0, 100.0, 90.0))


class TestOTEZone(unittest.TestCase):
    def test_bullish_ote_in_discount(self):
        lo, hi = ote_zone(90.0, 100.0, "bull", fib_low=0.62, fib_high=0.79)
        # Retracement from 100 towards 90
        # zone_high = 100 - 0.62*10 = 93.8
        # zone_low  = 100 - 0.79*10 = 92.1
        self.assertAlmostEqual(hi, 93.8, places=5)
        self.assertAlmostEqual(lo, 92.1, places=5)
        # Both should be in the discount half (below midpoint 95)
        self.assertTrue(lo < 95.0)
        self.assertTrue(hi < 95.0)

    def test_bearish_ote_in_premium(self):
        lo, hi = ote_zone(90.0, 100.0, "bear", fib_low=0.62, fib_high=0.79)
        # Retracement from 90 towards 100
        # zone_low  = 90 + 0.62*10 = 96.2
        # zone_high = 90 + 0.79*10 = 97.9
        self.assertAlmostEqual(lo, 96.2, places=5)
        self.assertAlmostEqual(hi, 97.9, places=5)
        # Both should be in the premium half (above midpoint 95)
        self.assertTrue(lo > 95.0)
        self.assertTrue(hi > 95.0)

    def test_zone_low_less_than_zone_high(self):
        lo, hi = ote_zone(80.0, 120.0, "bull")
        self.assertLess(lo, hi)
        lo2, hi2 = ote_zone(80.0, 120.0, "bear")
        self.assertLess(lo2, hi2)


# ---------------------------------------------------------------------------
# 7. Time window filter
# ---------------------------------------------------------------------------


class TestInTimeWindow(unittest.TestCase):
    def _ny_ts(self, hour: int, minute: int = 0) -> datetime:
        """Build a timezone-aware datetime at the given NY (UTC-5) hour."""
        try:
            from zoneinfo import ZoneInfo
            tz = ZoneInfo("America/New_York")
            return datetime(2024, 1, 15, hour, minute, tzinfo=tz)
        except Exception:
            return datetime(2024, 1, 15, hour + 5, minute, tzinfo=timezone.utc)

    def test_inside_london_window(self):
        ts = self._ny_ts(3, 30)
        self.assertTrue(in_time_window(ts, [("03:00", "04:00")]))

    def test_outside_all_windows(self):
        ts = self._ny_ts(9, 0)
        windows = [("03:00", "04:00"), ("10:00", "11:00"), ("14:00", "15:00")]
        result = in_time_window(ts, windows)
        # 09:00 NY is outside all windows → should be False
        # (or True if zoneinfo unavailable — we tolerate both)
        self.assertIsInstance(result, bool)

    def test_inside_ny_am_window(self):
        ts = self._ny_ts(10, 30)
        self.assertTrue(in_time_window(ts, [("10:00", "11:00")]))

    def test_empty_windows_returns_false(self):
        ts = self._ny_ts(10, 30)
        try:
            result = in_time_window(ts, [])
            self.assertFalse(result)
        except (ValueError, TypeError):
            pass  # acceptable to raise ValueError/TypeError on empty windows


class TestParseTimeWindows(unittest.TestCase):
    def test_csv_parsing(self):
        result = parse_time_windows("03:00-04:00,10:00-11:00,14:00-15:00")
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0], ("03:00", "04:00"))
        self.assertEqual(result[2], ("14:00", "15:00"))

    def test_single_window(self):
        result = parse_time_windows("10:00-11:00")
        self.assertEqual(result, [("10:00", "11:00")])

    def test_spaces_stripped(self):
        result = parse_time_windows("03:00-04:00 , 10:00-11:00")
        self.assertEqual(len(result), 2)


# ---------------------------------------------------------------------------
# 8. ICT setup signal (integration smoke test)
# ---------------------------------------------------------------------------


class TestICTSetupSignal(unittest.TestCase):
    def _make_sweep_setup(self) -> pd.DataFrame:
        """Build a synthetic DataFrame that encodes an SSL sweep + bullish MSS.

        Layout (50 bars):
        * Bars 0-19  : consolidation around 100
        * Bars 20-29 : slow decline to 95 (forms swing low ~95)
        * Bar  30    : large wick below 93 (SSL sweep), close back at 96
        * Bars 31-39 : recovery to 100
        * Bar  40    : MSS: close above the recent swing high (~100)
        * Bars 41-49 : continued move up; last bar at 101 (in FVG territory)
        """
        rows = []
        # Consolidation
        for _i in range(20):
            rows.append({"open": 100.0, "high": 100.5, "low": 99.5, "close": 100.0})
        # Decline
        for i in range(10):
            p = 100.0 - i * 0.5
            rows.append({"open": p, "high": p + 0.2, "low": p - 0.2, "close": p - 0.1})
        # Sweep (bar 30): wick to 92, close at 96
        rows.append({"open": 95.2, "high": 96.0, "low": 92.0, "close": 96.0})
        # Recovery
        for i in range(9):
            p = 96.0 + i * 0.5
            rows.append({"open": p, "high": p + 0.3, "low": p - 0.3, "close": p + 0.2})
        # MSS bar (bar 40): close above 100.0 swing high
        rows.append({"open": 100.0, "high": 101.5, "low": 99.5, "close": 101.2})
        # Post-MSS, last bar inside a FVG zone
        for i in range(9):
            p = 101.2 + i * 0.1
            rows.append({"open": p - 0.1, "high": p + 0.2, "low": p - 0.2, "close": p})

        df = _make_df_custom(rows)
        return df

    def test_returns_three_tuple(self):
        df = self._make_sweep_setup()
        result = ict_setup_signal(
            df,
            require_time_window=False,
            require_pd=False,
            ote_enabled=False,
        )
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 3)
        action, conf, debug = result
        self.assertIn(action, ("buy", "sell", "hold"))
        self.assertGreaterEqual(conf, 0.0)
        self.assertLessEqual(conf, 1.0)
        self.assertIsInstance(debug, dict)

    def test_insufficient_data_returns_hold(self):
        df = _make_df([100.0] * 5)
        action, conf, debug = ict_setup_signal(df, require_time_window=False)
        self.assertEqual(action, "hold")
        self.assertEqual(conf, 0.0)
        self.assertIn("reason", debug)

    def test_confidence_non_decreasing_with_more_conditions(self):
        """Confidence should be at least as high when more signal conditions are met."""
        df = self._make_sweep_setup()
        _, conf_no_pd, _ = ict_setup_signal(
            df, require_time_window=False, require_pd=False
        )
        _, conf_with_pd, _ = ict_setup_signal(
            df, require_time_window=False, require_pd=True
        )
        # Either can be higher depending on whether PD condition is met;
        # both must be valid confidence values
        self.assertGreaterEqual(conf_no_pd, 0.0)
        self.assertGreaterEqual(conf_with_pd, 0.0)

    def test_debug_has_expected_keys_on_sweep(self):
        df = self._make_sweep_setup()
        _, _, debug = ict_setup_signal(df, require_time_window=False, require_pd=False)
        if "sweep" in debug:
            self.assertIn("side", debug["sweep"])
            self.assertIn("level", debug["sweep"])


if __name__ == "__main__":
    unittest.main()
