"""Unit tests for entry families and exit packs."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hogan_bot.exit_packs import (
    ALL_EXIT_PACKS,
    EXIT_PACKS,
    T1_TREND,
    T2_MEAN_REVERT,
    T3_BALANCED,
    ExitPack,
)
from hogan_bot.strategy import StrategySignal
from hogan_bot.strategy_candidates import (
    ENTRY_FAMILIES,
    BollingerSqueezeBreakout,
    DonchianBreakout,
    EMAPullbackTrend,
    RSIPullbackReclaim,
    StrippedBaseline,
    get_entry_family,
)

# ── Helpers ──────────────────────────────────────────────────────────────────

def _make_candles(n: int = 300, trend: str = "up", seed: int = 42) -> pd.DataFrame:
    """Generate synthetic OHLCV candles with a controllable trend."""
    rng = np.random.RandomState(seed)
    base = 100.0
    prices = [base]
    drift = {"up": 0.0005, "down": -0.0005, "flat": 0.0}[trend]
    for _ in range(n - 1):
        ret = drift + rng.normal(0, 0.005)
        prices.append(prices[-1] * (1 + ret))
    close = np.array(prices)
    noise = rng.uniform(0.001, 0.008, size=n)
    high = close * (1 + noise)
    low = close * (1 - noise)
    opn = close * (1 + rng.uniform(-0.003, 0.003, size=n))
    vol = rng.uniform(100, 1000, size=n)
    ts = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "timestamp": ts,
        "open": opn,
        "high": high,
        "low": low,
        "close": close,
        "volume": vol,
    })


def _make_breakout_candles(direction: str = "up", n: int = 250, seed: int = 7) -> pd.DataFrame:
    """Create candles that consolidate then break out."""
    rng = np.random.RandomState(seed)
    flat_n = n - 5
    base = 100.0
    prices = [base + rng.normal(0, 0.1) for _ in range(flat_n)]
    if direction == "up":
        for _ in range(5):
            prices.append(prices[-1] * 1.015)
    else:
        for _ in range(5):
            prices.append(prices[-1] * 0.985)
    close = np.array(prices)
    noise = rng.uniform(0.001, 0.005, size=n)
    high = close * (1 + noise)
    low = close * (1 - noise)
    opn = close * (1 + rng.uniform(-0.002, 0.002, size=n))
    vol = rng.uniform(100, 500, size=n)
    ts = pd.date_range("2024-01-01", periods=n, freq="h")
    return pd.DataFrame({
        "timestamp": ts, "open": opn, "high": high,
        "low": low, "close": close, "volume": vol,
    })


# ── Entry Family: Protocol Compliance ────────────────────────────────────────

class TestEntryFamilyProtocol:
    """Every entry family must return a valid StrategySignal."""

    @pytest.mark.parametrize("key", list(ENTRY_FAMILIES.keys()))
    def test_returns_strategy_signal(self, key):
        family = get_entry_family(key)
        candles = _make_candles(300)
        sig = family.generate_signal(candles)
        assert isinstance(sig, StrategySignal)
        assert sig.action in ("buy", "sell", "hold")
        assert sig.confidence == 1.0
        assert sig.volume_ratio == 1.0
        assert sig.stop_distance_pct > 0

    @pytest.mark.parametrize("key", list(ENTRY_FAMILIES.keys()))
    def test_hold_on_short_data(self, key):
        family = get_entry_family(key)
        candles = _make_candles(5)
        sig = family.generate_signal(candles)
        assert sig.action == "hold"

    @pytest.mark.parametrize("key", list(ENTRY_FAMILIES.keys()))
    def test_has_name(self, key):
        family = get_entry_family(key)
        assert isinstance(family.name, str)
        assert len(family.name) > 0


# ── A. Donchian Breakout ────────────────────────────────────────────────────

class TestDonchianBreakout:
    def test_buy_on_new_high(self):
        candles = _make_breakout_candles("up")
        family = DonchianBreakout(lookback=20)
        sig = family.generate_signal(candles)
        assert sig.action == "buy"

    def test_sell_on_new_low(self):
        candles = _make_breakout_candles("down")
        family = DonchianBreakout(lookback=20)
        sig = family.generate_signal(candles)
        assert sig.action == "sell"

    def test_hold_in_range(self):
        candles = _make_candles(300, trend="flat", seed=99)
        family = DonchianBreakout(lookback=20)
        signals = []
        for i in range(100, 200):
            sig = family.generate_signal(candles.iloc[:i])
            signals.append(sig.action)
        hold_pct = signals.count("hold") / len(signals)
        assert hold_pct > 0.5, f"Expected mostly holds in flat market, got {hold_pct:.0%}"

    def test_symmetric_lookback(self):
        family = DonchianBreakout()
        assert family.lookback == 20


# ── B. RSI Pullback Reclaim ─────────────────────────────────────────────────

class TestRSIPullbackReclaim:
    def test_no_signal_without_trend(self):
        candles = _make_candles(300, trend="flat")
        family = RSIPullbackReclaim()
        sig = family.generate_signal(candles)
        assert sig.action == "hold"

    def test_reclaim_params(self):
        family = RSIPullbackReclaim()
        assert family.long_dip == 30.0
        assert family.long_reclaim == 35.0
        assert family.short_rip == 70.0
        assert family.short_reclaim == 65.0
        assert family.ema_fast == 50
        assert family.ema_slow == 200


# ── C. EMA Pullback Trend ──────────────────────────────────────────────────

class TestEMAPullbackTrend:
    def test_default_params(self):
        family = EMAPullbackTrend()
        assert family.fast == 8
        assert family.medium == 21
        assert family.slow == 55

    def test_returns_valid_signal_on_trend(self):
        candles = _make_candles(300, trend="up")
        family = EMAPullbackTrend()
        sig = family.generate_signal(candles)
        assert sig.action in ("buy", "sell", "hold")


# ── D. Bollinger Squeeze Breakout ──────────────────────────────────────────

class TestBollingerSqueezeBreakout:
    def test_default_params(self):
        family = BollingerSqueezeBreakout()
        assert family.period == 20
        assert family.bb_std == 2.0
        assert family.kc_mult == 1.5

    def test_signal_on_breakout(self):
        candles = _make_breakout_candles("up")
        family = BollingerSqueezeBreakout()
        sig = family.generate_signal(candles)
        assert sig.action in ("buy", "hold")


# ── E. Stripped Baseline (Control) ─────────────────────────────────────────

class TestStrippedBaseline:
    def test_neutral_metadata(self):
        candles = _make_candles(300, trend="up")
        family = StrippedBaseline()
        sig = family.generate_signal(candles)
        assert sig.confidence == 1.0
        assert sig.volume_ratio == 1.0

    def test_labeled_as_control(self):
        family = StrippedBaseline()
        assert family.name == "stripped_baseline"


# ── Exit Packs ──────────────────────────────────────────────────────────────

class TestExitPacks:
    def test_three_packs(self):
        assert len(ALL_EXIT_PACKS) == 3

    def test_registry_keys(self):
        assert set(EXIT_PACKS.keys()) == {"T1_trend", "T2_mean_revert", "T3_balanced"}

    def test_t1_trend_params(self):
        assert T1_TREND.stop_atr_mult == 2.0
        assert T1_TREND.take_profit_atr_mult == 0.0
        assert T1_TREND.trailing_stop_atr_mult == 2.5
        assert T1_TREND.max_hold_hours == 120.0
        assert not T1_TREND.has_take_profit
        assert T1_TREND.has_trailing_stop

    def test_t2_mean_revert_params(self):
        assert T2_MEAN_REVERT.stop_atr_mult == 1.25
        assert T2_MEAN_REVERT.take_profit_atr_mult == 2.0
        assert T2_MEAN_REVERT.trailing_stop_atr_mult == 0.0
        assert T2_MEAN_REVERT.max_hold_hours == 36.0
        assert T2_MEAN_REVERT.has_take_profit
        assert not T2_MEAN_REVERT.has_trailing_stop

    def test_t3_balanced_params(self):
        assert T3_BALANCED.stop_atr_mult == 1.5
        assert T3_BALANCED.take_profit_atr_mult == 3.0
        assert T3_BALANCED.trailing_stop_atr_mult == 0.0
        assert T3_BALANCED.max_hold_hours == 72.0
        assert T3_BALANCED.has_take_profit
        assert not T3_BALANCED.has_trailing_stop

    def test_frozen(self):
        with pytest.raises(AttributeError):
            T1_TREND.stop_atr_mult = 999.0  # type: ignore[misc]

    def test_is_exit_pack_type(self):
        for pack in ALL_EXIT_PACKS:
            assert isinstance(pack, ExitPack)


# ── get_entry_family registry ──────────────────────────────────────────────

class TestRegistry:
    def test_all_keys_resolve(self):
        for key in ENTRY_FAMILIES:
            f = get_entry_family(key)
            assert hasattr(f, "generate_signal")

    def test_unknown_key_raises(self):
        with pytest.raises(KeyError):
            get_entry_family("nonexistent")
