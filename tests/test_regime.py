"""Tests for hogan_bot.regime — market regime detection."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from hogan_bot.regime import (
    RegimeState,
    _atr_percentile_rank,
    _wilder_adx,
    detect_regime,
    effective_thresholds,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _candles(n: int = 200, trend: str = "up") -> pd.DataFrame:
    """Synthetic OHLCV with a clear up / down / flat trend."""
    np.random.seed(0)
    prices = [100.0]
    for _ in range(n - 1):
        if trend == "up":
            prices.append(prices[-1] * (1 + np.random.normal(0.002, 0.005)))
        elif trend == "down":
            prices.append(prices[-1] * (1 + np.random.normal(-0.002, 0.005)))
        else:  # flat
            prices.append(prices[-1] * (1 + np.random.normal(0.0, 0.003)))

    closes = np.array(prices)
    return pd.DataFrame({
        "open":   closes * (1 + np.random.normal(0, 0.001, n)),
        "high":   closes * (1 + abs(np.random.normal(0, 0.004, n))),
        "low":    closes * (1 - abs(np.random.normal(0, 0.004, n))),
        "close":  closes,
        "volume": np.random.uniform(1e6, 2e6, n),
        "ts_ms":  np.arange(n) * 300_000,
    })


def _volatile_candles(n: int = 200) -> pd.DataFrame:
    """Candles with a strong spike in volatility at the end."""
    df = _candles(n, trend="flat")
    # Make the last 20 bars very wide ranges
    df.loc[df.index[-20:], "high"] = df["close"].iloc[-20:] * 1.05
    df.loc[df.index[-20:], "low"] = df["close"].iloc[-20:] * 0.95
    return df


class _FakeConfig:
    use_regime_detection = True
    volume_threshold = 0.30
    ml_buy_threshold = 0.52
    ml_sell_threshold = 0.50
    trailing_stop_pct = 0.03
    take_profit_pct = 0.08
    regime_adx_trending = 25.0
    regime_adx_ranging = 20.0
    regime_atr_volatile_pct = 0.80


# ---------------------------------------------------------------------------
# Unit tests
# ---------------------------------------------------------------------------

class TestWilderADX:
    def test_adx_returns_series(self):
        candles = _candles(200, "up")
        adx, plus_di, minus_di = _wilder_adx(candles["high"], candles["low"], candles["close"])
        assert len(adx) == len(candles)

    def test_adx_range(self):
        candles = _candles(200, "up")
        adx, _, _ = _wilder_adx(candles["high"], candles["low"], candles["close"])
        valid = adx.dropna()
        assert (valid >= 0).all() and (valid <= 100).all()


class TestATRPercentile:
    def test_rank_in_unit_interval(self):
        rank = _atr_percentile_rank(_candles(200))
        assert 0.0 <= rank <= 1.0

    def test_high_volatility_gives_high_rank(self):
        rank = _atr_percentile_rank(_volatile_candles(200))
        assert rank >= 0.70, f"Expected high ATR rank for volatile candles, got {rank:.2f}"


class TestDetectRegime:
    def test_returns_regime_state(self):
        state = detect_regime(_candles(200))
        assert isinstance(state, RegimeState)
        assert state.regime in ("trending_up", "trending_down", "ranging", "volatile")

    def test_uptrend_detected(self):
        """Strong uptrend → trending_up or possibly volatile (ATR can spike in trends)."""
        # Use a very smooth uptrend to avoid volatile classification
        np.random.seed(42)
        n = 200
        prices = [100.0 + i * 0.5 for i in range(n)]
        df = pd.DataFrame({
            "open":   [p * 0.999 for p in prices],
            "high":   [p * 1.003 for p in prices],
            "low":    [p * 0.997 for p in prices],
            "close":  prices,
            "volume": [1e6] * n,
            "ts_ms":  list(range(n)),
        })
        state = detect_regime(df, adx_trending_threshold=20.0)
        assert state.regime == "trending_up", f"Expected trending_up, got {state.regime}"
        assert state.trend_direction == 1

    def test_downtrend_detected(self):
        # Use constant-percentage moves so ATR/close stays flat (no volatile trigger)
        np.random.seed(42)
        n = 200
        prices = [100.0 * (0.998 ** i) for i in range(n)]  # -0.2%/bar
        df = pd.DataFrame({
            "open":   [p * 1.001 for p in prices],
            "high":   [p * 1.003 for p in prices],
            "low":    [p * 0.997 for p in prices],
            "close":  prices,
            "volume": [1e6] * n,
            "ts_ms":  list(range(n)),
        })
        state = detect_regime(df, adx_trending_threshold=20.0)
        assert state.regime == "trending_down", f"Expected trending_down, got {state.regime}"
        assert state.trend_direction == -1

    def test_ranging_market(self):
        """Flat market with low noise → low ADX → ranging."""
        np.random.seed(0)
        n = 200
        # Oscillate around 100 with very small moves
        prices = [100.0 + np.sin(i * 0.3) * 0.5 for i in range(n)]
        df = pd.DataFrame({
            "open":   [p - 0.05 for p in prices],
            "high":   [p + 0.1 for p in prices],
            "low":    [p - 0.1 for p in prices],
            "close":  prices,
            "volume": [1e6] * n,
            "ts_ms":  list(range(n)),
        })
        state = detect_regime(df)
        assert state.regime == "ranging", f"Expected ranging, got {state.regime}"

    def test_volatile_candles(self):
        state = detect_regime(_volatile_candles(200))
        assert state.regime == "volatile"
        assert state.atr_pct_rank >= 0.80

    def test_confidence_in_unit_interval(self):
        for trend in ("up", "down", "flat"):
            state = detect_regime(_candles(200, trend))
            assert 0.0 <= state.confidence <= 1.0

    def test_btc_dominance_boosts_confidence(self):
        candles = _candles(200, "up")
        state_no_dom = detect_regime(candles, adx_trending_threshold=20.0)
        state_with_dom = detect_regime(
            candles, adx_trending_threshold=20.0, btc_dominance=65.0
        )
        if state_no_dom.regime in ("trending_up", "trending_down"):
            assert state_with_dom.confidence >= state_no_dom.confidence


class TestEffectiveThresholds:
    def test_disabled_returns_defaults(self):
        cfg = _FakeConfig()
        cfg.use_regime_detection = False
        state = detect_regime(_candles(200))
        eff = effective_thresholds(state, cfg)
        assert eff["ml_buy_threshold"] == cfg.ml_buy_threshold
        assert eff["volume_threshold"] == cfg.volume_threshold

    def test_trending_up_loosens_vol_threshold(self):
        cfg = _FakeConfig()
        np.random.seed(42)
        n = 200
        prices = [100.0 + i * 0.5 for i in range(n)]
        df = pd.DataFrame({
            "open": [p * 0.999 for p in prices], "high": [p * 1.003 for p in prices],
            "low": [p * 0.997 for p in prices], "close": prices,
            "volume": [1e6] * n, "ts_ms": list(range(n)),
        })
        state = detect_regime(df, adx_trending_threshold=20.0)
        eff = effective_thresholds(state, cfg)
        if state.regime == "trending_up":
            assert eff["volume_threshold"] == 1.2

    def test_volatile_halves_position_scale(self):
        cfg = _FakeConfig()
        state = detect_regime(_volatile_candles(200))
        assert state.regime == "volatile"
        eff = effective_thresholds(state, cfg)
        assert eff["position_scale"] == 0.50

    def test_low_confidence_returns_defaults(self):
        """When confidence is below threshold, keep config defaults."""
        cfg = _FakeConfig()
        state = RegimeState(
            regime="trending_up", adx=25.0, atr_pct_rank=0.5,
            trend_direction=1, ma_spread=0.01, confidence=0.3,
        )
        eff = effective_thresholds(state, cfg, min_confidence=0.50)
        assert eff["ml_buy_threshold"] == cfg.ml_buy_threshold
