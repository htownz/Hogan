"""Forecast head — directional probability over multiple horizons.

Produces structured forward-looking estimates that the policy layer
uses to decide *whether there is edge*, independent of the technical
signal's crossover/trend logic.

Outputs
-------
ForecastResult
    direction_prob : dict[str, float]
        Probability of positive return at 4h, 12h, 24h horizons.
    expected_return : dict[str, float]
        Expected return (%) at each horizon (simple historical analog).
    trend_persistence : float
        Probability the current trend direction continues (0-1).
    confidence : float
        Overall confidence in the forecast (0-1), based on feature coverage.

Architecture
------------
Uses the existing ``_feature_frame`` from ``ml.py`` to compute a rich
feature vector, then applies lightweight statistical models:

1. **Logistic trend persistence**: rolling MA slope + ADX -> trend continuation prob
2. **Volatility-scaled return expectation**: recent returns + vol regime -> expected move
3. **Multi-horizon probability**: feature-based directional estimates

These are intentionally simple and robust — no deep learning, no overfit.
The forecast head should be *calibrated* rather than *ambitious*.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from hogan_bot.indicators import compute_atr

logger = logging.getLogger(__name__)


@dataclass
class ForecastResult:
    """Structured forecast output for the policy layer."""
    direction_prob: dict[str, float] = field(default_factory=dict)
    expected_return: dict[str, float] = field(default_factory=dict)
    trend_persistence: float = 0.5
    confidence: float = 0.0

    @property
    def bullish_4h(self) -> float:
        return self.direction_prob.get("4h", 0.5)

    @property
    def bullish_12h(self) -> float:
        return self.direction_prob.get("12h", 0.5)

    @property
    def bullish_24h(self) -> float:
        return self.direction_prob.get("24h", 0.5)

    def summary(self) -> str:
        parts = []
        for h in ("4h", "12h", "24h"):
            p = self.direction_prob.get(h, 0.5)
            er = self.expected_return.get(h, 0.0)
            parts.append(f"{h}:{p:.0%}up/{er:+.2f}%")
        parts.append(f"persist={self.trend_persistence:.0%}")
        return " ".join(parts)


def _safe_sigmoid(x: float) -> float:
    """Numerically stable sigmoid."""
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    else:
        z = math.exp(x)
        return z / (1.0 + z)


def compute_forecast(candles: pd.DataFrame) -> ForecastResult:
    """Compute multi-horizon directional forecast from OHLCV data.

    Uses only causal features (no look-ahead). Designed for 1h candles
    but works on any timeframe — horizons are in bar-counts.
    """
    min_bars = 80
    if candles is None or len(candles) < min_bars:
        return ForecastResult(confidence=0.0)

    close = candles["close"].astype(float)
    high = candles["high"].astype(float)
    low = candles["low"].astype(float)
    volume = candles["volume"].astype(float)

    # --- Feature extraction (all causal) ---

    # Momentum features
    ret_1 = close.pct_change(1)
    ret_4 = close.pct_change(4)
    ret_12 = close.pct_change(12)
    ret_24 = close.pct_change(24)

    # MA trend
    ma_fast = close.rolling(12).mean()
    ma_slow = close.rolling(48).mean()
    ma_spread = ((ma_fast - ma_slow) / ma_slow.clip(lower=1e-9)).iloc[-1]

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0.0).ewm(com=13, min_periods=14, adjust=False).mean()
    loss = (-delta.clip(upper=0.0)).ewm(com=13, min_periods=14, adjust=False).mean()
    rs = gain / loss.clip(lower=1e-9)
    rsi = (100.0 - (100.0 / (1.0 + rs))).iloc[-1] / 100.0  # [0, 1]

    # Volatility
    vol_20 = float(ret_1.rolling(20).std().iloc[-1])
    vol_50 = float(ret_1.rolling(50).std().iloc[-1]) or vol_20
    vol_regime = vol_20 / max(vol_50, 1e-9)

    # ADX for trend strength
    atr = compute_atr(candles, window=14)
    atr_pct = float(atr.iloc[-1] / max(close.iloc[-1], 1e-9))

    # Volume profile
    vol_avg = float(volume.rolling(20).mean().iloc[-1])
    vol_ratio = float(volume.iloc[-1] / max(vol_avg, 1e-9))

    # Bollinger %B
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    bb_upper = bb_mid + 2.0 * bb_std
    bb_lower = bb_mid - 2.0 * bb_std
    bb_pct_b = float((close.iloc[-1] - bb_lower.iloc[-1]) / max(bb_upper.iloc[-1] - bb_lower.iloc[-1], 1e-9))

    # --- Trend persistence model ---
    # Higher ADX + consistent MA spread + moderate RSI = higher persistence
    trend_dir = 1.0 if ma_spread > 0 else -1.0

    # Logit-based persistence: positive features -> trend continues
    persistence_logit = (
        2.0 * abs(ma_spread) * 100.0        # stronger trend = more persistent
        + 0.5 * max(0, atr_pct * 100.0 - 1.0)  # some vol helps trends
        - 1.0 * vol_regime                    # excessive vol regime hurts
        + 0.3 * (0.5 - abs(rsi - 0.5)) * 2.0  # extreme RSI = reversal risk
    )
    trend_persistence = _safe_sigmoid(persistence_logit)

    # --- Multi-horizon directional probability ---
    # Combine momentum, mean-reversion, and trend features
    direction_prob = {}
    expected_return = {}

    for horizon_name, horizon_bars in [("4h", 4), ("12h", 12), ("24h", 24)]:
        # Momentum component: recent returns predict short-term direction
        recent_ret = float(close.pct_change(min(horizon_bars, len(close) - 1)).iloc[-1])

        # Mean-reversion component: extreme BB%B -> reversion pressure
        mr_pressure = -(bb_pct_b - 0.5) * 0.3  # oversold = bullish, overbought = bearish

        # Trend component: MA spread aligned with price direction
        trend_component = ma_spread * 20.0  # scale to comparable range

        # Volume confirmation: high volume amplifies the signal
        vol_boost = min(0.3, max(-0.1, (vol_ratio - 1.0) * 0.15))

        # Horizon-dependent weighting: short-term = momentum, long-term = trend
        if horizon_bars <= 4:
            logit = 0.5 * recent_ret * 50.0 + 0.3 * mr_pressure + 0.2 * trend_component
        elif horizon_bars <= 12:
            logit = 0.3 * recent_ret * 30.0 + 0.25 * mr_pressure + 0.45 * trend_component
        else:
            logit = 0.15 * recent_ret * 20.0 + 0.2 * mr_pressure + 0.65 * trend_component

        logit += vol_boost

        prob_up = _safe_sigmoid(logit)
        direction_prob[horizon_name] = round(prob_up, 4)

        # Expected return: directional probability * typical move size
        typical_move = vol_20 * math.sqrt(horizon_bars) * 100.0  # annualize
        expected_return[horizon_name] = round((2.0 * prob_up - 1.0) * typical_move, 4)

    # --- Confidence: based on data coverage and feature stability ---
    feature_count = 6  # momentum, MA, RSI, vol, BB, volume
    nan_count = sum(1 for v in [ma_spread, rsi, vol_20, bb_pct_b, vol_ratio, atr_pct]
                    if v != v)  # NaN check
    data_confidence = (feature_count - nan_count) / feature_count
    stability = max(0.0, 1.0 - abs(vol_regime - 1.0))  # stable vol = higher confidence
    confidence = round(data_confidence * (0.5 + 0.5 * stability), 4)

    return ForecastResult(
        direction_prob=direction_prob,
        expected_return=expected_return,
        trend_persistence=round(trend_persistence, 4),
        confidence=confidence,
    )
