"""Market regime detection for Hogan.

Classifies the current market into one of four regimes based on ADX strength,
ATR percentile rank, and trend direction.  Each regime has a corresponding
set of threshold overrides that the bot applies dynamically each iteration.

Regimes
-------
trending_up
    ADX ≥ threshold AND fast MA > slow MA.
    Strategy: ride the trend, wide stops, loose volume gate, easy longs.

trending_down
    ADX ≥ threshold AND fast MA < slow MA.
    Strategy: ride the trend, wide stops, easy shorts, hard longs.

ranging
    ADX < threshold AND ATR percentile < volatile threshold.
    Strategy: mean-reversion, tight targets, moderate thresholds.

volatile
    ATR percentile ≥ volatile threshold (regardless of ADX).
    Strategy: wait for high-volume confirmation, smaller positions, moderate R/R.

Usage
-----
    from hogan_bot.regime import detect_regime, effective_thresholds

    state = detect_regime(candles)
    eff = effective_thresholds(state, config)
    # Use eff["ml_buy_threshold"], eff["volume_threshold"], etc. for this bar
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Data structure
# ---------------------------------------------------------------------------

@dataclass
class RegimeState:
    """Snapshot of market regime at the current bar."""
    regime: str           # "trending_up" | "trending_down" | "ranging" | "volatile"
    adx: float            # ADX value (0–100)
    atr_pct_rank: float   # ATR percentile rank vs *atr_lookback* bars (0.0–1.0)
    trend_direction: int  # +1 up, -1 down, 0 flat
    ma_spread: float      # (fast_ma − slow_ma) / slow_ma — positive = uptrend
    confidence: float     # 0.0–1.0 composite confidence in the regime label
    btc_dominance: float | None = None  # latest BTC dominance % from DB (if available)
    fear_greed: float | None = None     # Fear & Greed index value (0–100, if available)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _wilder_adx(
    high: pd.Series,
    low: pd.Series,
    close: pd.Series,
    period: int = 14,
) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return (ADX, +DI, -DI) using Wilder's smoothing."""
    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)

    up_move = high.diff()
    down_move = -(low.diff())
    plus_dm = pd.Series(
        np.where((up_move > down_move) & (up_move > 0), up_move, 0.0),
        index=close.index,
    )
    minus_dm = pd.Series(
        np.where((down_move > up_move) & (down_move > 0), down_move, 0.0),
        index=close.index,
    )

    alpha = 1.0 / period
    atr14 = tr.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    plus_di = (
        100.0
        * plus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        / atr14.clip(lower=1e-9)
    )
    minus_di = (
        100.0
        * minus_dm.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
        / atr14.clip(lower=1e-9)
    )

    dx = (plus_di - minus_di).abs() / (plus_di + minus_di).clip(lower=1e-9) * 100.0
    adx = dx.ewm(alpha=alpha, min_periods=period, adjust=False).mean()
    return adx, plus_di, minus_di


def _atr_percentile_rank(candles: pd.DataFrame, lookback: int = 100) -> float:
    """Return the current ATR's percentile rank vs the last *lookback* bars.

    Uses *percentage* ATR (ATR / close) so the ranking is not inflated by
    price growth in a trending market.

    0.0 = current ATR% is the lowest it has been.
    1.0 = current ATR% is the highest it has been (spike in volatility).
    """
    high = candles["high"].astype(float)
    low = candles["low"].astype(float)
    close = candles["close"].astype(float)

    tr = pd.concat(
        [high - low, (high - close.shift()).abs(), (low - close.shift()).abs()],
        axis=1,
    ).max(axis=1)
    # Normalize by close price to get a percentage — prevents trend-induced inflation
    atr_pct = tr.ewm(span=14, adjust=False).mean() / close.clip(lower=1e-9)

    window = atr_pct.iloc[-lookback:]
    current = float(atr_pct.iloc[-1])
    if window.empty or window.std() < 1e-9:
        return 0.5
    rank = float((window < current).mean())
    return rank


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

_REGIME_HISTORY: list[str] = []
_REGIME_HYSTERESIS_BARS: int = 3


def reset_regime_history() -> None:
    """Clear the hysteresis buffer. Useful for tests and backtest initialization."""
    _REGIME_HISTORY.clear()


def detect_regime(
    candles: pd.DataFrame,
    adx_period: int = 14,
    atr_lookback: int = 100,
    fast_ma_period: int = 20,
    slow_ma_period: int = 50,
    adx_trending_threshold: float = 25.0,
    adx_ranging_threshold: float = 20.0,
    atr_volatile_pct: float = 0.80,
    btc_dominance: float | None = None,
    fear_greed: float | None = None,
    hysteresis_bars: int = _REGIME_HYSTERESIS_BARS,
) -> RegimeState:
    """Classify the current market regime from OHLCV candles.

    Parameters
    ----------
    candles:
        OHLCV DataFrame with at minimum 80 rows for reliable ADX.
    adx_period:
        Wilder smoothing window for ADX (default 14).
    atr_lookback:
        Number of bars to rank current ATR against (default 100).
    fast_ma_period / slow_ma_period:
        Period for the two MAs used to determine trend direction.
    adx_trending_threshold:
        ADX value above which the market is considered trending (default 25).
    adx_ranging_threshold:
        ADX value below which the market is considered ranging (default 20).
    atr_volatile_pct:
        ATR percentile rank (0–1) above which the market is considered volatile
        regardless of ADX (default 0.80 = top 20% of ATR readings).
    btc_dominance / fear_greed:
        Optional signals from external DB — used to boost regime confidence.
    """
    close = candles["close"].astype(float)
    high = candles["high"].astype(float)
    low = candles["low"].astype(float)

    # --- ADX ---
    adx_series, plus_di, minus_di = _wilder_adx(high, low, close, period=adx_period)
    adx_val = float(adx_series.iloc[-1]) if not adx_series.isna().iloc[-1] else 20.0
    adx_val = max(0.0, min(100.0, adx_val))

    # --- Trend direction via MA crossover ---
    fast_ma = close.rolling(fast_ma_period).mean().iloc[-1]
    slow_ma = close.rolling(slow_ma_period).mean().iloc[-1]
    ma_spread = 0.0
    if slow_ma and slow_ma > 0:
        ma_spread = float((fast_ma - slow_ma) / slow_ma)

    if plus_di.iloc[-1] > minus_di.iloc[-1]:
        trend_dir = 1
    elif minus_di.iloc[-1] > plus_di.iloc[-1]:
        trend_dir = -1
    else:
        trend_dir = 0

    # --- ATR percentile rank ---
    atr_rank = _atr_percentile_rank(candles, lookback=atr_lookback)

    # --- Regime classification ---
    # Volatile overrides everything when ATR is in the top *atr_volatile_pct* tier
    if atr_rank >= atr_volatile_pct:
        regime = "volatile"
        confidence = 0.5 + 0.5 * (atr_rank - atr_volatile_pct) / (1.0 - atr_volatile_pct + 1e-9)
    elif adx_val >= adx_trending_threshold:
        regime = "trending_up" if trend_dir >= 0 else "trending_down"
        # Confidence scales with how far ADX is above the threshold
        confidence = 0.5 + min(0.5, (adx_val - adx_trending_threshold) / 50.0)
    else:
        regime = "ranging"
        # Confidence scales with how far ADX is below the threshold
        confidence = 0.5 + min(0.5, (adx_ranging_threshold - min(adx_val, adx_ranging_threshold)) / adx_ranging_threshold)

    # --- Optional: boost/dampen confidence from macro signals ---
    if btc_dominance is not None:
        # Dominance > 60 → BTC-led trend, signals are more reliable for BTC
        if btc_dominance > 60 and regime in ("trending_up", "trending_down"):
            confidence = min(1.0, confidence + 0.05)
        # Dominance < 45 → altcoin season, signals less reliable
        elif btc_dominance < 45:
            confidence = max(0.3, confidence - 0.05)

    if fear_greed is not None:
        # Extreme fear (<20) or extreme greed (>80) may confirm a trend
        if fear_greed < 20 and regime == "trending_down":
            confidence = min(1.0, confidence + 0.05)
        elif fear_greed > 80 and regime == "trending_up":
            confidence = min(1.0, confidence + 0.05)
        # Fear & Greed in neutral zone (40–60) suggests ranging
        elif 40 <= fear_greed <= 60 and regime == "ranging":
            confidence = min(1.0, confidence + 0.05)

    # Hysteresis: require N consecutive bars of the new regime before switching
    _REGIME_HISTORY.append(regime)
    if len(_REGIME_HISTORY) > hysteresis_bars * 3:
        del _REGIME_HISTORY[:-hysteresis_bars * 3]

    if len(_REGIME_HISTORY) >= hysteresis_bars + 1:
        prev_regime = _REGIME_HISTORY[-(hysteresis_bars + 1)]
        recent = _REGIME_HISTORY[-hysteresis_bars:]
        if regime != prev_regime and not all(r == regime for r in recent):
            regime = prev_regime
            confidence *= 0.8

    return RegimeState(
        regime=regime,
        adx=adx_val,
        atr_pct_rank=atr_rank,
        trend_direction=trend_dir,
        ma_spread=ma_spread,
        confidence=float(confidence),
        btc_dominance=btc_dominance,
        fear_greed=fear_greed,
    )


# ---------------------------------------------------------------------------
# Threshold overrides per regime
# ---------------------------------------------------------------------------
# Strategy parameters (volume_threshold, trailing_stop, take_profit) use
# *multipliers* against the per-symbol Optuna base so that each instrument
# keeps its optimised proportions.  ML thresholds remain absolute because
# they operate on a fixed 0–1 probability scale.

def _build_regime_overrides() -> dict[str, dict]:
    """Build _REGIME_OVERRIDES from DEFAULT_REGIME_CONFIGS for backward compat."""
    from hogan_bot.config import DEFAULT_REGIME_CONFIGS
    out: dict[str, dict] = {}
    for regime, rc in DEFAULT_REGIME_CONFIGS.items():
        out[regime] = {
            "volume_threshold_mult": rc.volume_threshold_mult,
            "ml_buy_threshold": rc.ml_buy_threshold,
            "ml_sell_threshold": rc.ml_sell_threshold,
            "trailing_stop_mult": rc.trailing_stop_mult,
            "take_profit_mult": rc.take_profit_mult,
            "position_scale": rc.position_scale,
            "allow_longs": rc.allow_longs,
            "allow_shorts": rc.allow_shorts,
            "long_size_scale": rc.long_size_scale,
            "short_size_scale": rc.short_size_scale,
        }
    return out


_REGIME_OVERRIDES: dict[str, dict[str, float]] = _build_regime_overrides()


def effective_thresholds(
    state: RegimeState,
    config,
    min_confidence: float = 0.50,
) -> dict[str, float]:
    """Return the effective threshold values for this regime.

    Strategy-specific parameters (``volume_threshold``, ``trailing_stop_pct``,
    ``take_profit_pct``) are *scaled* from the per-symbol Optuna base via
    multipliers, so each instrument keeps its optimised proportions.
    ML thresholds are absolute (fixed probability scale).

    When regime ``confidence`` is below *min_confidence* or regime detection
    is disabled, returns the raw config defaults unchanged.

    The returned dict always contains:
    ``volume_threshold``, ``ml_buy_threshold``, ``ml_sell_threshold``,
    ``trailing_stop_pct``, ``take_profit_pct``, ``position_scale``.
    """
    defaults = {
        "volume_threshold":  config.volume_threshold,
        "ml_buy_threshold":  config.ml_buy_threshold,
        "ml_sell_threshold": config.ml_sell_threshold,
        "trailing_stop_pct": config.trailing_stop_pct,
        "take_profit_pct":   config.take_profit_pct,
        "position_scale":    1.0,
        "allow_longs":       True,
        "allow_shorts":      True,
        "long_size_scale":   1.0,
        "short_size_scale":  1.0,
    }

    if not getattr(config, "use_regime_detection", False):
        return defaults

    if state.confidence < min_confidence:
        return defaults

    overrides = _REGIME_OVERRIDES.get(state.regime, {})
    if not overrides:
        return defaults

    result = defaults.copy()
    # Multiplier-based keys: scale from the per-symbol base
    if "volume_threshold_mult" in overrides:
        result["volume_threshold"] = defaults["volume_threshold"] * overrides["volume_threshold_mult"]
    if "trailing_stop_mult" in overrides:
        result["trailing_stop_pct"] = defaults["trailing_stop_pct"] * overrides["trailing_stop_mult"]
    if "take_profit_mult" in overrides:
        result["take_profit_pct"] = defaults["take_profit_pct"] * overrides["take_profit_mult"]
    # Absolute keys: ML thresholds, position scale, side participation
    for key in ("ml_buy_threshold", "ml_sell_threshold", "position_scale",
                "allow_longs", "allow_shorts", "long_size_scale", "short_size_scale"):
        if key in overrides:
            result[key] = overrides[key]
    return result


# ---------------------------------------------------------------------------
# DB helpers — load latest macro signals for regime detection
# ---------------------------------------------------------------------------

def load_regime_signals(conn) -> dict[str, float | None]:
    """Load the latest BTC dominance and Fear & Greed from the SQLite DB.

    Returns a dict with keys ``btc_dominance`` and ``fear_greed``.
    Values are ``None`` if data is not available.
    """
    signals: dict[str, float | None] = {"btc_dominance": None, "fear_greed": None}
    try:
        cur = conn.cursor()
        # BTC dominance — stored by fetch_cmc.py as "cg_btc_dominance" or "cmc_btc_dominance"
        for metric in ("cg_btc_dominance", "cmc_btc_dominance"):
            cur.execute(
                "SELECT value FROM onchain_metrics WHERE metric=? ORDER BY date DESC LIMIT 1",
                (metric,),
            )
            row = cur.fetchone()
            if row:
                signals["btc_dominance"] = float(row[0])
                break

        # Fear & Greed — stored by fetch_feargreed.py as "fear_greed_value"
        cur.execute(
            "SELECT value FROM onchain_metrics WHERE metric='fear_greed_value' ORDER BY date DESC LIMIT 1"
        )
        row = cur.fetchone()
        if row:
            signals["fear_greed"] = float(row[0])
    except Exception:  # noqa: BLE001
        pass
    return signals
