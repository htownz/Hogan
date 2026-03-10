"""Multi-timeframe ensemble for Hogan.

Combines three timeframes into a single directional decision:

  daily  →  directional gate  (only trade in the direction of the daily trend)
  1h     →  primary signal    (Optuna-optimised entry/exit logic)
  30m    →  entry timing      (momentum confirmation for entry quality)

Architecture
------------
The daily candles provide a trend bias: bullish / bearish / neutral.
The 1h signal is the main strategy output (buy / sell / hold).
The 30m candles confirm momentum alignment before entry.

Ensemble rules
--------------
1. If daily bias OPPOSES the 1h signal → filter to hold.
2. If 30m does NOT confirm → scale confidence to *unconfirmed_scale* (default 0.6).
3. If daily bias ALIGNS and 30m confirms → full confidence.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class MTFBias:
    """Result of the multi-timeframe ensemble evaluation."""
    daily_trend: str       # "bullish" | "bearish" | "neutral"
    hourly_action: str     # raw 1h signal: "buy" | "sell" | "hold"
    m30_confirms: bool     # does 30m confirm the 1h direction?
    final_action: str      # ensemble output: "buy" | "sell" | "hold"
    confidence_mult: float # 0.0–1.0 multiplier applied to position sizing


# ---------------------------------------------------------------------------
# Daily trend bias
# ---------------------------------------------------------------------------

def daily_trend_bias(
    daily_candles: pd.DataFrame,
    fast_period: int = 10,
    slow_period: int = 30,
    crossover_lookback: int = 3,
) -> str:
    """Classify the daily trend as bullish / bearish / neutral.

    Uses a simple fast/slow MA crossover on daily closes.
    If the crossover occurred within *crossover_lookback* bars,
    the bias is neutral (transition zone).
    """
    if daily_candles is None or len(daily_candles) < slow_period + crossover_lookback:
        return "neutral"

    close = daily_candles["close"].astype(float)
    fast_ma = close.rolling(fast_period).mean()
    slow_ma = close.rolling(slow_period).mean()

    spread = fast_ma - slow_ma
    current_spread = float(spread.iloc[-1])

    recent = spread.iloc[-crossover_lookback:]
    signs = np.sign(recent.dropna().values)
    if len(signs) < 2:
        return "neutral"

    if not np.all(signs == signs[0]):
        return "neutral"

    if current_spread > 0:
        return "bullish"
    elif current_spread < 0:
        return "bearish"
    return "neutral"


# ---------------------------------------------------------------------------
# 30-minute confirmation
# ---------------------------------------------------------------------------

def m30_confirms(
    m30_candles: pd.DataFrame,
    action: str,
    fast_period: int = 8,
    rsi_period: int = 14,
) -> bool:
    """Check if 30m momentum aligns with the proposed action.

    For a buy: price above 30m fast MA AND RSI > 40  (not oversold reversal).
    For a sell: price below 30m fast MA AND RSI < 60  (not overbought reversal).
    """
    if m30_candles is None or len(m30_candles) < max(fast_period, rsi_period) + 1:
        return True  # not enough data → don't block the trade

    close = m30_candles["close"].astype(float)
    price = float(close.iloc[-1])
    fast_ma = float(close.rolling(fast_period).mean().iloc[-1])

    delta = close.diff()
    gain = delta.clip(lower=0).ewm(span=rsi_period, adjust=False).mean()
    loss = (-delta.clip(upper=0)).ewm(span=rsi_period, adjust=False).mean()
    rs = gain / loss.clip(lower=1e-9)
    rsi = float(100 - 100 / (1 + rs.iloc[-1]))

    if action == "buy":
        return price > fast_ma and rsi > 40
    elif action == "sell":
        return price < fast_ma and rsi < 60
    return True


# ---------------------------------------------------------------------------
# Ensemble
# ---------------------------------------------------------------------------

def evaluate_mtf(
    daily_candles: pd.DataFrame | None,
    hourly_action: str,
    m30_candles: pd.DataFrame | None,
    unconfirmed_scale: float = 0.60,
) -> MTFBias:
    """Run the full multi-timeframe ensemble.

    Parameters
    ----------
    daily_candles : DataFrame or None
        Daily OHLCV.  ``None`` disables the daily filter.
    hourly_action : str
        Raw 1h strategy action: ``"buy"`` / ``"sell"`` / ``"hold"``.
    m30_candles : DataFrame or None
        30-minute OHLCV.  ``None`` disables 30m confirmation.
    unconfirmed_scale : float
        Confidence multiplier when 30m does not confirm (default 0.60).

    Returns
    -------
    MTFBias
        Ensemble result with ``final_action`` and ``confidence_mult``.
    """
    if hourly_action == "hold":
        return MTFBias(
            daily_trend="neutral",
            hourly_action="hold",
            m30_confirms=False,
            final_action="hold",
            confidence_mult=0.0,
        )

    bias = daily_trend_bias(daily_candles) if daily_candles is not None else "neutral"

    # Gate: daily bias must not oppose the hourly signal
    if bias == "bearish" and hourly_action == "buy":
        logger.info("MTF FILTER: daily bearish opposes 1h buy → hold")
        return MTFBias(
            daily_trend=bias,
            hourly_action=hourly_action,
            m30_confirms=False,
            final_action="hold",
            confidence_mult=0.0,
        )
    if bias == "bullish" and hourly_action == "sell":
        logger.info("MTF FILTER: daily bullish opposes 1h sell → hold")
        return MTFBias(
            daily_trend=bias,
            hourly_action=hourly_action,
            m30_confirms=False,
            final_action="hold",
            confidence_mult=0.0,
        )

    confirmed = m30_confirms(m30_candles, hourly_action) if m30_candles is not None else True
    confidence = 1.0 if confirmed else unconfirmed_scale

    if bias != "neutral" and confirmed:
        logger.info(
            "MTF ALIGNED: daily=%s 1h=%s 30m=confirmed → full confidence",
            bias, hourly_action,
        )
    elif not confirmed:
        logger.info(
            "MTF PARTIAL: daily=%s 1h=%s 30m=NOT confirmed → confidence=%.2f",
            bias, hourly_action, confidence,
        )

    return MTFBias(
        daily_trend=bias,
        hourly_action=hourly_action,
        m30_confirms=confirmed,
        final_action=hourly_action,
        confidence_mult=confidence,
    )
