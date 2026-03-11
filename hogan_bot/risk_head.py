"""Risk head — volatility, adverse excursion, stop-hit, and hold-time estimates.

Produces structured risk estimates that the policy layer uses to decide
*whether the size and exits are acceptable*, independent of the signal.

Outputs
-------
RiskEstimate
    expected_vol_pct : float
        Expected annualised volatility over the next 24 bars.
    max_adverse_pct : float
        Estimated max adverse excursion (worst intra-trade drawdown) for a
        typical hold, as a percentage of entry price.
    stop_hit_prob : float
        Probability that the trailing stop (at *stop_pct*) gets hit before
        take-profit, based on historical volatility and stop distance.
    expected_hold_bars : float
        Expected number of bars until exit (stop, TP, or max-hold).
    regime_risk : str
        "low" / "medium" / "high" — composite risk classification.

Architecture
------------
All estimates are derived from recent OHLCV data using well-calibrated
statistical methods (no ML required for the first version):

1. **Volatility**: EWMA of returns, scaled to annualised %.
2. **Max adverse excursion**: historical distribution of worst intra-bar
   drawdowns during similar volatility regimes.
3. **Stop-hit probability**: geometric Brownian motion approximation.
4. **Hold-time**: historical average time-in-trade given vol regime.
"""
from __future__ import annotations

import logging
import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

from hogan_bot.indicators import compute_atr

logger = logging.getLogger(__name__)


@dataclass
class RiskEstimate:
    """Structured risk output for the policy layer."""
    expected_vol_pct: float = 0.0
    max_adverse_pct: float = 0.0
    stop_hit_prob: float = 0.5
    expected_hold_bars: float = 12.0
    regime_risk: str = "medium"
    position_scale: float = 1.0

    def summary(self) -> str:
        return (
            f"vol={self.expected_vol_pct:.1f}% "
            f"mae={self.max_adverse_pct:.2f}% "
            f"stop_hit={self.stop_hit_prob:.0%} "
            f"hold~{self.expected_hold_bars:.0f}bars "
            f"risk={self.regime_risk} "
            f"scale={self.position_scale:.2f}"
        )


def compute_risk(
    candles: pd.DataFrame,
    stop_pct: float = 0.02,
    tp_pct: float = 0.05,
    max_hold_bars: int = 24,
) -> RiskEstimate:
    """Compute risk estimates from OHLCV data.

    Parameters
    ----------
    candles : DataFrame
        OHLCV data (at least 60 bars for reliable estimates).
    stop_pct : float
        Trailing stop distance as decimal (0.02 = 2%).
    tp_pct : float
        Take-profit distance as decimal (0.05 = 5%).
    max_hold_bars : int
        Maximum hold time in bars.
    """
    min_bars = 60
    if candles is None or len(candles) < min_bars:
        return RiskEstimate()

    close = candles["close"].astype(float)
    high = candles["high"].astype(float)
    low = candles["low"].astype(float)

    ret_1 = close.pct_change(1).dropna()

    # --- Expected volatility ---
    # EWMA volatility with half-life of 12 bars, annualised for 1h (8760 bars/yr)
    ewma_vol = float(ret_1.ewm(halflife=12).std().iloc[-1])
    annualised_vol = ewma_vol * math.sqrt(8760) * 100.0  # as percentage

    # Shorter-term vol for risk regime classification
    vol_10 = float(ret_1.tail(10).std()) if len(ret_1) >= 10 else ewma_vol
    vol_50 = float(ret_1.tail(50).std()) if len(ret_1) >= 50 else ewma_vol
    vol_ratio = vol_10 / max(vol_50, 1e-9)

    # --- Max adverse excursion (MAE) ---
    # Compute worst intra-bar drawdown from close-to-low for recent history
    intra_bar_dd = (close - low) / close.clip(lower=1e-9)
    recent_dd = intra_bar_dd.tail(50)
    mae_95th = float(recent_dd.quantile(0.95)) * 100.0  # 95th percentile as %

    # Scale MAE by expected hold time and volatility
    # For multi-bar holds, MAE compounds approximately as sqrt(bars)
    hold_scaling = math.sqrt(min(max_hold_bars, 12))
    estimated_mae = mae_95th * hold_scaling

    # --- Stop-hit probability ---
    # GBM approximation: P(hit stop before TP) depends on vol and distance ratio
    # Using reflection principle for Brownian motion with drift
    if ewma_vol > 0 and stop_pct > 0 and tp_pct > 0:
        # Drift from recent momentum (last 12 bars)
        drift = float(ret_1.tail(12).mean()) if len(ret_1) >= 12 else 0.0

        # For GBM: ratio determines stop-hit probability
        # With drift mu and vol sigma, for barriers at -a (stop) and +b (TP):
        # P(hit -a first) ≈ (exp(2*mu*b/sigma^2) - 1) / (exp(2*mu*(a+b)/sigma^2) - 1)
        # Simplified: when drift ≈ 0, P ≈ b / (a + b) (symmetric random walk)
        if abs(drift) < 1e-6:
            stop_hit_prob = tp_pct / (stop_pct + tp_pct)
        else:
            exponent_b = 2.0 * drift * tp_pct / (ewma_vol ** 2)
            exponent_ab = 2.0 * drift * (stop_pct + tp_pct) / (ewma_vol ** 2)
            # Clamp to avoid overflow
            exponent_b = max(-20.0, min(20.0, exponent_b))
            exponent_ab = max(-20.0, min(20.0, exponent_ab))
            numerator = math.exp(exponent_b) - 1.0
            denominator = math.exp(exponent_ab) - 1.0
            if abs(denominator) < 1e-12:
                stop_hit_prob = 0.5
            else:
                stop_hit_prob = max(0.0, min(1.0, numerator / denominator))
    else:
        stop_hit_prob = 0.5

    # --- Expected hold time ---
    # Empirical: average bars until price moves stop_pct or tp_pct from current
    # Approximate using vol: E[T] ≈ (target_move / vol_per_bar)^2
    avg_target = (stop_pct + tp_pct) / 2.0
    if ewma_vol > 0:
        expected_hold = min(max_hold_bars, (avg_target / ewma_vol) ** 2)
    else:
        expected_hold = float(max_hold_bars)
    expected_hold = max(1.0, expected_hold)

    # --- Risk regime classification ---
    if vol_ratio > 1.5 or annualised_vol > 80.0:
        regime_risk = "high"
        position_scale = 0.5
    elif vol_ratio > 1.2 or annualised_vol > 50.0:
        regime_risk = "medium"
        position_scale = 0.75
    else:
        regime_risk = "low"
        position_scale = 1.0

    return RiskEstimate(
        expected_vol_pct=round(annualised_vol, 2),
        max_adverse_pct=round(estimated_mae, 4),
        stop_hit_prob=round(stop_hit_prob, 4),
        expected_hold_bars=round(expected_hold, 1),
        regime_risk=regime_risk,
        position_scale=round(position_scale, 2),
    )
