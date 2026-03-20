from __future__ import annotations

import logging
import math

logger = logging.getLogger(__name__)

_MAX_CONFIDENCE_SCALE = 1.50


def calculate_position_size(
    equity_usd: float,
    price: float,
    stop_distance_pct: float,
    max_risk_per_trade: float,
    max_allocation_pct: float,
    confidence_scale: float = 1.0,
    fee_rate: float = 0.0,
    atr_pct: float = 0.0,
    avg_atr_pct: float = 0.0,
) -> float:
    """Return coin amount based on risk and allocation constraints.

    *confidence_scale* (0.0–``_MAX_CONFIDENCE_SCALE``) multiplies the raw
    position size to allow ML-confidence-based dynamic sizing.  Values > 1.0
    increase size for high-conviction signals (e.g. ``ml_probability_sizer``
    returns up to 1.50).  Default ``1.0`` preserves the original behaviour.

    *fee_rate* — when provided, reduces size proportionally when the stop
    distance is tight relative to fees. When stop_distance_pct < 3 * fee_rate,
    fees eat a large share of the expected move so we scale down to limit damage.

    *atr_pct* / *avg_atr_pct* — when both are positive, apply volatility-
    adjusted sizing.  In high-vol regimes the position shrinks (inverse
    scaling) to keep dollar-risk constant; in low-vol periods it can grow
    slightly (up to 1.30x).  This keeps the *dollar volatility* of each
    position roughly constant regardless of market conditions.
    """
    if any(math.isnan(v) or math.isinf(v) for v in
           (equity_usd, price, stop_distance_pct, max_risk_per_trade,
            max_allocation_pct, confidence_scale)):
        logger.error(
            "NaN/Inf in position sizing inputs: equity=%.2f price=%.2f "
            "stop=%.4f risk=%.4f alloc=%.4f conf=%.4f",
            equity_usd, price, stop_distance_pct,
            max_risk_per_trade, max_allocation_pct, confidence_scale,
        )
        return 0.0

    if equity_usd <= 0 or price <= 0:
        return 0.0
    if stop_distance_pct <= 0 or max_risk_per_trade <= 0 or max_allocation_pct <= 0:
        return 0.0

    risk_budget_usd = equity_usd * max_risk_per_trade
    size_from_risk = risk_budget_usd / (price * stop_distance_pct)

    allocation_budget_usd = equity_usd * max_allocation_pct
    size_from_allocation = allocation_budget_usd / price

    raw = max(0.0, min(size_from_risk, size_from_allocation))

    if fee_rate > 0:
        fee_floor = 3.0 * fee_rate
        if stop_distance_pct < fee_floor:
            fee_scale = stop_distance_pct / fee_floor
            raw *= max(0.1, fee_scale)

    # Volatility-adjusted sizing: inverse-scale by current vs average ATR.
    # When volatility spikes (atr_pct >> avg_atr_pct), shrink position to
    # maintain roughly constant dollar-risk.  When volatility is low, allow
    # a modest increase (capped at 1.30x) to capture more upside in calm
    # markets without excessive leverage.
    if atr_pct > 0 and avg_atr_pct > 0:
        vol_ratio = atr_pct / avg_atr_pct
        # Inverse square-root scaling: smooths out spikes
        vol_scale = max(0.40, min(1.30, vol_ratio ** -0.5))
        raw *= vol_scale
        if vol_scale < 0.80 or vol_scale > 1.10:
            logger.debug(
                "VOL_SIZE_ADJUST: atr=%.4f avg=%.4f ratio=%.2f scale=%.2f",
                atr_pct, avg_atr_pct, vol_ratio, vol_scale,
            )

    return raw * max(0.0, min(_MAX_CONFIDENCE_SCALE, confidence_scale))


class DrawdownGuard:
    """Tracks equity and stops trading if max drawdown is breached."""

    def __init__(self, starting_equity: float, max_drawdown: float) -> None:
        self.peak_equity = starting_equity
        self.max_drawdown = max_drawdown

    def update_and_check(self, current_equity: float) -> bool:
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        if self.peak_equity <= 0:
            logger.error("DrawdownGuard: peak_equity=%.2f is non-positive — halting trading", self.peak_equity)
            return False

        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        if drawdown > self.max_drawdown:
            logger.warning(
                "DrawdownGuard BREACH: drawdown=%.2f%% exceeds max=%.2f%% (equity=%.2f peak=%.2f)",
                drawdown * 100, self.max_drawdown * 100, current_equity, self.peak_equity,
            )
            return False
        return True
