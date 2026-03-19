from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


def calculate_position_size(
    equity_usd: float,
    price: float,
    stop_distance_pct: float,
    max_risk_per_trade: float,
    max_allocation_pct: float,
    confidence_scale: float = 1.0,
    fee_rate: float = 0.0,
) -> float:
    """Return coin amount based on risk and allocation constraints.

    *confidence_scale* (0.0–1.0) multiplies the raw position size to allow
    ML-confidence-based dynamic sizing.  Default ``1.0`` preserves the
    original behaviour.

    *fee_rate* — when provided, reduces size proportionally when the stop
    distance is tight relative to fees. When stop_distance_pct < 3 * fee_rate,
    fees eat a large share of the expected move so we scale down to limit damage.
    """
    if equity_usd <= 0 or price <= 0:
        return 0.0
    if stop_distance_pct <= 0 or max_risk_per_trade <= 0 or max_allocation_pct <= 0:
        return 0.0

    risk_budget_usd = equity_usd * max_risk_per_trade
    size_from_risk = risk_budget_usd / (price * stop_distance_pct)

    allocation_budget_usd = equity_usd * max_allocation_pct
    size_from_allocation = allocation_budget_usd / price

    raw = max(0.0, min(size_from_risk, size_from_allocation))

    # Fee-aware scaling: when stop is tight relative to fees, reduce size
    if fee_rate > 0:
        fee_floor = 3.0 * fee_rate
        if stop_distance_pct < fee_floor:
            fee_scale = stop_distance_pct / fee_floor
            raw *= max(0.1, fee_scale)

    return raw * max(0.0, min(1.0, confidence_scale))


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
