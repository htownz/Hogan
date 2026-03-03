from __future__ import annotations


def calculate_position_size(
    equity_usd: float,
    price: float,
    stop_distance_pct: float,
    max_risk_per_trade: float,
    max_allocation_pct: float,
) -> float:
    """Return coin amount based on risk and allocation constraints."""

    if equity_usd <= 0 or price <= 0:
        return 0.0
    if stop_distance_pct <= 0 or max_risk_per_trade <= 0 or max_allocation_pct <= 0:
        return 0.0

    risk_budget_usd = equity_usd * max_risk_per_trade
    size_from_risk = risk_budget_usd / (price * stop_distance_pct)

    allocation_budget_usd = equity_usd * max_allocation_pct
    size_from_allocation = allocation_budget_usd / price

    return max(0.0, min(size_from_risk, size_from_allocation))


class DrawdownGuard:
    """Tracks equity and stops trading if max drawdown is breached."""

    def __init__(self, starting_equity: float, max_drawdown: float) -> None:
        self.peak_equity = starting_equity
        self.max_drawdown = max_drawdown

    def update_and_check(self, current_equity: float) -> bool:
        if current_equity > self.peak_equity:
            self.peak_equity = current_equity

        if self.peak_equity <= 0:
            return False

        drawdown = (self.peak_equity - current_equity) / self.peak_equity
        return drawdown <= self.max_drawdown
