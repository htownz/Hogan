from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Position:
    qty: float = 0.0
    avg_entry: float = 0.0


@dataclass
class PaperPortfolio:
    cash_usd: float
    fee_rate: float
    positions: dict[str, Position] = field(default_factory=dict)

    def total_equity(self, mark_prices: dict[str, float]) -> float:
        return self.cash_usd + sum(
            pos.qty * mark_prices.get(symbol, 0.0) for symbol, pos in self.positions.items()
        )

    def execute_buy(self, symbol: str, price: float, qty: float) -> bool:
        if qty <= 0 or price <= 0:
            return False
        cost = qty * price
        fee = cost * self.fee_rate
        total = cost + fee
        if total > self.cash_usd:
            return False

        self.cash_usd -= total
        pos = self.positions.get(symbol, Position())
        new_qty = pos.qty + qty
        pos.avg_entry = 0.0 if new_qty <= 0 else ((pos.qty * pos.avg_entry) + cost) / new_qty
        pos.qty = new_qty
        self.positions[symbol] = pos
        return True

    def execute_sell(self, symbol: str, price: float, qty: float) -> bool:
        pos = self.positions.get(symbol, Position())
        if qty <= 0 or price <= 0 or qty > pos.qty:
            return False

        proceeds = qty * price
        fee = proceeds * self.fee_rate
        self.cash_usd += proceeds - fee
        pos.qty -= qty
        if pos.qty <= 1e-12:
            self.positions.pop(symbol, None)
        else:
            self.positions[symbol] = pos
        return True
