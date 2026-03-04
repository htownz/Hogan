from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Position:
    qty: float = 0.0
    avg_entry: float = 0.0
    # Trailing-stop fraction (0 = disabled).  Stop fires when price falls
    # more than *trailing_stop_pct* below the highest price seen since entry.
    trailing_stop_pct: float = 0.0
    # Take-profit fraction above avg_entry (0 = disabled).
    take_profit_pct: float = 0.0
    # High-water mark updated by check_exits(); initialised to avg_entry.
    peak_price: float = 0.0


@dataclass
class PaperPortfolio:
    cash_usd: float
    fee_rate: float
    positions: dict[str, Position] = field(default_factory=dict)

    def total_equity(self, mark_prices: dict[str, float]) -> float:
        return self.cash_usd + sum(
            pos.qty * mark_prices.get(symbol, 0.0) for symbol, pos in self.positions.items()
        )

    def execute_buy(
        self,
        symbol: str,
        price: float,
        qty: float,
        trailing_stop_pct: float = 0.0,
        take_profit_pct: float = 0.0,
    ) -> bool:
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

        # Only update exit parameters when opening a fresh position or
        # adding to an existing one that has no stops yet.
        if trailing_stop_pct > 0:
            pos.trailing_stop_pct = trailing_stop_pct
            pos.peak_price = max(pos.peak_price, price)
        if take_profit_pct > 0:
            pos.take_profit_pct = take_profit_pct

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

    def check_exits(self, mark_prices: dict[str, float]) -> list[tuple[str, str]]:
        """Check open positions against current prices and return exit signals.

        Updates the ``peak_price`` high-water mark for each position, then
        returns a list of ``(symbol, reason)`` tuples where ``reason`` is
        ``"trailing_stop"`` or ``"take_profit"``.  Positions are *not*
        automatically closed — call :meth:`execute_sell` for each returned
        symbol after receiving the list.
        """
        exits: list[tuple[str, str]] = []
        for symbol, pos in list(self.positions.items()):
            px = mark_prices.get(symbol, 0.0)
            if px <= 0:
                continue

            # Update trailing high-water mark
            if pos.trailing_stop_pct > 0:
                if px > pos.peak_price:
                    pos.peak_price = px
                stop_level = pos.peak_price * (1.0 - pos.trailing_stop_pct)
                if px <= stop_level:
                    exits.append((symbol, "trailing_stop"))
                    continue

            # Check take-profit (small epsilon tolerates floating-point rounding)
            if pos.take_profit_pct > 0 and pos.avg_entry > 0:
                tp_level = pos.avg_entry * (1.0 + pos.take_profit_pct)
                if px >= tp_level * (1.0 - 1e-9):
                    exits.append((symbol, "take_profit"))

        return exits
