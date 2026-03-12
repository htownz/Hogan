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
    # Bars held since entry (incremented each check_exits call).
    bars_held: int = 0
    # Max adverse/favorable excursion as fraction of entry price
    max_adverse_pct: float = 0.0
    max_favorable_pct: float = 0.0


@dataclass
class ShortPosition:
    """Tracks a synthetic short position (paper trading only).

    P&L = (avg_entry - current_price) * qty
    Stop loss fires when price RISES above the trailing stop level.
    Take profit fires when price FALLS below avg_entry * (1 - take_profit_pct).
    """
    qty: float = 0.0
    avg_entry: float = 0.0
    # Stop fires when price rises more than trailing_stop_pct above the low-water mark.
    trailing_stop_pct: float = 0.0
    take_profit_pct: float = 0.0
    # Low-water mark updated by check_exits(); tracks the lowest price since short entry.
    trough_price: float = 0.0


@dataclass
class PaperPortfolio:
    cash_usd: float
    fee_rate: float
    positions: dict[str, Position] = field(default_factory=dict)
    short_positions: dict[str, ShortPosition] = field(default_factory=dict)

    def total_equity(self, mark_prices: dict[str, float]) -> float:
        long_value = sum(pos.qty * mark_prices.get(symbol, 0.0) for symbol, pos in self.positions.items())
        # Short P&L: positive when price has fallen below our entry (profitable)
        short_pnl = sum(
            (pos.avg_entry - mark_prices.get(symbol, 0.0)) * pos.qty
            for symbol, pos in self.short_positions.items()
        )
        return self.cash_usd + long_value + short_pnl

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

    def execute_short(
        self,
        symbol: str,
        price: float,
        qty: float,
        trailing_stop_pct: float = 0.0,
        take_profit_pct: float = 0.0,
    ) -> bool:
        """Open a synthetic short position.

        Cash is unchanged on entry (no collateral lock — paper-only model).
        P&L is realised when the position is covered.  Position size is
        limited to available cash as a safety guardrail.
        """
        if qty <= 0 or price <= 0:
            return False
        # Require at least 1× notional in free cash (1:1 margin guardrail)
        if qty * price > self.cash_usd:
            return False
        # Deduct entry fee only
        fee = qty * price * self.fee_rate
        self.cash_usd -= fee

        pos = self.short_positions.get(symbol, ShortPosition())
        new_qty = pos.qty + qty
        pos.avg_entry = ((pos.qty * pos.avg_entry) + (qty * price)) / new_qty
        pos.qty = new_qty
        if trailing_stop_pct > 0:
            pos.trailing_stop_pct = trailing_stop_pct
            pos.trough_price = min(pos.trough_price, price) if pos.trough_price > 0 else price
        if take_profit_pct > 0:
            pos.take_profit_pct = take_profit_pct
        self.short_positions[symbol] = pos
        return True

    def execute_cover(self, symbol: str, price: float, qty: float) -> bool:
        """Close (cover) a short position and realise P&L."""
        pos = self.short_positions.get(symbol)
        if pos is None or qty <= 0 or price <= 0 or qty > pos.qty:
            return False
        pnl = (pos.avg_entry - price) * qty
        fee = qty * price * self.fee_rate
        self.cash_usd += pnl - fee
        pos.qty -= qty
        if pos.qty <= 1e-12:
            self.short_positions.pop(symbol, None)
        else:
            self.short_positions[symbol] = pos
        return True

    def check_exits(
        self,
        mark_prices: dict[str, float],
        max_hold_bars: int = 0,
    ) -> list[tuple[str, str]]:
        """Check open long and short positions and return exit signals.

        Returns a list of ``(symbol, reason)`` tuples where ``reason`` is one of:
        ``"trailing_stop"``, ``"take_profit"``, ``"max_hold_time"``,
        ``"short_trailing_stop"``, or ``"short_take_profit"``.
        Positions are *not* automatically closed.
        """
        exits: list[tuple[str, str]] = []

        # ── Long positions ────────────────────────────────────────────────────
        for symbol, pos in list(self.positions.items()):
            px = mark_prices.get(symbol, 0.0)
            if px <= 0:
                continue
            pos.bars_held += 1
            if pos.avg_entry > 0:
                move_pct = (px - pos.avg_entry) / pos.avg_entry
                if move_pct < 0:
                    pos.max_adverse_pct = max(pos.max_adverse_pct, abs(move_pct))
                else:
                    pos.max_favorable_pct = max(pos.max_favorable_pct, move_pct)
            if max_hold_bars > 0 and pos.bars_held >= max_hold_bars:
                exits.append((symbol, "max_hold_time"))
                continue
            if pos.trailing_stop_pct > 0:
                if px > pos.peak_price:
                    pos.peak_price = px
                stop_level = pos.peak_price * (1.0 - pos.trailing_stop_pct)
                if px <= stop_level:
                    exits.append((symbol, "trailing_stop"))
                    continue
            if pos.take_profit_pct > 0 and pos.avg_entry > 0:
                tp_level = pos.avg_entry * (1.0 + pos.take_profit_pct)
                if px >= tp_level * (1.0 - 1e-9):
                    exits.append((symbol, "take_profit"))

        # ── Short positions ───────────────────────────────────────────────────
        for symbol, pos in list(self.short_positions.items()):
            px = mark_prices.get(symbol, 0.0)
            if px <= 0:
                continue
            # Trailing stop: tracks lowest price, fires when price rebounds up
            if pos.trailing_stop_pct > 0:
                if pos.trough_price <= 0 or px < pos.trough_price:
                    pos.trough_price = px
                stop_level = pos.trough_price * (1.0 + pos.trailing_stop_pct)
                if px >= stop_level:
                    exits.append((symbol, "short_trailing_stop"))
                    continue
            # Take profit: fires when price falls to entry × (1 - tp_pct)
            if pos.take_profit_pct > 0 and pos.avg_entry > 0:
                tp_level = pos.avg_entry * (1.0 - pos.take_profit_pct)
                if px <= tp_level * (1.0 + 1e-9):
                    exits.append((symbol, "short_take_profit"))

        return exits
