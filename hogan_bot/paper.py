from __future__ import annotations

import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


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
    # ATR at entry (for volatility expansion check in ExitEvaluator)
    entry_atr_pct: float = 0.0
    # Trailing stop only activates after price moves this far in your favor.
    # Prevents noise-triggered stops in the initial bars after entry.
    trail_activation_pct: float = 0.0
    trail_active: bool = False


@dataclass
class ShortPosition:
    """Tracks a synthetic short position (paper trading only).

    P&L = (avg_entry - current_price) * qty
    Stop loss fires when price RISES above the trailing stop level.
    Take profit fires when price FALLS below avg_entry * (1 - take_profit_pct).
    """
    qty: float = 0.0
    avg_entry: float = 0.0
    trailing_stop_pct: float = 0.0
    take_profit_pct: float = 0.0
    trough_price: float = 0.0
    bars_held: int = 0
    max_adverse_pct: float = 0.0
    max_favorable_pct: float = 0.0
    entry_atr_pct: float = 0.0
    trail_activation_pct: float = 0.0
    trail_active: bool = False


@dataclass
class PaperPortfolio:
    cash_usd: float
    fee_rate: float
    positions: dict[str, Position] = field(default_factory=dict)
    short_positions: dict[str, ShortPosition] = field(default_factory=dict)

    def total_equity(self, mark_prices: dict[str, float]) -> float:
        long_value = sum(pos.qty * mark_prices.get(symbol, 0.0) for symbol, pos in self.positions.items())
        short_pnl = sum(
            (pos.avg_entry - mark_prices.get(symbol, pos.avg_entry)) * pos.qty
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
        trail_activation_pct: float = 0.0,
    ) -> bool:
        if qty <= 0 or price <= 0:
            logger.warning("execute_buy rejected %s: invalid qty=%.6f or price=%.2f", symbol, qty, price)
            return False
        cost = qty * price
        fee = cost * self.fee_rate
        total = cost + fee
        if total > self.cash_usd:
            logger.warning("execute_buy rejected %s: insufficient cash (need %.2f, have %.2f)", symbol, total, self.cash_usd)
            return False

        self.cash_usd -= total
        pos = self.positions.get(symbol, Position())
        new_qty = pos.qty + qty
        pos.avg_entry = 0.0 if new_qty <= 0 else ((pos.qty * pos.avg_entry) + cost) / new_qty
        pos.qty = new_qty

        if trailing_stop_pct > 0:
            pos.trailing_stop_pct = trailing_stop_pct
            pos.trail_activation_pct = trail_activation_pct
            if trail_activation_pct <= 0:
                pos.peak_price = max(pos.peak_price, price)
        if take_profit_pct > 0:
            pos.take_profit_pct = take_profit_pct

        self.positions[symbol] = pos
        return True

    def execute_sell(self, symbol: str, price: float, qty: float) -> bool:
        pos = self.positions.get(symbol)
        if pos is None:
            logger.warning("execute_sell rejected %s: no open long position", symbol)
            return False
        if qty <= 0 or price <= 0 or qty > pos.qty:
            logger.warning("execute_sell rejected %s: invalid params (qty=%.6f, price=%.2f, held=%.6f)", symbol, qty, price, pos.qty)
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
        trail_activation_pct: float = 0.0,
    ) -> bool:
        """Open a synthetic short position.

        Cash is unchanged on entry (no collateral lock — paper-only model).
        P&L is realised when the position is covered.  Position size is
        limited to available cash as a safety guardrail.
        """
        if qty <= 0 or price <= 0:
            logger.warning("execute_short rejected %s: invalid qty=%.6f or price=%.2f", symbol, qty, price)
            return False
        if qty * price > self.cash_usd:
            logger.warning("execute_short rejected %s: insufficient cash (need %.2f, have %.2f)", symbol, qty * price, self.cash_usd)
            return False
        fee = qty * price * self.fee_rate
        self.cash_usd -= fee

        pos = self.short_positions.get(symbol, ShortPosition())
        new_qty = pos.qty + qty
        pos.avg_entry = ((pos.qty * pos.avg_entry) + (qty * price)) / new_qty
        pos.qty = new_qty
        if trailing_stop_pct > 0:
            pos.trailing_stop_pct = trailing_stop_pct
            pos.trail_activation_pct = trail_activation_pct
            if trail_activation_pct <= 0:
                pos.trough_price = min(pos.trough_price, price) if pos.trough_price > 0 else price
        if take_profit_pct > 0:
            pos.take_profit_pct = take_profit_pct
        self.short_positions[symbol] = pos
        return True

    def execute_cover(self, symbol: str, price: float, qty: float) -> bool:
        """Close (cover) a short position and realise P&L.

        Covers always succeed when the position exists — refusing to close a
        short with unlimited upside risk is worse than temporary negative cash.
        """
        pos = self.short_positions.get(symbol)
        if pos is None:
            logger.warning("execute_cover rejected %s: no open short position", symbol)
            return False
        if qty <= 0 or price <= 0 or qty > pos.qty:
            logger.warning("execute_cover rejected %s: invalid params (qty=%.6f, price=%.2f, held=%.6f)", symbol, qty, price, pos.qty)
            return False
        pnl = (pos.avg_entry - price) * qty
        fee = qty * price * self.fee_rate
        self.cash_usd += pnl - fee
        if self.cash_usd < 0:
            logger.warning(
                "execute_cover %s: cash negative after cover (cash=%.2f, pnl=%.2f, fee=%.2f) "
                "— short loss exceeded available cash",
                symbol, self.cash_usd, pnl, fee,
            )
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
        short_max_hold_bars: int = 0,
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
                logger.debug("check_exits: skipping long %s — no valid price", symbol)
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
                if not pos.trail_active and pos.trail_activation_pct > 0:
                    if pos.avg_entry > 0 and pos.max_favorable_pct >= pos.trail_activation_pct:
                        pos.trail_active = True
                        pos.peak_price = pos.avg_entry * (1.0 + pos.max_favorable_pct)
                elif pos.trail_activation_pct <= 0:
                    pos.trail_active = True
                if pos.trail_active:
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
                logger.debug("check_exits: skipping short %s — no valid price", symbol)
                continue
            pos.bars_held += 1
            if pos.avg_entry > 0:
                move_pct = (pos.avg_entry - px) / pos.avg_entry
                if move_pct < 0:
                    pos.max_adverse_pct = max(pos.max_adverse_pct, abs(move_pct))
                else:
                    pos.max_favorable_pct = max(pos.max_favorable_pct, move_pct)
            # Short max-loss guardrail: emergency exit if unrealized loss
            # exceeds 10% of entry value (shorts have unlimited upside risk).
            if pos.avg_entry > 0 and px > pos.avg_entry * 1.10:
                _short_loss_pct = (px - pos.avg_entry) / pos.avg_entry
                logger.warning(
                    "SHORT_MAX_LOSS %s — unrealized loss %.1f%% exceeds 10%% guardrail",
                    symbol, _short_loss_pct * 100,
                )
                exits.append((symbol, "short_max_loss"))
                continue
            _s_max = short_max_hold_bars if short_max_hold_bars > 0 else max_hold_bars
            if _s_max > 0 and pos.bars_held >= _s_max:
                exits.append((symbol, "short_max_hold_time"))
                continue
            if pos.trailing_stop_pct > 0:
                if not pos.trail_active and pos.trail_activation_pct > 0:
                    if pos.avg_entry > 0 and pos.max_favorable_pct >= pos.trail_activation_pct:
                        pos.trail_active = True
                        pos.trough_price = pos.avg_entry * (1.0 - pos.max_favorable_pct)
                elif pos.trail_activation_pct <= 0:
                    pos.trail_active = True
                if pos.trail_active:
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
