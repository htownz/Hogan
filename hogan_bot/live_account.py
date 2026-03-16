
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Tuple

from hogan_bot.exchange import ExchangeClient


@dataclass(frozen=True)
class LivePosition:
    symbol: str
    base: str
    quote: str
    qty: float


@dataclass(frozen=True)
class AccountState:
    ts_ms: int
    cash_quote: float
    equity_quote: float
    marks: Dict[str, float]
    positions: Dict[str, LivePosition]  # keyed by symbol


def _split_symbol(symbol: str) -> Tuple[str, str]:
    if "/" in symbol:
        base, quote = symbol.split("/", 1)
        return base, quote
    # fall back: BTCUSD
    return symbol[:-3], symbol[-3:]


def fetch_account_state(
    client: ExchangeClient,
    symbols: list[str],
    quote_ccy: str = "USD",
) -> AccountState:
    """Fetch balances and mark-to-market equity for spot accounts.

    - Pulls balances each tick (no shadow accounting).
    - Marks positions using last trade/close from fetch_ticker.
    - Treats quote currency balance as cash (USD/USDT/USDC depending on market).
    """
    ts_ms = int(time.time() * 1000)
    bal = client.fetch_balance()
    totals = bal.get("total") or bal.get("free") or {}
    # Cash
    cash = float(totals.get(quote_ccy, 0.0) or 0.0)

    marks: Dict[str, float] = {}
    positions: Dict[str, LivePosition] = {}
    equity = cash

    for sym in symbols:
        base, quote = _split_symbol(sym)
        qty = float(totals.get(base, 0.0) or 0.0)
        # ignore dust
        if abs(qty) < 1e-12:
            continue
        t = client.fetch_ticker(sym)
        px = float(t.get("last") or t.get("close") or t.get("bid") or t.get("ask") or 0.0)
        if px <= 0:
            continue
        marks[sym] = px
        positions[sym] = LivePosition(symbol=sym, base=base, quote=quote, qty=qty)
        equity += qty * px

    return AccountState(
        ts_ms=ts_ms,
        cash_quote=cash,
        equity_quote=equity,
        marks=marks,
        positions=positions,
    )
