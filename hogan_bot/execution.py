
from __future__ import annotations

import logging
import time
from dataclasses import dataclass

from hogan_bot.exchange import ExchangeClient
from hogan_bot.paper import PaperPortfolio
from hogan_bot.storage import record_order, record_fill, upsert_position

logger = logging.getLogger(__name__)


@dataclass
class ExecResult:
    ok: bool
    order_id: str | None = None
    error: str | None = None


class ExecutionEngine:
    """Uniform interface over paper + live routing.

    Safety principle:
      - paper mode is default
      - live mode requires explicit config + env flag
      - every order is journaled to SQLite

    Both subclasses own portfolio state mutation so callers never need
    to call ``portfolio.execute_buy/sell`` separately.
    """

    def buy(self, symbol: str, price: float, qty: float,
            trailing_stop_pct: float = 0.0,
            take_profit_pct: float = 0.0) -> ExecResult:  # pragma: no cover
        raise NotImplementedError

    def sell(self, symbol: str, price: float, qty: float) -> ExecResult:  # pragma: no cover
        raise NotImplementedError


class PaperExecution(ExecutionEngine):
    def __init__(self, portfolio: PaperPortfolio, conn=None, exchange_id: str = "paper"):
        self.portfolio = portfolio
        self.conn = conn
        self.exchange_id = exchange_id

    def buy(
        self,
        symbol: str,
        price: float,
        qty: float,
        trailing_stop_pct: float = 0.0,
        take_profit_pct: float = 0.0,
    ) -> ExecResult:
        ok = self.portfolio.execute_buy(
            symbol, price, qty,
            trailing_stop_pct=trailing_stop_pct,
            take_profit_pct=take_profit_pct,
        )
        if ok and self.conn is not None:
            ts_ms = int(time.time() * 1000)
            upsert_position(
                self.conn,
                symbol,
                self.portfolio.positions.get(symbol).qty,
                self.portfolio.positions.get(symbol).avg_entry,
                ts_ms,
            )
        return ExecResult(ok=ok)

    def sell(self, symbol: str, price: float, qty: float) -> ExecResult:
        ok = self.portfolio.execute_sell(symbol, price, qty)
        if ok and self.conn is not None:
            ts_ms = int(time.time() * 1000)
            pos = self.portfolio.positions.get(symbol)
            if pos is None:
                upsert_position(self.conn, symbol, 0.0, 0.0, ts_ms)
            else:
                upsert_position(self.conn, symbol, pos.qty, pos.avg_entry, ts_ms)
        return ExecResult(ok=ok)


class LiveExecution(ExecutionEngine):
    """CCXT-backed spot execution. Uses market orders by default.

    Notes:
      - For US users: Kraken / Coinbase / Gemini are typical.
      - This is intentionally conservative: no leverage, no margin, no derivatives.
    """

    def __init__(self, client: ExchangeClient, conn, exchange_id: str,
                 portfolio: PaperPortfolio | None = None):
        self.client = client
        self.conn = conn
        self.exchange_id = exchange_id
        self.portfolio = portfolio

    def buy(self, symbol: str, price: float, qty: float,
            trailing_stop_pct: float = 0.0,
            take_profit_pct: float = 0.0) -> ExecResult:
        try:
            order = self.client.create_market_order(symbol=symbol, side="buy", amount=qty)
            order["exchange"] = self.exchange_id
            record_order(self.conn, order)
            self._sync_fills(symbol)
            if self.portfolio is not None:
                self.portfolio.execute_buy(
                    symbol, price, qty,
                    trailing_stop_pct=trailing_stop_pct,
                    take_profit_pct=take_profit_pct,
                )
            return ExecResult(ok=True, order_id=str(order.get("id")))
        except Exception as exc:  # pragma: no cover
            logger.exception("Live buy failed: %s", exc)
            return ExecResult(ok=False, error=str(exc))

    def sell(self, symbol: str, price: float, qty: float) -> ExecResult:
        try:
            order = self.client.create_market_order(symbol=symbol, side="sell", amount=qty)
            order["exchange"] = self.exchange_id
            record_order(self.conn, order)
            self._sync_fills(symbol)
            if self.portfolio is not None:
                self.portfolio.execute_sell(symbol, price, qty)
            return ExecResult(ok=True, order_id=str(order.get("id")))
        except Exception as exc:  # pragma: no cover
            logger.exception("Live sell failed: %s", exc)
            return ExecResult(ok=False, error=str(exc))

    def _sync_fills(self, symbol: str | None = None) -> int:
        """Fetch and journal fills since the last recorded fill timestamp.

        Returns number of new fills recorded.
        """
        from hogan_bot.storage import load_latest_fill_ts

        since = load_latest_fill_ts(self.conn, self.exchange_id, symbol=symbol)
        since = max(0, int(since) + 1)
        new = 0
        trades = self.client.fetch_my_trades(symbol=symbol, since=since, limit=200)
        for t in trades:
            td = dict(t)
            td["exchange"] = self.exchange_id
            record_fill(self.conn, td)
            new += 1
        return new
