
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field

from hogan_bot.exchange import ExchangeClient
from hogan_bot.paper import PaperPortfolio, Position
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
            pos = self.portfolio.positions.get(symbol)
            if pos is not None:
                upsert_position(self.conn, symbol, pos.qty, pos.avg_entry, ts_ms)
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


# ---------------------------------------------------------------------------
# Smart passive-limit execution
# ---------------------------------------------------------------------------

@dataclass
class SmartExecConfig:
    """Configuration for passive limit execution."""
    max_reprices: int = 2
    reprice_wait_s: float = 5.0
    stale_timeout_s: float = 30.0
    post_only: bool = True
    taker_for_stops: bool = True
    chase_bps: float = 2.0


class SmartExecution(ExecutionEngine):
    """Passive limit-order execution with post-only + reprice logic.

    Default behavior:
    - Entry: post-only limit at best bid (buy) or best ask (sell)
    - Wait `reprice_wait_s` seconds; if unfilled, cancel and reprice up to
      `max_reprices` times, each time improving the price by `chase_bps` bps
    - If still unfilled after all reprices, cancel the stale setup instead of
      chasing with a market order
    - Stop/emergency exits always use taker (market) orders

    This reduces expected fill cost from ~spread/2 to near-zero for entries,
    while ensuring exits are never stuck.
    """

    def __init__(
        self,
        client: ExchangeClient,
        conn,
        exchange_id: str,
        portfolio: PaperPortfolio | None = None,
        config: SmartExecConfig | None = None,
    ):
        self.client = client
        self.conn = conn
        self.exchange_id = exchange_id
        self.portfolio = portfolio
        self.config = config or SmartExecConfig()

    def buy(
        self,
        symbol: str,
        price: float,
        qty: float,
        trailing_stop_pct: float = 0.0,
        take_profit_pct: float = 0.0,
    ) -> ExecResult:
        """Passive limit buy: post at best bid, reprice if needed."""
        cfg = self.config
        limit_price = price

        try:
            ob = self.client.fetch_order_book(symbol, depth=5)
            if ob.get("bids"):
                limit_price = ob["bids"][0][0]
        except Exception as exc:
            logger.debug("Order book fetch failed, using signal price: %s", exc)

        for attempt in range(1 + cfg.max_reprices):
            try:
                params = {"postOnly": True} if cfg.post_only else {}
                order = self.client.create_limit_order(
                    symbol, "buy", qty, limit_price, params=params,
                )
                order["exchange"] = self.exchange_id
                record_order(self.conn, order)
                order_id = str(order.get("id", ""))

                time.sleep(cfg.reprice_wait_s)

                open_orders = self.client.fetch_open_orders(symbol)
                still_open = any(str(o.get("id")) == order_id for o in open_orders)

                if not still_open:
                    self._sync_fills(symbol)
                    if self.portfolio is not None:
                        self.portfolio.execute_buy(
                            symbol, limit_price, qty,
                            trailing_stop_pct=trailing_stop_pct,
                            take_profit_pct=take_profit_pct,
                        )
                    logger.info(
                        "SMART_BUY filled %s at %.2f (attempt %d)",
                        symbol, limit_price, attempt + 1,
                    )
                    return ExecResult(ok=True, order_id=order_id)

                self.client.cancel_order(order_id, symbol)
                logger.debug(
                    "SMART_BUY reprice %d/%d for %s",
                    attempt + 1, cfg.max_reprices, symbol,
                )
                limit_price *= 1.0 + cfg.chase_bps / 10_000

            except Exception as exc:
                logger.warning("SmartExecution buy error: %s", exc)
                return ExecResult(ok=False, error=str(exc))

        logger.info(
            "SMART_BUY cancelled stale setup for %s after %d reprices",
            symbol, cfg.max_reprices,
        )
        return ExecResult(ok=False, error="stale_setup_cancelled")

    def sell(self, symbol: str, price: float, qty: float) -> ExecResult:
        """Passive limit sell: post at best ask, reprice if needed."""
        cfg = self.config
        limit_price = price

        try:
            ob = self.client.fetch_order_book(symbol, depth=5)
            if ob.get("asks"):
                limit_price = ob["asks"][0][0]
        except Exception as exc:
            logger.debug("Order book fetch failed, using signal price: %s", exc)

        for attempt in range(1 + cfg.max_reprices):
            try:
                params = {"postOnly": True} if cfg.post_only else {}
                order = self.client.create_limit_order(
                    symbol, "sell", qty, limit_price, params=params,
                )
                order["exchange"] = self.exchange_id
                record_order(self.conn, order)
                order_id = str(order.get("id", ""))

                time.sleep(cfg.reprice_wait_s)

                open_orders = self.client.fetch_open_orders(symbol)
                still_open = any(str(o.get("id")) == order_id for o in open_orders)

                if not still_open:
                    self._sync_fills(symbol)
                    if self.portfolio is not None:
                        self.portfolio.execute_sell(symbol, limit_price, qty)
                    logger.info(
                        "SMART_SELL filled %s at %.2f (attempt %d)",
                        symbol, limit_price, attempt + 1,
                    )
                    return ExecResult(ok=True, order_id=order_id)

                self.client.cancel_order(order_id, symbol)
                logger.debug(
                    "SMART_SELL reprice %d/%d for %s",
                    attempt + 1, cfg.max_reprices, symbol,
                )
                limit_price *= 1.0 - cfg.chase_bps / 10_000

            except Exception as exc:
                logger.warning("SmartExecution sell error: %s", exc)
                return ExecResult(ok=False, error=str(exc))

        logger.info(
            "SMART_SELL cancelled stale setup for %s after %d reprices",
            symbol, cfg.max_reprices,
        )
        return ExecResult(ok=False, error="stale_setup_cancelled")

    def market_sell(self, symbol: str, price: float, qty: float) -> ExecResult:
        """Emergency taker sell (for stops / forced exits)."""
        try:
            order = self.client.create_market_order(symbol, "sell", qty)
            order["exchange"] = self.exchange_id
            record_order(self.conn, order)
            self._sync_fills(symbol)
            if self.portfolio is not None:
                self.portfolio.execute_sell(symbol, price, qty)
            logger.info("TAKER_SELL %s qty=%.6f", symbol, qty)
            return ExecResult(ok=True, order_id=str(order.get("id")))
        except Exception as exc:
            logger.exception("Market sell failed: %s", exc)
            return ExecResult(ok=False, error=str(exc))

    def _sync_fills(self, symbol: str | None = None) -> int:
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


# ---------------------------------------------------------------------------
# Realistic paper execution (upgraded fill simulation)
# ---------------------------------------------------------------------------

@dataclass
class FillSimConfig:
    """Configuration for realistic paper fills."""
    slippage_bps: float = 5.0
    spread_half_bps: float = 3.0
    partial_fill_probability: float = 0.0
    min_fill_ratio: float = 0.7


class RealisticPaperExecution(ExecutionEngine):
    """Paper execution with spread + slippage simulation.

    Instead of filling at the exact signal price, this class simulates:
    - Spread impact: buys fill at `price * (1 + spread_half_bps/10000)`,
      sells fill at `price * (1 - spread_half_bps/10000)`
    - Slippage: additional adverse price impact of `slippage_bps` bps
    - Partial fills: configurable probability of only filling `min_fill_ratio`

    Use this in paper mode for more realistic P&L estimation.
    """

    def __init__(
        self,
        portfolio: PaperPortfolio,
        conn=None,
        exchange_id: str = "paper_realistic",
        config: FillSimConfig | None = None,
    ):
        self.portfolio = portfolio
        self.conn = conn
        self.exchange_id = exchange_id
        self.config = config or FillSimConfig()

    def _apply_slippage_buy(self, price: float) -> float:
        spread_cost = self.config.spread_half_bps / 10_000
        slip_cost = self.config.slippage_bps / 10_000
        return price * (1.0 + spread_cost + slip_cost)

    def _apply_slippage_sell(self, price: float) -> float:
        spread_cost = self.config.spread_half_bps / 10_000
        slip_cost = self.config.slippage_bps / 10_000
        return price * (1.0 - spread_cost - slip_cost)

    def _maybe_partial(self, qty: float) -> float:
        import random
        if self.config.partial_fill_probability > 0:
            if random.random() < self.config.partial_fill_probability:
                return qty * self.config.min_fill_ratio
        return qty

    def buy(
        self,
        symbol: str,
        price: float,
        qty: float,
        trailing_stop_pct: float = 0.0,
        take_profit_pct: float = 0.0,
    ) -> ExecResult:
        fill_price = self._apply_slippage_buy(price)
        fill_qty = self._maybe_partial(qty)

        ok = self.portfolio.execute_buy(
            symbol, fill_price, fill_qty,
            trailing_stop_pct=trailing_stop_pct,
            take_profit_pct=take_profit_pct,
        )
        if ok and self.conn is not None:
            ts_ms = int(time.time() * 1000)
            pos = self.portfolio.positions.get(symbol)
            if pos:
                upsert_position(self.conn, symbol, pos.qty, pos.avg_entry, ts_ms)

        if ok:
            logger.debug(
                "REALISTIC_BUY %s fill=%.2f (signal=%.2f, slip=%.1fbps) qty=%.6f",
                symbol, fill_price, price,
                (fill_price / price - 1) * 10_000, fill_qty,
            )
        return ExecResult(ok=ok)

    def sell(self, symbol: str, price: float, qty: float) -> ExecResult:
        fill_price = self._apply_slippage_sell(price)
        fill_qty = self._maybe_partial(qty)

        ok = self.portfolio.execute_sell(symbol, fill_price, fill_qty)
        if ok and self.conn is not None:
            ts_ms = int(time.time() * 1000)
            pos = self.portfolio.positions.get(symbol)
            if pos is None:
                upsert_position(self.conn, symbol, 0.0, 0.0, ts_ms)
            else:
                upsert_position(self.conn, symbol, pos.qty, pos.avg_entry, ts_ms)

        if ok:
            logger.debug(
                "REALISTIC_SELL %s fill=%.2f (signal=%.2f, slip=%.1fbps) qty=%.6f",
                symbol, fill_price, price,
                (1 - fill_price / price) * 10_000, fill_qty,
            )
        return ExecResult(ok=ok)
