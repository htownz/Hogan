"""Oanda execution adapter for Hogan.

Implements the :class:`ExecutionEngine` interface using the Oanda v20 REST API.
Supports both market and passive limit execution styles.

Usage::

    from hogan_bot.oanda_client import OandaClient
    from hogan_bot.oanda_execution import OandaExecution

    client = OandaClient()
    executor = OandaExecution(client=client, conn=db_conn)
    result = executor.buy("EUR/USD", 1.0850, 10_000)
"""
from __future__ import annotations

import logging
import time

from hogan_bot.execution import ExecutionEngine, ExecResult
from hogan_bot.fx_utils import pip_size
from hogan_bot.oanda_client import OandaClient
from hogan_bot.paper import PaperPortfolio
from hogan_bot.storage import record_order, upsert_position

logger = logging.getLogger(__name__)


class OandaExecution(ExecutionEngine):
    """Market-order execution via Oanda REST v20.

    Translates Hogan's (symbol, price, qty) interface into Oanda-native
    units-based orders.  For FX, `qty` is interpreted as the number of
    base-currency units (e.g. 10,000 EUR for EUR/USD).

    Attaches stop-loss and take-profit to orders when configured.
    """

    def __init__(
        self,
        client: OandaClient,
        conn=None,
        portfolio: PaperPortfolio | None = None,
        default_stop_pips: float = 50.0,
        default_tp_pips: float = 100.0,
    ):
        self.client = client
        self.conn = conn
        self.portfolio = portfolio
        self.default_stop_pips = default_stop_pips
        self.default_tp_pips = default_tp_pips

    def buy(
        self,
        symbol: str,
        price: float,
        qty: float,
        trailing_stop_pct: float = 0.0,
        take_profit_pct: float = 0.0,
    ) -> ExecResult:
        pip = pip_size(symbol)
        sl_price = price - self.default_stop_pips * pip if self.default_stop_pips > 0 else None
        tp_price = price + self.default_tp_pips * pip if self.default_tp_pips > 0 else None

        if take_profit_pct > 0:
            tp_price = price * (1 + take_profit_pct)
        if trailing_stop_pct > 0:
            trailing_sl = price * (1 - trailing_stop_pct)
            if sl_price is None or trailing_sl > sl_price:
                sl_price = trailing_sl

        try:
            result = self.client.create_market_order(
                symbol, units=abs(qty),
                stop_loss_price=sl_price,
                take_profit_price=tp_price,
            )

            # Journal the order
            if self.conn is not None:
                order_record = {
                    "id": result.get("orderFillTransaction", {}).get("id", ""),
                    "symbol": symbol,
                    "side": "buy",
                    "type": "market",
                    "amount": qty,
                    "price": price,
                    "exchange": "oanda",
                    "status": "filled" if result.get("orderFillTransaction") else "rejected",
                }
                try:
                    record_order(self.conn, order_record)
                except Exception as exc:
                    logger.debug("Order journal failed: %s", exc)

            fill = result.get("orderFillTransaction", {})
            if fill:
                fill_price = float(fill.get("price", price))
                fill_units = abs(float(fill.get("units", qty)))
                order_id = fill.get("id", "")

                if self.portfolio is not None:
                    self.portfolio.execute_buy(
                        symbol, fill_price, fill_units,
                        trailing_stop_pct=trailing_stop_pct,
                        take_profit_pct=take_profit_pct,
                    )
                if self.conn is not None:
                    ts_ms = int(time.time() * 1000)
                    upsert_position(self.conn, symbol, fill_units, fill_price, ts_ms)

                logger.info(
                    "OANDA_BUY %s units=%.0f fill=%.5f sl=%.5f tp=%.5f",
                    symbol, fill_units, fill_price,
                    sl_price or 0, tp_price or 0,
                )
                return ExecResult(ok=True, order_id=str(order_id))

            reject = result.get("orderRejectTransaction", {})
            reason = reject.get("rejectReason", "unknown")
            logger.warning("OANDA_BUY rejected: %s", reason)
            return ExecResult(ok=False, error=reason)

        except Exception as exc:
            logger.exception("Oanda buy failed: %s", exc)
            return ExecResult(ok=False, error=str(exc))

    def sell(self, symbol: str, price: float, qty: float) -> ExecResult:
        """Close a long position by selling units, or open a short."""
        try:
            open_trades = self.client.get_open_trades(symbol)
            long_trades = [t for t in open_trades if float(t.get("currentUnits", 0)) > 0]

            if long_trades:
                trade = long_trades[0]
                trade_id = trade["id"]
                available = abs(float(trade.get("currentUnits", 0)))
                close_units = str(int(min(qty, available)))
                result = self.client.close_trade(trade_id, units=close_units)

                close_txn = result.get("orderFillTransaction", {})
                if close_txn:
                    fill_price = float(close_txn.get("price", price))
                    realized_pl = float(close_txn.get("pl", 0))

                    if self.portfolio is not None:
                        self.portfolio.execute_sell(symbol, fill_price, float(close_units))
                    if self.conn is not None:
                        ts_ms = int(time.time() * 1000)
                        pos = self.portfolio.positions.get(symbol) if self.portfolio else None
                        if pos:
                            upsert_position(self.conn, symbol, pos.qty, pos.avg_entry, ts_ms)
                        else:
                            upsert_position(self.conn, symbol, 0.0, 0.0, ts_ms)

                    logger.info(
                        "OANDA_SELL %s units=%s fill=%.5f pnl=%.2f",
                        symbol, close_units, fill_price, realized_pl,
                    )
                    return ExecResult(ok=True, order_id=trade_id)

            result = self.client.create_market_order(symbol, units=-abs(qty))
            fill = result.get("orderFillTransaction", {})
            if fill:
                fill_price = float(fill.get("price", price))
                if self.portfolio is not None:
                    self.portfolio.execute_short(symbol, fill_price, abs(qty))
                if self.conn is not None:
                    order_record = {
                        "id": fill.get("id", ""),
                        "symbol": symbol, "side": "sell", "type": "market",
                        "amount": qty, "price": fill_price, "exchange": "oanda",
                        "status": "filled",
                    }
                    try:
                        record_order(self.conn, order_record)
                    except Exception:
                        pass
                logger.info("OANDA_SHORT %s units=%.0f fill=%.5f", symbol, qty, fill_price)
                return ExecResult(ok=True, order_id=str(fill.get("id", "")))

            return ExecResult(ok=False, error="no fill")

        except Exception as exc:
            logger.exception("Oanda sell failed: %s", exc)
            return ExecResult(ok=False, error=str(exc))
