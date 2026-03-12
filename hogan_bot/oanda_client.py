"""Oanda REST v20 trading client for Hogan.

Provides order management, position queries, and OHLCV candle fetching
on top of the Oanda v20 REST API.  No external SDK required — uses only
``urllib`` so there is zero additional dependency.

Environment variables::

    OANDA_ACCESS_TOKEN=<your-token>
    OANDA_ACCOUNT_ID=001-001-XXXXXXX-001
    OANDA_ENVIRONMENT=practice          # or "live"
"""
from __future__ import annotations

import json
import logging
import os
import time
from datetime import datetime, timezone, timedelta
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import pandas as pd

logger = logging.getLogger(__name__)

_TIMEOUT = 15

_ENVIRONMENTS = {
    "practice": "https://api-fxpractice.oanda.com",
    "live": "https://api-fxtrade.oanda.com",
}

_TF_MAP = {
    "1m": "M1", "5m": "M5", "15m": "M15", "30m": "M30",
    "1h": "H1", "2h": "H2", "4h": "H4", "8h": "H8",
    "1d": "D", "1w": "W", "1M": "M",
}


def _oanda_instrument(symbol: str) -> str:
    """Convert Hogan symbol 'EUR/USD' to Oanda instrument 'EUR_USD'."""
    return symbol.replace("/", "_")


def _hogan_symbol(instrument: str) -> str:
    """Convert Oanda instrument 'EUR_USD' to Hogan symbol 'EUR/USD'."""
    return instrument.replace("_", "/")


class OandaClient:
    """Oanda REST v20 trading client.

    Parameters
    ----------
    access_token : str or None
        Bearer token.  Defaults to ``OANDA_ACCESS_TOKEN`` env var.
    account_id : str or None
        Oanda account ID.  Defaults to ``OANDA_ACCOUNT_ID`` env var.
    environment : str
        ``"practice"`` (default) or ``"live"``.
    """

    def __init__(
        self,
        access_token: str | None = None,
        account_id: str | None = None,
        environment: str | None = None,
    ) -> None:
        self.token = (access_token or os.getenv("OANDA_ACCESS_TOKEN", "")).strip()
        self.account_id = (account_id or os.getenv("OANDA_ACCOUNT_ID", "")).strip()
        env = (environment or os.getenv("OANDA_ENVIRONMENT", "practice")).strip().lower()
        self.base_url = _ENVIRONMENTS.get(env, _ENVIRONMENTS["practice"])
        self.environment = env

        if not self.token:
            raise RuntimeError("OANDA_ACCESS_TOKEN not set")
        if not self.account_id:
            raise RuntimeError("OANDA_ACCOUNT_ID not set")

    # ------------------------------------------------------------------
    # HTTP helpers
    # ------------------------------------------------------------------

    def _request(self, method: str, path: str, body: dict | None = None) -> dict:
        url = f"{self.base_url}{path}"
        headers = {
            "Authorization": f"Bearer {self.token}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        data = json.dumps(body).encode() if body else None
        req = Request(url, data=data, headers=headers, method=method)
        try:
            with urlopen(req, timeout=_TIMEOUT) as resp:
                return json.loads(resp.read().decode())
        except HTTPError as exc:
            err_body = ""
            try:
                err_body = exc.read().decode()[:500]
            except Exception:
                pass
            raise URLError(f"HTTP {exc.code} {exc.reason}: {err_body}") from exc

    def _get(self, path: str) -> dict:
        return self._request("GET", path)

    def _post(self, path: str, body: dict) -> dict:
        return self._request("POST", path, body)

    def _put(self, path: str, body: dict) -> dict:
        return self._request("PUT", path, body)

    # ------------------------------------------------------------------
    # Account
    # ------------------------------------------------------------------

    def account_summary(self) -> dict[str, Any]:
        data = self._get(f"/v3/accounts/{self.account_id}/summary")
        return data.get("account", {})

    def fetch_balance(self) -> dict[str, float]:
        summary = self.account_summary()
        return {
            "balance": float(summary.get("balance", 0)),
            "nav": float(summary.get("NAV", 0)),
            "unrealized_pnl": float(summary.get("unrealizedPL", 0)),
            "margin_used": float(summary.get("marginUsed", 0)),
            "margin_available": float(summary.get("marginAvailable", 0)),
        }

    # ------------------------------------------------------------------
    # Pricing
    # ------------------------------------------------------------------

    def fetch_price(self, symbol: str) -> dict[str, float]:
        inst = _oanda_instrument(symbol)
        data = self._get(
            f"/v3/accounts/{self.account_id}/pricing?instruments={inst}"
        )
        for quote in data.get("prices", []):
            if quote.get("instrument") == inst:
                bids = quote.get("bids", [{}])
                asks = quote.get("asks", [{}])
                bid = float(bids[0].get("price", 0)) if bids else 0
                ask = float(asks[0].get("price", 0)) if asks else 0
                return {"bid": bid, "ask": ask, "mid": (bid + ask) / 2, "spread": ask - bid}
        return {"bid": 0, "ask": 0, "mid": 0, "spread": 0}

    def fetch_spread(self, symbol: str) -> float:
        """Return current spread as a fraction of mid-price."""
        p = self.fetch_price(symbol)
        mid = p.get("mid", 0)
        if mid <= 0:
            return 0.0
        return p.get("spread", 0) / mid

    # ------------------------------------------------------------------
    # Candles
    # ------------------------------------------------------------------

    def fetch_candles(
        self,
        symbol: str,
        timeframe: str = "1h",
        count: int = 500,
    ) -> pd.DataFrame:
        """Fetch OHLCV candles from Oanda.

        Returns a DataFrame with columns: timestamp, open, high, low, close, volume.
        """
        inst = _oanda_instrument(symbol)
        granularity = _TF_MAP.get(timeframe, "H1")
        count = min(count, 5000)

        data = self._get(
            f"/v3/instruments/{inst}/candles"
            f"?granularity={granularity}&count={count}&price=M"
        )

        rows = []
        for candle in data.get("candles", []):
            if not candle.get("complete", False):
                continue
            mid = candle.get("mid", {})
            rows.append({
                "timestamp": candle["time"],
                "open": float(mid.get("o", 0)),
                "high": float(mid.get("h", 0)),
                "low": float(mid.get("l", 0)),
                "close": float(mid.get("c", 0)),
                "volume": int(candle.get("volume", 0)),
            })

        if not rows:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        df = pd.DataFrame(rows)
        df["timestamp"] = pd.to_datetime(df["timestamp"], utc=True)
        return df

    # ------------------------------------------------------------------
    # Order management
    # ------------------------------------------------------------------

    def create_market_order(
        self,
        symbol: str,
        units: float,
        stop_loss_price: float | None = None,
        take_profit_price: float | None = None,
    ) -> dict:
        """Create a market order.  Positive units = buy, negative = sell."""
        inst = _oanda_instrument(symbol)
        order_body: dict[str, Any] = {
            "type": "MARKET",
            "instrument": inst,
            "units": str(int(units)) if abs(units) > 1 else str(units),
            "timeInForce": "FOK",
        }
        if stop_loss_price is not None:
            order_body["stopLossOnFill"] = {"price": f"{stop_loss_price:.5f}"}
        if take_profit_price is not None:
            order_body["takeProfitOnFill"] = {"price": f"{take_profit_price:.5f}"}

        data = self._post(
            f"/v3/accounts/{self.account_id}/orders",
            {"order": order_body},
        )

        fill = data.get("orderFillTransaction", {})
        if fill:
            logger.info(
                "OANDA FILL %s units=%s px=%s",
                inst, fill.get("units"), fill.get("price"),
            )
        return data

    def create_limit_order(
        self,
        symbol: str,
        units: float,
        price: float,
        stop_loss_price: float | None = None,
        take_profit_price: float | None = None,
    ) -> dict:
        """Create a limit order.  Positive units = buy, negative = sell."""
        inst = _oanda_instrument(symbol)
        order_body: dict[str, Any] = {
            "type": "LIMIT",
            "instrument": inst,
            "units": str(int(units)) if abs(units) > 1 else str(units),
            "price": f"{price:.5f}",
            "timeInForce": "GTC",
        }
        if stop_loss_price is not None:
            order_body["stopLossOnFill"] = {"price": f"{stop_loss_price:.5f}"}
        if take_profit_price is not None:
            order_body["takeProfitOnFill"] = {"price": f"{take_profit_price:.5f}"}

        return self._post(
            f"/v3/accounts/{self.account_id}/orders",
            {"order": order_body},
        )

    def cancel_order(self, order_id: str) -> dict:
        return self._put(
            f"/v3/accounts/{self.account_id}/orders/{order_id}/cancel",
            {},
        )

    def get_open_trades(self, symbol: str | None = None) -> list[dict]:
        data = self._get(f"/v3/accounts/{self.account_id}/openTrades")
        trades = data.get("trades", [])
        if symbol:
            inst = _oanda_instrument(symbol)
            trades = [t for t in trades if t.get("instrument") == inst]
        return trades

    def close_trade(self, trade_id: str, units: str | None = None) -> dict:
        """Close an open trade (full or partial)."""
        body: dict = {}
        if units is not None:
            body["units"] = units
        return self._put(
            f"/v3/accounts/{self.account_id}/trades/{trade_id}/close",
            body,
        )

    def get_open_orders(self, symbol: str | None = None) -> list[dict]:
        data = self._get(f"/v3/accounts/{self.account_id}/pendingOrders")
        orders = data.get("orders", [])
        if symbol:
            inst = _oanda_instrument(symbol)
            orders = [o for o in orders if o.get("instrument") == inst]
        return orders

    def __repr__(self) -> str:
        return f"OandaClient(account={self.account_id}, env={self.environment})"
