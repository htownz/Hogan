"""Exchange abstraction layer for Hogan.

Every exchange interaction goes through :class:`ExchangeClient`, which wraps
any exchange available in the CCXT library (110+ venues).  The exchange is
selected at construction time via the ``exchange_id`` string so the rest of
the codebase never imports ``ccxt`` directly.

Backward-compatible convenience subclass::

    client = KrakenClient(api_key, api_secret)   # identical to ExchangeClient("kraken", …)

Using your forked CCXT (https://github.com/htownz/ccxt)::

    pip install git+https://github.com/htownz/ccxt.git@master#egg=ccxt

The module imports ``ccxt`` at runtime so you can swap the installed version
without touching this file.
"""

from __future__ import annotations

import logging
from typing import Any

import ccxt
import pandas as pd

logger = logging.getLogger(__name__)

# OHLCV column names matching the CCXT standard.
_OHLCV_COLS = ["timestamp", "open", "high", "low", "close", "volume"]


class ExchangeClient:
    """Generic CCXT-backed exchange client.

    Parameters
    ----------
    exchange_id:
        Any exchange ID recognised by CCXT (e.g. ``"kraken"``, ``"binance"``,
        ``"bybit"``, ``"coinbase"``).  Case-insensitive.
    api_key / api_secret:
        Optional credentials.  Omit for public endpoints only.
    sandbox:
        When ``True`` the exchange's sandbox/testnet environment is used (if
        supported).

    Examples
    --------
    >>> client = ExchangeClient("binance")
    >>> df = client.fetch_ohlcv_df("BTC/USDT", timeframe="1h", limit=200)
    """

    def __init__(
        self,
        exchange_id: str = "kraken",
        api_key: str | None = None,
        api_secret: str | None = None,
        sandbox: bool = False,
    ) -> None:
        exchange_id = exchange_id.lower()
        try:
            exchange_class = getattr(ccxt, exchange_id)
        except AttributeError as exc:
            available = ", ".join(sorted(ccxt.exchanges)[:20])
            raise ValueError(
                f"Unknown exchange '{exchange_id}'.  A few valid IDs: {available} …"
            ) from exc

        self.exchange_id = exchange_id
        self._exchange: ccxt.Exchange = exchange_class(
            {
                "apiKey": api_key,
                "secret": api_secret,
                "enableRateLimit": True,
            }
        )
        if sandbox:
            self._exchange.set_sandbox_mode(True)

    # ------------------------------------------------------------------
    # Core data — used by strategy, ML pipeline, backtesting
    # ------------------------------------------------------------------

    def fetch_ohlcv_df(
        self,
        symbol: str,
        timeframe: str = "5m",
        limit: int = 500,
        since: int | None = None,
    ) -> pd.DataFrame:
        """Return a DataFrame of OHLCV bars.

        Parameters
        ----------
        symbol:   Trading pair, e.g. ``"BTC/USD"`` or ``"BTC/USDT"``.
        timeframe: CCXT timeframe string, e.g. ``"1m"``, ``"5m"``, ``"1h"``.
        limit:    Number of bars to fetch (exchange-dependent maximum).
        since:    Unix timestamp in **milliseconds** to start from.  ``None``
                  fetches the most recent *limit* bars.
        """
        rows = self._exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, limit=limit, since=since
        )
        if not rows:
            return pd.DataFrame(columns=_OHLCV_COLS)
        df = pd.DataFrame(rows, columns=_OHLCV_COLS)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
        return df

    # ------------------------------------------------------------------
    # Market snapshot
    # ------------------------------------------------------------------

    def fetch_ticker(self, symbol: str) -> dict[str, Any]:
        """Return the latest ticker for *symbol*.

        The returned dict contains at minimum ``last``, ``bid``, ``ask``,
        ``baseVolume``, and ``quoteVolume`` (when available).
        """
        return self._exchange.fetch_ticker(symbol)

    def fetch_order_book(self, symbol: str, depth: int = 20) -> dict[str, Any]:
        """Return the current order book up to *depth* levels.

        Returns a dict with ``bids`` and ``asks`` lists of ``[price, qty]``
        pairs and a ``timestamp`` field.
        """
        return self._exchange.fetch_order_book(symbol, limit=depth)

    def fetch_trades(self, symbol: str, limit: int = 100) -> list[dict]:
        """Return the most recent public trades for *symbol*."""
        return self._exchange.fetch_trades(symbol, limit=limit)

    # ------------------------------------------------------------------
    # Derivatives data (futures / perpetuals) — gracefully optional
    # ------------------------------------------------------------------

    def fetch_funding_rate(self, symbol: str) -> dict[str, Any] | None:
        """Return the current funding rate for a perpetual contract.

        Returns ``None`` when the exchange or instrument does not support
        funding rates (e.g. spot-only pairs, Kraken spot).

        The returned dict contains at minimum ``fundingRate`` and
        ``fundingTimestamp`` when supported.
        """
        if not self._exchange.has.get("fetchFundingRate"):
            logger.debug("%s does not support fetchFundingRate", self.exchange_id)
            return None
        try:
            return self._exchange.fetch_funding_rate(symbol)
        except ccxt.BaseError as exc:
            logger.debug("fetch_funding_rate(%s) failed: %s", symbol, exc)
            return None

    def fetch_open_interest(self, symbol: str) -> dict[str, Any] | None:
        """Return current open interest for a derivative contract.

        Returns ``None`` when unsupported.  When available, the dict
        contains ``openInterest`` (base units) and ``openInterestValue``
        (quote units).
        """
        if not self._exchange.has.get("fetchOpenInterest"):
            logger.debug("%s does not support fetchOpenInterest", self.exchange_id)
            return None
        try:
            return self._exchange.fetch_open_interest(symbol)
        except ccxt.BaseError as exc:
            logger.debug("fetch_open_interest(%s) failed: %s", symbol, exc)
            return None

    def fetch_funding_rate_history(
        self, symbol: str, limit: int = 100
    ) -> list[dict] | None:
        """Return historical funding rates.  Returns ``None`` when unsupported."""
        if not self._exchange.has.get("fetchFundingRateHistory"):
            return None
        try:
            return self._exchange.fetch_funding_rate_history(symbol, limit=limit)
        except ccxt.BaseError as exc:
            logger.debug("fetch_funding_rate_history(%s) failed: %s", symbol, exc)
            return None

    # ------------------------------------------------------------------
    # Discovery helpers
    # ------------------------------------------------------------------

    def list_symbols(self, quote: str | None = None) -> list[str]:
        """Return all active symbols, optionally filtered by quote currency.

        Calls ``load_markets()`` on the first invocation; subsequent calls
        use the cached market map.

        Parameters
        ----------
        quote:
            Filter to symbols with this quote currency, e.g. ``"USDT"`` or
            ``"USD"``.
        """
        markets = self._exchange.load_markets()
        syms = [s for s, m in markets.items() if m.get("active")]
        if quote:
            syms = [s for s in syms if s.endswith(f"/{quote}")]
        return sorted(syms)

    def list_timeframes(self) -> list[str]:
        """Return the timeframes supported by this exchange."""
        tf = getattr(self._exchange, "timeframes", {})
        return sorted(tf.keys()) if tf else []

    def supports(self, method: str) -> bool:
        """Return ``True`` when the exchange advertises support for *method*."""
        return bool(self._exchange.has.get(method))

    def market_info(self, symbol: str) -> dict[str, Any]:
        """Return CCXT market metadata for *symbol* (precision, limits, fees)."""
        markets = self._exchange.load_markets()
        if symbol not in markets:
            raise KeyError(f"Symbol '{symbol}' not found on {self.exchange_id}")
        return markets[symbol]

    # ------------------------------------------------------------------
    # Private / account endpoints (paper-mode safety)
    # ------------------------------------------------------------------

    def fetch_balance(self) -> dict[str, Any]:
        """Return account balances.  Requires valid API credentials."""
        return self._exchange.fetch_balance()

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return f"ExchangeClient(exchange_id={self.exchange_id!r})"


# ---------------------------------------------------------------------------
# Backward-compatible alias
# ---------------------------------------------------------------------------


class KrakenClient(ExchangeClient):
    """Thin alias for ``ExchangeClient("kraken", …)``.

    Preserved so existing code that imports ``KrakenClient`` continues to
    work without modification.
    """

    def __init__(
        self,
        api_key: str | None = None,
        api_secret: str | None = None,
    ) -> None:
        super().__init__("kraken", api_key=api_key, api_secret=api_secret)
