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
import time
from typing import Any

import ccxt
import pandas as pd

logger = logging.getLogger(__name__)

# OHLCV column names matching the CCXT standard.
_OHLCV_COLS = ["timestamp", "open", "high", "low", "close", "volume"]

# Many exchanges (Kraken, Bitstamp, …) return at most this many candles per
# REST request.  We paginate automatically when the caller asks for more.
_MAX_BARS_PER_REQUEST: int = 720

# Milliseconds per timeframe — used to compute the `since` offset when
# walking backwards through history.
_TF_MS: dict[str, int] = {
    "1m":   60_000,
    "3m":  180_000,
    "5m":  300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h":  3_600_000,
    "2h":  7_200_000,
    "4h":  14_400_000,
    "6h":  21_600_000,
    "8h":  28_800_000,
    "12h": 43_200_000,
    "1d":  86_400_000,
    "3d":  259_200_000,
    "1w":  604_800_000,
    "2w":  1_209_600_000,
}


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
        """Return a DataFrame of OHLCV bars, paginating automatically.

        Parameters
        ----------
        symbol:    Trading pair, e.g. ``"BTC/USD"`` or ``"BTC/USDT"``.
        timeframe: CCXT timeframe string, e.g. ``"1m"``, ``"5m"``, ``"1h"``.
        limit:     Total number of bars to return.  Values larger than an
                   exchange's per-request cap (e.g. 720 for Kraken) are
                   satisfied by chaining multiple ``fetch_ohlcv`` calls via the
                   ``since`` parameter, walking backwards through history.
        since:     Unix timestamp in **milliseconds** to start from.  ``None``
                   fetches the most recent *limit* bars.
        """
        # Validate timeframe before hitting the exchange — many exchanges (Kraken
        # in particular) return a cryptic 400 when given an unsupported interval.
        supported = getattr(self._exchange, "timeframes", {})
        if supported and timeframe not in supported:
            supported_list = ", ".join(sorted(supported.keys()))
            raise ValueError(
                f"Timeframe '{timeframe}' is not supported by {self.exchange_id}. "
                f"Supported timeframes: {supported_list}"
            )

        # ── Fast path: single request is enough ─────────────────────────────
        if limit <= _MAX_BARS_PER_REQUEST or since is not None:
            rows = self._exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                limit=min(limit, _MAX_BARS_PER_REQUEST),
                since=since,
            )
            return self._rows_to_df(rows)

        # ── Paginating path: walk backwards from "now" ───────────────────────
        bar_ms = _TF_MS.get(timeframe, 300_000)
        all_rows: list[list] = []

        # Step 1 — most-recent batch (no since)
        batch = self._exchange.fetch_ohlcv(
            symbol, timeframe=timeframe, limit=_MAX_BARS_PER_REQUEST
        )
        if not batch:
            return self._rows_to_df([])

        all_rows = list(batch)
        logger.debug("fetch_ohlcv_df page 1: %d bars", len(all_rows))

        # Minimum new bars required to consider a page "useful".
        # Kraken 5m always returns the same ~720 bars regardless of `since`
        # so we'd get < 5 genuinely new bars each loop — detect and stop.
        _MIN_PROGRESS = max(10, _MAX_BARS_PER_REQUEST // 20)
        _MAX_PAGES = 50  # hard safety cap

        # Step 2 — keep fetching older batches until we have enough
        page = 2
        while len(all_rows) < limit and page <= _MAX_PAGES:
            oldest_ts: int = all_rows[0][0]
            fetch_since = oldest_ts - _MAX_BARS_PER_REQUEST * bar_ms
            batch = self._exchange.fetch_ohlcv(
                symbol,
                timeframe=timeframe,
                limit=_MAX_BARS_PER_REQUEST,
                since=fetch_since,
            )
            if not batch:
                break

            # Drop bars that overlap with what we already have
            new_bars = [b for b in batch if b[0] < oldest_ts]
            if len(new_bars) < _MIN_PROGRESS:
                # Exchange isn't providing older data (Kraken 5m cap, etc.)
                logger.info(
                    "fetch_ohlcv_df: pagination stalled on %s %s after %d bars "
                    "(exchange history limit reached). "
                    "For more history use a longer timeframe (1h → 30 days, "
                    "4h → 4 months) or pre-fetch into the local DB with "
                    "`python -m hogan_bot.fetch_data`.",
                    self.exchange_id, timeframe, len(all_rows),
                )
                break

            all_rows = new_bars + all_rows
            logger.debug("fetch_ohlcv_df page %d: +%d bars  total=%d",
                         page, len(new_bars), len(all_rows))
            page += 1
            time.sleep(0.33)   # gentle rate limiting (~3 req/s)

        if len(all_rows) < limit:
            logger.info(
                "fetch_ohlcv_df: returned %d bars (requested %d). "
                "Consider --timeframe 1h or --from-db for deeper history.",
                len(all_rows), limit,
            )

        # Deduplicate, sort ascending, keep the most-recent `limit` rows
        seen: set[int] = set()
        unique: list[list] = []
        for row in sorted(all_rows, key=lambda x: x[0]):
            if row[0] not in seen:
                seen.add(row[0])
                unique.append(row)

        return self._rows_to_df(unique[-limit:])

    @staticmethod
    def _rows_to_df(rows: list[list]) -> pd.DataFrame:
        """Convert a list of raw OHLCV rows to a typed DataFrame."""
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
