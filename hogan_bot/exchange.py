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

import pandas as pd

try:
    import ccxt
except ImportError:
    ccxt = None  # type: ignore[assignment]

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
        if ccxt is None:
            raise ImportError(
                "ccxt is required for ExchangeClient. Install with: pip install ccxt"
            )
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
                "timeout": 30_000,  # 30s — Kraken can be slow; avoids ReadTimeout crash
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
        timeframe: str = "1h",
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

        def _fetch_with_retry(**kwargs):
            for attempt in range(3):
                try:
                    return self._exchange.fetch_ohlcv(symbol, timeframe=timeframe, **kwargs)
                except (ccxt.RequestTimeout, ccxt.NetworkError) as e:
                    if attempt < 2:
                        delay = 2 ** attempt
                        logger.warning(
                            "Exchange %s timeout (attempt %d/3), retry in %ds: %s",
                            self.exchange_id, attempt + 1, delay, e,
                        )
                        time.sleep(delay)
                    else:
                        raise

        # ── Fast path: single request is enough ─────────────────────────────
        if limit <= _MAX_BARS_PER_REQUEST or since is not None:
            rows = _fetch_with_retry(
                limit=min(limit, _MAX_BARS_PER_REQUEST),
                since=since,
            )
            return self._rows_to_df(rows)

        # ── Paginating path: walk backwards from "now" ───────────────────────
        bar_ms = _TF_MS.get(timeframe, 300_000)
        all_rows: list[list] = []

        # Step 1 — most-recent batch (no since)
        batch = _fetch_with_retry(limit=_MAX_BARS_PER_REQUEST)
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
            batch = _fetch_with_retry(limit=_MAX_BARS_PER_REQUEST, since=fetch_since)
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

    # ---------------------------------------------------------------------
    # Trading (spot) — intentionally minimal, for live routing in production
    # ---------------------------------------------------------------------

    def create_market_order(self, symbol: str, side: str, amount: float, params: dict | None = None) -> dict:
        """Create a market order. Returns the raw CCXT order dict."""
        if amount <= 0:
            raise ValueError("amount must be > 0")
        params = params or {}
        return self._exchange.create_order(symbol, "market", side, amount, None, params)

    def create_limit_order(self, symbol: str, side: str, amount: float, price: float, params: dict | None = None) -> dict:
        """Create a limit order. Returns the raw CCXT order dict."""
        if amount <= 0 or price <= 0:
            raise ValueError("amount and price must be > 0")
        params = params or {}
        return self._exchange.create_order(symbol, "limit", side, amount, price, params)

    def cancel_order(self, order_id: str, symbol: str | None = None, params: dict | None = None) -> dict:
        params = params or {}
        return self._exchange.cancel_order(order_id, symbol, params)

    def fetch_open_orders(self, symbol: str | None = None, limit: int | None = None) -> list[dict]:
        return self._exchange.fetch_open_orders(symbol, None, limit)

    def fetch_my_trades(self, symbol: str | None = None, since: int | None = None, limit: int = 100) -> list[dict]:
        """Fetch your historical trades (fills). 'since' is ms timestamp."""
        return self._exchange.fetch_my_trades(symbol=symbol, since=since, limit=limit)

    def fetch_ticker(self, symbol: str) -> dict:
        return self._exchange.fetch_ticker(symbol)

    # ------------------------------------------------------------------
    # Dunder helpers
    # ------------------------------------------------------------------

    def __repr__(self) -> str:  # pragma: no cover
        return f"ExchangeClient(exchange_id={self.exchange_id!r})"


# ---------------------------------------------------------------------------
# Backward-compatible alias
# ---------------------------------------------------------------------------

class KrakenClient(ExchangeClient):
    def __init__(self, api_key: str | None = None, api_secret: str | None = None, sandbox: bool = False):
        super().__init__("kraken", api_key=api_key, api_secret=api_secret, sandbox=sandbox)


class CoinbaseClient(ExchangeClient):
    """Coinbase Advanced Trade client with CDP (JWT) authentication.

    Coinbase now uses the Coinbase Developer Platform (CDP) API keys which
    authenticate via JWT signed with an EC private key, rather than a simple
    HMAC secret.  This class loads the key pair from the environment and
    converts the escaped ``\\n`` literals in the private key to real newlines
    before passing them to CCXT.

    Environment variables
    ---------------------
    ``COINBASE_CDP_KEY_NAME``
        Full key name, e.g.
        ``organizations/{org_id}/apiKeys/{key_id}``.
    ``COINBASE_CDP_PRIVATE_KEY``
        EC private key PEM string stored on one line with ``\\n`` literals.

    Fallback (limited / no-account access)
    ---------------------------------------
    ``COINBASE_KEY_ID`` + ``COINBASE_KEY_SECRET``
        Legacy base64 key pair — read-only market data only.

    Examples
    --------
    >>> client = CoinbaseClient()            # reads from .env
    >>> df = client.fetch_ohlcv_df("BTC/USD", timeframe="1h", limit=200)
    """

    def __init__(self, sandbox: bool = False) -> None:
        import os

        cdp_key_name = os.getenv("COINBASE_CDP_KEY_NAME", "").strip()
        cdp_private_key = os.getenv("COINBASE_CDP_PRIVATE_KEY", "").strip()

        if cdp_key_name and cdp_private_key:
            # Convert escaped \n literals → real newlines (common in .env files)
            cdp_private_key = cdp_private_key.replace("\\n", "\n")
            api_key = cdp_key_name
            api_secret = cdp_private_key
        else:
            # Fallback to legacy key pair (market data only)
            api_key = os.getenv("COINBASE_KEY_ID", "").strip() or None
            api_secret = os.getenv("COINBASE_KEY_SECRET", "").strip() or None

        super().__init__("coinbase", api_key=api_key, api_secret=api_secret, sandbox=sandbox)

    @classmethod
    def from_env(cls, sandbox: bool = False) -> "CoinbaseClient":
        """Convenience constructor identical to ``CoinbaseClient()``."""
        return cls(sandbox=sandbox)


class AlpacaClient:
    """Alpaca crypto trading client (paper and live).

    Alpaca uses its own SDK (``alpaca-py``) rather than CCXT, so this class
    provides the same ``buy`` / ``sell`` / ``get_balance`` / ``fetch_ohlcv_df``
    interface as :class:`ExchangeClient` but delegates to the Alpaca REST API.

    Install
    -------
    ``pip install alpaca-py``

    Environment variables
    ---------------------
    ``ALPACA_API_KEY``    API key from alpaca.markets paper-trading dashboard
    ``ALPACA_SECRET_KEY`` Corresponding secret key
    ``ALPACA_PAPER``      ``true`` (default) or ``false`` for live trading

    Paper trading is the default and requires zero real money.  To switch to
    live trading, set ``ALPACA_PAPER=false`` and replace with live keys.
    """

    def __init__(self, paper: bool | None = None) -> None:
        import os
        self._api_key = os.getenv("ALPACA_API_KEY", "").strip()
        self._secret_key = os.getenv("ALPACA_SECRET_KEY", "").strip()
        if not self._api_key or not self._secret_key:
            raise ValueError(
                "ALPACA_API_KEY and ALPACA_SECRET_KEY must be set in .env. "
                "Get free paper-trading keys at alpaca.markets."
            )
        if paper is None:
            paper = os.getenv("ALPACA_PAPER", "true").lower() != "false"
        self._paper = paper
        self._base_url = (
            "https://paper-api.alpaca.markets"
            if paper
            else "https://api.alpaca.markets"
        )
        # Lazy-loaded SDK clients
        self._trading_client = None
        self._crypto_data_client = None

    # ------------------------------------------------------------------
    # Internal SDK bootstrapping
    # ------------------------------------------------------------------

    def _trading(self):
        """Return a lazily constructed TradingClient."""
        if self._trading_client is None:
            try:
                from alpaca.trading.client import TradingClient
            except ImportError as exc:
                raise ImportError("Run: pip install alpaca-py") from exc
            self._trading_client = TradingClient(
                self._api_key, self._secret_key, paper=self._paper
            )
        return self._trading_client

    def _crypto_data(self):
        """Return a lazily constructed CryptoHistoricalDataClient."""
        if self._crypto_data_client is None:
            try:
                from alpaca.data import CryptoHistoricalDataClient
            except ImportError as exc:
                raise ImportError("Run: pip install alpaca-py") from exc
            self._crypto_data_client = CryptoHistoricalDataClient(
                self._api_key, self._secret_key
            )
        return self._crypto_data_client

    # ------------------------------------------------------------------
    # ExchangeClient-compatible interface
    # ------------------------------------------------------------------

    def get_balance(self) -> dict[str, float]:
        """Return {currency: amount} dict for the Alpaca account."""
        acct = self._trading().get_account()
        return {
            "USD": float(acct.cash),
            "equity": float(acct.equity),
            "buying_power": float(acct.buying_power),
        }

    def buy(self, symbol: str, amount_usd: float, order_type: str = "market") -> dict:
        """Place a notional buy order for *symbol* (e.g. 'BTC/USD').

        *amount_usd* is the dollar notional to spend.
        Returns the Alpaca order object as a dict.
        """
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        alpaca_sym = symbol.replace("/", "")
        req = MarketOrderRequest(
            symbol=alpaca_sym,
            notional=round(amount_usd, 2),
            side=OrderSide.BUY,
            time_in_force=TimeInForce.GTC,
        )
        order = self._trading().submit_order(req)
        logger.info("Alpaca BUY %s notional=%.2f order_id=%s", symbol, amount_usd, order.id)
        return order.__dict__

    def sell(self, symbol: str, qty: float, order_type: str = "market") -> dict:
        """Place a qty-based sell order for *symbol*.

        *qty* is the number of coins/tokens to sell.
        Returns the Alpaca order object as a dict.
        """
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
        alpaca_sym = symbol.replace("/", "")
        req = MarketOrderRequest(
            symbol=alpaca_sym,
            qty=qty,
            side=OrderSide.SELL,
            time_in_force=TimeInForce.GTC,
        )
        order = self._trading().submit_order(req)
        logger.info("Alpaca SELL %s qty=%.6f order_id=%s", symbol, qty, order.id)
        return order.__dict__

    def fetch_ohlcv_df(
        self,
        symbol: str,
        timeframe: str = "1h",
        limit: int = 500,
    ) -> pd.DataFrame:
        """Fetch OHLCV bars from Alpaca into a standard Hogan DataFrame.

        *timeframe* uses CCXT-style notation: ``1m``, ``5m``, ``15m``,
        ``1h``, ``4h``, ``1d``.  Returns a DataFrame with columns
        [timestamp, open, high, low, close, volume].
        """
        from alpaca.data.requests import CryptoBarsRequest
        from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
        from datetime import datetime, timezone, timedelta

        _TF: dict[str, object] = {
            "1m":  TimeFrame(1, TimeFrameUnit.Minute),
            "5m":  TimeFrame(5, TimeFrameUnit.Minute),
            "15m": TimeFrame(15, TimeFrameUnit.Minute),
            "30m": TimeFrame(30, TimeFrameUnit.Minute),
            "1h":  TimeFrame.Hour,
            "4h":  TimeFrame(4, TimeFrameUnit.Hour),
            "1d":  TimeFrame.Day,
        }
        tf = _TF.get(timeframe, TimeFrame.Hour)

        # Estimate start date from limit and timeframe
        _TF_MIN: dict[str, int] = {
            "1m": 1, "5m": 5, "15m": 15, "30m": 30,
            "1h": 60, "4h": 240, "1d": 1440,
        }
        mins_per_bar = _TF_MIN.get(timeframe, 60)
        total_minutes = limit * mins_per_bar
        start = datetime.now(tz=timezone.utc) - timedelta(minutes=total_minutes + 60)

        alpaca_sym = symbol.replace("/", "")
        req = CryptoBarsRequest(
            symbol_or_symbols=alpaca_sym,
            timeframe=tf,
            start=start,
        )
        bars_resp = self._crypto_data().get_crypto_bars(req)
        sym_bars = bars_resp.get(alpaca_sym) if hasattr(bars_resp, "get") else None
        if not sym_bars:
            return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

        rows = [
            {
                "timestamp": int(bar.timestamp.timestamp() * 1000),
                "open": float(bar.open),
                "high": float(bar.high),
                "low": float(bar.low),
                "close": float(bar.close),
                "volume": float(bar.volume),
            }
            for bar in sym_bars[-limit:]
        ]
        return pd.DataFrame(rows)

    def get_positions(self) -> list[dict]:
        """Return all open positions as a list of dicts."""
        positions = self._trading().get_all_positions()
        return [p.__dict__ for p in positions]

    def cancel_all_orders(self) -> None:
        """Cancel all open orders."""
        self._trading().cancel_orders()

    @classmethod
    def from_env(cls, paper: bool | None = None) -> "AlpacaClient":
        """Convenience constructor identical to ``AlpacaClient()``."""
        return cls(paper=paper)

