"""WebSocket Data Engine for Hogan — Phase 2a.

Provides a push-based candle stream that replaces the blocking REST poll in
``main.py``.  Two implementations share the same ``asyncio.Queue`` interface:

* ``LiveDataEngine``  — subscribes to CCXT Pro WebSocket ``watchOHLCV`` /
  ``watchTicker`` streams with exponential-backoff reconnect on disconnect.
* ``BacktestDataEngine`` — replays historical SQLite candles onto the same
  queue so strategy code is identical in backtest and live modes (Phase 2c).

Usage::

    engine = LiveDataEngine(exchange_id="kraken", symbols=["BTC/USD"],
                            timeframes=["5m", "1h"])
    async with engine:
        async for event in engine.stream():
            # event is a CandleEvent(symbol, timeframe, candle_df)
            ...

Environment variables:
    HOGAN_USE_REST_DATA=1  — Force REST polling instead of WebSocket (useful when
                             Kraken WS fails with "GET /0/public/Assets" or network errors).
    HOGAN_WS_FAIL_THRESHOLD=N — After N consecutive WebSocket failures, switch to REST.
                                 Default 5. Set to 0 to disable.
"""
from __future__ import annotations

import asyncio
import collections
import logging
import os
import time
from dataclasses import dataclass, field
from typing import AsyncIterator

import pandas as pd

logger = logging.getLogger(__name__)

_RECONNECT_BASE = 1.0   # seconds
_RECONNECT_MAX  = 60.0  # seconds
_DEAD_MAN_SECS  = 900   # 15 minutes — fire alert if no candle in this window


def _use_rest_data() -> bool:
    """Check if REST polling should be forced (bypass WebSocket)."""
    return os.getenv("HOGAN_USE_REST_DATA", "").strip().lower() in ("1", "true", "yes")


def _ws_fail_threshold() -> int:
    """After this many consecutive WS failures, switch to REST. 0 = never switch."""
    try:
        return int(os.getenv("HOGAN_WS_FAIL_THRESHOLD", "5"))
    except ValueError:
        return 5


@dataclass
class CandleEvent:
    """A single new (or updated) candle emitted by a DataEngine."""
    symbol: str
    timeframe: str
    candle: pd.Series          # index: open, high, low, close, volume + ts_ms
    received_at: float = field(default_factory=time.time)


class DataEngineBase:
    """Abstract base — subclasses implement ``_run``."""

    def __init__(
        self,
        symbols: list[str],
        timeframes: list[str],
        queue_maxsize: int = 1000,
    ) -> None:
        self.symbols = symbols
        self.timeframes = timeframes
        self._queue: asyncio.Queue[CandleEvent] = asyncio.Queue(maxsize=queue_maxsize)
        self._running = False

    async def __aenter__(self):
        self._running = True
        self._task = asyncio.create_task(self._run())
        return self

    async def __aexit__(self, *_):
        self._running = False
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass

    async def stream(self) -> AsyncIterator[CandleEvent]:
        """Yield CandleEvent objects as they arrive."""
        while self._running:
            try:
                event = await asyncio.wait_for(self._queue.get(), timeout=1.0)
                yield event
            except asyncio.TimeoutError:
                continue

    async def _run(self) -> None:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Ring buffer — keeps last N candles per (symbol, timeframe) in memory
# ---------------------------------------------------------------------------
class CandleRingBuffer:
    """Thread-safe (asyncio) ring buffer of the last N candle rows."""

    def __init__(self, maxlen: int = 500) -> None:
        self._bufs: dict[tuple[str, str], collections.deque] = {}
        self._maxlen = maxlen

    def push(self, symbol: str, timeframe: str, row: dict) -> None:
        key = (symbol, timeframe)
        if key not in self._bufs:
            self._bufs[key] = collections.deque(maxlen=self._maxlen)
        self._bufs[key].append(row)

    def to_df(self, symbol: str, timeframe: str) -> pd.DataFrame:
        key = (symbol, timeframe)
        buf = self._bufs.get(key)
        if not buf:
            return pd.DataFrame()
        df = pd.DataFrame(list(buf))
        if "ts_ms" in df.columns:
            df = df.drop_duplicates(subset=["ts_ms"], keep="last")
            df = df.sort_values("ts_ms").reset_index(drop=True)
        return df


# ---------------------------------------------------------------------------
# Live WebSocket engine (CCXT Pro)
# ---------------------------------------------------------------------------
class LiveDataEngine(DataEngineBase):
    """Subscribe to CCXT Pro WebSocket streams.

    Falls back to REST polling with exponential backoff if the WS
    disconnects or if ``ccxt.pro`` is unavailable.
    """

    def __init__(
        self,
        exchange_id: str,
        api_key: str = "",
        api_secret: str = "",
        symbols: list[str] | None = None,
        timeframes: list[str] | None = None,
        ring_buffer_len: int = 500,
        rest_fallback_interval: float = 30.0,
        queue_maxsize: int = 1000,
    ) -> None:
        super().__init__(
            symbols=symbols or ["BTC/USD"],
            timeframes=timeframes or ["1h"],
            queue_maxsize=queue_maxsize,
        )
        self.exchange_id = exchange_id
        self.api_key = api_key
        self.api_secret = api_secret
        self.rest_fallback_interval = rest_fallback_interval
        self.buffer = CandleRingBuffer(maxlen=ring_buffer_len)
        self._last_candle_ts: dict[str, float] = {}

    def _ccxt_options(self) -> dict:
        """CCXT options for Kraken and other exchanges (timeout, rate limit)."""
        return {
            "enableRateLimit": True,
            "timeout": 60_000,  # 60s — Kraken /0/public/Assets can be slow; avoids RequestTimeout
        }

    # ------------------------------------------------------------------
    async def _run(self) -> None:
        if _use_rest_data():
            logger.info("HOGAN_USE_REST_DATA=1 — using REST polling (WebSocket disabled).")
            await self._run_rest_fallback()
            return
        try:
            import ccxt.pro as ccxtpro  # type: ignore
            await self._run_ws(ccxtpro)
        except ImportError:
            logger.warning("ccxt.pro not installed — falling back to REST polling.")
            await self._run_rest_fallback()

    # ------------------------------------------------------------------
    async def _run_ws(self, ccxtpro) -> None:
        exchange_cls = getattr(ccxtpro, self.exchange_id, None)
        if exchange_cls is None:
            logger.error("CCXT Pro exchange %r not found; falling back to REST.", self.exchange_id)
            await self._run_rest_fallback()
            return

        cfg: dict = {
            **self._ccxt_options(),
        }
        if self.api_key:
            cfg["apiKey"] = self.api_key
        if self.api_secret:
            cfg["secret"] = self.api_secret

        backoff = _RECONNECT_BASE
        ws_fail_count = 0
        threshold = _ws_fail_threshold()

        while self._running:
            exchange = exchange_cls(cfg)
            try:
                tasks = [
                    asyncio.create_task(
                        self._watch_symbol(exchange, symbol, tf)
                    )
                    for symbol in self.symbols
                    for tf in self.timeframes
                ]
                await asyncio.gather(*tasks)
                ws_fail_count = 0
                backoff = _RECONNECT_BASE
            except Exception as exc:
                ws_fail_count += 1
                logger.warning(
                    "WS error (%s); reconnecting in %.1fs [fail %d/%s]: %s",
                    self.exchange_id, backoff,
                    ws_fail_count, str(threshold) if threshold else "∞", exc,
                )
                if threshold and ws_fail_count >= threshold:
                    logger.warning(
                        "WebSocket failed %d times — switching to REST polling. "
                        "Set HOGAN_USE_REST_DATA=1 to skip WS from the start.",
                        ws_fail_count,
                    )
                    await self._run_rest_fallback()
                    return
                try:
                    from hogan_bot.metrics import WS_RECONNECTS
                    for sym in self.symbols:
                        WS_RECONNECTS.labels(symbol=sym).inc()
                except Exception as exc:
                    logger.debug("WS_RECONNECTS metric update failed: %s", exc)
                await asyncio.sleep(backoff)
                backoff = min(backoff * 2, _RECONNECT_MAX)
            finally:
                try:
                    await exchange.close()
                except Exception as exc:
                    logger.debug("Exchange close error during teardown: %s", exc)

    async def _watch_symbol(self, exchange, symbol: str, timeframe: str) -> None:
        """Persistent WS subscription for one (symbol, timeframe) pair."""
        while self._running:
            try:
                ohlcv_list = await exchange.watchOHLCV(symbol, timeframe)
                for ohlcv in ohlcv_list:
                    ts_ms, o, h, lo, c, v = ohlcv
                    row = {
                        "ts_ms": ts_ms, "open": o, "high": h,
                        "low": lo, "close": c, "volume": v,
                    }
                    self.buffer.push(symbol, timeframe, row)
                    event = CandleEvent(
                        symbol=symbol,
                        timeframe=timeframe,
                        candle=pd.Series(row),
                    )
                    try:
                        self._queue.put_nowait(event)
                    except asyncio.QueueFull:
                        logger.warning("Candle queue full — dropping %s/%s event (consumer too slow)", symbol, timeframe)

                    self._last_candle_ts[symbol] = time.time()
                    try:
                        from hogan_bot.metrics import CANDLES_RECEIVED, DATA_LAG_SECONDS
                        CANDLES_RECEIVED.labels(symbol=symbol, timeframe=timeframe).inc()
                        DATA_LAG_SECONDS.labels(symbol=symbol).set(0)
                    except Exception as exc:
                        logger.debug("Candle metrics update failed: %s", exc)
            except Exception as exc:
                logger.warning("watchOHLCV error for %s/%s: %s", symbol, timeframe, exc)
                raise  # bubble up to trigger reconnect

    # ------------------------------------------------------------------
    async def _run_rest_fallback(self) -> None:
        """Poll REST every ``rest_fallback_interval`` seconds as a fallback."""
        try:
            import ccxt
        except ImportError:
            logger.error("Neither ccxt.pro nor ccxt is installed. Data engine offline.")
            return

        exchange_cls = getattr(ccxt, self.exchange_id, None)
        if exchange_cls is None:
            logger.error("CCXT exchange %r not found.", self.exchange_id)
            return

        cfg: dict = {
            **self._ccxt_options(),
        }
        if self.api_key:
            cfg["apiKey"] = self.api_key
        if self.api_secret:
            cfg["secret"] = self.api_secret

        exchange = exchange_cls(cfg)
        logger.info(
            "REST polling active: %s %s every %.0fs",
            self.exchange_id, self.symbols, self.rest_fallback_interval,
        )

        # Warmup: fetch enough historical candles for indicator computation
        _warmup_limit = 200
        for symbol in self.symbols:
            for tf in self.timeframes:
                try:
                    raw = exchange.fetch_ohlcv(symbol, tf, limit=_warmup_limit)
                    _seen: set[int] = set()
                    for ohlcv in raw:
                        ts_ms, o, h, l, c, v = ohlcv
                        if ts_ms in _seen:
                            continue
                        _seen.add(ts_ms)
                        row = {"ts_ms": ts_ms, "open": o, "high": h,
                               "low": l, "close": c, "volume": v}
                        self.buffer.push(symbol, tf, row)
                    logger.info(
                        "REST warmup: %s/%s loaded %d candles",
                        symbol, tf, len(_seen),
                    )
                    self._last_candle_ts[symbol] = time.time()
                except Exception as exc:
                    logger.warning("REST warmup error %s/%s: %s", symbol, tf, exc)

        # Emit a single event for the latest candle of the primary timeframe
        # so the event loop can begin evaluating immediately.
        for symbol in self.symbols:
            _primary_tf = self.timeframes[0] if self.timeframes else "1h"
            df = self.buffer.to_df(symbol, _primary_tf)
            if not df.empty:
                _last = df.iloc[-1].to_dict()
                event = CandleEvent(
                    symbol=symbol,
                    timeframe=_primary_tf,
                    candle=pd.Series(_last),
                )
                try:
                    self._queue.put_nowait(event)
                except asyncio.QueueFull:
                    logger.debug("Queue full during REST warmup for %s — dropping initial event", symbol)

        # Steady-state polling: fetch latest 2 candles per interval, with
        # timestamp de-duplication so the buffer stays clean.
        _known_ts: dict[tuple[str, str], set[int]] = {}
        for symbol in self.symbols:
            for tf in self.timeframes:
                df = self.buffer.to_df(symbol, tf)
                if not df.empty and "ts_ms" in df.columns:
                    _known_ts[(symbol, tf)] = set(df["ts_ms"].astype(int).tolist())
                else:
                    _known_ts[(symbol, tf)] = set()

        while self._running:
            for symbol in self.symbols:
                for tf in self.timeframes:
                    try:
                        raw = exchange.fetch_ohlcv(symbol, tf, limit=2)
                        for ohlcv in raw:
                            ts_ms, o, h, lo, c, v = ohlcv
                            ts_key = int(ts_ms)
                            if ts_key in _known_ts.get((symbol, tf), set()):
                                continue
                            _known_ts.setdefault((symbol, tf), set()).add(ts_key)
                            row = {"ts_ms": ts_ms, "open": o, "high": h,
                                   "low": lo, "close": c, "volume": v}
                            self.buffer.push(symbol, tf, row)
                            event = CandleEvent(
                                symbol=symbol,
                                timeframe=tf,
                                candle=pd.Series(row),
                            )
                            try:
                                self._queue.put_nowait(event)
                            except asyncio.QueueFull:
                                logger.warning("Candle queue full — dropping %s/%s REST event (consumer too slow)", symbol, tf)
                        self._last_candle_ts[symbol] = time.time()
                    except Exception as exc:
                        logger.warning("REST poll error %s/%s: %s", symbol, tf, exc)
            await asyncio.sleep(self.rest_fallback_interval)

    # ------------------------------------------------------------------
    def check_dead_man(self) -> list[str]:
        """Return list of symbols that haven't received a candle recently."""
        now = time.time()
        stale = []
        for sym in self.symbols:
            last = self._last_candle_ts.get(sym, 0)
            if now - last > _DEAD_MAN_SECS:
                stale.append(sym)
                try:
                    from hogan_bot.metrics import DATA_LAG_SECONDS
                    DATA_LAG_SECONDS.labels(symbol=sym).set(now - last)
                except Exception as exc:
                    logger.debug("DATA_LAG_SECONDS metric update failed: %s", exc)
        return stale


# ---------------------------------------------------------------------------
# Oanda data engine — REST polling via OandaClient
# ---------------------------------------------------------------------------
class OandaDataEngine(DataEngineBase):
    """Poll Oanda REST v20 candles for FX paper/live trading.

    Slots into the same ``DataEngineBase`` interface as ``LiveDataEngine``
    so the event loop can swap data sources without changing strategy code.

    Usage::

        from hogan_bot.oanda_client import OandaClient
        engine = OandaDataEngine(
            client=OandaClient(),
            symbols=["EUR/USD", "GBP/USD"],
            timeframes=["15m"],
        )
        async with engine:
            async for event in engine.stream():
                ...
    """

    def __init__(
        self,
        client,  # OandaClient (lazy import to avoid hard dep)
        symbols: list[str] | None = None,
        timeframes: list[str] | None = None,
        poll_interval: float = 30.0,
        ring_buffer_len: int = 500,
        queue_maxsize: int = 1000,
    ) -> None:
        super().__init__(
            symbols=symbols or ["EUR/USD"],
            timeframes=timeframes or ["15m"],
            queue_maxsize=queue_maxsize,
        )
        self._client = client
        self._poll_interval = poll_interval
        self.buffer = CandleRingBuffer(maxlen=ring_buffer_len)
        self._last_candle_ts: dict[str, float] = {}

    async def _run(self) -> None:
        logger.info(
            "OandaDataEngine polling: %s %s every %.0fs",
            self._client.account_id, self.symbols, self._poll_interval,
        )
        while self._running:
            for symbol in self.symbols:
                for tf in self.timeframes:
                    try:
                        df = await asyncio.get_event_loop().run_in_executor(
                            None, self._client.fetch_candles, symbol, tf, 2,
                        )
                        if df.empty:
                            continue
                        for _, row in df.iterrows():
                            ts_raw = row.get("timestamp")
                            if hasattr(ts_raw, "timestamp"):
                                ts_ms = int(ts_raw.timestamp() * 1000)
                            else:
                                ts_ms = int(ts_raw)
                            candle = {
                                "ts_ms": ts_ms,
                                "open": float(row["open"]),
                                "high": float(row["high"]),
                                "low": float(row["low"]),
                                "close": float(row["close"]),
                                "volume": int(row.get("volume", 0)),
                            }
                            self.buffer.push(symbol, tf, candle)
                            event = CandleEvent(
                                symbol=symbol,
                                timeframe=tf,
                                candle=pd.Series(candle),
                            )
                            try:
                                self._queue.put_nowait(event)
                            except asyncio.QueueFull:
                                logger.warning("Candle queue full — dropping %s/%s Oanda event (consumer too slow)", symbol, tf)
                        self._last_candle_ts[symbol] = time.time()
                    except Exception as exc:
                        logger.warning("Oanda poll error %s/%s: %s", symbol, tf, exc)
            await asyncio.sleep(self._poll_interval)

    def check_dead_man(self) -> list[str]:
        """Return list of symbols that haven't received a candle recently."""
        now = time.time()
        stale = []
        for sym in self.symbols:
            last = self._last_candle_ts.get(sym, 0)
            if now - last > _DEAD_MAN_SECS:
                stale.append(sym)
        return stale


# ---------------------------------------------------------------------------
# Backtest replay engine (Phase 2c)
# ---------------------------------------------------------------------------
class BacktestDataEngine(DataEngineBase):
    """Replay candles from SQLite onto the same CandleEvent queue.

    Strategy code can run unmodified against historical data:

        engine = BacktestDataEngine(conn, symbols=["BTC/USD"],
                                    timeframes=["5m"], start_ms=..., end_ms=...)
        async with engine:
            async for event in engine.stream():
                ...
    """

    def __init__(
        self,
        conn,  # sqlite3.Connection
        symbols: list[str],
        timeframes: list[str],
        start_ms: int | None = None,
        end_ms: int | None = None,
        speed: float = 0.0,  # 0 = as fast as possible
        queue_maxsize: int = 10_000,
    ) -> None:
        super().__init__(symbols=symbols, timeframes=timeframes, queue_maxsize=queue_maxsize)
        self._conn = conn
        self._start_ms = start_ms
        self._end_ms = end_ms
        self._speed = speed

    async def _run(self) -> None:
        from hogan_bot.storage import load_candles

        all_events: list[tuple[int, CandleEvent]] = []

        for symbol in self.symbols:
            for tf in self.timeframes:
                df = load_candles(self._conn, symbol, tf, limit=100_000)
                if df.empty:
                    continue
                if self._start_ms:
                    df = df[df["ts_ms"] >= self._start_ms]
                if self._end_ms:
                    df = df[df["ts_ms"] <= self._end_ms]
                for _, row in df.iterrows():
                    evt = CandleEvent(symbol=symbol, timeframe=tf, candle=row)
                    all_events.append((int(row["ts_ms"]), evt))

        all_events.sort(key=lambda x: x[0])
        logger.info("BacktestDataEngine: replaying %d candle events.", len(all_events))

        for ts_ms, event in all_events:
            if not self._running:
                break
            await self._queue.put(event)
            if self._speed > 0:
                await asyncio.sleep(self._speed)

        logger.info("BacktestDataEngine: replay complete.")
        self._running = False
