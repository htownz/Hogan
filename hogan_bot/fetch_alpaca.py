"""Alpaca market-data fetcher for Hogan.

Provides the following capabilities:

1. **Stock macro data** (SPY, QQQ, GLD, TLT) — a reliable alternative to
   yfinance.  Stores daily close into ``onchain_metrics`` alongside existing
   OpenBB/yfinance records.

2. **Crypto bid-ask spread** — latest orderbook for BTC/USD and ETH/USD,
   giving Hogan a real-time microstructure signal.

3. **Crypto OHLCV bars** — multi-timeframe historical bars (10m, 30m, 1h, 1d)
   for BTC/USD, ETH/USD, SOL/USD stored in the ``candles`` table.  Supplements
   the Kraken feed (Kraken caps at 720 5m bars; Alpaca holds years of history).

4. **Stock OHLCV candles** — SPY/QQQ/GLD/TLT stored in the ``candles`` table
   (symbol format "SPY/USD") so they can be used as macro context features.

5. **Bulk backfill** — ``--backfill-all`` fetches every configured symbol ×
   timeframe combination in a single command.

Requirements
------------
    pip install alpaca-py

Environment variables
---------------------
ALPACA_API_KEY      Alpaca API key (free paper-trading account)
ALPACA_SECRET_KEY   Alpaca secret key

Both keys come from https://alpaca.markets → Paper Trading → API keys.
Paper trading account is completely free — no credit card required.

Usage
-----
    python -m hogan_bot.fetch_alpaca                           # macro + spread
    python -m hogan_bot.fetch_alpaca --crypto-bars --days 30  # BTC/ETH 1d bars
    python -m hogan_bot.fetch_alpaca --backfill-all --days 365 # full MTF backfill
    python -m hogan_bot.fetch_alpaca --stock-only              # SPY + VIX only
    python -m hogan_bot.fetch_alpaca --stock-candles --days 365 # stock OHLCV candles
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
from datetime import date, timedelta, datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants — default symbols and timeframes for bulk backfill
# ---------------------------------------------------------------------------

# Crypto symbols available on Alpaca
CRYPTO_SYMBOLS: list[str] = ["BTC/USD", "ETH/USD", "SOL/USD"]

# Stock symbols to store as OHLCV candles (macro context features)
STOCK_SYMBOLS: list[str] = ["SPY", "QQQ", "GLD", "TLT"]

# Timeframes supported for crypto backfill
CRYPTO_TIMEFRAMES: list[str] = ["10Min", "30Min", "1Hour", "1Day"]

# Canonical label map: Alpaca timeframe string → stored timeframe label
_TF_LABEL: dict[str, str] = {
    "1Min":  "1m",
    "5Min":  "5m",
    "10Min": "10m",
    "15Min": "15m",
    "30Min": "30m",
    "1Hour": "1h",
    "4Hour": "4h",
    "1Day":  "1d",
}


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def _keys() -> tuple[str, str]:
    """Return (api_key, secret_key) from environment."""
    k = os.getenv("ALPACA_API_KEY", "").strip()
    s = os.getenv("ALPACA_SECRET_KEY", "").strip()
    return k, s


def _check_keys() -> None:
    k, s = _keys()
    if not k or not s:
        raise RuntimeError(
            "ALPACA_API_KEY / ALPACA_SECRET_KEY not set — "
            "create a free paper account at alpaca.markets"
        )


def _try_import() -> tuple[Any, Any, Any]:
    """Import alpaca-py clients; raise ImportError with install hint if absent."""
    try:
        from alpaca.data import StockHistoricalDataClient, CryptoHistoricalDataClient
        from alpaca.data.timeframe import TimeFrame
        return StockHistoricalDataClient, CryptoHistoricalDataClient, TimeFrame
    except ImportError as exc:
        raise ImportError(
            "alpaca-py is not installed.  Run:  pip install alpaca-py"
        ) from exc


# ---------------------------------------------------------------------------
# Stock data: SPY daily close + VIX daily close
# ---------------------------------------------------------------------------

def fetch_stock_bars(
    symbols: list[str],
    days: int = 10,
) -> list[tuple[str, str, float]]:
    """Fetch daily OHLCV bars for *symbols* from Alpaca stock data API.

    Returns a list of (date_str, metric_name, value) tuples ready to upsert
    into the ``onchain_metrics`` table.
    """
    _check_keys()
    StockClient, _, TimeFrame = _try_import()
    from alpaca.data.requests import StockBarsRequest

    k, s = _keys()
    client = StockClient(api_key=k, secret_key=s)

    end = date.today()
    start = end - timedelta(days=days + 5)  # buffer for weekends / holidays

    records: list[tuple[str, str, float]] = []

    try:
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=datetime(start.year, start.month, start.day, tzinfo=timezone.utc),
            end=datetime(end.year, end.month, end.day, tzinfo=timezone.utc),
        )
        bars = client.get_stock_bars(req)

        _METRIC = {
            "SPY": "spy_close",
            "VIX": "vix_close",
        }

        for sym in symbols:
            sym_bars = bars[sym] if hasattr(bars, "__getitem__") else getattr(bars, sym, [])
            metric = _METRIC.get(sym.upper(), f"{sym.lower()}_close")
            for bar in sym_bars:
                d = bar.timestamp.strftime("%Y-%m-%d")
                records.append((d, metric, float(bar.close)))

        logger.info("Alpaca stock bars: %d records for %s", len(records), symbols)
    except Exception as exc:
        logger.warning("Alpaca stock bars failed (%s): %s", symbols, exc)

    return records


# ---------------------------------------------------------------------------
# Crypto orderbook spread: real-time microstructure signal
# ---------------------------------------------------------------------------

def fetch_crypto_spread(
    symbols: list[str] | None = None,
) -> list[tuple[str, str, float]]:
    """Fetch the latest bid-ask spread for crypto symbols.

    The bid-ask spread (ask - bid) / mid-price is a real microstructure signal:
    - Tight spread (~0.01%) = healthy liquidity, normal market
    - Wide spread (>0.1%)  = stressed / illiquid market → caution on entries

    Returns (today_str, metric_name, value) tuples.
    """
    if symbols is None:
        symbols = ["BTC/USD", "ETH/USD"]

    _check_keys()
    _, CryptoClient, _ = _try_import()
    from alpaca.data.requests import CryptoLatestOrderbookRequest

    k, s = _keys()
    client = CryptoClient(api_key=k, secret_key=s)

    # Alpaca crypto orderbook API requires BTC/USD format (with slash)
    alpaca_symbols = symbols
    today = date.today().isoformat()
    records: list[tuple[str, str, float]] = []

    try:
        req = CryptoLatestOrderbookRequest(symbol_or_symbols=alpaca_symbols)
        books = client.get_crypto_latest_orderbook(req)

        for sym in symbols:
            try:
                book = books[sym]
            except (KeyError, TypeError):
                continue
            if book is None:
                continue
            # Best bid/ask
            best_bid = float(book.bids[0].price) if book.bids else None
            best_ask = float(book.asks[0].price) if book.asks else None
            if best_bid and best_ask and best_bid > 0:
                mid = (best_bid + best_ask) / 2.0
                spread_pct = (best_ask - best_bid) / mid * 100.0
                short = sym.split("/")[0].lower()  # "btc" or "eth"
                records.append((today, f"{short}_bid_ask_spread_pct", round(spread_pct, 6)))
                logger.info(
                    "Alpaca orderbook %s: bid=%.2f ask=%.2f spread=%.4f%%",
                    sym, best_bid, best_ask, spread_pct,
                )
    except Exception as exc:
        logger.warning("Alpaca crypto spread failed: %s", exc)

    return records


# ---------------------------------------------------------------------------
# Crypto OHLCV bars: supplement Kraken candles (multi-timeframe)
# ---------------------------------------------------------------------------

def _build_alpaca_timeframe(timeframe_str: str) -> Any:
    """Convert an Alpaca timeframe string to an Alpaca TimeFrame object."""
    _, _, TimeFrame = _try_import()
    from alpaca.data.timeframe import TimeFrame as TF, TimeFrameUnit

    _DIRECT = {
        "1Min":  TF(1, TimeFrameUnit.Minute),
        "1Hour": TF(1, TimeFrameUnit.Hour),
        "4Hour": TF(4, TimeFrameUnit.Hour),
        "1Day":  TF(1, TimeFrameUnit.Day),
        "1Week": TF(1, TimeFrameUnit.Week),
        "1Month": TF(1, TimeFrameUnit.Month),
    }
    if timeframe_str in _DIRECT:
        return _DIRECT[timeframe_str]
    if timeframe_str.endswith("Min"):
        minutes = int(timeframe_str[:-3])
        return TF(minutes, TimeFrameUnit.Minute)
    if timeframe_str.endswith("Hour"):
        hours = int(timeframe_str[:-4])
        return TF(hours, TimeFrameUnit.Hour)
    raise ValueError(f"Unknown Alpaca timeframe string: {timeframe_str!r}")


def fetch_crypto_bars(
    symbols: list[str] | None = None,
    days: int = 30,
    timeframe_str: str = "1Day",
    db_path: str | None = None,
) -> dict[str, int]:
    """Fetch Alpaca crypto OHLCV bars and upsert into the local DB.

    Supports any timeframe: 1Min, 5Min, 10Min, 15Min, 30Min, 1Hour, 4Hour, 1Day.

    Returns {symbol: rows_written} mapping.
    """
    if symbols is None:
        symbols = ["BTC/USD", "ETH/USD", "SOL/USD"]

    _check_keys()
    _, CryptoClient, _ = _try_import()
    from alpaca.data.requests import CryptoBarsRequest

    k, s = _keys()
    client = CryptoClient(api_key=k, secret_key=s)
    tf = _build_alpaca_timeframe(timeframe_str)
    tf_label = _TF_LABEL.get(timeframe_str, timeframe_str.lower())

    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=days)
    written: dict[str, int] = {}

    try:
        req = CryptoBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=tf,
            start=start,
            end=end,
        )
        bars_resp = client.get_crypto_bars(req)
    except Exception as exc:
        logger.warning("Alpaca crypto bars request failed (%s %s): %s", timeframe_str, symbols, exc)
        return written

    _db = db_path or os.getenv("HOGAN_DB_PATH", "data/hogan.db")
    conn = sqlite3.connect(_db)

    try:
        for sym in symbols:
            try:
                sym_bars = bars_resp[sym]
            except (KeyError, TypeError):
                written[sym] = 0
                continue
            if not sym_bars:
                written[sym] = 0
                continue

            rows = [
                (
                    sym, tf_label,
                    int(bar.timestamp.timestamp() * 1000),
                    float(bar.open), float(bar.high),
                    float(bar.low), float(bar.close),
                    float(bar.volume),
                )
                for bar in sym_bars
            ]
            conn.executemany(
                """INSERT OR REPLACE INTO candles
                   (symbol, timeframe, ts_ms, open, high, low, close, volume)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            conn.commit()
            written[sym] = len(rows)
            logger.info(
                "Alpaca crypto bars: wrote %d %s rows for %s",
                len(rows), tf_label, sym,
            )
    finally:
        conn.close()

    return written


# ---------------------------------------------------------------------------
# Stock OHLCV candles: SPY, QQQ, GLD, TLT → candles table
# ---------------------------------------------------------------------------

def fetch_stock_candles_to_db(
    symbols: list[str] | None = None,
    days: int = 365,
    timeframe_str: str = "1Day",
    db_path: str | None = None,
) -> dict[str, int]:
    """Fetch Alpaca stock OHLCV bars and store them in the ``candles`` table.

    Uses symbol format ``"SPY/USD"`` so the candles table can serve as
    macro context features alongside crypto candles.

    Parameters
    ----------
    symbols:
        List of Alpaca stock ticker symbols, e.g. ``["SPY", "QQQ", "GLD", "TLT"]``.
    days:
        How many calendar days of history to fetch.
    timeframe_str:
        Alpaca timeframe string.  ``"1Day"`` is most reliable on the free tier.
    db_path:
        Override for the SQLite DB path; defaults to ``HOGAN_DB_PATH``.

    Returns
    -------
    dict mapping ``"SYMBOL/USD"`` → rows written.
    """
    if symbols is None:
        symbols = STOCK_SYMBOLS

    _check_keys()
    StockClient, _, _ = _try_import()
    from alpaca.data.requests import StockBarsRequest

    k, s = _keys()
    client = StockClient(api_key=k, secret_key=s)
    tf = _build_alpaca_timeframe(timeframe_str)
    tf_label = _TF_LABEL.get(timeframe_str, timeframe_str.lower())

    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=days + 5)  # buffer for weekends/holidays
    written: dict[str, int] = {}

    try:
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=tf,
            start=start,
            end=end,
        )
        bars_resp = client.get_stock_bars(req)
    except Exception as exc:
        logger.warning("Alpaca stock candles request failed: %s", exc)
        return written

    _db = db_path or os.getenv("HOGAN_DB_PATH", "data/hogan.db")
    conn = sqlite3.connect(_db)

    try:
        for sym in symbols:
            try:
                sym_bars = bars_resp[sym] if hasattr(bars_resp, "__getitem__") else getattr(bars_resp, sym, [])
            except (KeyError, TypeError):
                written[f"{sym}/USD"] = 0
                continue
            if not sym_bars:
                written[f"{sym}/USD"] = 0
                continue

            db_sym = f"{sym}/USD"
            rows = [
                (
                    db_sym, tf_label,
                    int(bar.timestamp.timestamp() * 1000),
                    float(bar.open), float(bar.high),
                    float(bar.low), float(bar.close),
                    float(bar.volume),
                )
                for bar in sym_bars
            ]
            conn.executemany(
                """INSERT OR REPLACE INTO candles
                   (symbol, timeframe, ts_ms, open, high, low, close, volume)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            conn.commit()
            written[db_sym] = len(rows)
            logger.info(
                "Alpaca stock candles: wrote %d %s rows for %s",
                len(rows), tf_label, db_sym,
            )
    finally:
        conn.close()

    return written


# ---------------------------------------------------------------------------
# Bulk backfill: all symbols × timeframes in one call
# ---------------------------------------------------------------------------

def backfill_all_candles(
    db_path: str | None = None,
    crypto_symbols: list[str] | None = None,
    crypto_timeframes: list[str] | None = None,
    stock_symbols: list[str] | None = None,
    days: int = 365,
) -> dict[str, int]:
    """Backfill all configured symbols × timeframes into the DB.

    Iterates over every crypto symbol × timeframe combination plus stock
    daily candles.  Designed for the initial historical backfill or periodic
    deep refresh.

    Parameters
    ----------
    db_path:
        Override for the SQLite DB path; defaults to ``HOGAN_DB_PATH``.
    crypto_symbols:
        List of crypto pairs.  Defaults to :data:`CRYPTO_SYMBOLS`.
    crypto_timeframes:
        List of Alpaca timeframe strings.  Defaults to :data:`CRYPTO_TIMEFRAMES`.
    stock_symbols:
        List of stock tickers.  Defaults to :data:`STOCK_SYMBOLS`.
    days:
        Calendar days of history to fetch for each combination.

    Returns
    -------
    Aggregated ``{description: rows_written}`` mapping.
    """
    _check_keys()
    _db = db_path or os.getenv("HOGAN_DB_PATH", "data/hogan.db")
    cs = crypto_symbols or CRYPTO_SYMBOLS
    ctf = crypto_timeframes or CRYPTO_TIMEFRAMES
    ss = stock_symbols or STOCK_SYMBOLS

    total: dict[str, int] = {}

    # Crypto: each symbol × each timeframe
    for tf_str in ctf:
        tf_label = _TF_LABEL.get(tf_str, tf_str.lower())
        logger.info("Backfilling crypto %s × %s (%d days)…", cs, tf_label, days)
        result = fetch_crypto_bars(symbols=cs, days=days, timeframe_str=tf_str, db_path=_db)
        for sym, n in result.items():
            total[f"{sym}_{tf_label}"] = n

    # Stocks: daily candles
    logger.info("Backfilling stock daily candles %s (%d days)…", ss, days)
    stock_result = fetch_stock_candles_to_db(
        symbols=ss, days=days, timeframe_str="1Day", db_path=_db
    )
    for sym, n in stock_result.items():
        total[f"{sym}_1d"] = n

    total_rows = sum(total.values())
    logger.info("Backfill complete: %d total rows across %d series", total_rows, len(total))
    return total


# ---------------------------------------------------------------------------
# DB upsert for macro metrics
# ---------------------------------------------------------------------------

def _upsert_macro(conn: sqlite3.Connection, records: list[tuple[str, str, float]], symbol: str = "BTC/USD") -> int:
    """Insert or replace (symbol, date, metric, value) into onchain_metrics."""
    if not records:
        return 0
    conn.executemany(
        "INSERT OR REPLACE INTO onchain_metrics (symbol, date, metric, value) VALUES (?, ?, ?, ?)",
        [(symbol, d, m, v) for d, m, v in records],
    )
    conn.commit()
    return len(records)


# ---------------------------------------------------------------------------
# Convenience: fetch_all_alpaca — called by refresh_daily.py
# ---------------------------------------------------------------------------

def fetch_all_alpaca(
    db_path: str = "data/hogan.db",
    stock_days: int = 10,
    include_spread: bool = True,
    include_candles_incremental: bool = True,
) -> dict[str, int]:
    """Fetch SPY close, crypto bid-ask spreads, and incremental MTF candles.

    Called daily by ``refresh_daily.py``.  The ``include_candles_incremental``
    flag (default True) fetches 3 days of new bars for each configured crypto
    timeframe and 7 days of daily stock candles — lightweight enough to run
    every day alongside the other refresh steps.
    """
    _check_keys()

    db_path = os.getenv("HOGAN_DB_PATH", db_path)
    conn = sqlite3.connect(db_path)
    total: dict[str, int] = {}

    try:
        # Stock macro close into onchain_metrics
        stock_records = fetch_stock_bars(["SPY", "QQQ"], days=stock_days)
        n = _upsert_macro(conn, stock_records)
        total["stock_macro"] = n
        logger.info("Alpaca: %d stock macro records", n)

        # Crypto bid-ask spread
        if include_spread:
            spread_records = fetch_crypto_spread(["BTC/USD", "ETH/USD"])
            n = _upsert_macro(conn, spread_records)
            total["crypto_spread"] = n
            logger.info("Alpaca: %d spread records", n)

    finally:
        conn.close()

    # Incremental MTF candle refresh (3 days catches any missed bars)
    if include_candles_incremental:
        for tf_str in CRYPTO_TIMEFRAMES:
            result = fetch_crypto_bars(
                symbols=CRYPTO_SYMBOLS,
                days=3,
                timeframe_str=tf_str,
                db_path=db_path,
            )
            for sym, n in result.items():
                key = f"crypto_{_TF_LABEL.get(tf_str, tf_str)}_{sym.split('/')[0].lower()}"
                total[key] = total.get(key, 0) + n

        # Stock daily candles (7 days rolling)
        stock_result = fetch_stock_candles_to_db(
            symbols=STOCK_SYMBOLS, days=7, timeframe_str="1Day", db_path=db_path
        )
        for sym, n in stock_result.items():
            total[f"stock_candles_{sym.split('/')[0].lower()}"] = n

    return total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch Alpaca market data into Hogan DB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--db", default=os.getenv("HOGAN_DB_PATH", "data/hogan.db"),
                   help="SQLite DB path")
    p.add_argument("--stock-days", type=int, default=10,
                   help="Days of stock macro bars to fetch")
    p.add_argument("--stock-only", action="store_true",
                   help="Only fetch stock macro bars (skip spread and candles)")
    p.add_argument("--crypto-bars", action="store_true",
                   help="Fetch crypto OHLCV bars for --symbols at --timeframe")
    p.add_argument("--crypto-days", type=int, default=30,
                   help="Days of crypto bars when using --crypto-bars")
    p.add_argument(
        "--timeframe", default="1Day",
        choices=["1Min", "5Min", "10Min", "15Min", "30Min", "1Hour", "4Hour", "1Day"],
        help="Crypto bar timeframe",
    )
    p.add_argument(
        "--symbols",
        default=",".join(CRYPTO_SYMBOLS),
        help="Comma-separated crypto symbols for --crypto-bars",
    )
    p.add_argument("--stock-candles", action="store_true",
                   help="Fetch SPY/QQQ/GLD/TLT OHLCV candles into the candles table")
    p.add_argument("--stock-candles-days", type=int, default=365,
                   help="Days of stock candles to fetch")
    p.add_argument(
        "--backfill-all", action="store_true",
        help=(
            "Bulk-backfill ALL symbols × timeframes in one command. "
            "Use --days to control how far back to go."
        ),
    )
    p.add_argument("--days", type=int, default=365,
                   help="Days of history for --backfill-all")
    return p.parse_args()


if __name__ == "__main__":
    import sys
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = parse_args()

    result: dict[str, int] = {}

    # --backfill-all: one command to rule them all
    if args.backfill_all:
        logger.info("Starting full MTF backfill (%d days)…", args.days)
        result = backfill_all_candles(db_path=args.db, days=args.days)
        print(json.dumps(result, indent=2))
        sys.exit(0)

    # Standard daily snapshot (macro + spread + incremental candles)
    try:
        result.update(fetch_all_alpaca(
            db_path=args.db,
            stock_days=args.stock_days,
            include_spread=not args.stock_only,
            include_candles_incremental=not args.stock_only,
        ))
    except RuntimeError as e:
        logger.error("%s", e)
        sys.exit(1)

    # --crypto-bars: single-timeframe fetch
    if args.crypto_bars:
        syms = [s.strip() for s in args.symbols.split(",") if s.strip()]
        written = fetch_crypto_bars(
            symbols=syms,
            days=args.crypto_days,
            timeframe_str=args.timeframe,
            db_path=args.db,
        )
        result.update({f"crypto_bars_{k}": v for k, v in written.items()})

    # --stock-candles: OHLCV for SPY/QQQ/GLD/TLT
    if args.stock_candles:
        sc_result = fetch_stock_candles_to_db(
            symbols=STOCK_SYMBOLS,
            days=args.stock_candles_days,
            timeframe_str="1Day",
            db_path=args.db,
        )
        result.update({f"stock_candles_{k}": v for k, v in sc_result.items()})

    print(json.dumps(result, indent=2))
