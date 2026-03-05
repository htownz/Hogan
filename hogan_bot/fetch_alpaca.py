"""Alpaca market-data fetcher for Hogan.

Provides two capabilities:

1. **Stock macro data** (SPY, VIX) — a reliable alternative to yfinance that
   avoids the recurring MultiIndex column issues.  Stores into the
   ``onchain_metrics`` table alongside the existing OpenBB/yfinance records.

2. **Crypto bid-ask spread** — the latest orderbook for BTC/USD and ETH/USD,
   giving Hogan a real-time microstructure signal (wide spread = illiquid /
   stressed market; tight spread = healthy liquidity).

3. **Crypto OHLCV bars** — historical BTC/USD bars stored into the ``candles``
   table, supplementing the Kraken feed (useful when Kraken returns only 720
   bars, Alpaca holds more history on their data feed).

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
    python -m hogan_bot.fetch_alpaca           # macro + spread snapshot
    python -m hogan_bot.fetch_alpaca --crypto-bars --days 30   # backfill candles
    python -m hogan_bot.fetch_alpaca --stock-only              # SPY + VIX only
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
# Crypto OHLCV bars: supplement Kraken candles
# ---------------------------------------------------------------------------

def fetch_crypto_bars(
    symbols: list[str] | None = None,
    days: int = 30,
    timeframe_str: str = "1Day",
) -> dict[str, int]:
    """Fetch Alpaca crypto OHLCV bars and upsert into the local DB.

    Useful for:
    - Backfilling candles Kraken won't return (Kraken caps at 720 5m bars)
    - 1-hour or 1-day candles for multi-timeframe context

    Returns {symbol: rows_written} mapping.
    """
    if symbols is None:
        symbols = ["BTC/USD", "ETH/USD"]

    _check_keys()
    _, CryptoClient, TimeFrame = _try_import()
    from alpaca.data.requests import CryptoBarsRequest

    k, s = _keys()
    client = CryptoClient(api_key=k, secret_key=s)

    _TF_MAP = {
        "1Min": TimeFrame.Minute,
        "5Min": None,    # needs custom construction
        "15Min": None,
        "1Hour": TimeFrame.Hour,
        "1Day": TimeFrame.Day,
    }

    tf = _TF_MAP.get(timeframe_str, TimeFrame.Day)
    if tf is None:
        # Build a custom timeframe for 5m / 15m
        from alpaca.data.timeframe import TimeFrame as TF, TimeFrameUnit
        minutes = int(timeframe_str.replace("Min", ""))
        tf = TF(minutes, TimeFrameUnit.Minute)

    end = datetime.now(tz=timezone.utc)
    start = end - timedelta(days=days)

    # Alpaca crypto API requires BTC/USD format (with slash)
    alpaca_symbols = symbols
    written: dict[str, int] = {}

    try:
        req = CryptoBarsRequest(
            symbol_or_symbols=alpaca_symbols,
            timeframe=tf,
            start=start,
            end=end,
        )
        bars_resp = client.get_crypto_bars(req)
    except Exception as exc:
        logger.warning("Alpaca crypto bars request failed: %s", exc)
        return written

    db_path = os.getenv("HOGAN_DB_PATH", "data/hogan.db")
    conn = sqlite3.connect(db_path)

    try:
        for sym_orig in symbols:
            try:
                sym_bars = bars_resp[sym_orig]
            except (KeyError, TypeError):
                written[sym_orig] = 0
                continue
            if not sym_bars:
                written[sym_orig] = 0
                continue

            # Determine timeframe label (CCXT-style: 1m, 5m, 1h, 1d)
            _tf_label = {
                "1Min": "1m", "5Min": "5m", "15Min": "15m",
                "1Hour": "1h", "1Day": "1d",
            }.get(timeframe_str, timeframe_str.lower())

            rows = []
            for bar in sym_bars:
                ts_ms = int(bar.timestamp.timestamp() * 1000)
                rows.append((
                    sym_orig, _tf_label, ts_ms,
                    float(bar.open), float(bar.high),
                    float(bar.low), float(bar.close),
                    float(bar.volume),
                ))

            conn.executemany(
                """INSERT OR REPLACE INTO candles
                   (symbol, timeframe, ts_ms, open, high, low, close, volume)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                rows,
            )
            conn.commit()
            written[sym_orig] = len(rows)
            logger.info("Alpaca bars: wrote %d %s rows for %s", len(rows), _tf_label, sym_orig)

    finally:
        conn.close()

    return written


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
) -> dict[str, int]:
    """Fetch SPY close, VIX close, and crypto bid-ask spreads; store in DB.

    Called daily by ``refresh_daily.py``.  Does NOT fetch crypto OHLCV bars
    (use ``fetch_crypto_bars()`` for that separately, e.g. once per week).
    """
    _check_keys()

    db_path = os.getenv("HOGAN_DB_PATH", db_path)
    conn = sqlite3.connect(db_path)
    total: dict[str, int] = {}

    try:
        # --- Stock macro data ---
        stock_records = fetch_stock_bars(["SPY"], days=stock_days)
        n = _upsert_macro(conn, stock_records)
        total["stock_bars"] = n
        logger.info("Alpaca: %d stock macro records", n)

        # --- Crypto bid-ask spread ---
        if include_spread:
            spread_records = fetch_crypto_spread(["BTC/USD", "ETH/USD"])
            n = _upsert_macro(conn, spread_records)
            total["crypto_spread"] = n
            logger.info("Alpaca: %d spread records", n)

    finally:
        conn.close()

    return total


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch Alpaca market data into Hogan DB")
    p.add_argument("--db", default=os.getenv("HOGAN_DB_PATH", "data/hogan.db"))
    p.add_argument("--stock-days", type=int, default=10,
                   help="Days of stock bars to fetch (default 10)")
    p.add_argument("--stock-only", action="store_true",
                   help="Only fetch stock bars (skip spread)")
    p.add_argument("--crypto-bars", action="store_true",
                   help="Also fetch BTC/ETH crypto OHLCV bars (default: off)")
    p.add_argument("--crypto-days", type=int, default=30,
                   help="Days of crypto bars to fetch (default 30)")
    p.add_argument("--timeframe", default="1Day",
                   choices=["1Min", "5Min", "15Min", "1Hour", "1Day"],
                   help="Crypto bar timeframe (default 1Day)")
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

    try:
        result.update(fetch_all_alpaca(
            db_path=args.db,
            stock_days=args.stock_days,
            include_spread=not args.stock_only,
        ))
    except RuntimeError as e:
        logger.error("%s", e)
        sys.exit(1)

    if args.crypto_bars:
        written = fetch_crypto_bars(days=args.crypto_days, timeframe_str=args.timeframe)
        result.update({f"crypto_bars_{k}": v for k, v in written.items()})

    print(json.dumps(result, indent=2))
