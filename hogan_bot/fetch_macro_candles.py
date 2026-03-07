"""Macro-asset OHLCV candle fetcher for Hogan (via yfinance, no API key required).

Downloads historical bars for US equity ETFs, volatility indices, and bond
proxies and stores them in the ``candles`` table alongside crypto candles.
These assets provide critical macro context for the ML model:

  SPY/USD  — S&P 500 ETF (broad equity sentiment)
  QQQ/USD  — Nasdaq-100 ETF (tech-growth risk appetite)
  GLD/USD  — Gold ETF (inflation / risk-off hedge)
  SLV/USD  — Silver ETF (industrial metals + inflation)
  TLT/USD  — 20-Year Treasury Bond ETF (interest rate sensitivity)
  UUP/USD  — Invesco Dollar Bull ETF (DXY proxy — dollar strength)
  VIX/USD  — CBOE Volatility Index (fear gauge, close = VIX level)
  TNX/USD  — 10-Year Treasury Yield (close = yield %, ×10 for scale)

Timeframes
----------
  1d  — Up to 10 years of history (most useful for macro context)
  1h  — Up to 2 years of history (yfinance hard limit for hourly data)

Usage
-----
    # Full historical backfill (run once):
    python -m hogan_bot.fetch_macro_candles --backfill

    # Incremental update (last 7 days):
    python -m hogan_bot.fetch_macro_candles

    # Custom backfill period:
    python -m hogan_bot.fetch_macro_candles --backfill --period 5y
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any

import pandas as pd

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Asset registry
# ---------------------------------------------------------------------------

# (db_symbol, yfinance_ticker, description)
MACRO_ASSETS: list[tuple[str, str, str]] = [
    ("SPY/USD",  "SPY",       "S&P 500 ETF"),
    ("QQQ/USD",  "QQQ",       "Nasdaq-100 ETF"),
    ("GLD/USD",  "GLD",       "Gold ETF"),
    ("SLV/USD",  "SLV",       "Silver ETF"),
    ("TLT/USD",  "TLT",       "20-Year Treasury Bond ETF"),
    ("UUP/USD",  "UUP",       "Dollar Bull ETF (DXY proxy)"),
    ("VIX/USD",  "^VIX",      "CBOE Volatility Index"),
    ("TNX/USD",  "^TNX",      "10-Year Treasury Yield"),
]

# Timeframes to fetch during a full backfill
BACKFILL_TIMEFRAMES: list[tuple[str, str]] = [
    ("1d", "10y"),   # daily — 10 years
    ("1h", "2y"),    # hourly — 2 years (yfinance max)
]

# Timeframes to refresh daily (incremental)
INCREMENTAL_TIMEFRAMES: list[tuple[str, str]] = [
    ("1d", "30d"),   # catch any missed daily bars
    ("1h", "7d"),    # last week of hourly bars
]

# ---------------------------------------------------------------------------
# yfinance helpers
# ---------------------------------------------------------------------------

_TF_TO_YF: dict[str, str] = {
    "1m": "1m", "5m": "5m", "10m": "10m", "15m": "15m", "30m": "30m",
    "1h": "60m", "4h": "4h",
    "1d": "1d", "1w": "1wk",
}


def _fetch_yf(yf_ticker: str, timeframe: str, period: str) -> pd.DataFrame:
    """Download OHLCV from Yahoo Finance and return normalised DataFrame."""
    try:
        import yfinance as yf
    except ImportError as exc:
        raise ImportError("yfinance is not installed. Run: pip install yfinance") from exc

    yf_interval = _TF_TO_YF.get(timeframe)
    if yf_interval is None:
        raise ValueError(f"Unsupported timeframe {timeframe!r} for yfinance")

    df = yf.download(
        yf_ticker,
        period=period,
        interval=yf_interval,
        progress=False,
        auto_adjust=True,
    )

    if df.empty:
        return pd.DataFrame()

    # Flatten MultiIndex columns (yfinance >= 0.2.38 with single ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Volume": "volume",
    })
    df.index.name = "timestamp"
    df = df.reset_index()

    # Ensure UTC-aware timestamps
    ts_col = df["timestamp"]
    if hasattr(ts_col.dt, "tz") and ts_col.dt.tz is None:
        df["timestamp"] = ts_col.dt.tz_localize("UTC")
    else:
        df["timestamp"] = ts_col.dt.tz_convert("UTC")

    df = df[["timestamp", "open", "high", "low", "close", "volume"]].copy()
    df = df.dropna(subset=["close"])
    df["volume"] = df["volume"].fillna(0.0)  # VIX/^TNX have no volume
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# DB storage
# ---------------------------------------------------------------------------

def _upsert_candles(
    conn: sqlite3.Connection,
    db_symbol: str,
    timeframe: str,
    df: pd.DataFrame,
) -> int:
    """Write rows to the candles table; return rows written."""
    if df.empty:
        return 0
    rows = [
        (
            db_symbol, timeframe,
            int(row.timestamp.timestamp() * 1000),
            float(row.open), float(row.high), float(row.low),
            float(row.close), float(row.volume),
        )
        for row in df.itertuples(index=False)
    ]
    conn.executemany(
        """INSERT OR REPLACE INTO candles
           (symbol, timeframe, ts_ms, open, high, low, close, volume)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        rows,
    )
    conn.commit()
    return len(rows)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def backfill_macro_candles(
    db_path: str | None = None,
    assets: list[tuple[str, str, str]] | None = None,
    timeframes: list[tuple[str, str]] | None = None,
) -> dict[str, int]:
    """Full historical backfill for all macro assets.

    Parameters
    ----------
    db_path : str, optional
        Override the SQLite DB path (defaults to ``HOGAN_DB_PATH``).
    assets : list of (db_symbol, yf_ticker, description), optional
        Defaults to :data:`MACRO_ASSETS`.
    timeframes : list of (timeframe, period), optional
        Defaults to :data:`BACKFILL_TIMEFRAMES`.

    Returns
    -------
    dict mapping "SYMBOL_TF" to rows written.
    """
    _db = db_path or os.getenv("HOGAN_DB_PATH", "data/hogan.db")
    _assets = assets or MACRO_ASSETS
    _tfs = timeframes or BACKFILL_TIMEFRAMES
    conn = sqlite3.connect(_db)
    total: dict[str, int] = {}

    try:
        for db_sym, yf_tick, desc in _assets:
            for tf, period in _tfs:
                try:
                    df = _fetch_yf(yf_tick, tf, period)
                    n = _upsert_candles(conn, db_sym, tf, df)
                    key = f"{db_sym}_{tf}"
                    total[key] = n
                    logger.info(
                        "Macro candles: wrote %d %s rows for %s (%s)",
                        n, tf, db_sym, desc,
                    )
                except Exception as exc:
                    logger.warning(
                        "Macro candles: failed %s %s — %s", db_sym, tf, exc
                    )
                    total[f"{db_sym}_{tf}"] = 0
    finally:
        conn.close()

    total_rows = sum(total.values())
    logger.info("Macro candle backfill complete: %d rows across %d series", total_rows, len(total))
    return total


def fetch_all_macro_candles(
    db_path: str | None = None,
    assets: list[tuple[str, str, str]] | None = None,
) -> dict[str, int]:
    """Incremental daily update — fetch the last 7–30 days for each asset.

    Designed to be called from ``refresh_daily.py`` as a lightweight daily
    step that keeps the macro candle data current.
    """
    return backfill_macro_candles(
        db_path=db_path,
        assets=assets,
        timeframes=INCREMENTAL_TIMEFRAMES,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch macro-asset OHLCV candles into the Hogan DB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--db", default=os.getenv("HOGAN_DB_PATH", "data/hogan.db"))
    p.add_argument(
        "--backfill",
        action="store_true",
        help="Full historical backfill (1d × 10y + 1h × 2y). Run once.",
    )
    p.add_argument(
        "--period",
        default=None,
        help="Override the lookback period for all timeframes (e.g. 5y, 2y, 1y).",
    )
    p.add_argument(
        "--timeframe",
        default=None,
        choices=["1d", "1h"],
        help="Fetch only this timeframe (default: all).",
    )
    return p.parse_args()


if __name__ == "__main__":
    import json
    import sys

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()

    if args.backfill:
        tfs = BACKFILL_TIMEFRAMES
        if args.period:
            tfs = [(tf, args.period) for tf, _ in tfs]
        if args.timeframe:
            tfs = [(tf, p) for tf, p in tfs if tf == args.timeframe]
        result = backfill_macro_candles(db_path=args.db, timeframes=tfs)
    else:
        tfs = INCREMENTAL_TIMEFRAMES
        if args.timeframe:
            tfs = [(tf, p) for tf, p in tfs if tf == args.timeframe]
        result = fetch_all_macro_candles(db_path=args.db)

    total = sum(result.values())
    print(json.dumps({"total_rows": total, "detail": result}, indent=2))
    sys.exit(0 if total >= 0 else 1)
