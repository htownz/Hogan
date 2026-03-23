#!/usr/bin/env python
"""Backfill historical candles from Oanda into the Hogan SQLite DB.

Usage:
    python scripts/backfill_oanda.py --symbol GBP/USD --timeframe 1h --days 800
    python scripts/backfill_oanda.py --symbol GBP/USD --timeframe 1h --from 2024-03-01 --to 2026-03-20
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv
load_dotenv()

from hogan_bot.oanda_client import OandaClient
from hogan_bot.storage import upsert_candles


def _rfc3339(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S.000000000Z")


def _chunk_size(timeframe: str) -> tuple[int, timedelta]:
    """Return (max_bars_per_request, timedelta_per_chunk) for a given TF."""
    tf_seconds = {
        "1m": 60, "5m": 300, "15m": 900, "30m": 1800,
        "1h": 3600, "4h": 14400, "1d": 86400,
    }
    secs = tf_seconds.get(timeframe, 3600)
    bars = 4500
    return bars, timedelta(seconds=secs * bars)


def backfill(
    symbol: str,
    timeframe: str,
    start: datetime,
    end: datetime,
    db_path: str = "data/hogan.db",
):
    client = OandaClient()
    conn = sqlite3.connect(db_path)

    _, chunk_delta = _chunk_size(timeframe)
    total_inserted = 0
    cursor = start

    print(f"Backfilling {symbol} {timeframe} from {start.date()} to {end.date()}")
    print(f"  Chunk size: {chunk_delta}")

    while cursor < end:
        chunk_end = min(cursor + chunk_delta, end)
        from_str = _rfc3339(cursor)
        to_str = _rfc3339(chunk_end)

        try:
            df = client.fetch_candles(
                symbol=symbol,
                timeframe=timeframe,
                from_time=from_str,
                to_time=to_str,
                count=None,
            )
        except Exception as exc:
            print(f"  ERROR at {cursor.date()}: {exc}")
            time.sleep(2)
            cursor = chunk_end
            continue

        if df is not None and not df.empty:
            upsert_candles(conn, symbol, timeframe, df)
            total_inserted += len(df)
            print(f"  {cursor.date()} -> {chunk_end.date()}: {len(df)} bars (total: {total_inserted})")
        else:
            print(f"  {cursor.date()} -> {chunk_end.date()}: 0 bars")

        cursor = chunk_end
        time.sleep(0.5)

    conn.close()
    print(f"\nDone. Total bars inserted/updated: {total_inserted}")


def main():
    parser = argparse.ArgumentParser(description="Backfill Oanda candles")
    parser.add_argument("--symbol", default="GBP/USD")
    parser.add_argument("--timeframe", default="1h")
    parser.add_argument("--days", type=int, default=None,
                        help="Days of history from today (alternative to --from/--to)")
    parser.add_argument("--from", dest="from_date", default=None,
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--to", dest="to_date", default=None,
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--db", default="data/hogan.db")
    args = parser.parse_args()

    end = datetime.now(timezone.utc)
    if args.to_date:
        end = datetime.strptime(args.to_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)

    if args.from_date:
        start = datetime.strptime(args.from_date, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    elif args.days:
        start = end - timedelta(days=args.days)
    else:
        start = end - timedelta(days=800)

    backfill(args.symbol, args.timeframe, start, end, args.db)


if __name__ == "__main__":
    main()
