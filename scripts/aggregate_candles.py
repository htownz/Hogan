#!/usr/bin/env python
"""Aggregate 1h candles to 4h and store in the same DB with timeframe='4h'.

Usage:
    python scripts/aggregate_candles.py --db data/hogan.db --assets BTC/USD ETH/USD SOL/USD
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def aggregate_1h_to_4h(conn: sqlite3.Connection, symbol: str) -> int:
    query = """
        SELECT ts_ms, open, high, low, close, volume
        FROM candles
        WHERE symbol = ? AND timeframe = '1h'
        ORDER BY ts_ms
    """
    df = pd.read_sql_query(query, conn, params=[symbol])
    if df.empty:
        return 0

    df["dt"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df = df.set_index("dt")

    agg = df.resample("4h").agg({
        "ts_ms": "first",
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last",
        "volume": "sum",
    }).dropna()

    agg = agg.reset_index(drop=True)
    agg["symbol"] = symbol
    agg["timeframe"] = "4h"
    agg["ts_ms"] = agg["ts_ms"].astype(int)

    cursor = conn.cursor()
    for _, row in agg.iterrows():
        cursor.execute("""
            INSERT OR REPLACE INTO candles (symbol, timeframe, ts_ms, open, high, low, close, volume)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (row["symbol"], row["timeframe"], row["ts_ms"],
              row["open"], row["high"], row["low"], row["close"], row["volume"]))

    conn.commit()
    return len(agg)


def main():
    parser = argparse.ArgumentParser(description="Aggregate 1h -> 4h candles")
    parser.add_argument("--db", default="data/hogan.db")
    parser.add_argument("--assets", nargs="+", default=["BTC/USD", "ETH/USD", "SOL/USD"])
    args = parser.parse_args()

    conn = sqlite3.connect(args.db)

    for asset in args.assets:
        count = aggregate_1h_to_4h(conn, asset)
        print(f"{asset}: {count} 4h candles written")

    conn.close()
    print("Done.")


if __name__ == "__main__":
    main()
