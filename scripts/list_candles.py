"""Print a summary of all candle series stored in the Hogan DB."""
import argparse
import sqlite3
from datetime import datetime, timezone


def main() -> None:
    p = argparse.ArgumentParser(description="List candle series in the Hogan DB")
    p.add_argument("--db", default="data/hogan.db")
    args = p.parse_args()

    conn = sqlite3.connect(args.db)
    rows = conn.execute(
        """
        SELECT symbol, timeframe, COUNT(*) as n, MIN(ts_ms), MAX(ts_ms)
        FROM candles
        GROUP BY symbol, timeframe
        ORDER BY symbol, timeframe
        """
    ).fetchall()
    conn.close()

    if not rows:
        print("No candles found in the database.")
        return

    print(f"{'Symbol':<14} {'TF':<6} {'Bars':>8}  {'Oldest':<20} {'Newest'}")
    print("-" * 72)
    for sym, tf, n, ts_min, ts_max in rows:
        oldest = datetime.fromtimestamp(ts_min / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        newest = datetime.fromtimestamp(ts_max / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M")
        print(f"{sym:<14} {tf:<6} {n:>8}  {oldest:<20} {newest}")

    total = sum(r[2] for r in rows)
    print("-" * 72)
    print(f"{'TOTAL':<14} {'':<6} {total:>8}  rows across {len(rows)} series")


if __name__ == "__main__":
    main()
