#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shutil
import sqlite3
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="Create trimmed canary DB snapshot")
    p.add_argument("--src", default="data/hogan.db")
    p.add_argument("--dst", default="data/hogan_vnext_canary.db")
    p.add_argument("--symbol", default="BTC/USD")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--keep-bars", type=int, default=6000)
    args = p.parse_args()

    src = Path(args.src)
    dst = Path(args.dst)
    shutil.copy2(src, dst)

    conn = sqlite3.connect(dst)
    try:
        conn.execute(
            """
            DELETE FROM candles
            WHERE symbol = ?
              AND timeframe = ?
              AND ts_ms NOT IN (
                SELECT ts_ms
                FROM candles
                WHERE symbol = ?
                  AND timeframe = ?
                ORDER BY ts_ms DESC
                LIMIT ?
              )
            """,
            (args.symbol, args.timeframe, args.symbol, args.timeframe, int(args.keep_bars)),
        )
        conn.commit()
        n = conn.execute(
            "SELECT COUNT(*) FROM candles WHERE symbol = ? AND timeframe = ?",
            (args.symbol, args.timeframe),
        ).fetchone()[0]
    finally:
        conn.close()

    print(f"canary_db={dst} symbol={args.symbol} timeframe={args.timeframe} bars={n}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
