"""Migrate SQLite candle history into TimescaleDB/Postgres.

This intentionally migrates only the ``candles`` time-series table. Trading
state, paper trades, fills, swarm rows, and operational state remain in SQLite
until the candle path is validated in production/paper.
"""
from __future__ import annotations

import argparse
import os

from hogan_bot.candle_store import TimescaleCandleStore
from hogan_bot.storage import available_symbols, get_connection, load_candles


def _parse_symbol_filter(raw: str | None) -> set[tuple[str, str]] | None:
    if not raw:
        return None
    out: set[tuple[str, str]] = set()
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if ":" not in item:
            raise ValueError("symbol filters must look like BTC/USD:1h")
        symbol, timeframe = item.rsplit(":", 1)
        out.add((symbol.strip(), timeframe.strip()))
    return out


def migrate(sqlite_db: str, database_url: str, *, only: str | None = None) -> int:
    filters = _parse_symbol_filter(only)
    src = get_connection(sqlite_db)
    dst = TimescaleCandleStore(database_url)
    total = 0
    try:
        for symbol, timeframe, count in available_symbols(src):
            if filters is not None and (symbol, timeframe) not in filters:
                continue
            df = load_candles(src, symbol, timeframe)
            written = dst.upsert_candles(symbol, timeframe, df)
            total += written
            print(f"{symbol} {timeframe}: migrated {written}/{count} candles")
    finally:
        src.close()
        dst.close()
    return total


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sqlite-db", default=os.getenv("HOGAN_DB_PATH", "data/hogan.db"))
    parser.add_argument(
        "--database-url",
        default=os.getenv("HOGAN_DATABASE_URL", ""),
        help="Timescale/Postgres URL, e.g. postgresql://hogan:hogan@localhost:5432/hogan",
    )
    parser.add_argument(
        "--only",
        help="Comma-separated symbol:timeframe filters, e.g. BTC/USD:1h,ETH/USD:1m",
    )
    args = parser.parse_args()
    if not args.database_url:
        raise SystemExit("--database-url or HOGAN_DATABASE_URL is required")
    total = migrate(args.sqlite_db, args.database_url, only=args.only)
    print(f"done: migrated {total} candles")


if __name__ == "__main__":
    main()
