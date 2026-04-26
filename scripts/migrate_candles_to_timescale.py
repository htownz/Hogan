"""Migrate SQLite candle history into TimescaleDB/Postgres.

This intentionally migrates only the ``candles`` time-series table. Trading
state, paper trades, fills, swarm rows, and operational state remain in SQLite
until the candle path is validated in production/paper.
"""
from __future__ import annotations

import argparse
import os
from dataclasses import dataclass

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


@dataclass(frozen=True)
class MigrationResult:
    symbol: str
    timeframe: str
    source_count: int
    migrated_count: int
    destination_count: int | None = None

    @property
    def verified(self) -> bool:
        return self.destination_count is None or self.destination_count >= self.source_count


def _migration_plan(src, filters: set[tuple[str, str]] | None) -> list[tuple[str, str, int]]:
    plan: list[tuple[str, str, int]] = []
    for symbol, timeframe, count in available_symbols(src):
        if filters is not None and (symbol, timeframe) not in filters:
            continue
        plan.append((symbol, timeframe, count))
    return plan


def migrate(
    sqlite_db: str,
    database_url: str,
    *,
    only: str | None = None,
    dry_run: bool = False,
    verify: bool = False,
) -> list[MigrationResult]:
    filters = _parse_symbol_filter(only)
    src = get_connection(sqlite_db)
    dst = None if dry_run else TimescaleCandleStore(database_url)
    results: list[MigrationResult] = []
    try:
        for symbol, timeframe, count in _migration_plan(src, filters):
            if dry_run:
                results.append(MigrationResult(symbol, timeframe, count, 0, None))
                print(f"{symbol} {timeframe}: would migrate {count} candles")
                continue

            df = load_candles(src, symbol, timeframe)
            written = dst.upsert_candles(symbol, timeframe, df) if dst is not None else 0
            destination_count = (
                dst.candle_count(symbol, timeframe)
                if verify and dst is not None
                else None
            )
            results.append(
                MigrationResult(
                    symbol=symbol,
                    timeframe=timeframe,
                    source_count=count,
                    migrated_count=written,
                    destination_count=destination_count,
                )
            )
            print(f"{symbol} {timeframe}: migrated {written}/{count} candles")
            if verify and destination_count is not None:
                status = "ok" if destination_count >= count else "mismatch"
                print(f"{symbol} {timeframe}: verify {status} ({destination_count}/{count})")
    finally:
        src.close()
        if dst is not None:
            dst.close()
    return results


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
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show the symbol/timeframe migration plan without writing to Timescale.",
    )
    parser.add_argument(
        "--verify",
        action="store_true",
        help="After each copy, compare destination candle count against source count.",
    )
    args = parser.parse_args()
    if not args.database_url and not args.dry_run:
        raise SystemExit("--database-url or HOGAN_DATABASE_URL is required")
    results = migrate(
        args.sqlite_db,
        args.database_url,
        only=args.only,
        dry_run=args.dry_run,
        verify=args.verify,
    )
    failed = [r for r in results if not r.verified]
    if failed:
        raise SystemExit(
            "verification failed for: "
            + ", ".join(f"{r.symbol}:{r.timeframe}" for r in failed)
        )
    total = sum(r.migrated_count for r in results)
    print(f"done: migrated {total} candles")


if __name__ == "__main__":
    main()
