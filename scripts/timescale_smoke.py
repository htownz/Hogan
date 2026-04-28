"""Optional Timescale integration smoke test.

Creates a tiny SQLite candle fixture, migrates it into Timescale/Postgres,
and reads it back through ``TimescaleCandleStore``. This requires a live
Timescale/Postgres URL and is intended for VPS/manual validation.
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import tempfile
from pathlib import Path

import pandas as pd

from hogan_bot.candle_store import TimescaleCandleStore
from hogan_bot.storage import _create_schema, upsert_candles
from scripts.migrate_candles_to_timescale import migrate


def _fixture() -> pd.DataFrame:
    ts = pd.date_range("2024-01-01", periods=3, freq="1h", tz="UTC")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": [100.0, 101.0, 102.0],
            "high": [101.0, 102.0, 103.0],
            "low": [99.0, 100.0, 101.0],
            "close": [100.5, 101.5, 102.5],
            "volume": [10.0, 11.0, 12.0],
        }
    )


def create_sqlite_fixture(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        _create_schema(conn)
        upsert_candles(conn, "BTC/USD", "1h", _fixture())
    finally:
        conn.close()


def run_smoke(database_url: str, sqlite_path: Path) -> int:
    create_sqlite_fixture(sqlite_path)
    results = migrate(str(sqlite_path), database_url, only="BTC/USD:1h", verify=True)
    if len(results) != 1 or not results[0].verified:
        raise SystemExit("Timescale migration verification failed")
    store = TimescaleCandleStore(database_url)
    try:
        loaded = store.load_candles("BTC/USD", "1h", limit=3)
        if len(loaded) < 3:
            raise SystemExit(f"Expected at least 3 migrated candles, got {len(loaded)}")
    finally:
        store.close()
    print("timescale smoke: ok")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--database-url", default=os.getenv("HOGAN_DATABASE_URL", ""))
    parser.add_argument("--sqlite-db", help="Optional path for the generated SQLite fixture")
    args = parser.parse_args(argv)
    if not args.database_url:
        raise SystemExit("--database-url or HOGAN_DATABASE_URL is required")
    if args.sqlite_db:
        return run_smoke(args.database_url, Path(args.sqlite_db))
    with tempfile.TemporaryDirectory() as tmp:
        return run_smoke(args.database_url, Path(tmp) / "hogan_timescale_smoke.db")


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
