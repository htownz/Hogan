"""CLI tool to fetch OHLCV candles from Kraken and upsert them into the local
SQLite database.

Usage::

    # Fetch 5 000 5-minute BTC/USD bars and store them
    python -m hogan_bot.fetch_data --symbol BTC/USD --timeframe 5m --limit 5000

    # Use a custom database path
    python -m hogan_bot.fetch_data --symbol ETH/USD --db data/hogan.db

    # Refresh multiple symbols in one invocation
    python -m hogan_bot.fetch_data --symbol BTC/USD ETH/USD --limit 2000

Output is a JSON object per symbol printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import os
import sys

from hogan_bot.exchange import KrakenClient
from hogan_bot.storage import candle_count, get_connection, upsert_candles


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Fetch Kraken OHLCV candles and upsert into local SQLite DB",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--symbol",
        nargs="+",
        default=["BTC/USD"],
        metavar="SYMBOL",
        help="One or more trading pairs, e.g. BTC/USD ETH/USD",
    )
    p.add_argument("--timeframe", default="5m", help="OHLCV bar interval, e.g. 1m 5m 1h")
    p.add_argument("--limit", type=int, default=5000, help="Number of bars to fetch per symbol")
    p.add_argument(
        "--db",
        default=os.path.join("data", "hogan.db"),
        help="Path to the SQLite database file",
    )
    return p


def fetch_and_store(
    symbol: str,
    timeframe: str,
    limit: int,
    db_path: str,
) -> dict:
    """Fetch *limit* bars for *symbol* and upsert into *db_path*.

    Returns a summary dict with fetch/store statistics.
    """
    client = KrakenClient(api_key=None, api_secret=None)
    df = client.fetch_ohlcv_df(symbol, timeframe=timeframe, limit=limit)

    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = get_connection(db_path)
    before = candle_count(conn, symbol, timeframe)
    upsert_candles(conn, symbol, timeframe, df)
    after = candle_count(conn, symbol, timeframe)
    conn.close()

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "fetched": len(df),
        "new_rows": after - before,
        "total_stored": after,
    }


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    results = []
    errors = []

    for symbol in args.symbol:
        try:
            summary = fetch_and_store(symbol, args.timeframe, args.limit, args.db)
            results.append(summary)
            print(json.dumps(summary), flush=True)
        except Exception as exc:  # noqa: BLE001
            msg = {"symbol": symbol, "error": str(exc)}
            errors.append(msg)
            print(json.dumps(msg), file=sys.stderr, flush=True)

    if errors:
        sys.exit(1)


if __name__ == "__main__":
    main()
