"""CLI tool to fetch OHLCV candles from Kraken and upsert them into the local
SQLite database.

Usage::

    # Fetch 5 000 5-minute BTC/USD bars and store them
    python -m hogan_bot.fetch_data --symbol BTC/USD --timeframe 5m --limit 5000

    # Use a custom database path
    python -m hogan_bot.fetch_data --symbol ETH/USD --db data/hogan.db

    # Refresh multiple symbols in one invocation
    python -m hogan_bot.fetch_data --symbol BTC/USD ETH/USD --limit 2000

    # Backfill up to 20k 1h bars (paginates in 720-bar chunks)
    python -m hogan_bot.fetch_data --symbol BTC/USD --timeframe 1h --backfill --target-bars 20000

Output is a JSON object per symbol printed to stdout.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time

from hogan_bot.exchange import ExchangeClient
from hogan_bot.storage import candle_count, get_connection, oldest_ts_ms, upsert_candles

logger = logging.getLogger(__name__)

# Milliseconds per timeframe — used for pagination when walking backward.
_TF_MS: dict[str, int] = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
}

_CHUNK_SIZE = 720  # Kraken/exchange per-request limit


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
    p.add_argument("--timeframe", default="1h", help="OHLCV bar interval, e.g. 1m 5m 1h")
    p.add_argument("--limit", type=int, default=5000, help="Number of bars to fetch per symbol")
    p.add_argument(
        "--exchange",
        default=os.getenv("HOGAN_EXCHANGE", "kraken"),
        help="CCXT exchange ID, e.g. kraken binance bybit coinbase (default: kraken)",
    )
    p.add_argument(
        "--db",
        default=os.path.join("data", "hogan.db"),
        help="Path to the SQLite database file",
    )
    p.add_argument(
        "--backfill",
        action="store_true",
        help="Paginate in chunks to reach --target-bars (for deep history)",
    )
    p.add_argument(
        "--target-bars",
        type=int,
        default=None,
        help="Target bar count when --backfill (default: 20k for 1h, 40k for 30m)",
    )
    return p


def backfill_and_store(
    symbol: str,
    timeframe: str,
    target_bars: int,
    db_path: str,
    exchange_id: str = "kraken",
) -> dict:
    """Fetch up to *target_bars* by paginating in chunks, upserting each chunk.

    Walks backward from the most recent data (or from DB's oldest bar when
    extending). Stops when target is reached or the exchange returns no new data.
    """
    bar_ms = _TF_MS.get(timeframe, 3_600_000)
    client = ExchangeClient(exchange_id, api_key=None, api_secret=None)

    os.makedirs(os.path.dirname(os.path.abspath(db_path)), exist_ok=True)
    conn = get_connection(db_path)
    before = candle_count(conn, symbol, timeframe)

    total_fetched = 0
    chunk_num = 0
    min_progress = max(10, _CHUNK_SIZE // 20)

    # Start from DB's oldest (extend backward) or from "now" (most recent first)
    since_ms: int | None = None
    db_oldest = oldest_ts_ms(conn, symbol, timeframe)
    if db_oldest is not None and before < target_bars:
        # Extend backward: fetch bars before our oldest
        since_ms = db_oldest - _CHUNK_SIZE * bar_ms
        logger.info(
            "Backfill %s %s: extending from DB oldest (%d bars), fetching backward",
            symbol, timeframe, before,
        )

    while before + total_fetched < target_bars:
        chunk_num += 1
        df = client.fetch_ohlcv_df(
            symbol, timeframe=timeframe, limit=_CHUNK_SIZE, since=since_ms
        )
        if df.empty:
            logger.info("Backfill %s %s: chunk %d empty, stopping", symbol, timeframe, chunk_num)
            break

        upsert_candles(conn, symbol, timeframe, df)
        n = len(df)
        total_fetched += n

        oldest_ts = df["timestamp"].iloc[0]
        oldest_ms = int(oldest_ts.timestamp() * 1000)

        if n < min_progress:
            logger.info(
                "Backfill %s %s: chunk %d had only %d bars (exchange limit?), stopping",
                symbol, timeframe, chunk_num, n,
            )
            break

        # Next chunk: bars before current oldest
        since_ms = oldest_ms - _CHUNK_SIZE * bar_ms
        print(
            f"  chunk {chunk_num}: +{n} bars -> {before + total_fetched} total",
            flush=True,
        )

        if before + total_fetched >= target_bars:
            break
        time.sleep(0.4)  # rate limit

    after = candle_count(conn, symbol, timeframe)
    conn.close()

    return {
        "symbol": symbol,
        "timeframe": timeframe,
        "fetched": total_fetched,
        "new_rows": after - before,
        "total_stored": after,
    }


def fetch_and_store(
    symbol: str,
    timeframe: str,
    limit: int,
    db_path: str,
    exchange_id: str = "kraken",
) -> dict:
    """Fetch *limit* bars for *symbol* and upsert into *db_path*.

    Returns a summary dict with fetch/store statistics.
    """
    client = ExchangeClient(exchange_id, api_key=None, api_secret=None)
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


def _default_target_bars(timeframe: str) -> int:
    """Default target bar count for backfill by timeframe."""
    return 40_000 if timeframe == "30m" else 20_000


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    results = []
    errors = []

    target_bars = args.target_bars
    if args.backfill and target_bars is None:
        target_bars = _default_target_bars(args.timeframe)

    for symbol in args.symbol:
        try:
            if args.backfill:
                summary = backfill_and_store(
                    symbol, args.timeframe, target_bars, args.db, args.exchange
                )
            else:
                summary = fetch_and_store(
                    symbol, args.timeframe, args.limit, args.db, args.exchange
                )
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
