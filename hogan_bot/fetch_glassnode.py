"""Fetch on-chain market structure metrics from the Glassnode API.

Requires a paid Glassnode API key (Standard tier or higher, ~$29/mo).
Set ``GLASSNODE_KEY`` in your ``.env`` file.

Metrics fetched (daily)
-----------------------
``glassnode_exchange_netflow``
    BTC flowing net into (+) or out of (-) exchanges per day.
    Normalised: value / 21_000_000 (total BTC supply) → tiny fraction.
    Positive = net inflow (bearish — whales moving to sell).
    Negative = net outflow (bullish — self-custody / HODLing).

``glassnode_realized_dist``
    (Current Price - Realized Price) / Realized Price.
    0 = trading at realized price; >0 = above cost basis (bull regime).
    Realized Price = aggregate cost basis of all BTC in existence.

``glassnode_puell_multiple``
    Daily miner revenue (USD) / 365-day MA of daily revenue.
    <0.5 = miners stressed / capitulating (historically good buy zones).
    >4   = miners extremely profitable (historically peak territory).
    Normalised: log1p(value) clipped to [-2, 4] before storage.

Glassnode API Reference
-----------------------
    https://docs.glassnode.com/basic-api/endpoints
    Base URL: https://api.glassnode.com/v1/metrics/

Usage
-----
    # Add to .env:  GLASSNODE_KEY=xxxx
    # Daily refresh
    python -m hogan_bot.fetch_glassnode

    # Historical backfill (defaults to 365 days)
    python -m hogan_bot.fetch_glassnode --days 365
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_BASE_URL = "https://api.glassnode.com/v1/metrics"
_TIMEOUT = 30
_SLEEP = 1.0   # 1 s between calls — well within Standard rate limits

_TOTAL_BTC_SUPPLY = 21_000_000.0


def _get_key() -> str:
    key = os.getenv("GLASSNODE_KEY", "").strip()
    if not key:
        sys.exit(
            "GLASSNODE_KEY is not set.\n"
            "1. Subscribe at https://glassnode.com/pricing\n"
            "2. Add  GLASSNODE_KEY=<your_key>  to your .env"
        )
    return key


def _fetch_metric(
    api_key: str,
    category: str,
    metric: str,
    asset: str = "BTC",
    since_ts: int | None = None,
    until_ts: int | None = None,
) -> list[dict]:
    """Fetch a single Glassnode metric timeseries.

    Returns a list of ``{"t": unix_timestamp, "v": float}`` dicts.
    """
    params = f"a={asset}&api_key={api_key}&i=24h"
    if since_ts:
        params += f"&s={since_ts}"
    if until_ts:
        params += f"&u={until_ts}"
    url = f"{_BASE_URL}/{category}/{metric}?{params}"

    for attempt in range(3):
        try:
            req = Request(url, headers={"Accept": "application/json"})
            with urlopen(req, timeout=_TIMEOUT) as resp:
                return json.loads(resp.read().decode())
        except HTTPError as exc:
            if exc.code == 429 and attempt < 2:
                wait = 60 * (attempt + 1)
                print(f"  Rate limited — waiting {wait}s ...")
                time.sleep(wait)
                continue
            if exc.code == 403:
                raise RuntimeError(
                    f"HTTP 403 — {category}/{metric} may require a higher Glassnode tier.\n"
                    "Check https://glassnode.com/pricing for metric availability."
                ) from exc
            raise RuntimeError(f"HTTP {exc.code} fetching {category}/{metric}") from exc
        except URLError as exc:
            if attempt == 2:
                raise RuntimeError(f"Request failed {category}/{metric}: {exc}") from exc
            time.sleep(4 ** attempt)
    return []


def _ts_to_date(ts: int) -> str:
    return datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")


def fetch_and_store(
    symbol: str = "BTC/USD",
    db_path: str = "data/hogan.db",
    days: int = 365,
) -> dict[str, int]:
    """Fetch all three Glassnode metrics and upsert into ``onchain_metrics``.

    Parameters
    ----------
    days:
        Number of historical days to fetch.

    Returns
    -------
    dict
        ``{metric_name: rows_written}``
    """
    from hogan_bot.storage import get_connection, upsert_onchain

    api_key = _get_key()
    now = datetime.now(timezone.utc)
    since_ts = int((now - timedelta(days=days)).timestamp())
    until_ts = int(now.timestamp())

    print(f"Fetching Glassnode on-chain metrics ({days} days) ...")
    records: list[tuple[str, str, float]] = []
    written: dict[str, int] = {}

    # ── 1. Exchange Net Flow ────────────────────────────────────────────
    try:
        data = _fetch_metric(api_key, "transactions", "transfers_volume_exchanges_net",
                             since_ts=since_ts, until_ts=until_ts)
        time.sleep(_SLEEP)
        netflow_rows = 0
        for pt in data:
            val = pt.get("v")
            if val is None:
                continue
            normalised = float(val) / _TOTAL_BTC_SUPPLY
            records.append((_ts_to_date(pt["t"]), "glassnode_exchange_netflow", normalised))
            netflow_rows += 1
        print(f"  glassnode_exchange_netflow:  {netflow_rows} rows")
        written["glassnode_exchange_netflow"] = netflow_rows
    except Exception as exc:
        print(f"  Warning: exchange netflow failed — {exc}")

    # ── 2. Realized Price Distance ──────────────────────────────────────
    try:
        # Fetch both current price and realized price
        rp_data = _fetch_metric(api_key, "market", "price_realized_usd",
                                since_ts=since_ts, until_ts=until_ts)
        price_data = _fetch_metric(api_key, "market", "price_usd_close",
                                   since_ts=since_ts, until_ts=until_ts)
        time.sleep(_SLEEP)

        price_map = {_ts_to_date(pt["t"]): float(pt["v"])
                     for pt in price_data if pt.get("v") is not None}
        rp_rows = 0
        for pt in rp_data:
            if pt.get("v") is None:
                continue
            date_str = _ts_to_date(pt["t"])
            realized = float(pt["v"])
            current = price_map.get(date_str)
            if current and realized > 0:
                dist = (current - realized) / realized
                records.append((date_str, "glassnode_realized_dist", float(dist)))
                rp_rows += 1
        print(f"  glassnode_realized_dist:     {rp_rows} rows")
        written["glassnode_realized_dist"] = rp_rows
    except Exception as exc:
        print(f"  Warning: realized price distance failed — {exc}")

    # ── 3. Puell Multiple ───────────────────────────────────────────────
    try:
        data = _fetch_metric(api_key, "mining", "puell_multiple",
                             since_ts=since_ts, until_ts=until_ts)
        time.sleep(_SLEEP)
        puell_rows = 0
        for pt in data:
            val = pt.get("v")
            if val is None:
                continue
            # log1p to compress the right tail, clip to [-2, 4]
            log_val = float(math.log1p(max(float(val), 0.0)))
            clipped = max(-2.0, min(4.0, log_val))
            records.append((_ts_to_date(pt["t"]), "glassnode_puell_multiple", clipped))
            puell_rows += 1
        print(f"  glassnode_puell_multiple:    {puell_rows} rows")
        written["glassnode_puell_multiple"] = puell_rows
    except Exception as exc:
        print(f"  Warning: Puell Multiple failed — {exc}")

    if records:
        conn = get_connection(db_path)
        upsert_onchain(conn, symbol, records)
        conn.close()

    return written


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch Glassnode on-chain metrics (exchange flow, realized price, Puell)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="BTC/USD", help="Symbol for DB storage")
    p.add_argument("--db", default="data/hogan.db", help="SQLite database path")
    p.add_argument("--days", type=int, default=365, help="Number of historical days to fetch")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fetch_and_store(symbol=args.symbol, db_path=args.db, days=args.days)


if __name__ == "__main__":
    main()
