"""Fetch the Crypto Fear & Greed Index from Alternative.me.

No API key required.  Free endpoint with no enforced rate limits.

Endpoint
--------
    GET https://api.alternative.me/fng/?limit=N&format=json

Metric stored
-------------
    ``fear_greed_value`` — raw integer 0-100 (Extreme Fear=0, Extreme Greed=100)
    stored daily in the ``onchain_metrics`` table.

    In ``build_ext_features()`` this is divided by 100 → [0, 1].

Interpretation
--------------
    < 20  Extreme Fear  (contrarian buy signal)
    20-40 Fear
    40-60 Neutral
    60-80 Greed
    > 80  Extreme Greed (contrarian sell signal)

Usage
-----
    # Backfill up to 5 years of history (first run)
    python -m hogan_bot.fetch_feargreed --backfill

    # Fetch today only (cron / daily refresh)
    python -m hogan_bot.fetch_feargreed
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_FNG_URL = "https://api.alternative.me/fng/"
_TIMEOUT = 20
_METRIC = "fear_greed_value"


def _fetch(limit: int = 1) -> list[dict]:
    """Return a list of ``{date: "YYYY-MM-DD", value: int}`` dicts."""
    url = f"{_FNG_URL}?limit={limit}&format=json&date_format=us"
    for attempt in range(3):
        try:
            req = Request(url, headers={"Accept": "application/json"})
            with urlopen(req, timeout=_TIMEOUT) as resp:
                data = json.loads(resp.read().decode())
            records = data.get("data", [])
            result = []
            for item in records:
                raw_date = item.get("timestamp", "")
                # Alternative.me returns US format: "MM-DD-YYYY" with date_format=us
                try:
                    dt = datetime.strptime(raw_date, "%m-%d-%Y")
                    date_str = dt.strftime("%Y-%m-%d")
                except ValueError:
                    # fallback: treat as unix timestamp
                    try:
                        ts = int(raw_date)
                        dt = datetime.fromtimestamp(ts, tz=timezone.utc)
                        date_str = dt.strftime("%Y-%m-%d")
                    except (ValueError, OSError):
                        continue
                result.append({"date": date_str, "value": int(item.get("value", 0))})
            return result
        except HTTPError as exc:
            if exc.code == 429 and attempt < 2:
                time.sleep(10 * (attempt + 1))
                continue
            raise RuntimeError(f"HTTP {exc.code} fetching Fear & Greed data") from exc
        except URLError as exc:
            if attempt == 2:
                raise RuntimeError(f"Request failed: {exc}") from exc
            time.sleep(4 ** attempt)
    return []


def fetch_and_store(
    symbol: str = "BTC/USD",
    db_path: str = "data/hogan.db",
    backfill: bool = False,
) -> int:
    """Fetch Fear & Greed data and upsert into ``onchain_metrics``.

    Parameters
    ----------
    backfill:
        When True, fetch up to 2000 days (~5.5 years) of history.
        Otherwise only fetch today's reading.

    Returns
    -------
    int
        Number of rows written.
    """
    from hogan_bot.storage import get_connection, upsert_onchain

    limit = 2000 if backfill else 1
    label = f"{limit} days" if backfill else "today"
    print(f"Fetching Fear & Greed Index ({label}) ...")

    records_raw = _fetch(limit=limit)
    if not records_raw:
        print("  No data returned.")
        return 0

    rows = [(r["date"], _METRIC, float(r["value"])) for r in records_raw]
    conn = get_connection(db_path)
    try:
        written = upsert_onchain(conn, symbol, rows)
    finally:
        conn.close()

    if records_raw:
        latest = max(records_raw, key=lambda r: r["date"])
        print(f"  Latest: {latest['date']}  value={latest['value']}")
    print(f"  {len(rows)} records upserted.")
    return written


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch Crypto Fear & Greed Index from Alternative.me",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="BTC/USD", help="Symbol for DB storage")
    p.add_argument("--db", default="data/hogan.db", help="SQLite database path")
    p.add_argument(
        "--backfill", action="store_true",
        help="Fetch ~5 years of historical data instead of just today",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fetch_and_store(symbol=args.symbol, db_path=args.db, backfill=args.backfill)


if __name__ == "__main__":
    main()
