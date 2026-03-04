"""Fetch social intelligence and developer activity from Santiment.

Requires a paid Santiment API key (Sanbase Pro ~$49/mo or higher).
Set ``SANTIMENT_KEY`` in your ``.env`` file.

Santiment uses a GraphQL API at https://api.santiment.net/graphql

Metrics fetched (daily)
-----------------------
``santiment_social_vol_chg``
    24-hour % change in total BTC social volume (mentions across
    Twitter/X, Reddit, Telegram).  Positive = rising mentions.
    Normalised: clipped to [-1, 1] after dividing by 100.
    Social volume spikes often precede price moves by 1-2 days.

``santiment_dev_activity_chg``
    7-day % change in Bitcoin Core GitHub development activity
    (commit frequency, PR activity, contributor count).
    Positive = rising dev activity (longer-term confidence signal).
    Normalised: clipped to [-1, 1] after dividing by 100.

Santiment API Reference
-----------------------
    https://academy.santiment.net/sanapi/
    GraphQL endpoint: https://api.santiment.net/graphql
    Slug for Bitcoin: "bitcoin"

Usage
-----
    # Add to .env:  SANTIMENT_KEY=xxxx
    # Daily refresh
    python -m hogan_bot.fetch_santiment

    # Historical backfill
    python -m hogan_bot.fetch_santiment --days 365
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

_GRAPHQL_URL = "https://api.santiment.net/graphql"
_TIMEOUT = 30
_SLEEP = 1.5
_SLUG = "bitcoin"


def _get_key() -> str:
    key = os.getenv("SANTIMENT_KEY", "").strip()
    if not key:
        sys.exit(
            "SANTIMENT_KEY is not set.\n"
            "1. Subscribe at https://app.santiment.net/pricing\n"
            "2. Add  SANTIMENT_KEY=<your_key>  to your .env"
        )
    return key


def _gql_request(api_key: str, query: str) -> dict:
    """Execute a GraphQL query against the Santiment API."""
    payload = json.dumps({"query": query}).encode()
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Apikey {api_key}",
    }
    for attempt in range(3):
        try:
            req = Request(_GRAPHQL_URL, data=payload, headers=headers, method="POST")
            with urlopen(req, timeout=_TIMEOUT) as resp:
                return json.loads(resp.read().decode())
        except HTTPError as exc:
            if exc.code == 429 and attempt < 2:
                wait = 30 * (attempt + 1)
                print(f"  Rate limited — waiting {wait}s ...")
                time.sleep(wait)
                continue
            raise RuntimeError(f"HTTP {exc.code} from Santiment GraphQL") from exc
        except URLError as exc:
            if attempt == 2:
                raise RuntimeError(f"Request failed: {exc}") from exc
            time.sleep(4 ** attempt)
    return {}


def _fetch_social_volume(api_key: str, from_dt: str, to_dt: str) -> list[dict]:
    """Fetch daily total BTC social volume timeseries."""
    query = f"""{{
      getMetric(metric: "social_volume_total") {{
        timeseriesData(
          slug: "{_SLUG}"
          from: "{from_dt}"
          to: "{to_dt}"
          interval: "1d"
        ) {{
          datetime
          value
        }}
      }}
    }}"""
    resp = _gql_request(api_key, query)
    return resp.get("data", {}).get("getMetric", {}).get("timeseriesData", [])


def _fetch_dev_activity(api_key: str, from_dt: str, to_dt: str) -> list[dict]:
    """Fetch daily Bitcoin Core developer activity timeseries."""
    query = f"""{{
      getMetric(metric: "dev_activity") {{
        timeseriesData(
          slug: "{_SLUG}"
          from: "{from_dt}"
          to: "{to_dt}"
          interval: "1d"
        ) {{
          datetime
          value
        }}
      }}
    }}"""
    resp = _gql_request(api_key, query)
    return resp.get("data", {}).get("getMetric", {}).get("timeseriesData", [])


def _pct_change_series(data: list[dict]) -> list[tuple[str, float]]:
    """Convert a raw timeseries into daily % change tuples ``(date, pct_chg)``."""
    results: list[tuple[str, float]] = []
    prev_val: float | None = None
    for item in data:
        raw_dt = item.get("datetime", "")
        try:
            dt = datetime.fromisoformat(raw_dt.replace("Z", "+00:00"))
            date_str = dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
        val = item.get("value")
        if val is None:
            prev_val = None
            continue
        val = float(val)
        if prev_val is not None and prev_val > 0:
            pct = (val - prev_val) / prev_val * 100.0
            results.append((date_str, pct))
        prev_val = val
    return results


def fetch_and_store(
    symbol: str = "BTC/USD",
    db_path: str = "data/hogan.db",
    days: int = 365,
) -> dict[str, int]:
    """Fetch Santiment social + dev metrics and upsert into ``onchain_metrics``.

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
    from_dt = (now - timedelta(days=days + 1)).strftime("%Y-%m-%dT00:00:00Z")
    to_dt = now.strftime("%Y-%m-%dT00:00:00Z")

    print(f"Fetching Santiment social + dev metrics ({days} days) ...")
    records: list[tuple[str, str, float]] = []
    written: dict[str, int] = {}

    # ── 1. Social Volume % Change ────────────────────────────────────────
    try:
        sv_raw = _fetch_social_volume(api_key, from_dt, to_dt)
        time.sleep(_SLEEP)
        sv_changes = _pct_change_series(sv_raw)
        sv_rows = 0
        for date_str, pct in sv_changes:
            # Clip to [-100, 100] then divide by 100 → [-1, 1]
            normalised = max(-100.0, min(100.0, pct)) / 100.0
            records.append((date_str, "santiment_social_vol_chg", normalised))
            sv_rows += 1
        print(f"  santiment_social_vol_chg:   {sv_rows} rows")
        written["santiment_social_vol_chg"] = sv_rows
    except Exception as exc:
        print(f"  Warning: social volume failed — {exc}")

    # ── 2. Developer Activity % Change (7-day rolling before %-change) ──
    try:
        da_raw = _fetch_dev_activity(api_key, from_dt, to_dt)
        time.sleep(_SLEEP)

        import pandas as pd
        if da_raw:
            dates = []
            vals = []
            for item in da_raw:
                raw_dt = item.get("datetime", "")
                try:
                    dt = datetime.fromisoformat(raw_dt.replace("Z", "+00:00"))
                    dates.append(dt.strftime("%Y-%m-%d"))
                    vals.append(float(item["value"]) if item.get("value") is not None else None)
                except (ValueError, TypeError):
                    continue

            s = pd.Series(vals, index=dates, dtype=float)
            rolling7 = s.rolling(7).mean()
            pct_chg = rolling7.pct_change() * 100.0

            da_rows = 0
            for date_str, pct in pct_chg.dropna().items():
                normalised = max(-100.0, min(100.0, float(pct))) / 100.0
                records.append((date_str, "santiment_dev_activity_chg", normalised))
                da_rows += 1
            print(f"  santiment_dev_activity_chg: {da_rows} rows")
            written["santiment_dev_activity_chg"] = da_rows
        else:
            print("  santiment_dev_activity_chg: no data returned")
    except Exception as exc:
        print(f"  Warning: developer activity failed — {exc}")

    if records:
        conn = get_connection(db_path)
        upsert_onchain(conn, symbol, records)
        conn.close()

    return written


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch Santiment social volume and developer activity (paid API)",
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
