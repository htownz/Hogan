"""Fetch on-chain metrics from the CryptoQuant API.

Requires a CryptoQuant API key stored in the environment variable
``CRYPTOQUANT_KEY`` (or in the ``.env`` file).  Data is upserted into the
``onchain_metrics`` SQLite table as daily observations.

Metrics fetched
---------------
* **MVRV z-score**  — Market Value to Realised Value z-score; values above
  +7 historically mark cycle tops, below 0 mark bottoms.
  Endpoint: ``/v1/btc/market-data/mvrv``

* **SOPR**          — Spent Output Profit Ratio; > 1 means holders are
  selling at a profit.
  Endpoint: ``/v1/btc/market-data/sopr``

* **Active addresses** (7-day MA % change) — proxy for network activity.
  Endpoint: ``/v1/btc/network-data/active-addresses``

Usage
-----
    # Set key first:
    export CRYPTOQUANT_KEY=your_key_here

    python -m hogan_bot.fetch_onchain --symbol BTC/USD

    # Custom history window
    python -m hogan_bot.fetch_onchain --symbol BTC/USD --days 365
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from urllib.error import URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

_CQ_BASE = "https://api.cryptoquant.com"
_REQUEST_TIMEOUT = 20
_DEFAULT_DAYS = 365


def _get_api_key() -> str:
    key = os.getenv("CRYPTOQUANT_KEY", "").strip()
    if not key:
        sys.exit(
            "CRYPTOQUANT_KEY environment variable is not set.\n"
            "Add it to your .env file:  CRYPTOQUANT_KEY=your_key_here"
        )
    return key


def _fetch_json(url: str, api_key: str) -> dict:
    """GET *url* with Bearer auth, up to 3 retries."""
    req = Request(url, headers={"Authorization": f"Bearer {api_key}"})
    for attempt in range(3):
        try:
            with urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
                return json.loads(resp.read().decode())
        except URLError as exc:
            if attempt == 2:
                raise RuntimeError(f"Request failed: {url}  ({exc})") from exc
            time.sleep(2 ** attempt)
    return {}


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def _date_range_params(days: int) -> tuple[str, str]:
    """Return ISO-8601 date strings for *now-days* to *now*."""
    end = datetime.now(timezone.utc).date()
    start = end - timedelta(days=days)
    return start.isoformat(), end.isoformat()


def fetch_mvrv(api_key: str, symbol: str = "BTC/USD", days: int = _DEFAULT_DAYS) -> list[tuple[str, str, float]]:
    """Return ``(date, 'mvrv_zscore', z_score)`` records.

    CryptoQuant's MVRV endpoint returns raw MVRV; we compute a rolling
    365-day z-score from the series to produce the z-score feature.
    """
    start, end = _date_range_params(days + 365)  # extra year for z-score window
    url = (
        f"{_CQ_BASE}/v1/btc/market-data/mvrv"
        f"?window=day&from={start}&to={end}&limit=9999"
    )
    data = _fetch_json(url, api_key)
    items = data.get("result", {}).get("data", data.get("data", []))

    series: list[tuple[str, float]] = []
    for item in items:
        try:
            date = str(item.get("date", item.get("timestamp", "")))[:10]
            val = float(item.get("mvrv", item.get("value", 0.0)))
            series.append((date, val))
        except (KeyError, TypeError, ValueError):
            continue

    series.sort()
    if not series:
        return []

    # Compute rolling 365-day z-score
    dates_arr = [s[0] for s in series]
    vals_arr = np.array([s[1] for s in series], dtype=np.float64)
    window = 365
    records: list[tuple[str, str, float]] = []
    for i in range(window - 1, len(series)):
        w = vals_arr[max(0, i - window + 1): i + 1]
        mu, sigma = float(np.mean(w)), float(np.std(w))
        z = (vals_arr[i] - mu) / max(sigma, 1e-9)
        records.append((dates_arr[i], "mvrv_zscore", float(z)))
    return records


def fetch_sopr(api_key: str, symbol: str = "BTC/USD", days: int = _DEFAULT_DAYS) -> list[tuple[str, str, float]]:
    """Return ``(date, 'sopr', value)`` records."""
    start, end = _date_range_params(days)
    url = (
        f"{_CQ_BASE}/v1/btc/market-data/sopr"
        f"?window=day&from={start}&to={end}&limit=9999"
    )
    data = _fetch_json(url, api_key)
    items = data.get("result", {}).get("data", data.get("data", []))

    records: list[tuple[str, str, float]] = []
    for item in items:
        try:
            date = str(item.get("date", item.get("timestamp", "")))[:10]
            val = float(item.get("sopr", item.get("value", 1.0)))
            records.append((date, "sopr", val))
        except (KeyError, TypeError, ValueError):
            continue
    return records


def fetch_active_addresses(api_key: str, symbol: str = "BTC/USD", days: int = _DEFAULT_DAYS) -> list[tuple[str, str, float]]:
    """Return ``(date, 'active_addr_ma7_pct_change', value)`` records.

    Computes a 7-day moving average of active addresses and returns the
    day-over-day percentage change of that MA.
    """
    start, end = _date_range_params(days + 14)  # extra days for MA warm-up
    url = (
        f"{_CQ_BASE}/v1/btc/network-data/active-addresses"
        f"?window=day&from={start}&to={end}&limit=9999"
    )
    data = _fetch_json(url, api_key)
    items = data.get("result", {}).get("data", data.get("data", []))

    raw: list[tuple[str, float]] = []
    for item in items:
        try:
            date = str(item.get("date", item.get("timestamp", "")))[:10]
            val = float(item.get("activeAddresses", item.get("value", 0.0)))
            raw.append((date, val))
        except (KeyError, TypeError, ValueError):
            continue

    raw.sort()
    if not raw:
        return []

    dates_arr = [r[0] for r in raw]
    vals_arr = np.array([r[1] for r in raw], dtype=np.float64)

    # 7-day rolling mean
    ma7 = pd.Series(vals_arr).rolling(7, min_periods=7).mean().values
    records: list[tuple[str, str, float]] = []
    for i in range(1, len(raw)):
        if np.isnan(ma7[i]) or np.isnan(ma7[i - 1]) or ma7[i - 1] == 0:
            continue
        pct = float((ma7[i] - ma7[i - 1]) / ma7[i - 1])
        records.append((dates_arr[i], "active_addr_ma7_pct_change", pct))
    return records


# ---------------------------------------------------------------------------
# Main fetch routine
# ---------------------------------------------------------------------------

def fetch_onchain(
    symbol: str = "BTC/USD",
    days: int = _DEFAULT_DAYS,
    db_path: str = "data/hogan.db",
) -> dict[str, int]:
    """Fetch MVRV, SOPR, and active-address metrics and upsert into the DB.

    Parameters
    ----------
    symbol:
        Trading symbol (currently only ``"BTC/USD"`` is supported).
    days:
        Number of historical days to fetch.
    db_path:
        Path to the SQLite database.

    Returns
    -------
    dict
        Rows written per metric.
    """
    from hogan_bot.storage import get_connection, upsert_onchain

    api_key = _get_api_key()
    results: dict[str, int] = {}

    print("Fetching MVRV z-score ...")
    try:
        mvrv_records = fetch_mvrv(api_key, symbol=symbol, days=days)
        print(f"  Got {len(mvrv_records)} MVRV records")
    except RuntimeError as exc:
        print(f"  WARNING: {exc}")
        mvrv_records = []

    print("Fetching SOPR ...")
    try:
        sopr_records = fetch_sopr(api_key, symbol=symbol, days=days)
        print(f"  Got {len(sopr_records)} SOPR records")
    except RuntimeError as exc:
        print(f"  WARNING: {exc}")
        sopr_records = []

    print("Fetching active addresses ...")
    try:
        addr_records = fetch_active_addresses(api_key, symbol=symbol, days=days)
        print(f"  Got {len(addr_records)} active-address records")
    except RuntimeError as exc:
        print(f"  WARNING: {exc}")
        addr_records = []

    conn = get_connection(db_path)
    try:
        for metric, records in [
            ("mvrv_zscore", mvrv_records),
            ("sopr", sopr_records),
            ("active_addr_ma7_pct_change", addr_records),
        ]:
            if records:
                written = upsert_onchain(conn, symbol, records)
            else:
                written = 0
            results[metric] = written
    finally:
        conn.close()

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch CryptoQuant on-chain metrics (MVRV, SOPR, active addresses)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="BTC/USD",
                   help="Trading symbol (currently only BTC/USD is supported)")
    p.add_argument("--days", type=int, default=_DEFAULT_DAYS,
                   help="Number of historical days to fetch")
    p.add_argument("--db", default="data/hogan.db",
                   help="Path to the SQLite database file")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    result = fetch_onchain(
        symbol=args.symbol,
        days=args.days,
        db_path=args.db,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
