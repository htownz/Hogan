"""Fetch Kraken Futures derivatives data (funding rate + open interest).

Uses the Kraken Futures **public** REST API — no authentication required for
historical analytics endpoints.  Data is upserted into the
``derivatives_metrics`` SQLite table.

Endpoints used
--------------
Funding rate history:
    GET https://futures.kraken.com/api/charts/v1/analytics/funding
    ?symbol=PF_XBTUSD&from=<unix_s>&to=<unix_s>

Open interest:
    GET https://futures.kraken.com/api/charts/v1/analytics/openInterest
    ?symbol=PF_XBTUSD&from=<unix_s>&to=<unix_s>

Usage
-----
    python -m hogan_bot.fetch_derivatives --symbol BTC/USD

    # Limit history window
    python -m hogan_bot.fetch_derivatives --symbol BTC/USD --days 30
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from typing import Any
from urllib.error import URLError
from urllib.request import urlopen

import pandas as pd

# Kraken Futures API base URL
_KF_BASE = "https://futures.kraken.com/api/charts/v1/analytics"

# CCXT-style symbol -> Kraken Futures perpetual symbol
_SYMBOL_MAP: dict[str, str] = {
    "BTC/USD": "PF_XBTUSD",
    "ETH/USD": "PF_ETHUSD",
    "SOL/USD": "PF_SOLUSD",
}

_DEFAULT_DAYS = 90
_REQUEST_TIMEOUT = 15


def _ccxt_to_kf(symbol: str) -> str:
    mapped = _SYMBOL_MAP.get(symbol)
    if mapped is None:
        raise ValueError(
            f"No Kraken Futures mapping for '{symbol}'.  "
            f"Supported: {list(_SYMBOL_MAP)}"
        )
    return mapped


def _fetch_json(url: str) -> Any:
    """Fetch JSON from *url* with a simple retry (up to 3 attempts)."""
    for attempt in range(3):
        try:
            with urlopen(url, timeout=_REQUEST_TIMEOUT) as resp:
                return json.loads(resp.read().decode())
        except URLError as exc:
            if attempt == 2:
                raise RuntimeError(f"Failed to fetch {url}: {exc}") from exc
            time.sleep(2 ** attempt)
    return None


# ---------------------------------------------------------------------------
# Fetch helpers
# ---------------------------------------------------------------------------

def fetch_funding_rate(
    kf_symbol: str,
    from_ts: int,
    to_ts: int,
) -> list[tuple[int, str, float]]:
    """Return funding-rate records as ``(ts_ms, 'funding_rate', value)`` tuples.

    The Kraken Futures API returns ``fundingRate`` as an 8-hour rate.
    We store it as-is and normalise to a per-step value in the feature layer.
    """
    url = (
        f"{_KF_BASE}/funding?symbol={kf_symbol}"
        f"&from={from_ts}&to={to_ts}"
    )
    data = _fetch_json(url)

    records: list[tuple[int, str, float]] = []
    # Response schema varies; try multiple known shapes
    items = (
        data.get("rates")
        or data.get("data")
        or data.get("result", {}).get("rates")
        or []
    )
    for item in items:
        try:
            ts_ms = int(item.get("timestamp", item.get("time", 0))) * 1000
            rate = float(item.get("fundingRate", item.get("rate", 0.0)))
            if ts_ms > 0:
                records.append((ts_ms, "funding_rate", rate))
        except (KeyError, TypeError, ValueError):
            continue
    return records


def fetch_open_interest(
    kf_symbol: str,
    from_ts: int,
    to_ts: int,
) -> list[tuple[int, str, float]]:
    """Return OI records as ``(ts_ms, 'open_interest_pct_change', value)`` tuples.

    Converts raw OI to 1-day percentage change before storing.
    """
    url = (
        f"{_KF_BASE}/openInterest?symbol={kf_symbol}"
        f"&from={from_ts}&to={to_ts}"
    )
    data = _fetch_json(url)

    raw_oi: list[tuple[int, float]] = []
    items = (
        data.get("openInterest")
        or data.get("data")
        or data.get("result", {}).get("openInterest")
        or []
    )
    for item in items:
        try:
            ts_ms = int(item.get("timestamp", item.get("time", 0))) * 1000
            oi = float(item.get("openInterest", item.get("value", 0.0)))
            if ts_ms > 0:
                raw_oi.append((ts_ms, oi))
        except (KeyError, TypeError, ValueError):
            continue

    # Convert to pct change
    raw_oi.sort()
    records: list[tuple[int, str, float]] = []
    for i in range(1, len(raw_oi)):
        ts_ms, oi_now = raw_oi[i]
        _, oi_prev = raw_oi[i - 1]
        pct = (oi_now - oi_prev) / max(abs(oi_prev), 1e-9)
        records.append((ts_ms, "open_interest_pct_change", pct))
    return records


# ---------------------------------------------------------------------------
# Main fetch routine
# ---------------------------------------------------------------------------

def fetch_derivatives(
    symbol: str = "BTC/USD",
    days: int = _DEFAULT_DAYS,
    db_path: str = "data/hogan.db",
) -> dict[str, int]:
    """Fetch funding rate + OI for *symbol* and upsert into the DB.

    Parameters
    ----------
    symbol:
        CCXT-style symbol (e.g. ``"BTC/USD"``).
    days:
        Number of historical days to fetch.
    db_path:
        Path to the SQLite database.

    Returns
    -------
    dict
        ``{"funding_rate": <rows_written>, "open_interest_pct_change": <rows_written>}``
    """
    from hogan_bot.storage import get_connection, upsert_derivatives

    kf_symbol = _ccxt_to_kf(symbol)
    now = int(datetime.now(timezone.utc).timestamp())
    from_ts = now - days * 86_400

    print(f"Fetching funding rate for {kf_symbol} ...")
    fr_records = fetch_funding_rate(kf_symbol, from_ts, now)
    print(f"  Got {len(fr_records)} funding-rate records")

    print(f"Fetching open interest for {kf_symbol} ...")
    oi_records = fetch_open_interest(kf_symbol, from_ts, now)
    print(f"  Got {len(oi_records)} OI records")

    conn = get_connection(db_path)
    fr_written = upsert_derivatives(conn, symbol, fr_records) if fr_records else 0
    oi_written = upsert_derivatives(conn, symbol, oi_records) if oi_records else 0
    conn.close()

    return {"funding_rate": fr_written, "open_interest_pct_change": oi_written}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch Kraken Futures derivatives data (funding rate + OI)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="BTC/USD",
                   help="CCXT-style trading pair, e.g. BTC/USD")
    p.add_argument("--days", type=int, default=_DEFAULT_DAYS,
                   help="Number of historical days to fetch")
    p.add_argument("--db", default="data/hogan.db",
                   help="Path to the SQLite database file")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    try:
        result = fetch_derivatives(
            symbol=args.symbol,
            days=args.days,
            db_path=args.db,
        )
        print(json.dumps(result, indent=2))
    except (ValueError, RuntimeError) as exc:
        sys.exit(str(exc))


if __name__ == "__main__":
    main()
