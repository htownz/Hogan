"""Fetch Kraken Futures derivatives data (funding rate + open interest).

Uses the Kraken Futures **public** REST API v3 — no authentication required.
Each call fetches the current live snapshot (fundingRate, openInterest) from
the tickers endpoint and upserts a row for today into ``derivatives_metrics``.
OI % change is computed relative to the previous row stored in the DB.

Endpoint used
-------------
    GET https://futures.kraken.com/derivatives/api/v3/tickers

Usage
-----
    python -m hogan_bot.fetch_derivatives --symbol BTC/USD
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

import numpy as np

# Kraken Futures v3 tickers endpoint (public, no auth required)
_KF_TICKERS_URL = "https://futures.kraken.com/derivatives/api/v3/tickers"

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


def _fetch_ticker(kf_symbol: str) -> dict:
    """Return the live ticker dict for *kf_symbol* from the v3 tickers endpoint."""
    data = _fetch_json(_KF_TICKERS_URL)
    tickers = data.get("tickers", [])
    for t in tickers:
        if t.get("symbol") == kf_symbol:
            return t
    raise RuntimeError(
        f"Symbol {kf_symbol!r} not found in Kraken Futures tickers response"
    )


# ---------------------------------------------------------------------------
# Main fetch routine
# ---------------------------------------------------------------------------

def fetch_derivatives(
    symbol: str = "BTC/USD",
    days: int = _DEFAULT_DAYS,
    db_path: str = "data/hogan.db",
) -> dict[str, int]:
    """Fetch today's funding rate + OI snapshot for *symbol* and upsert into the DB.

    Uses the Kraken Futures v3 ``/tickers`` endpoint which returns the current
    live ``fundingRate`` and ``openInterest``.  OI % change is computed from
    the most recent raw-OI row already in the DB.

    Parameters
    ----------
    symbol:
        CCXT-style symbol (e.g. ``"BTC/USD"``).
    days:
        Unused — kept for backwards-compatible CLI signature.
    db_path:
        Path to the SQLite database.

    Returns
    -------
    dict
        ``{"funding_rate": <rows_written>, "open_interest_pct_change": <rows_written>}``
    """
    from hogan_bot.storage import get_connection, load_derivatives, upsert_derivatives

    kf_symbol = _ccxt_to_kf(symbol)
    print(f"Fetching live ticker for {kf_symbol} ...")

    ticker = _fetch_ticker(kf_symbol)
    funding_rate = float(ticker.get("fundingRate", 0.0))
    open_interest = float(ticker.get("openInterest", 0.0))
    now_ms = int(datetime.now(timezone.utc).timestamp() * 1000)

    print(f"  fundingRate    = {funding_rate:.6f}")
    print(f"  openInterest   = {open_interest:.4f} BTC")

    conn = get_connection(db_path)

    # Store funding rate
    fr_records = [(now_ms, "funding_rate", funding_rate)]
    fr_written = upsert_derivatives(conn, symbol, fr_records)

    # Store raw OI (BTC)
    oi_raw_records = [(now_ms, "open_interest_btc", open_interest)]
    upsert_derivatives(conn, symbol, oi_raw_records)

    # Compute OI % change from previous stored raw value
    oi_written = 0
    try:
        prev_raw = load_derivatives(conn, symbol, "open_interest_btc")
        prev_raw = prev_raw.sort_values("ts_ms")
        if len(prev_raw) >= 2:
            oi_prev = float(prev_raw.iloc[-2]["value"])
            oi_now = float(prev_raw.iloc[-1]["value"])
            pct = (oi_now - oi_prev) / max(abs(oi_prev), 1e-9)
            pct = float(np.clip(pct, -1.0, 1.0))
        else:
            pct = 0.0
        oi_records = [(now_ms, "open_interest_pct_change", pct)]
        oi_written = upsert_derivatives(conn, symbol, oi_records)
        print(f"  OI pct change  = {pct:+.4f}")
    except Exception as exc:
        print(f"  Warning: could not compute OI pct change: {exc}")

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
                   help="(unused) kept for CLI compatibility")
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
