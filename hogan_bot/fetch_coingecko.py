"""Fetch market intelligence from the CoinGecko API (Demo tier).

Requires the environment variable ``COINGECKO_KEY`` (stored in ``.env``).
Demo tier: 30 calls/minute, 10 000 calls/month — more than enough for
daily fetches.

Endpoints used
--------------
``GET /global``
    BTC dominance, stablecoin dominance, 24h market-cap change.

``GET /global/decentralized_finance_defi``
    DeFi market-cap dominance.

``GET /coins/bitcoin?localization=false&tickers=false``
    7-day return, 30-day return, ATH % distance, community sentiment.

``GET /coins/bitcoin/market_chart?vs_currency=usd&days=N&interval=daily``
    Used for historical backfill (365 days of daily price + market-cap).

Features extracted (stored in ``onchain_metrics`` as daily records)
-------------------------------------------------------------------
+-------------------------+-------------------------------------------+
| cg_btc_dominance        | BTC % of total crypto market cap (0-100)  |
| cg_stablecoin_dominance | USDT + USDC % of market cap  (0-100)      |
| cg_mcap_change_24h      | Global crypto market 24 h % change        |
| cg_defi_dominance       | DeFi % of total market cap  (0-100)       |
| cg_btc_ath_pct          | Distance from BTC ATH in %  (≤ 0)        |
| cg_btc_sentiment_up     | Community % of bullish votes  (0-100)     |
+-------------------------+-------------------------------------------+

Usage
-----
    # Set key in .env:  COINGECKO_KEY=CG-xxxx
    # or export:        export COINGECKO_KEY=CG-xxxx

    # Fetch today's metrics
    python -m hogan_bot.fetch_coingecko

    # + backfill 365 days of price-based historical metrics
    python -m hogan_bot.fetch_coingecko --backfill

    # Backfill with custom window
    python -m hogan_bot.fetch_coingecko --backfill --days 180
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timezone
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import numpy as np
import pandas as pd

_CG_BASE = "https://api.coingecko.com/api/v3"
_REQUEST_TIMEOUT = 20
_RATE_LIMIT_SLEEP = 2.5   # seconds between calls (safe for 30 req/min limit)

# Metric names stored in onchain_metrics; order matters (indices used in features_mtf)
CG_METRIC_NAMES: list[str] = [
    "cg_btc_dominance",
    "cg_stablecoin_dominance",
    "cg_mcap_change_24h",
    "cg_defi_dominance",
    "cg_btc_ath_pct",
    "cg_btc_sentiment_up",
]


def _get_api_key() -> str:
    key = os.getenv("COINGECKO_KEY", "").strip()
    if not key:
        sys.exit(
            "COINGECKO_KEY environment variable is not set.\n"
            "Add it to your .env file:  COINGECKO_KEY=CG-xxxxxxxxxxxx\n"
            "Get a free Demo key at: https://www.coingecko.com/en/api"
        )
    return key


# ---------------------------------------------------------------------------
# Low-level HTTP helper
# ---------------------------------------------------------------------------

class CoinGeckoClient:
    """Thin wrapper around the CoinGecko REST API with rate-limiting."""

    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self._last_call: float = 0.0

    def _get(self, path: str, params: dict | None = None) -> dict:
        """GET ``_CG_BASE + path`` with the API key and retry logic."""
        query = f"?x_cg_demo_api_key={self.api_key}"
        if params:
            query += "&" + "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{_CG_BASE}{path}{query}"

        # Honour rate limit
        elapsed = time.monotonic() - self._last_call
        if elapsed < _RATE_LIMIT_SLEEP:
            time.sleep(_RATE_LIMIT_SLEEP - elapsed)

        for attempt in range(3):
            try:
                req = Request(url, headers={"Accept": "application/json"})
                with urlopen(req, timeout=_REQUEST_TIMEOUT) as resp:
                    self._last_call = time.monotonic()
                    return json.loads(resp.read().decode())
            except HTTPError as exc:
                if exc.code == 429:
                    wait = 60 * (attempt + 1)
                    print(f"  Rate limited — waiting {wait}s …")
                    time.sleep(wait)
                    continue
                raise RuntimeError(f"HTTP {exc.code} fetching {url}") from exc
            except URLError as exc:
                if attempt == 2:
                    raise RuntimeError(f"Request failed: {url}  ({exc})") from exc
                time.sleep(4 ** attempt)
        return {}


# ---------------------------------------------------------------------------
# Feature fetch helpers
# ---------------------------------------------------------------------------

def _fetch_global(client: CoinGeckoClient) -> dict[str, float]:
    """Return BTC dominance, stablecoin dominance, and 24h market-cap change."""
    data = client._get("/global").get("data", {})
    pct = data.get("market_cap_percentage", {})
    btc_dom = float(pct.get("btc", 0.0))
    usdt = float(pct.get("usdt", 0.0))
    usdc = float(pct.get("usdc", 0.0))
    stable_dom = usdt + usdc
    mcap_change = float(data.get("market_cap_change_percentage_24h_usd", 0.0))
    return {
        "cg_btc_dominance": btc_dom,
        "cg_stablecoin_dominance": stable_dom,
        "cg_mcap_change_24h": mcap_change,
    }


def _fetch_defi(client: CoinGeckoClient) -> dict[str, float]:
    """Return DeFi dominance % of total market cap."""
    data = client._get("/global/decentralized_finance_defi").get("data", {})
    defi_dom = float(data.get("defi_dominance", 0.0))
    return {"cg_defi_dominance": defi_dom}


def _fetch_bitcoin_detail(client: CoinGeckoClient) -> dict[str, float]:
    """Return ATH %, 7-day return, and community sentiment from /coins/bitcoin."""
    data = client._get(
        "/coins/bitcoin",
        {
            "localization": "false",
            "tickers": "false",
            "community_data": "false",
            "developer_data": "false",
        },
    )
    md = data.get("market_data", {})
    ath_pct = float(
        md.get("ath_change_percentage", {}).get("usd", 0.0)
    )
    sentiment_up = float(data.get("sentiment_votes_up_percentage", 50.0))
    return {
        "cg_btc_ath_pct": ath_pct,
        "cg_btc_sentiment_up": sentiment_up,
    }


# ---------------------------------------------------------------------------
# Live daily fetch
# ---------------------------------------------------------------------------

def fetch_today(
    client: CoinGeckoClient,
    symbol: str = "BTC/USD",
    db_path: str = "data/hogan.db",
) -> dict[str, float]:
    """Fetch all 6 live metrics and upsert into ``onchain_metrics``.

    Returns a dict of ``{metric_name: value}`` for today's date.
    """
    from hogan_bot.storage import get_connection, upsert_onchain

    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    print(f"Fetching CoinGecko global metrics for {today} …")

    results: dict[str, float] = {}
    results.update(_fetch_global(client))
    results.update(_fetch_defi(client))
    results.update(_fetch_bitcoin_detail(client))

    records = [(today, metric, value) for metric, value in results.items()]
    conn = get_connection(db_path)
    upsert_onchain(conn, symbol, records)
    conn.close()

    for metric, value in results.items():
        print(f"  {metric:<30s} {value:.4f}")

    return results


# ---------------------------------------------------------------------------
# Historical backfill from market_chart
# ---------------------------------------------------------------------------

def _fetch_market_chart(client: CoinGeckoClient, days: int = 365) -> pd.DataFrame:
    """Fetch daily BTC price + market-cap history via market_chart endpoint.

    Returns a DataFrame with columns: ``date``, ``price``, ``market_cap``.
    """
    data = client._get(
        "/coins/bitcoin/market_chart",
        {"vs_currency": "usd", "days": str(days), "interval": "daily"},
    )
    prices = data.get("prices", [])
    mcaps = data.get("market_caps", [])

    df_prices = pd.DataFrame(prices, columns=["ts_ms", "price"])
    df_mcaps = pd.DataFrame(mcaps, columns=["ts_ms", "market_cap"])

    df = df_prices.merge(df_mcaps, on="ts_ms", how="inner")
    df["date"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True).dt.strftime("%Y-%m-%d")
    df = df.sort_values("date").reset_index(drop=True)
    return df[["date", "price", "market_cap"]]


def backfill_historical(
    client: CoinGeckoClient,
    symbol: str = "BTC/USD",
    db_path: str = "data/hogan.db",
    days: int = 365,
) -> dict[str, int]:
    """Backfill price-derived historical metrics into ``onchain_metrics``.

    Computes from ``/coins/bitcoin/market_chart``:
    - ``cg_btc_ath_pct``      — (price - rolling_max) / rolling_max × 100
    - ``cg_mcap_change_24h``  — 1-day % change in BTC market cap

    Other metrics (dominance, sentiment) require live endpoints and are
    only stored for the current date via :func:`fetch_today`.

    Parameters
    ----------
    days:
        Number of historical days to backfill (max 365 on Demo tier).

    Returns
    -------
    dict
        ``{metric_name: rows_written}``
    """
    from hogan_bot.storage import get_connection, upsert_onchain

    print(f"Backfilling {days} days of historical CoinGecko metrics …")
    df = _fetch_market_chart(client, days=days)
    print(f"  Downloaded {len(df)} daily bars from market_chart")

    # Use the current ATH as the historical reference price
    # ($126,080 as of Oct 2025 — updated periodically by fetch_today)
    ath_endpoint = client._get("/coins/bitcoin", {
        "localization": "false", "tickers": "false",
        "community_data": "false", "developer_data": "false",
    })
    current_ath = float(
        ath_endpoint.get("market_data", {})
        .get("ath", {})
        .get("usd", 126_080.0)
    )
    print(f"  BTC all-time high: ${current_ath:,.0f}")

    prices = df["price"].values.astype(np.float64)
    mcaps = df["market_cap"].values.astype(np.float64)
    dates = df["date"].tolist()

    records: list[tuple[str, str, float]] = []

    # cg_btc_ath_pct: distance below current ATH
    for i, (date, price) in enumerate(zip(dates, prices)):
        ath_pct = (price - current_ath) / current_ath * 100.0
        records.append((date, "cg_btc_ath_pct", float(ath_pct)))

    # cg_mcap_change_24h: 1-day pct change of BTC market cap
    for i in range(1, len(df)):
        prev_mcap = mcaps[i - 1]
        curr_mcap = mcaps[i]
        if prev_mcap > 0:
            pct = (curr_mcap - prev_mcap) / prev_mcap * 100.0
            records.append((dates[i], "cg_mcap_change_24h", float(pct)))

    conn = get_connection(db_path)
    if records:
        upsert_onchain(conn, symbol, records)
    conn.close()

    ath_written = sum(1 for r in records if r[1] == "cg_btc_ath_pct")
    mcap_written = sum(1 for r in records if r[1] == "cg_mcap_change_24h")
    print(f"  cg_btc_ath_pct:     {ath_written} rows written")
    print(f"  cg_mcap_change_24h: {mcap_written} rows written")
    return {"cg_btc_ath_pct": ath_written, "cg_mcap_change_24h": mcap_written}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch CoinGecko market intelligence (BTC dominance, ATH, sentiment, etc.)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="BTC/USD",
                   help="Trading symbol for DB storage")
    p.add_argument("--db", default="data/hogan.db",
                   help="Path to the SQLite database file")
    p.add_argument(
        "--backfill",
        action="store_true",
        help="Also backfill historical price-derived metrics (ATH pct, mcap change)",
    )
    p.add_argument(
        "--days", type=int, default=365,
        help="Number of historical days to backfill (max 365 for Demo tier)",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    api_key = _get_api_key()
    client = CoinGeckoClient(api_key)

    result = fetch_today(client, symbol=args.symbol, db_path=args.db)

    if args.backfill:
        hist = backfill_historical(
            client, symbol=args.symbol, db_path=args.db, days=args.days
        )
        result.update(hist)

    print(json.dumps({k: round(v, 6) for k, v in result.items()}, indent=2))


if __name__ == "__main__":
    main()
