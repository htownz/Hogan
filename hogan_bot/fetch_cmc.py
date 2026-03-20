"""CoinMarketCap (CMC) data fetcher for Hogan.

Replaces the CoinGecko market-intelligence signals (indices 6-11 in
EXT_FEATURE_NAMES) so the RL-agent ext-feature vector gets real values
instead of zeros when COINGECKO_KEY is absent.

Also stores additional CMC-only metrics for future feature use.

Data fetched (all from free Basic plan, ~3-5 credits per daily run):
───────────────────────────────────────────────────────────────────────
Global metrics  (1 credit)
  cg_btc_dominance       BTC % share of total crypto market cap
  cg_stablecoin_dominance Stablecoin % of total market cap
  cg_mcap_change_24h     Total crypto market cap 24h % change
  cg_defi_dominance      DeFi market cap % of total
  cmc_eth_dominance      ETH % of total market cap (new signal)
  cmc_altcoin_mcap_b     Altcoin market cap USD billions
  cmc_total_mcap_b       Total crypto market cap USD trillions
  cmc_total_vol_b        Total 24h trading volume USD billions

BTC + ETH quotes  (2 credits)
  cmc_btc_pct_1h / 24h / 7d   BTC price % change multi-timeframe
  cmc_eth_pct_1h / 24h / 7d   ETH price % change multi-timeframe
  cmc_btc_vol_b / cmc_eth_vol_b  24h trading volume billions

These metric names are stored in the ``onchain_metrics`` table so
features_mtf.py picks them up automatically at inference time.

Plan:
  * cg_btc_dominance, cg_stablecoin_dominance, cg_mcap_change_24h,
    cg_defi_dominance are stored under those EXACT names so they fill
    the CoinGecko slots in EXT_FEATURE_NAMES without any code change.

Credit budget:  ~3 credits/day → ~90/month  (Basic plan: 10,000/month)

Environment
-----------
CMC_API_KEY   CoinMarketCap Pro API key (free Basic plan at pro.coinmarketcap.com)

Usage
-----
    python -m hogan_bot.fetch_cmc
    python -m hogan_bot.fetch_cmc --days 30   # backfill (1 credit per day, estimated)
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sqlite3
import urllib.error
import urllib.request
from datetime import date

logger = logging.getLogger(__name__)

_BASE_URL = "https://pro-api.coinmarketcap.com"


# ---------------------------------------------------------------------------
# Internal HTTP helper
# ---------------------------------------------------------------------------

def _get(path: str, key: str, params: dict[str, str] | None = None) -> dict:
    url = _BASE_URL + path
    if params:
        url += "?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(
        url,
        headers={
            "X-CMC_PRO_API_KEY": key,
            "Accept": "application/json",
        },
    )
    with urllib.request.urlopen(req, timeout=15) as r:
        return json.loads(r.read())


import urllib.parse  # noqa: E402 (needed above)

# ---------------------------------------------------------------------------
# Global market metrics  (1 credit)
# ---------------------------------------------------------------------------

def fetch_global_metrics(key: str) -> list[tuple[str, str, float]]:
    """Fetch total market cap, BTC/ETH dominance, DeFi & stablecoin share.

    Returns (date_str, metric_name, value) triples for ``onchain_metrics``.
    """
    resp = _get("/v1/global-metrics/quotes/latest", key)
    data = resp.get("data")
    if not data:
        logger.warning("CMC global-metrics: missing 'data' key in response")
        return []
    q = (data.get("quote") or {}).get("USD")
    if not q:
        logger.warning("CMC global-metrics: missing quote.USD in response")
        return []
    today = date.today().isoformat()

    total_mcap = q.get("total_market_cap") or 0.0
    defi_mcap  = q.get("defi_market_cap", 0.0) or 0.0
    stable_mcap = q.get("stablecoin_market_cap", 0.0) or 0.0
    altcoin_mcap = q.get("altcoin_market_cap", 0.0) or 0.0
    total_vol   = q.get("total_volume_24h", 0.0) or 0.0
    btc_dom     = data.get("btc_dominance", 0.0) or 0.0
    eth_dom     = data.get("eth_dominance", 0.0) or 0.0
    mcap_chg    = q.get("total_market_cap_yesterday_percentage_change", 0.0) or 0.0

    defi_pct    = defi_mcap   / total_mcap * 100.0 if total_mcap > 0 else 0.0
    stable_pct  = stable_mcap / total_mcap * 100.0 if total_mcap > 0 else 0.0

    records = [
        # ── Fill existing CoinGecko feature slots (same metric names) ──────
        (today, "cg_btc_dominance",        btc_dom),
        (today, "cg_stablecoin_dominance", stable_pct),
        (today, "cg_mcap_change_24h",      mcap_chg),
        (today, "cg_defi_dominance",       defi_pct),
        # ── CMC-specific extras ─────────────────────────────────────────────
        (today, "cmc_eth_dominance",       eth_dom),
        (today, "cmc_altcoin_mcap_b",      altcoin_mcap / 1e9),
        (today, "cmc_total_mcap_b",        total_mcap   / 1e9),
        (today, "cmc_total_vol_b",         total_vol    / 1e9),
    ]
    logger.info(
        "CMC global: BTC dom=%.1f%%  ETH dom=%.1f%%  MCap=%.2fT  MCap24h=%.2f%%",
        btc_dom, eth_dom, total_mcap / 1e12 if total_mcap else 0.0, mcap_chg,
    )
    return records


# ---------------------------------------------------------------------------
# BTC + ETH quotes  (2 credits)
# ---------------------------------------------------------------------------

def fetch_coin_quotes(key: str, symbols: list[str] | None = None) -> list[tuple[str, str, float]]:
    """Fetch BTC and ETH market metrics: market cap, % changes, volume.

    These capture cross-timeframe momentum from CMC's aggregated price feed
    (more exchanges than Kraken alone).
    """
    if symbols is None:
        symbols = ["BTC", "ETH"]

    resp = _get(
        "/v2/cryptocurrency/quotes/latest",
        key,
        {"symbol": ",".join(symbols)},
    )
    data = resp.get("data")
    if not data:
        logger.warning("CMC coin-quotes: missing 'data' key in response")
        return []

    today = date.today().isoformat()
    records: list[tuple[str, str, float]] = []

    for sym in symbols:
        entries = data.get(sym)
        if not entries:
            continue
        q = (entries[0].get("quote") or {}).get("USD")
        if not q:
            continue
        prefix = sym.lower()  # "btc" or "eth"

        records.extend([
            (today, f"cmc_{prefix}_mcap_b",    (q.get("market_cap")    or 0.0) / 1e9),
            (today, f"cmc_{prefix}_vol_b",     (q.get("volume_24h")    or 0.0) / 1e9),
            (today, f"cmc_{prefix}_pct_1h",    q.get("percent_change_1h",  0.0) or 0.0),
            (today, f"cmc_{prefix}_pct_24h",   q.get("percent_change_24h", 0.0) or 0.0),
            (today, f"cmc_{prefix}_pct_7d",    q.get("percent_change_7d",  0.0) or 0.0),
        ])

    if records:
        btc_24h = next((v for _, m, v in records if m == "cmc_btc_pct_24h"), 0.0)
        eth_24h = next((v for _, m, v in records if m == "cmc_eth_pct_24h"), 0.0)
        logger.info("CMC quotes: BTC 24h=%.2f%%  ETH 24h=%.2f%%", btc_24h, eth_24h)

    return records


# ---------------------------------------------------------------------------
# DB upsert
# ---------------------------------------------------------------------------

def _upsert(conn: sqlite3.Connection, symbol: str, records: list[tuple[str, str, float]]) -> int:
    """Insert or replace (symbol, date, metric, value) into onchain_metrics."""
    if not records:
        return 0
    conn.executemany(
        "INSERT OR REPLACE INTO onchain_metrics (symbol, date, metric, value) VALUES (?, ?, ?, ?)",
        [(symbol, d, m, v) for d, m, v in records],
    )
    conn.commit()
    return len(records)


# ---------------------------------------------------------------------------
# Top-level: fetch_all_cmc — called by refresh_daily.py
# ---------------------------------------------------------------------------

def fetch_all_cmc(
    symbol: str = "BTC/USD",
    db_path: str = "data/hogan.db",
) -> dict[str, int]:
    """Fetch all CMC signals and store in onchain_metrics.

    Called daily by ``refresh_daily.py``.  Uses ~3 API credits per call.
    """
    key = os.getenv("CMC_API_KEY", "").strip()
    if not key:
        raise RuntimeError("CMC_API_KEY not set — skipping CoinMarketCap")

    db_path = os.getenv("HOGAN_DB_PATH", db_path)
    conn = sqlite3.connect(db_path)
    written: dict[str, int] = {}

    try:
        # Global metrics (fills CoinGecko slots + CMC extras)
        records = fetch_global_metrics(key)
        n = _upsert(conn, symbol, records)
        written["global_metrics"] = n

        # BTC + ETH per-coin quotes
        records = fetch_coin_quotes(key, ["BTC", "ETH"])
        n = _upsert(conn, symbol, records)
        written["coin_quotes"] = n

    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")[:300]
        raise RuntimeError(f"CMC API HTTP {exc.code}: {exc.reason} — {body}") from exc
    finally:
        conn.close()

    total = sum(written.values())
    logger.info("CMC: wrote %d records total", total)
    return written


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Fetch CoinMarketCap signals into Hogan DB")
    p.add_argument("--db", default=os.getenv("HOGAN_DB_PATH", "data/hogan.db"))
    p.add_argument("--symbol", default="BTC/USD")
    return p.parse_args()


if __name__ == "__main__":
    import sys
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    args = _parse_args()
    try:
        result = fetch_all_cmc(symbol=args.symbol, db_path=args.db)
        print(json.dumps(result, indent=2))
    except RuntimeError as e:
        logger.error("%s", e)
        sys.exit(1)
