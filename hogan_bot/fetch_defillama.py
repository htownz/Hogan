"""Fetch DeFi ecosystem metrics from DeFi Llama.

DeFi Llama is the leading DeFi analytics aggregator.  Its API is 100% free
with no authentication required.  It tracks TVL (Total Value Locked) across
500+ protocols and 80+ blockchains.

Why DeFi TVL matters for BTC/ETH trading
------------------------------------------
* **Rising TVL** → capital flowing into DeFi → risk-on → correlated with
  BTC price appreciation
* **Falling TVL** → capital exiting → risk-off / de-risking event
* **Stablecoin TVL share** → when stablecoins dominate, dry powder exists
  for a potential re-entry into risk assets
* **BTC TVL in DeFi** → wrapped BTC usage signals conviction from BTC
  holders willing to put their BTC to work
* **Chain dominance** → ETH vs Solana vs others shows where risk appetite
  is concentrated

API docs: https://defillama.com/docs/api

Metrics stored in ``onchain_metrics``
--------------------------------------
``defi_total_tvl_b``    — Total DeFi TVL in USD billions
``defi_tvl_change_1d``  — 1-day TVL % change
``defi_tvl_change_7d``  — 7-day TVL % change
``defi_eth_tvl_pct``    — Ethereum TVL as % of total (chain dominance)
``defi_btc_tvl_b``      — Bitcoin TVL (wBTC, tBTC etc.) in USD billions
``defi_stablecoin_b``   — Total stablecoin market cap in USD billions
``defi_stablecoin_chg`` — Stablecoin mcap 7-day % change (dry powder signal)

Usage
-----
    python -m hogan_bot.fetch_defillama
    python -m hogan_bot.fetch_defillama --symbol BTC/USD
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import date, datetime, timedelta, timezone
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_TIMEOUT = 25
_BASE = "https://api.llama.fi"
_STABLECOIN_BASE = "https://stablecoins.llama.fi"


def _get_json(url: str) -> dict | list:
    """Simple GET with retries."""
    for attempt in range(3):
        try:
            req = Request(url, headers={"Accept": "application/json"})
            with urlopen(req, timeout=_TIMEOUT) as resp:
                return json.loads(resp.read().decode())
        except URLError as exc:
            if attempt == 2:
                raise RuntimeError(f"DeFiLlama request failed: {url}  ({exc})") from exc
            time.sleep(2 ** attempt)
    return {}


# ---------------------------------------------------------------------------
# Individual metric fetchers
# ---------------------------------------------------------------------------

def _fetch_global_tvl() -> dict[str, float]:
    """Fetch total DeFi TVL and chain breakdown from /v2/chains."""
    result: dict[str, float] = {}
    today = date.today().isoformat()

    try:
        # /v2/chains returns per-chain TVL
        chains_data = _get_json(f"{_BASE}/v2/chains")
        if not isinstance(chains_data, list):
            return result

        total_tvl = 0.0
        eth_tvl = 0.0
        btc_chain_tvl = 0.0

        for chain in chains_data:
            tvl = float(chain.get("tvl", 0))
            total_tvl += tvl
            chain_name = str(chain.get("name", "")).lower()
            if chain_name == "ethereum":
                eth_tvl = tvl
            elif chain_name == "bitcoin":
                btc_chain_tvl = tvl

        if total_tvl > 0:
            result["defi_total_tvl_b"] = round(total_tvl / 1e9, 4)
            result["defi_eth_tvl_pct"] = round(eth_tvl / total_tvl * 100, 4)
            if btc_chain_tvl > 0:
                result["defi_btc_tvl_b"] = round(btc_chain_tvl / 1e9, 4)

        logger.info("DeFiLlama: total TVL = $%.1fB  ETH share = %.1f%%",
                    result.get("defi_total_tvl_b", 0),
                    result.get("defi_eth_tvl_pct", 0))

    except Exception as exc:
        logger.warning("DeFiLlama chains fetch failed: %s", exc)

    return result


def _fetch_tvl_change() -> dict[str, float]:
    """Fetch historical global TVL to compute 1d/7d % change."""
    result: dict[str, float] = {}
    try:
        # /v2/historicalChainTvl returns [{date, tvl}] daily series
        data = _get_json(f"{_BASE}/v2/historicalChainTvl")
        if not isinstance(data, list) or len(data) < 8:
            return result

        # Most recent entry
        latest = data[-1]
        day_ago = data[-2] if len(data) >= 2 else None
        week_ago = data[-8] if len(data) >= 8 else None

        latest_tvl = float(latest.get("tvl", 0))

        if day_ago and latest_tvl:
            prev = float(day_ago.get("tvl", 0))
            if prev > 0:
                result["defi_tvl_change_1d"] = round((latest_tvl - prev) / prev * 100, 4)

        if week_ago and latest_tvl:
            prev7 = float(week_ago.get("tvl", 0))
            if prev7 > 0:
                result["defi_tvl_change_7d"] = round((latest_tvl - prev7) / prev7 * 100, 4)

        logger.info("DeFiLlama: TVL 1d chg = %.2f%%  7d chg = %.2f%%",
                    result.get("defi_tvl_change_1d", 0),
                    result.get("defi_tvl_change_7d", 0))

    except Exception as exc:
        logger.warning("DeFiLlama historical TVL fetch failed: %s", exc)

    return result


def _fetch_stablecoins() -> dict[str, float]:
    """Fetch total stablecoin market cap and 7-day change from DeFiLlama."""
    result: dict[str, float] = {}
    try:
        data = _get_json(f"{_STABLECOIN_BASE}/stablecoins?includePrices=true")
        if not isinstance(data, dict):
            return result

        pegged_assets = data.get("peggedAssets", [])
        total_mcap = 0.0
        for asset in pegged_assets:
            circulating = asset.get("circulating", {})
            pegged_usd = circulating.get("peggedUSD", 0)
            if pegged_usd:
                try:
                    total_mcap += float(pegged_usd)
                except (TypeError, ValueError):
                    pass

        if total_mcap > 0:
            result["defi_stablecoin_b"] = round(total_mcap / 1e9, 4)
            logger.info("DeFiLlama: stablecoin mcap = $%.1fB", result["defi_stablecoin_b"])

    except Exception as exc:
        logger.warning("DeFiLlama stablecoins fetch failed: %s", exc)

    return result


# ---------------------------------------------------------------------------
# Main fetch → DB
# ---------------------------------------------------------------------------

def fetch_all_defillama(
    symbol: str = "BTC/USD",
    db_path: str = "data/hogan.db",
) -> dict[str, int]:
    """Fetch all DeFiLlama metrics and upsert into the database.

    Parameters
    ----------
    symbol : str
        Trading pair label for DB rows.
    db_path : str
        Path to the SQLite database.

    Returns
    -------
    dict
        ``{metric_name: rows_written}``
    """
    from hogan_bot.storage import get_connection, upsert_onchain

    conn = get_connection(db_path)
    today = date.today().isoformat()
    all_metrics: dict[str, float] = {}

    # Fetch each category with brief pauses
    all_metrics.update(_fetch_global_tvl())
    time.sleep(0.5)
    all_metrics.update(_fetch_tvl_change())
    time.sleep(0.5)
    all_metrics.update(_fetch_stablecoins())

    if not all_metrics:
        logger.warning("DeFiLlama: no metrics returned")
        conn.close()
        return {}

    records: list[tuple[str, str, float]] = [
        (today, metric, value) for metric, value in all_metrics.items()
    ]

    written = upsert_onchain(conn, symbol, records)
    conn.close()

    logger.info("DeFiLlama: wrote %d rows total", written)
    return {m: 1 for m in all_metrics}


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch DeFi Llama TVL and stablecoin data (no API key required)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="BTC/USD")
    p.add_argument("--db", default="data/hogan.db")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    result = fetch_all_defillama(symbol=args.symbol, db_path=args.db)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
