"""Fetch crypto fundamentals from the Messari API.

Messari provides institutional-grade on-chain and market data for BTC/ETH.
Free tier allows ~1000 requests/day at https://messari.io/api.

Auth
----
    Set ``MESSARI_KEY`` in your .env file.
    All requests send the key as the ``x-messari-api-key`` HTTP header.

Metrics fetched
---------------
* **NVT ratio**        — Network Value to Transactions; high = overvalued relative
                         to on-chain activity. >100 historically signals tops.
* **Realized cap**     — Sum of all UTXO values at last-moved price; proxy for
                         cost basis of all BTC holders.
* **Stock-to-flow**    — Scarcity ratio (circulating supply / annual issuance).
* **Annualized vol**   — 30-day annualized realized volatility from Messari.
* **Developer score**  — Commit activity percentile (0-100) — proxy for project health.

All metrics are stored as daily rows in ``onchain_metrics`` under metric names
``messari_nvt``, ``messari_realized_cap_usd``, ``messari_stock_to_flow``,
``messari_vol_30d``, ``messari_dev_score``.

Usage
-----
    python -m hogan_bot.fetch_messari --symbol BTC
    python -m hogan_bot.fetch_messari --symbol ETH --days 90
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_BASE = "https://data.messari.io/api"
_TIMEOUT = 20
_DEFAULT_DAYS = 30


def _get_key() -> str:
    key = os.getenv("MESSARI_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "MESSARI_KEY not set.\n"
            "Free key at https://messari.io/api — add to .env as MESSARI_KEY=..."
        )
    return key


def _get(path: str, api_key: str, params: str = "") -> dict:
    """GET from Messari API with auth header and up to 3 retries."""
    url = f"{_BASE}{path}{('?' + params) if params else ''}"
    req = Request(url, headers={"x-messari-api-key": api_key})
    for attempt in range(3):
        try:
            with urlopen(req, timeout=_TIMEOUT) as resp:
                return json.loads(resp.read().decode())
        except URLError as exc:
            if attempt == 2:
                raise RuntimeError(f"Messari request failed: {url}  ({exc})") from exc
            time.sleep(2 ** attempt)
    return {}


# ---------------------------------------------------------------------------
# Metric fetchers
# ---------------------------------------------------------------------------

def _slug(symbol: str) -> str:
    """Convert 'BTC/USD' → 'bitcoin', 'ETH/USD' → 'ethereum'."""
    _MAP = {
        "BTC": "bitcoin",
        "ETH": "ethereum",
        "SOL": "solana",
        "BNB": "binance-coin",
        "XRP": "xrp",
        "ADA": "cardano",
        "DOGE": "dogecoin",
        "LTC": "litecoin",
    }
    base = symbol.split("/")[0].upper()
    return _MAP.get(base, base.lower())


def fetch_market_metrics(
    api_key: str,
    symbol: str = "BTC/USD",
) -> list[tuple[str, str, float]]:
    """Fetch current-day market metrics snapshot from Messari.

    Returns a list of (date, metric_name, value) tuples for today only
    (Messari free tier returns a single snapshot, not historical series).
    """
    asset = _slug(symbol)
    try:
        data = _get(f"/v1/assets/{asset}/metrics", api_key)
    except RuntimeError as exc:
        logger.warning("Messari market metrics failed: %s", exc)
        return []

    today = datetime.now(timezone.utc).date().isoformat()
    metrics_data = data.get("data", {}).get("market_data", {})
    on_chain = data.get("data", {}).get("on_chain_data", {})
    misc = data.get("data", {}).get("misc_data", {})

    records: list[tuple[str, str, float]] = []

    def _safe(d: dict, key: str, name: str) -> None:
        val = d.get(key)
        if val is not None:
            try:
                records.append((today, name, float(val)))
            except (TypeError, ValueError):
                pass

    _safe(metrics_data, "price_usd", "messari_price_usd")
    _safe(metrics_data, "real_volume_last_24_hours", "messari_real_vol_24h")
    _safe(on_chain, "nvt_20_day_ma_ntv", "messari_nvt")
    _safe(on_chain, "realized_capitalization_usd", "messari_realized_cap_usd")
    _safe(misc, "asset_age_days", "messari_asset_age_days")

    return records


def fetch_roi_metrics(
    api_key: str,
    symbol: str = "BTC/USD",
) -> list[tuple[str, str, float]]:
    """Fetch ROI and volatility metrics."""
    asset = _slug(symbol)
    try:
        data = _get(f"/v1/assets/{asset}/metrics/roi-data", api_key)
    except RuntimeError as exc:
        logger.warning("Messari ROI metrics failed: %s", exc)
        return []

    today = datetime.now(timezone.utc).date().isoformat()
    roi_data = data.get("data", {})
    records: list[tuple[str, str, float]] = []

    def _safe(key: str, name: str) -> None:
        val = roi_data.get(key)
        if val is not None:
            try:
                records.append((today, name, float(val)))
            except (TypeError, ValueError):
                pass

    _safe("percent_change_last_1_week", "messari_roi_1w")
    _safe("percent_change_last_1_month", "messari_roi_1m")
    _safe("percent_change_last_3_months", "messari_roi_3m")
    _safe("volatility_stats.volatility_last_30_days", "messari_vol_30d")

    return records


def fetch_developer_activity(
    api_key: str,
    symbol: str = "BTC/USD",
) -> list[tuple[str, str, float]]:
    """Fetch developer activity score (0-100 percentile)."""
    asset = _slug(symbol)
    try:
        data = _get(f"/v1/assets/{asset}/metrics/developer-activity", api_key)
    except RuntimeError as exc:
        logger.warning("Messari dev activity failed: %s", exc)
        return []

    today = datetime.now(timezone.utc).date().isoformat()
    dev_data = data.get("data", {})
    records: list[tuple[str, str, float]] = []

    val = dev_data.get("developer_activity_score")
    if val is not None:
        try:
            records.append((today, "messari_dev_score", float(val)))
        except (TypeError, ValueError):
            pass

    return records


def fetch_all_messari(
    symbol: str = "BTC/USD",
    db_path: str = "data/hogan.db",
) -> dict[str, int]:
    """Fetch all Messari metrics and upsert into the database.

    Parameters
    ----------
    symbol : str
        Trading pair (e.g. ``"BTC/USD"``).
    db_path : str
        Path to the SQLite database.

    Returns
    -------
    dict
        Rows written per metric group.
    """
    from hogan_bot.storage import get_connection, upsert_onchain

    api_key = _get_key()
    conn = get_connection(db_path)
    results: dict[str, int] = {}

    for group_name, fetch_fn in [
        ("market_metrics", fetch_market_metrics),
        ("roi_metrics", fetch_roi_metrics),
        ("developer_activity", fetch_developer_activity),
    ]:
        logger.info("Messari: fetching %s for %s ...", group_name, symbol)
        try:
            records = fetch_fn(api_key, symbol=symbol)
            if records:
                written = upsert_onchain(conn, symbol, records)
                results[group_name] = written
                logger.info("  wrote %d rows", written)
            else:
                results[group_name] = 0
                logger.info("  no data returned")
        except Exception as exc:
            logger.warning("  %s failed: %s", group_name, exc)
            results[group_name] = 0

        time.sleep(0.3)  # be polite to the free-tier rate limit

    conn.close()
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch Messari crypto fundamentals (NVT, realized cap, ROI, dev activity)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="BTC/USD", help="Trading pair")
    p.add_argument("--days", type=int, default=_DEFAULT_DAYS,
                   help="Lookback days (informational only — Messari returns current snapshot)")
    p.add_argument("--db", default="data/hogan.db", help="SQLite database path")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    result = fetch_all_messari(symbol=args.symbol, db_path=args.db)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
