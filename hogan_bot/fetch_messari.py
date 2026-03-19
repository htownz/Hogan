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
from datetime import datetime, timezone
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_BASE = "https://data.messari.io/api/v2"   # v1 is deprecated; v2 requires paid plan
_BASE_V1 = "https://data.messari.io/api/v1"   # kept for fallback
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


def _get(path: str, api_key: str, params: str = "", base: str | None = None) -> dict:
    """GET from Messari API with auth header and up to 3 retries."""
    from urllib.error import HTTPError
    resolved_base = base or _BASE
    url = f"{resolved_base}{path}{('?' + params) if params else ''}"
    req = Request(url, headers={"x-messari-api-key": api_key})
    for attempt in range(3):
        try:
            with urlopen(req, timeout=_TIMEOUT) as resp:
                return json.loads(resp.read().decode())
        except HTTPError as exc:
            body = ""
            try:
                body = exc.read().decode()[:200]
            except Exception:
                pass
            raise RuntimeError(
                f"Messari request failed: {url}  (HTTP {exc.code}: {exc.reason}) — {body}"
            ) from exc
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

    Tries v2 asset metrics endpoint first, then falls back to v1.
    Returns a list of (date, metric_name, value) tuples for today only.

    Note: Messari v1 API free tier was deprecated; v2 requires a Pro plan.
    If both return 401, the API key may be invalid or requires upgrading.
    """
    asset = _slug(symbol)
    data: dict = {}

    # Try v2 first, fall back to v1
    for path, base in [
        (f"/assets/{asset}/metrics", _BASE),         # v2
        (f"/v1/assets/{asset}/metrics", _BASE_V1),   # v1 fallback
    ]:
        try:
            data = _get(path, api_key, base=base)
            if data.get("data"):
                break
        except RuntimeError as exc:
            logger.debug("Messari %s failed: %s", path, exc)
            continue

    if not data.get("data"):
        logger.warning(
            "Messari market metrics: no data returned. "
            "The free tier may be limited — check https://messari.io/api for plan details."
        )
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
    """Fetch ROI and volatility metrics.

    Note: The v1 /metrics/roi-data endpoint was removed.  We extract ROI
    from the main /metrics response (already fetched by fetch_market_metrics).
    This function is retained for interface compatibility but always returns [].
    """
    # roi-data endpoint was removed from Messari v1 and not re-added in v2.
    # ROI data (1h/24h/7d changes) is already covered by CoinMarketCap.
    return []


def fetch_developer_activity(
    api_key: str,
    symbol: str = "BTC/USD",
) -> list[tuple[str, str, float]]:
    """Fetch developer activity score (0-100 percentile).

    Note: The v1 /metrics/developer-activity endpoint was removed.
    Returns [] — this metric is no longer available on the free tier.
    """
    # /metrics/developer-activity was removed from Messari v1 and not re-added.
    return []


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

    total_written = 0
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
                total_written += written
                logger.info("  wrote %d rows", written)
            else:
                results[group_name] = 0
                logger.info("  no data returned")
        except Exception as exc:
            logger.warning("  %s failed: %s", group_name, exc)
            results[group_name] = 0

        time.sleep(0.3)  # be polite to the free-tier rate limit

    conn.close()

    if total_written == 0:
        raise RuntimeError(
            "Messari returned 0 records across all metric groups. "
            "The API key may be invalid or the free tier is restricted. "
            "Check your MESSARI_KEY or visit https://messari.io/api"
        )

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
