"""Fetch Bitcoin on-chain metrics from Blockchain.com and Mempool.space.

Both APIs are completely free with no authentication required.

Why on-chain data matters
--------------------------
* **Hash rate** — miner security investment; drops before capitulation events
* **Mining difficulty** — auto-adjusts every 2016 blocks; % change signals
  miner confidence in forward price
* **Mempool size** — network congestion proxy; spikes precede fee surges and
  often accompany price volatility
* **Transaction count** — network utilization; declining tx during price run
  is a bearish divergence signal
* **Average fee (USD)** — activity signal; very high fees = peak demand,
  very low fees = network dormancy

Data sources
------------
blockchain.com/api/charts  — no key, JSON REST
mempool.space/api          — no key, JSON REST (Bitcoin mempool specialist)

Metrics stored in ``onchain_metrics``
--------------------------------------
``btc_hashrate_eh``     — Hash rate in EH/s (divide by 1000 to normalize)
``btc_difficulty``      — Current difficulty (divide by 1e13 to normalize)
``btc_mempool_mb``      — Unconfirmed transaction backlog in MB
``btc_tx_count``        — 24-hour confirmed transaction count
``btc_avg_fee_usd``     — Average transaction fee in USD
``btc_active_addr``     — Unique active addresses in last 24 hours

Usage
-----
    python -m hogan_bot.fetch_blockchain
    python -m hogan_bot.fetch_blockchain --days 90   # historical backfill
"""
from __future__ import annotations

import argparse
import json
import logging
import time
from datetime import date, datetime, timezone
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_TIMEOUT = 20
_BLOCKCHAIN_BASE = "https://api.blockchain.info/charts"
_MEMPOOL_BASE = "https://mempool.space/api"


def _get_json(url: str) -> dict | list:
    """Simple GET with retries."""
    for attempt in range(3):
        try:
            req = Request(url, headers={"Accept": "application/json"})
            with urlopen(req, timeout=_TIMEOUT) as resp:
                return json.loads(resp.read().decode())
        except URLError as exc:
            if attempt == 2:
                raise RuntimeError(f"Request failed: {url}  ({exc})") from exc
            time.sleep(2 ** attempt)
    return {}


# ---------------------------------------------------------------------------
# Blockchain.com chart series
# ---------------------------------------------------------------------------

def _fetch_blockchain_chart(
    chart_name: str,
    days: int = 30,
) -> list[tuple[str, float]]:
    """Fetch a blockchain.com chart series as (date_str, value) pairs."""
    url = (
        f"{_BLOCKCHAIN_BASE}/{chart_name}"
        f"?timespan={days}days&sampled=true&metadata=false&cors=true&format=json"
    )
    try:
        data = _get_json(url)
    except RuntimeError as exc:
        logger.warning("blockchain.com chart %s failed: %s", chart_name, exc)
        return []

    values = data.get("values", []) if isinstance(data, dict) else []
    results: list[tuple[str, float]] = []
    for point in values:
        try:
            ts = int(point["x"])
            val = float(point["y"])
            d = datetime.fromtimestamp(ts, tz=timezone.utc).strftime("%Y-%m-%d")
            results.append((d, val))
        except (KeyError, TypeError, ValueError):
            continue
    return results


# ---------------------------------------------------------------------------
# Mempool.space (real-time snapshot)
# ---------------------------------------------------------------------------

def _fetch_mempool_snapshot() -> dict[str, float]:
    """Fetch current mempool statistics from mempool.space.

    Returns a dict with keys: mempool_mb, tx_count, fee_fastestSat, fee_halfHourSat
    """
    result: dict[str, float] = {}
    try:
        data = _get_json(f"{_MEMPOOL_BASE}/mempool")
        if isinstance(data, dict):
            vsize = data.get("vsize", 0)  # bytes
            result["mempool_mb"] = round(vsize / 1_000_000, 4)
            result["tx_count"] = float(data.get("count", 0))
    except Exception as exc:
        logger.warning("mempool.space mempool snapshot failed: %s", exc)

    try:
        fees = _get_json(f"{_MEMPOOL_BASE}/v1/fees/recommended")
        if isinstance(fees, dict):
            result["fee_fastest_sat"] = float(fees.get("fastestFee", 0))
            result["fee_hour_sat"] = float(fees.get("hourFee", 0))
    except Exception as exc:
        logger.warning("mempool.space fees failed: %s", exc)

    return result


def _fetch_btc_price_for_fee_usd() -> float:
    """Get current BTC/USD price from Blockchain.com to convert fee sat→USD."""
    try:
        data = _get_json("https://blockchain.info/ticker")
        if isinstance(data, dict):
            usd = data.get("USD", {})
            return float(usd.get("last", 0))
    except Exception:
        pass
    return 0.0


# ---------------------------------------------------------------------------
# Main fetch
# ---------------------------------------------------------------------------

def fetch_all_blockchain(
    symbol: str = "BTC/USD",
    db_path: str = "data/hogan.db",
    days: int = 30,
) -> dict[str, int]:
    """Fetch all Bitcoin on-chain metrics and upsert into the database.

    Parameters
    ----------
    symbol : str
        Trading pair label for DB rows (informational).
    db_path : str
        Path to the SQLite database.
    days : int
        Historical lookback days for chart series.

    Returns
    -------
    dict
        ``{metric_name: rows_written}``
    """
    from hogan_bot.storage import get_connection, upsert_onchain

    conn = get_connection(db_path)
    results: dict[str, int] = {}
    today = date.today().isoformat()

    # ── Blockchain.com chart series ──────────────────────────────────────────
    chart_series = [
        ("hash-rate",         "btc_hashrate_eh",  1.0),      # TH/s → EH/s (÷1e6)
        ("difficulty",        "btc_difficulty",   1.0),
        ("n-transactions",    "btc_tx_count",     1.0),
        ("n-unique-addresses","btc_active_addr",  1.0),
    ]

    for chart_name, metric_name, _divisor in chart_series:
        logger.info("blockchain.com: fetching %s ...", chart_name)
        series = _fetch_blockchain_chart(chart_name, days=days)
        if not series:
            results[metric_name] = 0
            continue

        records: list[tuple[str, str, float]] = []
        for d, v in series:
            # Convert hash-rate from TH/s to EH/s for normalization
            if metric_name == "btc_hashrate_eh":
                v = v / 1_000_000.0
            records.append((d, metric_name, v))

        if records:
            written = upsert_onchain(conn, symbol, records)
            results[metric_name] = written
            logger.info("  wrote %d rows", written)
        else:
            results[metric_name] = 0
        time.sleep(0.3)

    # ── Mempool.space real-time snapshot (stored as today's value) ───────────
    logger.info("mempool.space: fetching mempool snapshot ...")
    snapshot = _fetch_mempool_snapshot()

    mempool_records: list[tuple[str, str, float]] = []
    if "mempool_mb" in snapshot:
        mempool_records.append((today, "btc_mempool_mb", snapshot["mempool_mb"]))
        logger.info("  mempool size: %.2f MB", snapshot["mempool_mb"])
    if "fee_fastest_sat" in snapshot:
        # Convert sat/vbyte → estimated USD fee for a 250-vbyte tx
        btc_price = _fetch_btc_price_for_fee_usd()
        if btc_price > 0:
            fee_usd = snapshot["fee_fastest_sat"] * 250 * btc_price / 1e8
            mempool_records.append((today, "btc_avg_fee_usd", round(fee_usd, 4)))
            logger.info("  avg fee (fast): $%.2f USD", fee_usd)
        # Also store raw sat/vbyte
        mempool_records.append((today, "btc_fee_sat_vbyte", snapshot["fee_fastest_sat"]))

    if mempool_records:
        written = upsert_onchain(conn, symbol, mempool_records)
        results["btc_mempool"] = written

    conn.close()
    return results


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch Bitcoin on-chain metrics (blockchain.com + mempool.space, no key)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="BTC/USD")
    p.add_argument("--db", default="data/hogan.db")
    p.add_argument("--days", type=int, default=30,
                   help="Lookback days for historical chart data")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    result = fetch_all_blockchain(symbol=args.symbol, db_path=args.db, days=args.days)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
