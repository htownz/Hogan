"""Fetch on-chain analytics from Dune Analytics.

Dune allows querying curated public dashboards via their REST API.
We run pre-built public queries that provide Bitcoin and macro signals.

Auth
----
    Set ``DUNE_API_KEY`` in your .env file.
    API docs: https://dune.com/docs/api/

Queries used (all public, no modifications needed)
---------------------------------------------------
* **BTC Exchange Outflow** (query 2417357)
    Measures Bitcoin flowing off exchanges — high outflow = accumulation (bullish).
    Columns: day, net_flow_btc

* **BTC Whale Count** (query 1258228)
    Wallets holding ≥ 100 BTC — tracks smart money accumulation/distribution.
    Columns: date, whale_count_change_pct

* **ETH Gas Price Trend** (query 2392226)
    Average gas price proxy for network activity (used as macro context).
    Columns: day, avg_gas_gwei

All metrics are stored daily in ``onchain_metrics`` as:
    ``dune_btc_exchange_netflow``, ``dune_btc_whale_pct``, ``dune_eth_gas_gwei``

Usage
-----
    python -m hogan_bot.fetch_dune
    python -m hogan_bot.fetch_dune --query 2417357 --name dune_btc_exchange_netflow
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_BASE = "https://api.dune.com/api/v1"
_TIMEOUT = 30
_POLL_INTERVAL = 3   # seconds between execution status polls
_MAX_POLLS = 20      # ~60s maximum wait

# Pre-selected public queries mapping (query_id, metric_name, column_with_value)
#
# NOTE: Dune public query IDs change over time as authors update their dashboards.
# If a query fails, browse https://dune.com to find an active replacement and update
# the ID here or pass a custom query list to fetch_all_dune().
#
# Currently configured:
#   2417357  — BTC exchange net-flow (may need replacement if schema changed)
# Removed 1258228 (whale count) — was returning HTTP 400 consistently.
_DEFAULT_QUERIES: list[tuple[int, str, str]] = [
    # BTC net exchange flow — positive = inflow (bearish), negative = outflow (bullish)
    (2417357, "dune_btc_exchange_netflow", "net_flow_btc"),
]


def _get_key() -> str:
    key = os.getenv("DUNE_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "DUNE_API_KEY not set.\n"
            "Get a key at https://dune.com/settings/api — add to .env as DUNE_API_KEY=..."
        )
    return key


def _post(path: str, api_key: str, body: dict | None = None) -> dict:
    url = f"{_BASE}{path}"
    data = json.dumps(body or {}).encode("utf-8") if body is not None else b"{}"
    req = Request(
        url,
        data=data,
        headers={"X-Dune-API-Key": api_key, "Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode())


def _get_req(path: str, api_key: str) -> dict:
    url = f"{_BASE}{path}"
    req = Request(url, headers={"X-Dune-API-Key": api_key})
    with urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode())


# ---------------------------------------------------------------------------
# Execution helpers
# ---------------------------------------------------------------------------

def _execute_query(query_id: int, api_key: str) -> str:
    """Submit a query execution and return the execution_id."""
    result = _post(f"/query/{query_id}/execute", api_key)
    return result["execution_id"]


def _wait_for_result(execution_id: str, api_key: str) -> dict:
    """Poll until the execution completes and return the results dict."""
    for _ in range(_MAX_POLLS):
        status = _get_req(f"/execution/{execution_id}/status", api_key)
        state = status.get("state", "")
        if state == "QUERY_STATE_COMPLETED":
            return _get_req(f"/execution/{execution_id}/results", api_key)
        if state in ("QUERY_STATE_FAILED", "QUERY_STATE_CANCELLED"):
            raise RuntimeError(f"Dune execution {execution_id} ended with state: {state}")
        time.sleep(_POLL_INTERVAL)
    raise TimeoutError(f"Dune execution {execution_id} did not complete within timeout")


def _extract_latest_value(
    rows: list[dict],
    value_col: str,
    date_col: str = "day",
) -> tuple[str, float] | None:
    """Return (date_str, value) for the most recent row that has a valid value."""
    candidates: list[tuple[str, float]] = []
    for row in rows:
        try:
            raw_date = str(row.get(date_col, row.get("date", row.get("block_date", ""))))[:10]
            val = float(row[value_col])
            if raw_date:
                candidates.append((raw_date, val))
        except (KeyError, TypeError, ValueError):
            continue
    if not candidates:
        return None
    return max(candidates, key=lambda x: x[0])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def _get_cached_results(query_id: int, api_key: str) -> dict:
    """Fetch the latest cached results for a query without consuming execution credits.

    Uses ``GET /api/v1/query/{id}/results`` — returns the most-recent cached run.
    Falls back to executing if no cached results exist.
    """
    return _get_req(f"/query/{query_id}/results", api_key)


def run_query(
    query_id: int,
    metric_name: str,
    value_col: str,
    api_key: str,
    symbol: str = "BTC/USD",
) -> list[tuple[str, str, float]]:
    """Fetch a Dune query and return (date, metric_name, value) records.

    Strategy:
    1. Try ``GET /query/{id}/results`` for cached results (free, instant).
    2. Fall back to ``POST /query/{id}/execute`` + poll if no cached data.

    Only the most-recent row is returned (snapshot pattern).
    """
    logger.info("Dune: fetching query %d (%s) ...", query_id, metric_name)

    result: dict | None = None

    # ── Step 1: try cached results (no credits consumed) ──────────────────
    try:
        result = _get_cached_results(query_id, api_key)
        rows = result.get("result", {}).get("rows", [])
        if rows:
            logger.info("  using cached results (%d rows)", len(rows))
        else:
            logger.info("  no cached results — triggering execution")
            result = None
    except (URLError, KeyError) as exc:
        logger.info("  cached results unavailable (%s) — triggering execution", exc)
        result = None

    # ── Step 2: execute if no cached data ─────────────────────────────────
    if result is None:
        try:
            exec_id = _execute_query(query_id, api_key)
            result = _wait_for_result(exec_id, api_key)
        except (RuntimeError, TimeoutError, URLError) as exc:
            logger.warning("Dune query %d failed: %s", query_id, exc)
            return []

    rows = result.get("result", {}).get("rows", [])
    if not rows:
        logger.warning("Dune query %d returned no rows", query_id)
        return []

    latest = _extract_latest_value(rows, value_col)
    if latest is None:
        logger.warning("Dune query %d: column '%s' not found in rows", query_id, value_col)
        return []

    date_str, val = latest
    logger.info("  %s = %.4f on %s", metric_name, val, date_str)
    return [(date_str, metric_name, val)]


def fetch_all_dune(
    symbol: str = "BTC/USD",
    db_path: str = "data/hogan.db",
    queries: list[tuple[int, str, str]] | None = None,
) -> dict[str, int]:
    """Run all default Dune queries and upsert results into the database.

    Parameters
    ----------
    symbol : str
        Trading pair (informational — queries are BTC-focused by default).
    db_path : str
        Path to the SQLite database.
    queries : list[tuple[int, str, str]], optional
        Custom list of (query_id, metric_name, value_column) triples.
        Defaults to ``_DEFAULT_QUERIES``.

    Returns
    -------
    dict
        Rows written per metric name.
    """
    from hogan_bot.storage import get_connection, upsert_onchain

    api_key = _get_key()
    conn = get_connection(db_path)
    active_queries = queries or _DEFAULT_QUERIES
    results: dict[str, int] = {}

    total_written = 0
    for query_id, metric_name, value_col in active_queries:
        records = run_query(query_id, metric_name, value_col, api_key, symbol=symbol)
        if records:
            written = upsert_onchain(conn, symbol, records)
            results[metric_name] = written
            total_written += written
        else:
            results[metric_name] = 0
        time.sleep(1.0)  # avoid hammering the API

    conn.close()

    if active_queries and total_written == 0:
        raise RuntimeError(
            "Dune returned 0 records for all queries. "
            "Query IDs may be outdated — browse https://dune.com to find active replacements. "
            "Note: Dune metrics are not in the ML feature vector, so this does not affect model quality."
        )

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch Dune Analytics on-chain data (BTC exchange flow, whale count, etc.)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="BTC/USD", help="Trading pair context")
    p.add_argument("--db", default="data/hogan.db", help="SQLite database path")
    p.add_argument(
        "--query", type=int, metavar="ID",
        help="Run a specific Dune query ID (requires --name and --col)",
    )
    p.add_argument("--name", default="dune_custom", help="Metric name for custom query")
    p.add_argument("--col", default="value", help="Column name containing the numeric value")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    if args.query:
        api_key = _get_key()
        from hogan_bot.storage import get_connection, upsert_onchain
        records = run_query(args.query, args.name, args.col, api_key, symbol=args.symbol)
        if records:
            conn = get_connection(args.db)
            written = upsert_onchain(conn, args.symbol, records)
            conn.close()
            print(json.dumps({args.name: written}, indent=2))
        else:
            print("No data returned.")
    else:
        result = fetch_all_dune(symbol=args.symbol, db_path=args.db)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
