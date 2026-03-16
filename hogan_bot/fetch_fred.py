"""Fetch Federal Reserve (FRED) macroeconomic data for Hogan.

FRED (Federal Reserve Economic Data) is operated by the St. Louis Fed and
provides 800,000+ free economic time series.  For crypto trading, the most
useful series are interest rates, money supply, inflation, and the yield curve.

Why macro matters for BTC
--------------------------
* Rising 10Y yields → risk-off → crypto sell pressure
* Inverted yield curve (T10Y2Y < 0) → recession signal → risk-off
* Expanding M2 money supply → liquidity flood → crypto bullish
* High CPI → stagflation risk OR inflation hedge narrative (split signal)
* Fed rate hikes → tightening → headwind for speculative assets

Auth
----
    Free API key from https://fred.stlouisfed.org/docs/api/api_key.html
    (instant, no credit card, no payment)
    Set ``FRED_API_KEY`` in your .env file.

Series fetched
--------------
``fred_dgs10``         — 10-Year Treasury Constant Maturity Rate (%)
``fred_dgs2``          — 2-Year Treasury Constant Maturity Rate (%)
``fred_t10y2y``        — 10Y-2Y Yield Spread (recession indicator; <0 = inverted)
``fred_fedfunds``      — Effective Federal Funds Rate (%)
``fred_m2_yoy``        — M2 Money Supply Year-over-Year % change (liquidity proxy)
``fred_cpi_yoy``       — CPI Year-over-Year % change (inflation)
``fred_dxy``           — Trade Weighted US Dollar Index (broad, goods) — DTWEXBGS

All stored daily in ``onchain_metrics``.  Values are already in % or index
units — normalization happens in ``features_mtf.py``.

Usage
-----
    python -m hogan_bot.fetch_fred
    python -m hogan_bot.fetch_fred --days 365   # backfill one year
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import date, datetime, timedelta
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_BASE = "https://api.stlouisfed.org/fred/series/observations"
_TIMEOUT = 20
_DEFAULT_DAYS = 60

# (FRED series ID, metric name stored in DB, description)
_SERIES: list[tuple[str, str, str]] = [
    ("DGS10",    "fred_dgs10",    "10-Year Treasury Yield (%)"),
    ("DGS2",     "fred_dgs2",     "2-Year Treasury Yield (%)"),
    ("T10Y2Y",   "fred_t10y2y",   "10Y-2Y Yield Spread (%; <0 = inverted)"),
    ("FEDFUNDS", "fred_fedfunds", "Effective Federal Funds Rate (%)"),
    ("M2SL",     "fred_m2_yoy",   "M2 Money Supply YoY % change (computed)"),
    ("CPIAUCSL", "fred_cpi_yoy",  "CPI YoY % change (computed)"),
    ("DTWEXBGS", "fred_dxy",      "Trade-Weighted US Dollar Index (broad)"),
]


def _get_key() -> str:
    key = os.getenv("FRED_API_KEY", "").strip()
    if not key:
        raise RuntimeError(
            "FRED_API_KEY not set.\n"
            "Get a free key (instant) at: https://fred.stlouisfed.org/docs/api/api_key.html\n"
            "Then add to .env:  FRED_API_KEY=<your_key>"
        )
    return key


def _fetch_series(
    series_id: str,
    api_key: str,
    start_date: str,
    end_date: str,
) -> list[tuple[str, float]]:
    """Return list of (date_str, value) for a FRED series."""
    url = (
        f"{_BASE}?series_id={series_id}"
        f"&observation_start={start_date}"
        f"&observation_end={end_date}"
        f"&api_key={api_key}"
        "&file_type=json"
        "&sort_order=asc"
    )
    for attempt in range(3):
        try:
            req = Request(url, headers={"Accept": "application/json"})
            with urlopen(req, timeout=_TIMEOUT) as resp:
                data = json.loads(resp.read().decode())
            observations = data.get("observations", [])
            result: list[tuple[str, float]] = []
            for obs in observations:
                val_str = obs.get("value", ".")
                if val_str == ".":
                    continue  # FRED uses "." for missing values
                try:
                    result.append((obs["date"], float(val_str)))
                except (KeyError, ValueError):
                    continue
            return result
        except URLError as exc:
            if attempt == 2:
                raise RuntimeError(f"FRED request failed for {series_id}: {exc}") from exc
            time.sleep(2 ** attempt)
    return []


def _yoy_pct_change(observations: list[tuple[str, float]]) -> list[tuple[str, float]]:
    """Convert a level series into YoY % change.

    FRED M2 and CPI are levels — we want YoY % change as the feature.
    Requires at least 13 months of data.
    """
    if len(observations) < 13:
        return []
    by_date = {d: v for d, v in observations}
    dates = sorted(by_date.keys())
    result: list[tuple[str, float]] = []
    for d in dates:
        # Find the observation from ~365 days ago
        prior_d = str(
            (datetime.strptime(d, "%Y-%m-%d") - timedelta(days=365)).date()
        )
        # FRED data is monthly — search within ±35 days
        prior_val = None
        for delta in range(0, 36):
            candidate = str(
                (datetime.strptime(prior_d, "%Y-%m-%d") + timedelta(days=delta)).date()
            )
            if candidate in by_date:
                prior_val = by_date[candidate]
                break
            candidate2 = str(
                (datetime.strptime(prior_d, "%Y-%m-%d") - timedelta(days=delta)).date()
            )
            if candidate2 in by_date:
                prior_val = by_date[candidate2]
                break
        if prior_val and prior_val != 0:
            yoy = ((by_date[d] - prior_val) / abs(prior_val)) * 100.0
            result.append((d, round(yoy, 4)))
    return result


def fetch_all_fred(
    symbol: str = "BTC/USD",
    db_path: str = "data/hogan.db",
    days: int = _DEFAULT_DAYS,
) -> dict[str, int]:
    """Fetch all configured FRED series and upsert into the database.

    Parameters
    ----------
    symbol : str
        Trading pair label for DB rows.
    db_path : str
        Path to the SQLite database.
    days : int
        Lookback window.  Use 395+ when fetching M2/CPI to ensure enough
        history for the YoY % change computation.

    Returns
    -------
    dict
        ``{metric_name: rows_written}``
    """
    from hogan_bot.storage import get_connection, upsert_onchain

    api_key = _get_key()
    end_date = date.today().isoformat()
    # Add 13 months of buffer for YoY calculations on M2/CPI
    start_date = (date.today() - timedelta(days=max(days, 395))).isoformat()

    conn = get_connection(db_path)
    results: dict[str, int] = {}

    for series_id, metric_name, description in _SERIES:
        logger.info("FRED: fetching %s (%s) ...", series_id, description)
        try:
            raw = _fetch_series(series_id, api_key, start_date, end_date)
            if not raw:
                logger.info("  no data returned")
                results[metric_name] = 0
                continue

            # M2 and CPI: convert level → YoY % change
            if metric_name in ("fred_m2_yoy", "fred_cpi_yoy"):
                observations = _yoy_pct_change(raw)
                # Trim to requested window after YoY calc
                cutoff = (date.today() - timedelta(days=days)).isoformat()
                observations = [(d, v) for d, v in observations if d >= cutoff]
            else:
                observations = raw

            records: list[tuple[str, str, float]] = [
                (d, metric_name, v) for d, v in observations
            ]
            if records:
                written = upsert_onchain(conn, symbol, records)
                results[metric_name] = written
                logger.info("  wrote %d rows", written)
            else:
                results[metric_name] = 0

        except Exception as exc:
            logger.warning("  %s failed: %s", series_id, exc)
            results[metric_name] = 0

        time.sleep(0.2)  # FRED rate limit: 120 req/min, this is very conservative

    conn.close()
    return results


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch FRED macroeconomic data (rates, yield curve, M2, CPI, DXY)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="BTC/USD")
    p.add_argument("--db", default="data/hogan.db")
    p.add_argument("--days", type=int, default=_DEFAULT_DAYS,
                   help="Lookback days (use 400+ for initial backfill with YoY series)")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    result = fetch_all_fred(symbol=args.symbol, db_path=args.db, days=args.days)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
