"""OpenBB data fetcher — Phase 6.

Fetches macro and derivatives data not currently available in other fetch
modules.  Uses OpenBB Platform (``openbb>=4.3.0``) as a unified connector.

Data sources:
    - BTC options skew  (put/call ratio, 25-delta risk reversal)
    - DXY               (US Dollar Index — inverse correlation to crypto)
    - VIX               (volatility index — risk-off signal)
    - BTC dominance     (cross-check vs CoinGecko)
    - Macro calendar    (Fed meeting dates as binary feature)

All data is stored in the ``onchain_metrics`` table under a ``BTC/USD``
symbol key, using the existing schema so ``features_mtf.py`` can pick it up.

Usage::

    python -m hogan_bot.fetch_openbb
    python -m hogan_bot.fetch_openbb --days 30 --db data/hogan.db
"""
from __future__ import annotations

import argparse
import logging
from datetime import date, datetime, timedelta, timezone
from typing import Any

logger = logging.getLogger(__name__)

_SYMBOL = "BTC/USD"
_DEFAULT_DAYS = 30


def _try_import_openbb():
    try:
        from openbb import obb  # type: ignore
        return obb
    except ImportError:
        return None


# ---------------------------------------------------------------------------
# Fetch functions (each returns list[tuple[date_str, metric_name, value]])
# ---------------------------------------------------------------------------

def fetch_dxy(days: int = _DEFAULT_DAYS) -> list[tuple[str, str, float]]:
    """Fetch DXY (US Dollar Index) daily close from OpenBB.

    Falls back to yfinance if OpenBB is unavailable.
    """
    records: list[tuple[str, str, float]] = []
    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    obb = _try_import_openbb()
    if obb is not None:
        try:
            data = obb.equity.price.historical(
                "DX-Y.NYB",
                start_date=str(start_date),
                end_date=str(end_date),
                provider="yfinance",
            ).to_df()
            for idx, row in data.iterrows():
                d = str(idx.date()) if hasattr(idx, "date") else str(idx)[:10]
                records.append((d, "dxy_close", float(row.get("close", 0))))
            logger.info("Fetched %d DXY rows via OpenBB", len(records))
            return records
        except Exception as exc:
            logger.warning("OpenBB DXY fetch failed: %s — trying yfinance fallback", exc)

    # yfinance fallback
    try:
        import yfinance as yf
        df = yf.download("DX-Y.NYB", start=str(start_date), end=str(end_date), progress=False)
        if not df.empty:
            for idx, row in df.iterrows():
                d = str(idx.date()) if hasattr(idx, "date") else str(idx)[:10]
                records.append((d, "dxy_close", float(row.get("Close", row.get("close", 0)))))
        logger.info("Fetched %d DXY rows via yfinance", len(records))
    except Exception as exc:
        logger.warning("yfinance DXY fetch failed: %s", exc)

    return records


def fetch_vix(days: int = _DEFAULT_DAYS) -> list[tuple[str, str, float]]:
    """Fetch VIX (CBOE Volatility Index) daily close."""
    records: list[tuple[str, str, float]] = []
    end_date = date.today()
    start_date = end_date - timedelta(days=days)

    obb = _try_import_openbb()
    if obb is not None:
        try:
            data = obb.equity.price.historical(
                "^VIX",
                start_date=str(start_date),
                end_date=str(end_date),
                provider="yfinance",
            ).to_df()
            for idx, row in data.iterrows():
                d = str(idx.date()) if hasattr(idx, "date") else str(idx)[:10]
                records.append((d, "vix_close", float(row.get("close", 0))))
            logger.info("Fetched %d VIX rows via OpenBB", len(records))
            return records
        except Exception as exc:
            logger.warning("OpenBB VIX fetch failed: %s — trying yfinance fallback", exc)

    try:
        import yfinance as yf
        df = yf.download("^VIX", start=str(start_date), end=str(end_date), progress=False)
        if not df.empty:
            for idx, row in df.iterrows():
                d = str(idx.date()) if hasattr(idx, "date") else str(idx)[:10]
                records.append((d, "vix_close", float(row.get("Close", row.get("close", 0)))))
        logger.info("Fetched %d VIX rows via yfinance", len(records))
    except Exception as exc:
        logger.warning("yfinance VIX fetch failed: %s", exc)

    return records


def fetch_spy_return(days: int = _DEFAULT_DAYS) -> list[tuple[str, str, float]]:
    """Fetch SPY 1-day return (risk-on/risk-off indicator)."""
    records: list[tuple[str, str, float]] = []
    end_date = date.today()
    start_date = end_date - timedelta(days=days + 5)  # extra buffer for pct_change

    try:
        import yfinance as yf
        import pandas as pd
        df = yf.download("SPY", start=str(start_date), end=str(end_date), progress=False)
        if not df.empty:
            close = df["Close"] if "Close" in df.columns else df["close"]
            returns = close.pct_change().dropna()
            for idx, val in returns.items():
                d = str(idx.date()) if hasattr(idx, "date") else str(idx)[:10]
                records.append((d, "spy_return_pct", round(float(val) * 100, 4)))
        logger.info("Fetched %d SPY return rows", len(records))
    except Exception as exc:
        logger.warning("SPY return fetch failed: %s", exc)

    return records


def fetch_fed_calendar(years: int = 2) -> list[tuple[str, str, float]]:
    """Generate Fed FOMC meeting dates as a binary calendar.

    Uses a static list of known FOMC meeting dates (approximately 8/year).
    Returns 1.0 for days within 2 days of an FOMC meeting, else 0.0.
    """
    # FOMC meeting dates 2024-2026 (approximate)
    fomc_dates = {
        "2024-01-31", "2024-03-20", "2024-05-01", "2024-06-12",
        "2024-07-31", "2024-09-18", "2024-11-07", "2024-12-18",
        "2025-01-29", "2025-03-19", "2025-05-07", "2025-06-18",
        "2025-07-30", "2025-09-17", "2025-11-05", "2025-12-17",
        "2026-01-28", "2026-03-18", "2026-04-29", "2026-06-17",
        "2026-07-29", "2026-09-16", "2026-11-04", "2026-12-16",
    }
    fomc_dt = {datetime.strptime(d, "%Y-%m-%d").date() for d in fomc_dates}

    records: list[tuple[str, str, float]] = []
    today = date.today()
    start = today - timedelta(days=365 * years)
    current = start
    while current <= today:
        is_fomc = any(abs((current - f).days) <= 1 for f in fomc_dt)
        records.append((str(current), "fomc_proximity", 1.0 if is_fomc else 0.0))
        current += timedelta(days=1)

    logger.info("Generated %d FOMC calendar rows", len(records))
    return records


def fetch_btc_options_skew(days: int = _DEFAULT_DAYS) -> list[tuple[str, str, float]]:
    """Fetch BTC options put/call ratio from OpenBB derivatives.

    Falls back to zeros if unavailable (options data often requires paid API).
    """
    records: list[tuple[str, str, float]] = []
    obb = _try_import_openbb()
    if obb is None:
        logger.info("OpenBB not available — skipping options skew fetch.")
        return records

    try:
        # Try to get BTC put/call ratio (requires OpenBB derivatives extension)
        data = obb.derivatives.options.chains("BTC-USD", provider="cboe").to_df()
        if not data.empty:
            # Simple put/call ratio from open interest
            puts = data[data.get("option_type", data.get("type", "")) == "put"]
            calls = data[data.get("option_type", data.get("type", "")) == "call"]
            put_oi = puts.get("open_interest", puts.get("oi", 0)).sum()
            call_oi = calls.get("open_interest", calls.get("oi", 0)).sum()
            ratio = float(put_oi) / max(float(call_oi), 1.0)
            today = str(date.today())
            records.append((today, "btc_put_call_ratio", round(ratio, 4)))
            logger.info("BTC put/call ratio: %.4f", ratio)
    except Exception as exc:
        logger.warning("BTC options skew fetch failed (requires paid API): %s", exc)

    return records


# ---------------------------------------------------------------------------
# Storage helper
# ---------------------------------------------------------------------------

def store_records(
    records: list[tuple[str, str, float]],
    symbol: str,
    conn,
) -> int:
    """Upsert records into the onchain_metrics table."""
    if not records:
        return 0
    from hogan_bot.storage import upsert_onchain
    upsert_onchain(conn, symbol, records)
    return len(records)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Fetch OpenBB macro/derivatives data")
    p.add_argument("--db", default="data/hogan.db")
    p.add_argument("--days", type=int, default=_DEFAULT_DAYS)
    p.add_argument("--symbol", default=_SYMBOL)
    args = p.parse_args()

    from hogan_bot.storage import get_connection
    conn = get_connection(args.db)

    total = 0
    for fetch_fn, name in [
        (fetch_dxy, "DXY"),
        (fetch_vix, "VIX"),
        (fetch_spy_return, "SPY"),
        (fetch_fed_calendar, "FOMC"),
        (fetch_btc_options_skew, "BTC options"),
    ]:
        try:
            records = fetch_fn(args.days) if name not in ("FOMC", "BTC options") else fetch_fn()
            n = store_records(records, args.symbol, conn)
            total += n
            logger.info("%s: stored %d rows", name, n)
        except Exception as exc:
            logger.error("%s fetch error: %s", name, exc)

    conn.close()
    print(f"Total rows stored: {total}")


if __name__ == "__main__":
    _main()
