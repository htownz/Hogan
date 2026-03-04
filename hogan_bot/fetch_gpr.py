"""Fetch and import the Caldara & Iacoviello Geopolitical Risk (GPR) Index.

The GPR Index quantifies geopolitical risk by counting news articles
mentioning war, terrorism, and geopolitical tensions across major newspapers.
It is free academic data maintained by the U.S. Federal Reserve Board.

Source
------
    https://www.matteoiacoviello.com/gpr.htm
    Daily data file: ``data_gpr_daily_recent.xls``
    Updated quarterly; data lags the current date by ~1 month.

Metric stored
-------------
    ``gpr_index`` — normalized GPR score stored daily in ``onchain_metrics``.

    Normalisation: (value - baseline_mean) / baseline_std using the
    2000-2019 average as the stable reference period.  Typical range ≈ [-1, 5].
    In ``build_ext_features()`` this is used raw (within the ±10 clip).

    A reading above 0 means above-average geopolitical stress.
    A reading above 2 indicates an elevated-risk episode (e.g. Gulf War: ~10,
    9/11: ~6, COVID-2020: ~3, Ukraine invasion 2022: ~4).

Usage
-----
    # Download and import (creates/updates onchain_metrics)
    python -m hogan_bot.fetch_gpr

    # Force re-download even if cached file exists
    python -m hogan_bot.fetch_gpr --force
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlretrieve

_GPR_URL = "https://www.matteoiacoviello.com/gpr_files/data_gpr_daily_recent.xls"
_CACHE_PATH = Path("data") / "gpr_daily_cache.xls"
_METRIC = "gpr_index"

# 2000-2019 baseline statistics for normalisation (pre-computed from the paper)
_BASELINE_MEAN = 100.0   # GPR is already indexed to 2000-2019 = 100
_BASELINE_STD = 50.0     # approximate long-run std dev


def _download(force: bool = False) -> Path:
    """Download the GPR Excel file; return its local path."""
    _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    if _CACHE_PATH.exists() and not force:
        age_days = (time.time() - _CACHE_PATH.stat().st_mtime) / 86400
        if age_days < 30:
            print(f"  Using cached file ({age_days:.0f} days old): {_CACHE_PATH}")
            return _CACHE_PATH
    print(f"  Downloading GPR data from {_GPR_URL} ...")
    try:
        urlretrieve(_GPR_URL, _CACHE_PATH)
        print(f"  Saved to {_CACHE_PATH}")
    except URLError as exc:
        raise RuntimeError(f"Failed to download GPR data: {exc}") from exc
    return _CACHE_PATH


def _parse_xls(path: Path):
    """Parse the GPR Excel file and return a list of (date_str, gpr_value) tuples.

    Handles both legacy ``.xls`` (xlrd) and modern ``.xlsx`` (openpyxl) formats.
    """
    rows: list = []
    if path.suffix.lower() == ".xls":
        # Legacy binary Excel format — requires xlrd
        try:
            import xlrd
            wb = xlrd.open_workbook(str(path))
            ws = wb.sheet_by_index(0)
            rows = [ws.row_values(i) for i in range(ws.nrows)]
        except ImportError as exc:
            raise RuntimeError(
                "xlrd is required to read .xls files.\n"
                "Run: pip install xlrd"
            ) from exc
    else:
        # Modern .xlsx — use openpyxl
        try:
            import openpyxl
            wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
            ws = wb.active
            rows = list(ws.iter_rows(values_only=True))
            wb.close()
        except ImportError as exc:
            raise RuntimeError(
                "openpyxl is required to read .xlsx files.\n"
                "Run: pip install openpyxl"
            ) from exc

    results: list[tuple[str, float]] = []
    import datetime as dt_module

    for i, row in enumerate(rows):
        if len(row) < 3:
            continue
        raw_date = row[0]
        raw_val = row[2]   # column 2 = 'GPRD' (smoothed daily GPR index)

        if raw_date is None or raw_val is None:
            continue

        # Parse date — GPR files use YYYYMMDD integer strings (e.g. '19850101')
        date_str: str | None = None
        if isinstance(raw_date, str):
            raw_date = raw_date.strip()
            for fmt in ("%Y%m%d", "%Y-%m-%d", "%m/%d/%Y", "%Y/%m/%d", "%d-%m-%Y"):
                try:
                    date_str = dt_module.datetime.strptime(raw_date, fmt).strftime("%Y-%m-%d")
                    break
                except ValueError:
                    continue
        elif isinstance(raw_date, (dt_module.date, dt_module.datetime)):
            date_str = raw_date.strftime("%Y-%m-%d")
        elif isinstance(raw_date, (int, float)):
            # Try as Excel serial number (openpyxl path)
            try:
                import openpyxl.utils.datetime as oxl_dt
                date_obj = oxl_dt.from_excel(int(raw_date))
                date_str = date_obj.strftime("%Y-%m-%d")
            except Exception:
                pass

        if date_str is None:
            continue

        try:
            gpr_val = float(raw_val)
        except (ValueError, TypeError):
            continue

        if gpr_val <= 0:
            continue

        # Normalise: GPR is indexed to 100 = 2000-2019 average
        # Returns z-score-like value; most readings fall in [-1, 5]
        normalised = (gpr_val - _BASELINE_MEAN) / _BASELINE_STD

        results.append((date_str, normalised))

    return results


def fetch_and_store(
    symbol: str = "BTC/USD",
    db_path: str = "data/hogan.db",
    force: bool = False,
) -> int:
    """Download GPR data and upsert into ``onchain_metrics``.

    Returns
    -------
    int
        Number of rows written.
    """
    from hogan_bot.storage import get_connection, upsert_onchain

    print("Fetching Caldara & Iacoviello GPR Index ...")
    xls_path = _download(force=force)
    parsed = _parse_xls(xls_path)

    if not parsed:
        print("  Warning: no parseable rows found in GPR file.")
        return 0

    rows = [(date_str, _METRIC, val) for date_str, val in parsed]
    conn = get_connection(db_path)
    written = upsert_onchain(conn, symbol, rows)
    conn.close()

    if parsed:
        latest_date = max(r[0] for r in parsed)
        latest_raw_val = next(r[1] for r in reversed(parsed))
        latest_actual = latest_raw_val * _BASELINE_STD + _BASELINE_MEAN
        print(f"  {len(rows)} daily GPR records upserted")
        print(f"  Latest date: {latest_date}  GPR={latest_actual:.1f}  (z={latest_raw_val:.2f})")
    return written


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Download and import the Caldara & Iacoviello GPR Index",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="BTC/USD", help="Symbol for DB storage")
    p.add_argument("--db", default="data/hogan.db", help="SQLite database path")
    p.add_argument("--force", action="store_true", help="Re-download even if cached file is recent")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    fetch_and_store(symbol=args.symbol, db_path=args.db, force=args.force)


if __name__ == "__main__":
    main()
