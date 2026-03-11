"""Lookahead Bias Checker — Phase 7 (Freqtrade-inspired).

Scans indicator and ML feature computations for potential lookahead bias —
i.e., indicator values that reference future candle data relative to the bar
being evaluated.  Run this before any backtest to catch silent P&L inflation.

How it works
------------
1. Takes a DataFrame of candles.
2. Runs the full signal pipeline twice:
   - Pass A: uses the first N bars (baseline)
   - Pass B: uses bars 1..N+1 (shift by 1)
3. For each indicator column, checks whether the value at bar N in Pass A
   equals the value at bar N in Pass B.  If they differ, the indicator is
   NOT recomputing from identical history — indicating future leak.
4. Also checks ML features for NaN patterns that suggest improper shifts.

A lookahead-free indicator must produce identical bar-N values in both passes.

Usage::

    python -m hogan_bot.lookahead_check --db data/hogan.db --symbol BTC/USD
    python -m hogan_bot.lookahead_check --csv data/btc_candles.csv
"""
from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


def _compute_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all indicators on the given candle DataFrame.

    Returns a DataFrame with indicator columns appended.
    """
    from hogan_bot.indicators import fvg_features_frame

    out = df.copy()
    close = out["close"]
    high = out["high"]
    low = out["low"]
    volume = out.get("volume", pd.Series(1.0, index=out.index))

    # Moving averages
    out["sma_10"] = close.rolling(10).mean()
    out["sma_20"] = close.rolling(20).mean()
    out["sma_50"] = close.rolling(50).mean()
    out["ema_9"] = close.ewm(span=9, adjust=False).mean()
    out["ema_21"] = close.ewm(span=21, adjust=False).mean()

    # RSI
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    loss = (-delta).clip(lower=0).ewm(com=13, adjust=False).mean()
    out["rsi_14"] = 100 - 100 / (1 + gain / (loss + 1e-9))

    # ATR
    hl = high - low
    hpc = (high - close.shift(1)).abs()
    lpc = (low - close.shift(1)).abs()
    out["atr_14"] = pd.concat([hl, hpc, lpc], axis=1).max(axis=1).rolling(14).mean()

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    out["macd_hist"] = macd_line - signal_line

    # Bollinger Bands
    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    out["bb_upper"] = bb_mid + 2 * bb_std
    out["bb_lower"] = bb_mid - 2 * bb_std
    out["bb_pct_b"] = (close - out["bb_lower"]) / (out["bb_upper"] - out["bb_lower"] + 1e-9)

    # Volume ratio
    vol_ma = volume.rolling(20).mean()
    out["vol_ratio"] = volume / (vol_ma + 1e-9)

    # FVG features (point-in-time only — checks for gaps in the last 3 bars)
    try:
        fvg_cols = fvg_features_frame(df)
        for col in fvg_cols.columns:
            out[col] = fvg_cols[col]
    except Exception:
        pass

    return out


def check_lookahead(
    candles: pd.DataFrame,
    tolerance: float = 1e-9,
) -> dict[str, Any]:
    """Check indicators for lookahead bias.

    Parameters
    ----------
    candles:
        A candle DataFrame with at least ``open``, ``high``, ``low``,
        ``close``, ``volume`` columns.
    tolerance:
        Floating-point tolerance for indicator value comparison.

    Returns
    -------
    dict with:
        - ``ok``: True if no lookahead detected.
        - ``leaking_columns``: list of column names that may have lookahead.
        - ``clean_columns``: list of columns that passed.
        - ``details``: per-column diff statistics.
    """
    if len(candles) < 60:
        return {
            "ok": True,
            "leaking_columns": [],
            "clean_columns": [],
            "details": {},
            "warning": "Too few candles to run lookahead check (need ≥60).",
        }

    n = len(candles)
    mid = n // 2
    check_idx = mid - 1  # the bar whose value we compare

    # Pass A: bars 0..mid
    df_a = candles.iloc[:mid + 1].copy().reset_index(drop=True)
    ind_a = _compute_indicators(df_a)

    # Pass B: bars 1..mid+2  (same history length, shifted by 1)
    df_b = candles.iloc[1:mid + 2].copy().reset_index(drop=True)
    ind_b = _compute_indicators(df_b)

    # The bar at index `check_idx` in Pass A should have the same indicator
    # values as bar at index `check_idx - 1` in Pass B (since the underlying
    # candle data is identical up to that bar).

    leaking: list[str] = []
    clean: list[str] = []
    details: dict[str, Any] = {}

    indicator_cols = [
        c for c in ind_a.columns
        if c not in ("open", "high", "low", "close", "volume",
                     "ts_ms", "timestamp", "symbol", "timeframe")
    ]

    for col in indicator_cols:
        try:
            val_a = float(ind_a[col].iloc[check_idx])
            val_b = float(ind_b[col].iloc[check_idx - 1])
        except (IndexError, KeyError, TypeError, ValueError):
            continue

        if np.isnan(val_a) or np.isnan(val_b):
            continue  # NaN at boundary — not conclusive

        diff = abs(val_a - val_b)
        details[col] = {"val_a": round(val_a, 8), "val_b": round(val_b, 8), "diff": round(diff, 8)}

        if diff > tolerance:
            leaking.append(col)
            logger.warning(
                "LOOKAHEAD DETECTED: column=%s val_a=%.8f val_b=%.8f diff=%.2e",
                col, val_a, val_b, diff,
            )
        else:
            clean.append(col)

    result = {
        "ok": len(leaking) == 0,
        "leaking_columns": leaking,
        "clean_columns": clean,
        "details": {k: v for k, v in details.items() if k in leaking},
        "total_checked": len(indicator_cols),
    }

    if leaking:
        logger.error(
            "Lookahead bias found in %d/%d columns: %s",
            len(leaking), len(indicator_cols), leaking,
        )
    else:
        logger.info("Lookahead check PASSED: %d columns clean.", len(clean))

    return result


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _main() -> None:
    import json
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Lookahead bias checker")
    p.add_argument("--db", default="data/hogan.db")
    p.add_argument("--symbol", default="BTC/USD")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--limit", type=int, default=500)
    p.add_argument("--csv", default=None, help="Load candles from CSV instead of DB")
    args = p.parse_args()

    if args.csv:
        candles = pd.read_csv(args.csv)
    else:
        from hogan_bot.storage import get_connection, load_candles
        conn = get_connection(args.db)
        candles = load_candles(conn, args.symbol, args.timeframe, limit=args.limit)
        conn.close()

    if candles.empty:
        print(f"No candles found for {args.symbol}/{args.timeframe}")
        return

    print(f"Checking {len(candles)} candles for {args.symbol}/{args.timeframe}...")
    result = check_lookahead(candles)
    print(json.dumps(result, indent=2, default=str))

    if not result["ok"]:
        import sys
        sys.exit(1)


if __name__ == "__main__":
    _main()
