"""Exit reason attribution analysis.

Queries paper_trades and candles to evaluate exit strategy effectiveness:
- Per-reason statistics: count, avg PnL, avg MFE, avg MAE, avg bars_held
- Breakdown by regime (using exit_regime)
- Future-bar holdout: for closed trades, computes what-if PnL 1/2/4 bars after close

Usage:
    python scripts/analyze_exits.py
    python scripts/analyze_exits.py --db path/to/hogan.db
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_trades(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT trade_id, symbol, side, entry_price, exit_price, qty,
               realized_pnl, pnl_pct, close_reason,
               open_ts_ms, close_ts_ms,
               max_adverse_pct, max_favorable_pct, bars_held, exit_regime,
               entry_atr_pct
        FROM paper_trades
        WHERE exit_price IS NOT NULL
        """,
        conn,
    )


def _load_candles(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT symbol, ts_ms, close FROM candles ORDER BY symbol, ts_ms",
        conn,
    )


def _future_bar_returns(trades: pd.DataFrame, candles: pd.DataFrame) -> pd.DataFrame:
    """For each closed trade, compute returns 1/2/4 bars after exit."""
    results = []
    for sym in candles["symbol"].unique():
        sym_candles = candles[candles["symbol"] == sym].sort_values("ts_ms").reset_index(drop=True)
        sym_trades = trades[trades["symbol"] == sym]

        ts_list = sym_candles["ts_ms"].values
        close_list = sym_candles["close"].values

        for _, t in sym_trades.iterrows():
            exit_ts = t["close_ts_ms"]
            exit_px = t["exit_price"]
            side = t["side"]
            if pd.isna(exit_ts) or pd.isna(exit_px) or exit_px <= 0:
                continue

            idx = ts_list.searchsorted(exit_ts, side="right")
            row = {"trade_id": t["trade_id"]}
            for offset in (1, 2, 4):
                if idx + offset < len(close_list):
                    future_px = close_list[idx + offset]
                    if side == "long":
                        row[f"future_{offset}bar_pct"] = (future_px - exit_px) / exit_px
                    else:
                        row[f"future_{offset}bar_pct"] = (exit_px - future_px) / exit_px
                else:
                    row[f"future_{offset}bar_pct"] = None
            results.append(row)

    if not results:
        return pd.DataFrame(columns=["trade_id", "future_1bar_pct", "future_2bar_pct", "future_4bar_pct"])
    return pd.DataFrame(results)


def _fmt(val, decimals=4, pct=False):
    if pd.isna(val):
        return "   n/a"
    if pct:
        return f"{val * 100:+7.2f}%"
    return f"{val:8.{decimals}f}"


def analyze(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    trades = _load_trades(conn)
    if trades.empty:
        print("No closed paper trades found.")
        return

    candles = _load_candles(conn)

    print("=" * 80)
    print("EXIT REASON ATTRIBUTION REPORT")
    print("=" * 80)
    print(f"\nTotal closed trades: {len(trades)}")
    print()

    grouped = trades.groupby("close_reason")
    print(f"{'Reason':<20} {'Count':>6} {'Avg PnL%':>9} {'Avg MFE%':>9} {'Avg MAE%':>9} {'Avg Bars':>9}")
    print("-" * 70)
    for reason, grp in sorted(grouped, key=lambda x: -len(x[1])):
        print(
            f"{reason:<20} {len(grp):>6} "
            f"{_fmt(grp['pnl_pct'].mean(), pct=True):>9} "
            f"{_fmt(grp['max_favorable_pct'].mean(), pct=True):>9} "
            f"{_fmt(grp['max_adverse_pct'].mean(), pct=True):>9} "
            f"{_fmt(grp['bars_held'].mean(), 1):>9}"
        )

    has_regime = trades["exit_regime"].notna().any()
    if has_regime:
        print("\n\n--- By Regime ---\n")
        for regime, rgrp in trades.groupby("exit_regime"):
            if pd.isna(regime):
                continue
            print(f"\n  Regime: {regime} ({len(rgrp)} trades)")
            sub = rgrp.groupby("close_reason")
            print(f"  {'Reason':<20} {'Count':>6} {'Avg PnL%':>9} {'Avg MFE%':>9} {'Avg MAE%':>9}")
            print(f"  {'-'*60}")
            for reason, grp in sorted(sub, key=lambda x: -len(x[1])):
                print(
                    f"  {reason:<20} {len(grp):>6} "
                    f"{_fmt(grp['pnl_pct'].mean(), pct=True):>9} "
                    f"{_fmt(grp['max_favorable_pct'].mean(), pct=True):>9} "
                    f"{_fmt(grp['max_adverse_pct'].mean(), pct=True):>9}"
                )

    if not candles.empty:
        print("\n\n--- Future-Bar Holdout (what if held longer?) ---\n")
        fb = _future_bar_returns(trades, candles)
        if not fb.empty:
            merged = trades.merge(fb, on="trade_id", how="left")
            print(f"{'Reason':<20} {'Count':>6} {'Actual PnL%':>11} {'+1 bar':>9} {'+2 bars':>9} {'+4 bars':>9}")
            print("-" * 75)
            for reason, grp in sorted(merged.groupby("close_reason"), key=lambda x: -len(x[1])):
                print(
                    f"{reason:<20} {len(grp):>6} "
                    f"{_fmt(grp['pnl_pct'].mean(), pct=True):>11} "
                    f"{_fmt(grp['future_1bar_pct'].mean(), pct=True):>9} "
                    f"{_fmt(grp['future_2bar_pct'].mean(), pct=True):>9} "
                    f"{_fmt(grp['future_4bar_pct'].mean(), pct=True):>9}"
                )
            print("\n  (+) means additional gains if held, (-) means saved by exiting")
        else:
            print("  No candle data available for holdout analysis.")

    conn.close()
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze exit effectiveness")
    parser.add_argument("--db", default=str(ROOT / "hogan.db"), help="Path to hogan.db")
    args = parser.parse_args()
    analyze(args.db)
