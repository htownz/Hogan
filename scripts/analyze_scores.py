"""Per-regime post-trade score diagnostics.

Joins decision_log with paper_trades to analyze direction_score,
quality_score, and size_score for winners vs losers, by regime.
Flags actionable patterns.

Usage:
    python scripts/analyze_scores.py
    python scripts/analyze_scores.py --db path/to/hogan.db
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _load_data(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query(
        """
        SELECT
            d.id AS decision_id,
            d.regime,
            d.final_action,
            d.direction_score,
            d.quality_score,
            d.size_score,
            d.quality_components_json,
            d.block_reasons_json,
            t.realized_pnl,
            t.pnl_pct,
            t.close_reason,
            t.max_favorable_pct,
            t.max_adverse_pct,
            t.bars_held,
            t.exit_regime
        FROM decision_log d
        INNER JOIN paper_trades t ON d.linked_trade_id = t.trade_id
        WHERE t.exit_price IS NOT NULL
          AND d.final_action != 'hold'
        """,
        conn,
    )


def _fmt(val, pct=False):
    if pd.isna(val):
        return "   n/a"
    if pct:
        return f"{val * 100:+7.2f}%"
    return f"{val:7.3f}"


def analyze(db_path: str) -> None:
    conn = sqlite3.connect(db_path)
    df = _load_data(conn)
    conn.close()

    if df.empty:
        print("No linked decision→trade data found.")
        return

    df["winner"] = df["pnl_pct"] > 0

    print("=" * 80)
    print("PER-REGIME SCORE DIAGNOSTICS")
    print("=" * 80)
    print(f"\nTotal linked trades: {len(df)}")
    print(f"Winners: {df['winner'].sum()} | Losers: {(~df['winner']).sum()}")

    print("\n\n--- Overall: Winners vs Losers ---\n")
    for label, grp in df.groupby("winner"):
        tag = "WINNERS" if label else "LOSERS"
        print(
            f"  {tag:<10} n={len(grp):>4}  "
            f"dir={_fmt(grp['direction_score'].mean())}  "
            f"qual={_fmt(grp['quality_score'].mean())}  "
            f"size={_fmt(grp['size_score'].mean())}  "
            f"avg_pnl={_fmt(grp['pnl_pct'].mean(), pct=True)}"
        )

    regimes = [r for r in df["regime"].unique() if pd.notna(r)]
    if regimes:
        print("\n\n--- By Regime ---\n")
        for regime in sorted(regimes):
            rgrp = df[df["regime"] == regime]
            print(f"\n  [{regime}] ({len(rgrp)} trades)")
            print(f"  {'':>10} {'Count':>6} {'Dir':>8} {'Qual':>8} {'Size':>8} {'Avg PnL%':>10}")
            print(f"  {'-'*55}")
            for label, sgrp in rgrp.groupby("winner"):
                tag = "WINNERS" if label else "LOSERS"
                print(
                    f"  {tag:<10} {len(sgrp):>6} "
                    f"{_fmt(sgrp['direction_score'].mean()):>8} "
                    f"{_fmt(sgrp['quality_score'].mean()):>8} "
                    f"{_fmt(sgrp['size_score'].mean()):>8} "
                    f"{_fmt(sgrp['pnl_pct'].mean(), pct=True):>10}"
                )

    print("\n\n--- Actionable Patterns ---\n")
    findings = []

    high_dir_low_qual_losers = df[(df["direction_score"] > 0.3) & (df["quality_score"] < 0.3) & (~df["winner"])]
    if len(high_dir_low_qual_losers) > 2:
        findings.append(
            f"  [!] {len(high_dir_low_qual_losers)} trades had high direction (>0.3) + low quality (<0.3) → lost money. "
            f"Quality gate may need tightening."
        )

    high_qual_low_size_winners = df[(df["quality_score"] > 0.5) & (df["size_score"] < 0.3) & df["winner"]]
    if len(high_qual_low_size_winners) > 2:
        findings.append(
            f"  [!] {len(high_qual_low_size_winners)} trades had high quality (>0.5) + low size (<0.3) → won but under-sized. "
            f"Position sizing may be too conservative."
        )

    for regime in regimes:
        rgrp = df[df["regime"] == regime]
        rwin = rgrp["winner"].mean() if len(rgrp) > 3 else None
        if rwin is not None and rwin < 0.35:
            findings.append(
                f"  [!] Regime '{regime}' has {rwin*100:.0f}% win rate ({len(rgrp)} trades). "
                f"Consider stronger filtering in this regime."
            )

    big_losers_volatile = df[(df["regime"] == "volatile") & (df["pnl_pct"] < -0.02) & (df["size_score"] > 0.5)]
    if len(big_losers_volatile) > 1:
        findings.append(
            f"  [!] {len(big_losers_volatile)} big losers in volatile regime with high size_score. "
            f"Volatile regime sizing may need dampening."
        )

    if findings:
        for f in findings:
            print(f)
    else:
        print("  No obvious patterns detected. (May need more trade data.)")

    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Per-regime score diagnostics")
    parser.add_argument("--db", default=str(ROOT / "hogan.db"), help="Path to hogan.db")
    args = parser.parse_args()
    analyze(args.db)
