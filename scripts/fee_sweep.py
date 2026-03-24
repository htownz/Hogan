#!/usr/bin/env python
"""Fee sensitivity sweep — find the break-even fee rate for top entry families.

Runs the top 3 tournament entries x T1_trend on BTC/USD at multiple fee levels
to determine exactly where the gross edge becomes net-profitable.

Usage:
    python scripts/fee_sweep.py --db data/hogan.db
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hogan_bot.exit_packs import T1_TREND
from scripts.strategy_matrix import (
    passes_promotion_gate,
    passes_screen_gate,
    run_matrix,
)

FEE_LEVELS = [0.0, 0.0002, 0.0005, 0.0008, 0.001, 0.0015, 0.002, 0.0026]
SLIP_LEVELS_BPS = [0.0, 1.0, 2.0, 3.0, 5.0]

TOP_ENTRIES = ["C_ema_pullback", "D_bb_squeeze", "E_baseline"]


def main():
    parser = argparse.ArgumentParser(description="Fee Sensitivity Sweep")
    parser.add_argument("--db", default="data/hogan.db")
    parser.add_argument("--output-dir", default="reports/tournament")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    exit_packs = [T1_TREND]

    rows = []
    print("=" * 90)
    print("FEE SENSITIVITY SWEEP")
    print("=" * 90)
    print(f"  Entries: {TOP_ENTRIES}")
    print("  Exit:    T1_trend")
    print(f"  Fee levels: {FEE_LEVELS}")
    print("  Slippage: fixed at 5 bps (only fee varies)")
    print("=" * 90)

    for fee_rate in FEE_LEVELS:
        label = f"fee={fee_rate:.4f}"
        t0 = time.time()

        zero_cost = (fee_rate == 0.0)
        results = run_matrix(
            db_path=args.db,
            assets=["BTC/USD"],
            entry_keys=TOP_ENTRIES,
            exit_packs=exit_packs,
            n_splits=5,
            zero_cost=zero_cost,
            long_only=False,
            custom_fee=None if zero_cost else fee_rate,
            custom_slip=None if zero_cost else 5.0,
        )

        for r in results:
            row = {
                "fee_rate": fee_rate,
                "fee_pct": f"{fee_rate*100:.2f}%",
                "entry": r.entry,
                "exit_pack": r.exit_pack,
                "mean_net_return_pct": r.mean_net_return_pct,
                "mean_calmar": r.mean_calmar,
                "mean_profit_factor": r.mean_profit_factor,
                "max_drawdown_pct": r.max_drawdown_pct,
                "total_trades": r.total_trades,
                "positive_windows": r.positive_windows,
                "screen_gate": passes_screen_gate(r),
                "promotion_gate": passes_promotion_gate(r),
            }
            rows.append(row)
            status = "PROMO" if row["promotion_gate"] else ("PASS" if row["screen_gate"] else "fail")
            print(f"  {label} | {r.entry:20s} | net={r.mean_net_return_pct:+7.2f}% | "
                  f"calmar={r.mean_calmar:+6.2f} | PF={r.mean_profit_factor:.2f} | "
                  f"DD={r.max_drawdown_pct:5.1f}% | {status}")

        elapsed = time.time() - t0
        print(f"  --- {label} done in {elapsed:.0f}s ---\n")

    # Save CSV
    csv_path = os.path.join(args.output_dir, "fee_sweep.csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved: {csv_path}")

    # Print break-even summary
    print("\n" + "=" * 90)
    print("BREAK-EVEN ANALYSIS")
    print("=" * 90)
    for entry in TOP_ENTRIES:
        entry_rows = [r for r in rows if r["entry"] == entry]
        profitable = [r for r in entry_rows if r["mean_net_return_pct"] > 0]
        if profitable:
            max_fee = max(r["fee_rate"] for r in profitable)
            print(f"  {entry:20s}: profitable up to fee={max_fee:.4f} ({max_fee*100:.2f}% per side)")
        else:
            print(f"  {entry:20s}: NOT profitable at any tested fee level")

        passing = [r for r in entry_rows if r["screen_gate"]]
        if passing:
            max_fee_pass = max(r["fee_rate"] for r in passing)
            print(f"  {'':20s}  screen gate passes up to fee={max_fee_pass:.4f} ({max_fee_pass*100:.2f}%)")

        promoted = [r for r in entry_rows if r["promotion_gate"]]
        if promoted:
            max_fee_promo = max(r["fee_rate"] for r in promoted)
            print(f"  {'':20s}  promotion gate passes up to fee={max_fee_promo:.4f} ({max_fee_promo*100:.2f}%)")
        print()

    print("Current Kraken fee: 0.26% per side (0.0026)")
    print("Kraken Pro maker:   0.16% per side (0.0016)")
    print("Binance VIP1:       0.08% per side (0.0008)")
    print("Binance VIP0 maker: 0.10% per side (0.0010)")
    print()


if __name__ == "__main__":
    main()
