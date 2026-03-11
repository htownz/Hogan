#!/usr/bin/env python
"""Out-of-sample validation with correct chronological splits.

IMPORTANT: load_candles(limit=N) returns the most recent N bars, not the first
N bars of a larger dataset. So existing opt_*.json files were optimized on the
latest window. This script:

1. Loads a larger fixed window (e.g. 20,000 bars)
2. Defines explicit chronological splits:
   - Train = oldest chunk (e.g. 0–8000)
   - Validation = next chunk (e.g. 8000–10000)
   - Test = newest untouched chunk (e.g. 10000–20000)
3. Reruns optimization on the train split only
4. Selects best config on validation
5. Reports final metrics on test

Usage::

    python scripts/validate_oos.py --symbol BTC/USD --timeframe 1h --total-bars 20000

Requires: data/hogan.db with sufficient candles. Run backfill first.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hogan_bot.backtest import run_backtest_on_candles
from hogan_bot.config import load_config
from hogan_bot.optimize import _run_trial, optimize
from hogan_bot.storage import get_connection, load_candles


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="OOS validation with chronological train/val/test splits",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="BTC/USD")
    p.add_argument("--timeframe", default="1h")
    p.add_argument("--db", default="data/hogan.db")
    p.add_argument(
        "--total-bars",
        type=int,
        default=20000,
        help="Total bars to load (most recent from DB). Must be >= train + val + test.",
    )
    p.add_argument("--train-bars", type=int, default=8000, help="Train split size (oldest)")
    p.add_argument("--val-bars", type=int, default=2000, help="Validation split size")
    p.add_argument("--test-bars", type=int, default=2000, help="Test split size (newest)")
    p.add_argument(
        "--auto-split",
        action="store_true",
        help="Scale train/val/test to fit available bars when total < train+val+test",
    )
    p.add_argument("--trials", type=int, default=50, help="Optuna trials on train split")
    p.add_argument("--metric", default="sharpe", choices=["sharpe", "sortino", "calmar"])
    p.add_argument("--out", default=None, help="Write report JSON to this path")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()

    conn = get_connection(args.db)
    # load_candles returns most recent N bars, sorted oldest→newest
    all_candles = load_candles(conn, args.symbol, args.timeframe, limit=args.total_bars)
    conn.close()

    if all_candles is None or all_candles.empty:
        print(f"ERROR: No candles for {args.symbol}/{args.timeframe}. Run backfill first.")
        sys.exit(1)

    n = len(all_candles)
    train_sz = args.train_bars
    val_sz = args.val_bars
    test_sz = args.test_bars
    needed = train_sz + val_sz + test_sz

    if n < needed:
        if args.auto_split and n >= 3000:
            # Scale down proportionally: train/val/test ratio preserved
            scale = n / needed
            train_sz = max(2000, int(train_sz * scale))
            val_sz = max(500, int(val_sz * scale))
            test_sz = n - train_sz - val_sz
            if test_sz < 500:
                test_sz = max(500, n // 5)
                val_sz = max(500, n // 5)
                train_sz = n - val_sz - test_sz
            print(f"Auto-scaled splits to fit {n} bars: train={train_sz} val={val_sz} test={test_sz}")
        else:
            print(
                f"ERROR: Need {needed} bars, got {n}. "
                f"Use --auto-split to scale down, or run: python -m hogan_bot.fetch_data "
                f"--symbol {args.symbol} --timeframe {args.timeframe} --limit {args.total_bars}"
            )
            sys.exit(1)

    # Chronological splits (oldest first)
    train_end = train_sz
    val_end = train_sz + val_sz
    test_end = train_sz + val_sz + test_sz

    train_candles = all_candles.iloc[:train_end].copy()
    val_candles = all_candles.iloc[train_end:val_end].copy()
    test_candles = all_candles.iloc[val_end:test_end].copy()

    print(f"Loaded {n} bars. Splits: train={len(train_candles)} val={len(val_candles)} test={len(test_candles)}")

    # 1. Optimize on train only
    metric_map = {"sharpe": "sharpe_ratio", "sortino": "sortino_ratio", "calmar": "calmar_ratio"}
    metric_key = metric_map[args.metric]
    result = optimize(
        train_candles,
        args.symbol,
        cfg,
        timeframe=args.timeframe,
        trials=args.trials,
        metric=metric_key,
        top_k=5,
        verbose=True,
    )

    best_config = result.get("best_config", {})
    if not best_config:
        print("No feasible config found on train split.")
        sys.exit(1)

    # 2. Evaluate best config on validation
    val_row = _run_trial(val_candles, args.symbol, best_config, cfg, timeframe=args.timeframe)
    val_score = val_row.get(metric_key, float("-inf"))
    if "error" in val_row:
        print(f"Validation error: {val_row['error']}")
        sys.exit(1)

    # 3. Final report on test
    test_row = _run_trial(test_candles, args.symbol, best_config, cfg, timeframe=args.timeframe)
    test_score = test_row.get(metric_key)
    if "error" in test_row:
        print(f"Test error: {test_row['error']}")
        sys.exit(1)

    report = {
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "metric": args.metric,
        "train_bars": train_sz,
        "val_bars": val_sz,
        "test_bars": test_sz,
        "train_best_score": result.get("best_score"),
        "val_score": val_score,
        "test_score": test_score,
        "best_config": best_config,
        "test_metrics": {
            k: test_row.get(k)
            for k in (
                "total_return_pct",
                "max_drawdown_pct",
                "sharpe_ratio",
                "sortino_ratio",
                "calmar_ratio",
                "trades",
                "win_rate",
            )
            if k in test_row
        },
    }

    print("\n" + "=" * 60)
    print("OOS VALIDATION REPORT")
    print("=" * 60)
    train_score = report.get("train_best_score")
    print(f"  Train best {args.metric}: {train_score:.4f}" if train_score is not None else f"  Train best {args.metric}: N/A")
    print(f"  Validation {args.metric}: {val_score:.4f}" if val_score is not None else f"  Validation {args.metric}: N/A")
    print(f"  Test {args.metric}:       {test_score:.4f}" if test_score is not None else f"  Test {args.metric}: N/A")
    print(f"  Test return: {report['test_metrics'].get('total_return_pct', 'N/A')}%")
    print(f"  Test trades: {report['test_metrics'].get('trades', 'N/A')}")
    print("=" * 60)

    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
        print(f"\nReport written to {args.out}")


if __name__ == "__main__":
    main()
