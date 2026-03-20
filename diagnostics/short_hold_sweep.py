"""Sweep short max hold hours to find the optimal value.

Tests different short_max_hold_hours values with ML sizer + macro sitout.
"""
import sqlite3
import sys
import time
import logging
import json
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hogan_bot.walk_forward import WFConfig, walk_forward_validate
from hogan_bot.macro_sitout import MacroSitout

logging.basicConfig(level=logging.WARNING, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

DB_PATH = "data/hogan.db"

SHORT_HOLD_VALUES = [4.0, 6.0, 8.0, 10.0, 12.0, 16.0, 24.0]


def main():
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(
        "SELECT ts_ms, open, high, low, close, volume FROM candles "
        "WHERE symbol = 'BTC/USD' AND timeframe = '1h' ORDER BY ts_ms",
        conn,
    )
    df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    logger.info("Loaded %d candles", len(df))

    sitout = MacroSitout.from_db(conn)
    conn.close()

    results = []

    for hold_h in SHORT_HOLD_VALUES:
        t0 = time.perf_counter()
        logger.info("Short max hold = %.0fh ...", hold_h)
        sys.stdout.flush()

        try:
            cfg = WFConfig(
                n_splits=5,
                use_ml_filter=False,
                use_ml_as_sizer=True,
                use_macro_sitout=True,
                short_max_hold_hours=hold_h,
            )

            report = walk_forward_validate(df, cfg, macro_sitout=sitout)
            s = report.summary()
        except Exception as exc:
            logger.error("  hold_h=%.0f FAILED: %s", hold_h, exc)
            continue
        elapsed = time.perf_counter() - t0

        per_window = []
        total_max_hold_exits = 0
        total_stop_exits = 0
        total_tp_exits = 0
        for w in report.windows:
            f = w.signal_funnel or {}
            mh = f.get("short_covered_max_hold", 0)
            st = f.get("short_covered_stop", 0)
            tp = f.get("short_covered_tp", 0)
            total_max_hold_exits += mh
            total_stop_exits += st
            total_tp_exits += tp
            per_window.append({
                "idx": w.window_idx,
                "ret": round(w.total_return_pct, 4),
                "sharpe": round(w.sharpe, 4) if w.sharpe else None,
                "trades": w.trades,
                "win_rate": round(w.win_rate, 4),
                "short_max_hold_exits": mh,
                "short_stop_exits": st,
            })

        row = {
            "short_max_hold_hours": hold_h,
            "mean_return_pct": s["mean_return_pct"],
            "mean_sharpe": s["mean_sharpe"],
            "total_trades": s["total_trades"],
            "n_positive": s["n_positive"],
            "worst_drawdown_pct": s["worst_drawdown_pct"],
            "short_max_hold_exits": total_max_hold_exits,
            "short_stop_exits": total_stop_exits,
            "per_window": per_window,
        }
        results.append(row)
        logger.info(
            "  -> ret=%+.2f%%  sharpe=%.2f  trades=%d  pos=%d/5  mh_exits=%d  (%.0fs)",
            s["mean_return_pct"], s["mean_sharpe"],
            s["total_trades"], s["n_positive"],
            total_max_hold_exits, elapsed,
        )
        sys.stdout.flush()

    print(f"\n{'=' * 90}")
    print("SHORT MAX HOLD SWEEP RESULTS")
    print(f"{'=' * 90}")
    print(f"{'Hold':>6} {'Return':>8} {'Sharpe':>8} {'Trades':>7} {'Win':>5} {'MH_Exit':>8} {'Stop':>6} {'DD':>6}")
    print(f"{'-' * 90}")
    for r in sorted(results, key=lambda x: x["mean_return_pct"], reverse=True):
        print(
            f"{r['short_max_hold_hours']:>5.0f}h "
            f"{r['mean_return_pct']:>+7.2f}% "
            f"{r['mean_sharpe']:>7.2f} "
            f"{r['total_trades']:>7} "
            f"{r['n_positive']:>3}/5 "
            f"{r['short_max_hold_exits']:>7} "
            f"{r['short_stop_exits']:>5} "
            f"{r['worst_drawdown_pct']:>5.1f}%"
        )

    print(f"\nPer-window detail for top 3:")
    top3 = sorted(results, key=lambda x: x["mean_return_pct"], reverse=True)[:3]
    for r in top3:
        print(f"\n  Hold={r['short_max_hold_hours']:.0f}h (ret={r['mean_return_pct']:+.2f}%):")
        for w in r["per_window"]:
            print(f"    W{w['idx']}: ret={w['ret']:+.4f}%  sharpe={w['sharpe']}  trades={w['trades']}  win={w['win_rate']:.0%}  mh={w['short_max_hold_exits']}  stop={w['short_stop_exits']}")

    print(f"{'=' * 90}")

    out_path = Path("diagnostics/short_hold_sweep.json")
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
