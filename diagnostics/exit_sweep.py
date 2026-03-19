"""Sweep trailing stop and take profit parameters across walk-forward windows.

Runs the walk-forward validation with different TS/TP combinations to find
the optimal exit parameters.  Uses no-ML + macro-sitout as the baseline.
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

TRAILING_STOP_VALUES = [0.015, 0.020, 0.025, 0.030, 0.040]
TAKE_PROFIT_VALUES = [0.030, 0.040, 0.054, 0.070, 0.090]


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
    total = len(TRAILING_STOP_VALUES) * len(TAKE_PROFIT_VALUES)
    idx = 0

    for ts_pct in TRAILING_STOP_VALUES:
        for tp_pct in TAKE_PROFIT_VALUES:
            idx += 1
            t0 = time.perf_counter()
            logger.info(
                "[%d/%d] TS=%.1f%% TP=%.1f%% ...",
                idx, total, ts_pct * 100, tp_pct * 100,
            )
            sys.stdout.flush()

            import os
            os.environ["HOGAN_TRAILING_STOP_PCT"] = str(ts_pct)
            os.environ["HOGAN_TAKE_PROFIT_PCT"] = str(tp_pct)

            cfg = WFConfig(
                n_splits=5,
                use_ml_filter=False,
                use_macro_sitout=True,
            )

            report = walk_forward_validate(df, cfg, macro_sitout=sitout)
            s = report.summary()
            elapsed = time.perf_counter() - t0

            row = {
                "trailing_stop_pct": ts_pct,
                "take_profit_pct": tp_pct,
                "mean_return_pct": s["mean_return_pct"],
                "mean_sharpe": s["mean_sharpe"],
                "total_trades": s["total_trades"],
                "n_positive": s["n_positive"],
                "worst_drawdown_pct": s["worst_drawdown_pct"],
                "passes_gate": s["passes_gate"],
                "per_window": [
                    {
                        "idx": w.window_idx,
                        "ret": round(w.total_return_pct, 4),
                        "sharpe": round(w.sharpe, 4) if w.sharpe else None,
                        "trades": w.trades,
                        "win_rate": round(w.win_rate, 4),
                    }
                    for w in report.windows
                ],
            }
            results.append(row)
            logger.info(
                "  -> ret=%+.2f%%  sharpe=%.2f  trades=%d  pos=%d/5  (%.0fs)",
                s["mean_return_pct"], s["mean_sharpe"],
                s["total_trades"], s["n_positive"], elapsed,
            )
            sys.stdout.flush()

    results.sort(key=lambda r: r["mean_return_pct"], reverse=True)

    print(f"\n{'=' * 80}")
    print("EXIT PARAMETER SWEEP RESULTS")
    print(f"{'=' * 80}")
    print(f"{'TS%':>6} {'TP%':>6} {'Return':>8} {'Sharpe':>8} {'Trades':>7} {'Win':>5} {'DD':>6} {'Gate':>5}")
    print(f"{'-' * 80}")
    for r in results:
        print(
            f"{r['trailing_stop_pct']*100:>5.1f}% "
            f"{r['take_profit_pct']*100:>5.1f}% "
            f"{r['mean_return_pct']:>+7.2f}% "
            f"{r['mean_sharpe']:>7.2f} "
            f"{r['total_trades']:>7} "
            f"{r['n_positive']:>3}/5 "
            f"{r['worst_drawdown_pct']:>5.1f}% "
            f"{'PASS' if r['passes_gate'] else 'FAIL':>5}"
        )
    print(f"{'=' * 80}\n")

    out_path = Path("diagnostics/exit_sweep_results.json")
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    logger.info("Results saved to %s", out_path)


if __name__ == "__main__":
    main()
