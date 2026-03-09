"""Massive retraining orchestration script.

Runs a comprehensive model training pipeline:
  1. Refreshes all data sources
  2. Builds multi-symbol training sets with extended features
  3. Tests multiple labeling horizons
  4. Trains and compares all model types via walk-forward CV
  5. Promotes the best-performing model
  6. Runs a before/after backtest comparison

Usage:
    python scripts/massive_retrain.py
    python scripts/massive_retrain.py --skip-refresh      # skip data fetching
    python scripts/massive_retrain.py --symbols BTC/USD   # single symbol
    python scripts/massive_retrain.py --horizons 12 24    # specific horizons
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hogan_bot.storage import get_connection, load_candles
from hogan_bot.ml import (
    build_training_set,
    walk_forward_cv,
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
    train_lightgbm,
    _FEATURE_COLUMNS,
)
from hogan_bot.config import load_config

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("massive_retrain")

MODEL_TYPES = ["logreg", "random_forest", "xgboost", "lightgbm"]
DEFAULT_HORIZONS = [6, 12, 24, 48]
DEFAULT_SYMBOLS = ["BTC/USD", "ETH/USD", "SOL/USD"]


def refresh_data(db_path: str) -> None:
    """Run all data fetchers to ensure fresh data."""
    import subprocess
    logger.info("=== Refreshing all data sources ===")
    result = subprocess.run(
        [sys.executable, "refresh_daily.py"],
        capture_output=True, text=True, timeout=600,
        cwd=str(Path(__file__).resolve().parent.parent),
    )
    if result.returncode != 0:
        logger.warning("Data refresh had errors (non-fatal):\n%s", result.stderr[-500:] if result.stderr else "")
    else:
        logger.info("Data refresh complete")


def build_multi_symbol_dataset(
    conn, symbols: list[str], timeframe: str, limit: int,
    horizon_bars: int, fee_rate: float,
) -> tuple:
    """Concatenate training sets across multiple symbols."""
    import pandas as pd
    all_X, all_y = [], []
    for sym in symbols:
        candles = load_candles(conn, sym, timeframe, limit=limit)
        if candles.empty:
            logger.warning("No candles for %s/%s — skipping", sym, timeframe)
            continue
        X, y, cols = build_training_set(candles, horizon_bars=horizon_bars, db_conn=conn, fee_rate=fee_rate)
        if X is not None and len(X) > 0:
            all_X.append(X)
            all_y.append(y)
            logger.info("  %s: %d rows (%.0f%% positive)", sym, len(X), 100 * y.mean())
    if not all_X:
        return None, None, _FEATURE_COLUMNS
    return pd.concat(all_X, ignore_index=True), pd.concat(all_y, ignore_index=True), _FEATURE_COLUMNS


def train_and_evaluate(
    model_type: str, candles_concat, model_path: str,
    horizon_bars: int, db_conn=None,
) -> tuple:
    """Train a model using the standard training functions and return (model_result_dict, metrics)."""
    import tempfile
    tmp_path = os.path.join(tempfile.gettempdir(), f"hogan_candidate_{model_type}_{horizon_bars}.pkl")
    t0 = time.perf_counter()
    try:
        if model_type == "logreg":
            result = train_logistic_regression(
                candles_concat, tmp_path, horizon_bars=horizon_bars,
                db_conn=db_conn, prune_features=True, max_features=40,
            )
        elif model_type == "random_forest":
            result = train_random_forest(
                candles_concat, tmp_path, horizon_bars=horizon_bars,
                db_conn=db_conn, prune_features=True, max_features=40,
            )
        elif model_type == "xgboost":
            result = train_xgboost(
                candles_concat, tmp_path, horizon_bars=horizon_bars,
                db_conn=db_conn, prune_features=True, max_features=40,
            )
        elif model_type == "lightgbm":
            result = train_lightgbm(
                candles_concat, tmp_path, horizon_bars=horizon_bars,
                db_conn=db_conn,
            )
        else:
            return None, {}, ""
    except Exception as exc:
        logger.warning("  %s training failed: %s", model_type, exc)
        return None, {}, ""

    elapsed = time.perf_counter() - t0
    metrics = {k: v for k, v in result.items() if k != "model_path"}
    metrics["train_seconds"] = round(elapsed, 1)

    return result, metrics, tmp_path


def run_walk_forward(candles_concat, model_type: str, horizon_bars: int,
                     n_splits: int = 5, db_conn=None, fee_rate: float = 0.0026) -> list[dict]:
    """Run purged walk-forward CV for a given model type."""
    try:
        results = walk_forward_cv(
            candles_concat,
            horizon_bars=horizon_bars,
            n_splits=n_splits,
            model_type=model_type,
            db_conn=db_conn,
            fee_rate=fee_rate,
        )
        return results
    except Exception as exc:
        logger.warning("  Walk-forward CV failed for %s: %s", model_type, exc)
        return []


def run_backtest_comparison(conn, model_path: str, symbol: str = "BTC/USD") -> dict:
    """Run a quick backtest to compare before/after."""
    try:
        from hogan_bot.backtest import run_backtest_on_candles
        from hogan_bot.ml import load_model
        candles = load_candles(conn, symbol, "5m", limit=10000)
        if candles.empty:
            return {}
        config = load_config()
        model = load_model(model_path) if os.path.exists(model_path) else None
        result = run_backtest_on_candles(
            candles,
            starting_balance_usd=config.starting_balance_usd,
            fee_rate=config.fee_rate,
            short_ma_window=config.short_ma_window,
            long_ma_window=config.long_ma_window,
            volume_threshold=config.volume_threshold,
            trailing_stop_pct=config.trailing_stop_pct,
            take_profit_pct=config.take_profit_pct,
            ml_model=model,
            ml_buy_threshold=config.ml_buy_threshold,
            ml_sell_threshold=config.ml_sell_threshold,
        )
        return {
            "total_return_pct": round(result.total_return_pct, 2),
            "max_drawdown_pct": round(result.max_drawdown_pct, 2),
            "trades": result.trades,
            "win_rate": round(result.win_rate, 2) if result.win_rate else 0,
            "sharpe": round(result.sharpe_ratio, 3) if result.sharpe_ratio else 0,
        }
    except Exception as exc:
        logger.warning("Backtest failed: %s", exc)
        return {}


def main() -> None:
    parser = argparse.ArgumentParser(description="Massive model retraining pipeline")
    parser.add_argument("--skip-refresh", action="store_true", help="Skip data refresh step")
    parser.add_argument("--symbols", nargs="+", default=DEFAULT_SYMBOLS)
    parser.add_argument("--horizons", nargs="+", type=int, default=DEFAULT_HORIZONS)
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--limit", type=int, default=100000)
    parser.add_argument("--db", default="data/hogan.db")
    parser.add_argument("--fee-rate", type=float, default=0.0026)
    parser.add_argument("--model-path", default="models/hogan_logreg.pkl")
    parser.add_argument("--cv-splits", type=int, default=5)
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("MASSIVE RETRAINING PIPELINE")
    logger.info("Symbols: %s | TF: %s | Horizons: %s", args.symbols, args.timeframe, args.horizons)
    logger.info("=" * 70)

    # Step 1: Data refresh
    if not args.skip_refresh:
        refresh_data(args.db)

    conn = get_connection(args.db)

    # Step 2: Backtest BEFORE (baseline)
    logger.info("\n=== BASELINE BACKTEST (current model) ===")
    bt_before = run_backtest_comparison(conn, args.model_path)
    if bt_before:
        logger.info("  Before: return=%.2f%% dd=%.2f%% trades=%d win=%.0f%% sharpe=%.3f",
                     bt_before.get("total_return_pct", 0), bt_before.get("max_drawdown_pct", 0),
                     bt_before.get("trades", 0), bt_before.get("win_rate", 0) * 100,
                     bt_before.get("sharpe", 0))

    # Step 3: Build combined candle dataset for multi-symbol training
    import pandas as pd
    candles_all = []
    for sym in args.symbols:
        c = load_candles(conn, sym, args.timeframe, limit=args.limit)
        if not c.empty:
            candles_all.append(c)
            logger.info("  Loaded %s: %d bars", sym, len(c))
    if not candles_all:
        logger.error("No candle data available — aborting")
        return
    candles_concat = pd.concat(candles_all, ignore_index=True)
    logger.info("  Total combined candles: %d", len(candles_concat))

    # Step 4: Train across all horizons and model types
    best_overall = {"roc_auc": 0.0, "model_path": "", "model_type": "", "horizon": 0, "metrics": {}}
    all_results = []

    for horizon in args.horizons:
        logger.info("\n=== HORIZON: %d bars ===", horizon)

        # Quick dataset size check
        X, y, cols = build_multi_symbol_dataset(
            conn, args.symbols, args.timeframe, args.limit,
            horizon, args.fee_rate,
        )
        if X is None:
            logger.warning("  No training data for horizon=%d — skipping", horizon)
            continue
        logger.info("  Dataset: %d rows x %d features (%.0f%% positive)",
                     len(X), len(cols), 100 * y.mean())

        # Walk-forward CV for each model type
        for mtype in MODEL_TYPES:
            logger.info("  Training %s (horizon=%d) ...", mtype, horizon)

            cv_results = run_walk_forward(candles_concat, mtype, horizon, args.cv_splits, db_conn=conn, fee_rate=args.fee_rate)
            if cv_results:
                avg_auc = sum(r.get("roc_auc", 0.5) for r in cv_results) / len(cv_results)
                avg_acc = sum(r.get("accuracy", 0.5) for r in cv_results) / len(cv_results)
                logger.info("    CV: avg_auc=%.4f avg_acc=%.4f (%d folds)", avg_auc, avg_acc, len(cv_results))
            else:
                avg_auc = 0.5

            # Full train + hold-out evaluation
            result, metrics, tmp_path = train_and_evaluate(
                mtype, candles_concat, args.model_path,
                horizon, db_conn=conn,
            )
            if metrics:
                roc = metrics.get("roc_auc", 0.5)
                logger.info("    Hold-out: auc=%.4f acc=%.4f prec=%.4f rec=%.4f f1=%.4f (%.1fs)",
                             roc, metrics.get("accuracy", 0), metrics.get("precision", 0),
                             metrics.get("recall", 0), metrics.get("f1", 0),
                             metrics.get("train_seconds", 0))

                result_entry = {
                    "model_type": mtype, "horizon": horizon,
                    "cv_avg_auc": round(avg_auc, 4), **metrics,
                }
                all_results.append(result_entry)

                if roc > best_overall["roc_auc"]:
                    best_overall = {
                        "roc_auc": roc, "model_path": tmp_path,
                        "model_type": mtype, "horizon": horizon,
                        "metrics": metrics,
                    }

    # Step 5: Promote best model
    if best_overall["model_path"] and os.path.exists(best_overall["model_path"]):
        logger.info("\n=== BEST MODEL ===")
        logger.info("  Type: %s | Horizon: %d | ROC AUC: %.4f",
                     best_overall["model_type"], best_overall["horizon"],
                     best_overall["roc_auc"])
        logger.info("  Metrics: %s", json.dumps(best_overall["metrics"], indent=2))

        import shutil
        model_path = args.model_path
        shutil.copy2(best_overall["model_path"], model_path)
        logger.info("  Promoted to: %s", model_path)

        # Update registry
        try:
            import hashlib
            with open(model_path, "rb") as f:
                model_hash = hashlib.md5(f.read()).hexdigest()[:16]
            registry_entry = {
                "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
                "model_path": model_path,
                "model_hash": model_hash,
                "model_type": best_overall["model_type"],
                "symbol": ",".join(args.symbols),
                "timeframe": args.timeframe,
                "horizon_bars": best_overall["horizon"],
                "features": len(_FEATURE_COLUMNS),
                "metrics": best_overall["metrics"],
            }
            with open("models/registry.jsonl", "a") as f:
                f.write(json.dumps(registry_entry) + "\n")
            logger.info("  Registry updated")
        except Exception as exc:
            logger.warning("  Registry update failed: %s", exc)
    else:
        logger.warning("No model trained successfully")

    # Step 5: Backtest AFTER
    logger.info("\n=== POST-TRAINING BACKTEST ===")
    bt_after = run_backtest_comparison(conn, args.model_path)
    if bt_after:
        logger.info("  After:  return=%.2f%% dd=%.2f%% trades=%d win=%.0f%% sharpe=%.3f",
                     bt_after.get("total_return_pct", 0), bt_after.get("max_drawdown_pct", 0),
                     bt_after.get("trades", 0), bt_after.get("win_rate", 0) * 100,
                     bt_after.get("sharpe", 0))

    # Step 6: Summary
    logger.info("\n" + "=" * 70)
    logger.info("TRAINING SUMMARY")
    logger.info("=" * 70)
    if all_results:
        logger.info("%-15s %-8s %-10s %-10s %-10s %-10s", "Model", "Horizon", "ROC AUC", "Accuracy", "F1", "CV AUC")
        for r in sorted(all_results, key=lambda x: x.get("roc_auc", 0), reverse=True):
            logger.info("%-15s %-8d %-10.4f %-10.4f %-10.4f %-10.4f",
                         r["model_type"], r["horizon"],
                         r.get("roc_auc", 0), r.get("accuracy", 0),
                         r.get("f1", 0), r.get("cv_avg_auc", 0))
    if bt_before and bt_after:
        delta_ret = bt_after.get("total_return_pct", 0) - bt_before.get("total_return_pct", 0)
        logger.info("\nBacktest improvement: %.2f%% return delta", delta_ret)
    logger.info("=" * 70)


if __name__ == "__main__":
    main()
