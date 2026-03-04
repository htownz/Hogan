"""Walk-forward retraining for Hogan ML models.

Rolling-window walk-forward strategy
-------------------------------------
Each retrain cycle uses the most recent ``--window-bars`` bars.  Because the
window rolls forward with time, the model continuously adapts to the current
market regime rather than being dominated by stale historical patterns.

Promotion gate
--------------
A newly trained candidate model replaces the production model **only** when it
improves the ``--promotion-metric`` by at least ``--min-improvement`` over the
current registry best.  Every run — promoted or not — is logged in the model
registry so the complete history is always inspectable.

Safety
------
Training writes to a timestamped *candidate* path first.  The production path
is only overwritten on successful promotion, so a failed or regressing run
never corrupts the live model.

Usage examples
--------------
One-shot retrain (fetches live data from the configured exchange)::

    python -m hogan_bot.retrain

Load candles from local SQLite DB (faster, offline, good for cron)::

    python -m hogan_bot.retrain --from-db

Retrain every 24 hours (blocking loop)::

    python -m hogan_bot.retrain --schedule 24

Retrain every 6 hours on Binance BTC/USDT with XGBoost::

    python -m hogan_bot.retrain --exchange binance --symbol BTC/USDT \\
        --model-type xgboost --schedule 6

Evaluate without touching anything::

    python -m hogan_bot.retrain --dry-run
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from hogan_bot.exchange import ExchangeClient
from hogan_bot.ml import (
    train_lightgbm,
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
)
from hogan_bot.registry import ModelRegistry
from hogan_bot.storage import get_connection, load_candles

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Argument parser
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Walk-forward retraining for Hogan ML models",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="BTC/USD", help="Trading pair to train on")
    p.add_argument("--timeframe", default="5m", help="Bar interval")
    p.add_argument(
        "--exchange",
        default="kraken",
        help="CCXT exchange ID for live data fetches (ignored when --from-db is set)",
    )
    p.add_argument(
        "--window-bars",
        type=int,
        default=5000,
        metavar="N",
        help="Rolling window size: train on the N most recent bars",
    )
    p.add_argument("--horizon-bars", type=int, default=3, help="Prediction horizon in bars")
    p.add_argument(
        "--model-type",
        choices=["logreg", "random_forest", "xgboost", "lightgbm"],
        default="logreg",
        help="Model family to train",
    )
    p.add_argument("--model-path", default="models/hogan_logreg.pkl", help="Production model path")
    p.add_argument(
        "--tune",
        action="store_true",
        help="Run C hyper-parameter search (logreg only)",
    )
    p.add_argument(
        "--promotion-metric",
        default="roc_auc",
        choices=["roc_auc", "accuracy", "f1", "precision", "recall"],
        help="Metric used to decide whether to promote the new model",
    )
    p.add_argument(
        "--min-improvement",
        type=float,
        default=0.005,
        metavar="DELTA",
        help="Minimum metric gain over registry best required for promotion",
    )
    p.add_argument(
        "--registry-path",
        default="models/registry.jsonl",
        help="Path to the JSONL model registry",
    )
    p.add_argument(
        "--from-db",
        action="store_true",
        help="Load candles from local SQLite DB instead of fetching live data",
    )
    p.add_argument(
        "--db",
        default="data/hogan.db",
        help="SQLite DB path (used when --from-db is set)",
    )
    p.add_argument(
        "--schedule",
        type=float,
        default=None,
        metavar="HOURS",
        help="Run a retrain every HOURS hours (blocking loop). Omit for one-shot.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Train and evaluate but do NOT overwrite the production model or update the registry",
    )
    return p


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _fetch_candles(args: argparse.Namespace) -> pd.DataFrame:
    """Return a DataFrame of candles, either from DB or live exchange."""
    if args.from_db:
        conn = get_connection(args.db)
        df = load_candles(conn, args.symbol, args.timeframe, limit=args.window_bars)
        conn.close()
        if df.empty:
            raise RuntimeError(
                f"No candles found in the local DB for {args.symbol}/{args.timeframe}. "
                "Run `python -m hogan_bot.fetch_data` first to populate the database."
            )
        logger.info("Loaded %d bars from DB (%s / %s)", len(df), args.symbol, args.timeframe)
        return df

    client = ExchangeClient(args.exchange)
    df = client.fetch_ohlcv_df(args.symbol, timeframe=args.timeframe, limit=args.window_bars)
    logger.info(
        "Fetched %d bars from %s (%s / %s)", len(df), args.exchange, args.symbol, args.timeframe
    )
    return df


def _train_to_candidate(args: argparse.Namespace, candles: pd.DataFrame) -> tuple[dict, str]:
    """Train a model and write it to a timestamped candidate path.

    Returns ``(metrics_dict, candidate_path)``.  Writing to a separate
    candidate path ensures the production model is never touched unless
    promotion is explicitly decided.
    """
    ts = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    model_dir = Path(args.model_path).parent
    model_stem = Path(args.model_path).stem
    candidate_path = str(model_dir / f"{model_stem}_candidate_{ts}.pkl")

    model_dir.mkdir(parents=True, exist_ok=True)

    if args.model_type == "random_forest":
        metrics = train_random_forest(
            candles, model_path=candidate_path, horizon_bars=args.horizon_bars
        )
    elif args.model_type == "xgboost":
        metrics = train_xgboost(
            candles, model_path=candidate_path, horizon_bars=args.horizon_bars
        )
    elif args.model_type == "lightgbm":
        metrics = train_lightgbm(
            candles, model_path=candidate_path, horizon_bars=args.horizon_bars
        )
    else:
        metrics = train_logistic_regression(
            candles,
            model_path=candidate_path,
            horizon_bars=args.horizon_bars,
            tune_hyperparams=args.tune,
        )

    return metrics, candidate_path


def _get_current_best_score(registry: ModelRegistry, metric: str) -> float | None:
    """Return the current best value for *metric* in *registry*, or ``None``."""
    best = registry.best(metric)
    if best is None:
        return None
    return best.get("metrics", {}).get(metric)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def retrain_once(args: argparse.Namespace) -> dict:
    """Execute one walk-forward retrain cycle.

    Steps
    -----
    1. Fetch the ``window_bars`` most recent candles (live or from DB).
    2. Train a candidate model to a temporary path.
    3. Compare candidate score against the registry best.
    4. Promote (copy to production path + log to registry) when the score
       improves by at least ``min_improvement``.
    5. Clean up the temporary candidate file.

    Returns a JSON-serialisable dict summarising the run.
    """
    start_ts = datetime.now(tz=timezone.utc).isoformat()
    logger.info(
        "Retrain start — symbol=%s timeframe=%s window=%d model=%s dry_run=%s",
        args.symbol,
        args.timeframe,
        args.window_bars,
        args.model_type,
        args.dry_run,
    )

    # 1. Fetch candles
    candles = _fetch_candles(args)

    # 2. Train candidate
    metrics, candidate_path = _train_to_candidate(args, candles)
    new_score: float = float(metrics.get(args.promotion_metric, 0.0))
    logger.info(
        "Candidate trained — %s=%.4f  path=%s",
        args.promotion_metric,
        new_score,
        candidate_path,
    )

    # 3. Determine whether to promote
    registry = ModelRegistry(registry_path=args.registry_path)
    current_score = _get_current_best_score(registry, args.promotion_metric)
    threshold = (current_score or 0.0) + args.min_improvement
    should_promote = current_score is None or new_score >= threshold

    # Build result dict (feature_importances omitted — too large for JSON log)
    result: dict = {
        "timestamp": start_ts,
        "symbol": args.symbol,
        "timeframe": args.timeframe,
        "exchange": args.exchange,
        "window_bars": args.window_bars,
        "horizon_bars": args.horizon_bars,
        "model_type": args.model_type,
        "promotion_metric": args.promotion_metric,
        "min_improvement": args.min_improvement,
        "new_score": new_score,
        "current_score": float(current_score) if current_score is not None else None,
        "promoted": False,
        "dry_run": args.dry_run,
    }
    # Surface scalar metrics from the training run (accuracy, f1, …)
    for k, v in metrics.items():
        if k not in ("feature_importances", "model_type") and isinstance(v, (int, float)):
            result.setdefault(k, float(v))

    # 4. Act on the decision
    try:
        if args.dry_run:
            result["message"] = (
                f"Dry run — {args.promotion_metric}={new_score:.4f}"
                + (f" vs current {current_score:.4f}" if current_score is not None else " (no current model)")
            )
            logger.info("Dry run complete. %s", result["message"])

        elif should_promote:
            shutil.copy2(candidate_path, args.model_path)
            registry.log(
                metrics,
                model_path=args.model_path,
                symbol=args.symbol,
                timeframe=args.timeframe,
                horizon_bars=args.horizon_bars,
            )
            result["promoted"] = True
            improvement = new_score - (current_score or 0.0)
            if current_score is None:
                result["message"] = "Promoted — first model in registry"
            else:
                result["message"] = (
                    f"Promoted — {args.promotion_metric} {new_score:.4f} "
                    f"(+{improvement:.4f} over {current_score:.4f})"
                )
            logger.info("Model promoted → %s", args.model_path)

        else:
            improvement = new_score - (current_score or 0.0)
            result["message"] = (
                f"Not promoted — {args.promotion_metric} {new_score:.4f} "
                f"vs {current_score:.4f} (need +{args.min_improvement:.4f})"
            )
            logger.info("Model NOT promoted. %s", result["message"])

    finally:
        # 5. Always clean up the candidate file
        candidate = Path(candidate_path)
        if candidate.exists():
            candidate.unlink()

    return result


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _build_parser().parse_args(argv)

    if args.schedule is not None:
        interval_secs = args.schedule * 3600.0
        logger.info(
            "Scheduled retraining every %.1f hours. Press Ctrl-C to stop.", args.schedule
        )
        while True:
            try:
                result = retrain_once(args)
                print(json.dumps(result, indent=2), flush=True)
            except KeyboardInterrupt:
                logger.info("Retrain scheduler stopped by user.")
                break
            except Exception as exc:  # noqa: BLE001
                logger.error("Retrain cycle failed: %s", exc)
            logger.info("Sleeping %.0f s until next retrain…", interval_secs)
            try:
                time.sleep(interval_secs)
            except KeyboardInterrupt:
                logger.info("Retrain scheduler stopped by user.")
                break
    else:
        result = retrain_once(args)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
