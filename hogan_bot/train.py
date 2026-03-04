from __future__ import annotations

import argparse
import json
import statistics

from hogan_bot.exchange import KrakenClient
from hogan_bot.ml import (
    train_logistic_regression,
    train_random_forest,
    train_xgboost,
    walk_forward_cv,
)
from hogan_bot.registry import ModelRegistry


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Hogan ML directional model")
    parser.add_argument("--symbol", default="BTC/USD")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--horizon-bars", type=int, default=3)
    parser.add_argument("--model-path", default="models/hogan_logreg.pkl")
    parser.add_argument(
        "--model-type",
        choices=["logreg", "random_forest", "xgboost"],
        default="logreg",
        help="Classifier: logistic regression (default), random forest, or xgboost",
    )
    parser.add_argument(
        "--tune",
        action="store_true",
        help="Run C grid-search hyper-parameter tuning (logreg only)",
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Run walk-forward cross-validation instead of training",
    )
    parser.add_argument("--cv-splits", type=int, default=5)
    parser.add_argument(
        "--registry-path",
        default="models/registry.jsonl",
        help="Path to the model registry JSONL file",
    )
    parser.add_argument(
        "--no-registry",
        action="store_true",
        help="Skip writing to the model registry",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = KrakenClient(api_key=None, api_secret=None)
    candles = client.fetch_ohlcv_df(args.symbol, timeframe=args.timeframe, limit=args.limit)

    if args.cv:
        folds = walk_forward_cv(candles, horizon_bars=args.horizon_bars, n_splits=args.cv_splits)
        output: dict = (
            {
                "cv_folds": folds,
                "mean_accuracy": statistics.mean(f["accuracy"] for f in folds),
                "mean_roc_auc": statistics.mean(f["roc_auc"] for f in folds),
            }
            if folds
            else {"cv_folds": [], "error": "No folds could be evaluated"}
        )
        print(json.dumps(output, indent=2))
        return

    if args.model_type == "random_forest":
        metrics = train_random_forest(
            candles, model_path=args.model_path, horizon_bars=args.horizon_bars
        )
    elif args.model_type == "xgboost":
        metrics = train_xgboost(
            candles, model_path=args.model_path, horizon_bars=args.horizon_bars
        )
    else:
        metrics = train_logistic_regression(
            candles,
            model_path=args.model_path,
            horizon_bars=args.horizon_bars,
            tune_hyperparams=args.tune,
        )

    if not args.no_registry:
        registry = ModelRegistry(registry_path=args.registry_path)
        registry.log(
            metrics,
            model_path=args.model_path,
            symbol=args.symbol,
            timeframe=args.timeframe,
            horizon_bars=args.horizon_bars,
        )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
