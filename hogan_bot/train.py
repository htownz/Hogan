from __future__ import annotations

import argparse
import json
import statistics

from hogan_bot.exchange import KrakenClient
from hogan_bot.ml import train_logistic_regression, train_random_forest, walk_forward_cv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Hogan ML directional model")
    parser.add_argument("--symbol", default="BTC/USD")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--horizon-bars", type=int, default=3)
    parser.add_argument("--model-path", default="models/hogan_logreg.pkl")
    parser.add_argument(
        "--model-type",
        choices=["logreg", "random_forest"],
        default="logreg",
        help="Classifier type: logistic regression (default) or random forest",
    )
    parser.add_argument(
        "--cv",
        action="store_true",
        help="Run walk-forward cross-validation and print fold metrics instead of training",
    )
    parser.add_argument(
        "--cv-splits",
        type=int,
        default=5,
        help="Number of folds for walk-forward CV (default 5)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = KrakenClient(api_key=None, api_secret=None)
    candles = client.fetch_ohlcv_df(args.symbol, timeframe=args.timeframe, limit=args.limit)

    if args.cv:
        folds = walk_forward_cv(candles, horizon_bars=args.horizon_bars, n_splits=args.cv_splits)
        if folds:
            mean_acc = statistics.mean(f["accuracy"] for f in folds)
            mean_auc = statistics.mean(f["roc_auc"] for f in folds)
            output = {
                "cv_folds": folds,
                "mean_accuracy": mean_acc,
                "mean_roc_auc": mean_auc,
            }
        else:
            output = {"cv_folds": [], "error": "No folds could be evaluated"}
        print(json.dumps(output, indent=2))
        return

    if args.model_type == "random_forest":
        metrics = train_random_forest(
            candles, model_path=args.model_path, horizon_bars=args.horizon_bars
        )
    else:
        metrics = train_logistic_regression(
            candles, model_path=args.model_path, horizon_bars=args.horizon_bars
        )

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
