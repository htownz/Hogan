from __future__ import annotations

import argparse
import json

from hogan_bot.exchange import KrakenClient
from hogan_bot.ml import train_logistic_regression


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Hogan ML directional model")
    parser.add_argument("--symbol", default="BTC/USD")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument("--limit", type=int, default=5000)
    parser.add_argument("--horizon-bars", type=int, default=3)
    parser.add_argument("--model-path", default="models/hogan_logreg.pkl")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = KrakenClient(api_key=None, api_secret=None)
    candles = client.fetch_ohlcv_df(args.symbol, timeframe=args.timeframe, limit=args.limit)
    metrics = train_logistic_regression(candles, model_path=args.model_path, horizon_bars=args.horizon_bars)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
