
from __future__ import annotations

import argparse
import logging

from hogan_bot.exchange import ExchangeClient
from hogan_bot.ml_advanced import save_artifact, train_advanced_ensemble

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def main() -> None:
    p = argparse.ArgumentParser(description="Train Hogan advanced ensemble model.")
    p.add_argument("--exchange", default="kraken")
    p.add_argument("--symbol", default="BTC/USD")
    p.add_argument("--timeframe", default="5m")
    p.add_argument("--limit", type=int, default=20000)
    p.add_argument("--horizon", type=int, default=48)
    p.add_argument("--label-k", type=float, default=2.0)
    p.add_argument("--n-regimes", type=int, default=3)
    p.add_argument("--out", default="models/hogan_advanced_ensemble.pkl")
    args = p.parse_args()

    client = ExchangeClient(args.exchange)
    df = client.fetch_ohlcv_df(args.symbol, timeframe=args.timeframe, limit=args.limit)
    artifact, metrics = train_advanced_ensemble(
        df,
        horizon=args.horizon,
        label_k=args.label_k,
        n_regimes=args.n_regimes,
    )
    save_artifact(artifact, args.out)
    logging.info("Saved: %s", args.out)
    logging.info("Metrics: %s", metrics)


if __name__ == "__main__":
    main()
