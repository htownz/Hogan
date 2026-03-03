from __future__ import annotations

import argparse
import json

from hogan_bot.backtest import run_backtest_on_candles
from hogan_bot.config import load_config
from hogan_bot.exchange import KrakenClient
from hogan_bot.ml import load_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Hogan backtest on Kraken OHLCV")
    parser.add_argument("--symbol", default="BTC/USD")
    parser.add_argument("--timeframe", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--use-ml", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()

    timeframe = args.timeframe or cfg.timeframe
    limit = args.limit or cfg.ohlcv_limit

    client = KrakenClient(cfg.kraken_api_key, cfg.kraken_api_secret)
    candles = client.fetch_ohlcv_df(args.symbol, timeframe=timeframe, limit=limit)

    ml_model = load_model(cfg.ml_model_path) if args.use_ml else None

    result = run_backtest_on_candles(
        candles=candles,
        symbol=args.symbol,
        starting_balance_usd=cfg.starting_balance_usd,
        aggressive_allocation=cfg.aggressive_allocation,
        max_risk_per_trade=cfg.max_risk_per_trade,
        max_drawdown=cfg.max_drawdown,
        short_ma_window=cfg.short_ma_window,
        long_ma_window=cfg.long_ma_window,
        volume_window=cfg.volume_window,
        volume_threshold=cfg.volume_threshold,
        fee_rate=cfg.fee_rate,
        ml_model=ml_model,
        ml_buy_threshold=cfg.ml_buy_threshold,
        ml_sell_threshold=cfg.ml_sell_threshold,
    )

    print(json.dumps(result.__dict__, indent=2))


if __name__ == "__main__":
    main()
