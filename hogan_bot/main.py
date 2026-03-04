from __future__ import annotations

import argparse
import logging
import time
from datetime import datetime

from hogan_bot.config import BotConfig, load_config
from hogan_bot.exchange import KrakenClient
from hogan_bot.decision import apply_ml_filter
from hogan_bot.ml import TrainedModel, load_model, predict_up_probability
from hogan_bot.paper import PaperPortfolio
from hogan_bot.risk import DrawdownGuard, calculate_position_size
from hogan_bot.strategy import generate_signal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def is_allowed_trading_time(trade_weekends: bool) -> bool:
    if trade_weekends:
        return True
    return datetime.utcnow().weekday() < 5



def run_iteration(
    config: BotConfig,
    client: KrakenClient,
    portfolio: PaperPortfolio,
    guard: DrawdownGuard,
    ml_model: TrainedModel | None,
) -> bool:
    if not is_allowed_trading_time(config.trade_weekends):
        logging.info("Weekend pause enabled; sleeping.")
        return True

    candles_by_symbol = {}
    mark_prices = {}

    for symbol in config.symbols:
        candles = client.fetch_ohlcv_df(symbol, timeframe=config.timeframe, limit=config.ohlcv_limit)
        if candles.empty:
            logging.warning("No candles for %s", symbol)
            continue
        candles_by_symbol[symbol] = candles
        mark_prices[symbol] = float(candles["close"].iloc[-1])

    if not mark_prices:
        logging.warning("No symbols had market data this cycle")
        return True

    equity = portfolio.total_equity(mark_prices)
    if not guard.update_and_check(equity):
        logging.error("Max drawdown breached. Halting bot. equity=%.2f peak=%.2f", equity, guard.peak_equity)
        return False

    for symbol, candles in candles_by_symbol.items():
        signal = generate_signal(
            candles,
            short_window=config.short_ma_window,
            long_window=config.long_ma_window,
            volume_window=config.volume_window,
            volume_threshold=config.volume_threshold,
            use_ema_clouds=config.use_ema_clouds,
            ema_fast_short=config.ema_fast_short,
            ema_fast_long=config.ema_fast_long,
            ema_slow_short=config.ema_slow_short,
            ema_slow_long=config.ema_slow_long,
            use_fvg=config.use_fvg,
            fvg_min_gap_pct=config.fvg_min_gap_pct,
            signal_mode=config.signal_mode,
        )
        px = mark_prices[symbol]

        up_prob = None
        action = signal.action
        if config.use_ml_filter and ml_model is not None:
            up_prob = predict_up_probability(candles, ml_model)
            action = apply_ml_filter(signal.action, up_prob, config.ml_buy_threshold, config.ml_sell_threshold)

        size = calculate_position_size(
            equity_usd=equity,
            price=px,
            stop_distance_pct=signal.stop_distance_pct,
            max_risk_per_trade=config.max_risk_per_trade,
            max_allocation_pct=config.aggressive_allocation,
        )

        if action == "buy":
            executed = portfolio.execute_buy(symbol, px, size)
            logging.info(
                "BUY symbol=%s px=%.2f qty=%.6f ok=%s vol_ratio=%.2f conf=%.2f ml_up=%.3f equity=%.2f cash=%.2f",
                symbol,
                px,
                size,
                executed,
                signal.volume_ratio,
                signal.confidence,
                up_prob if up_prob is not None else -1.0,
                equity,
                portfolio.cash_usd,
            )
        elif action == "sell":
            current_qty = portfolio.positions.get(symbol).qty if symbol in portfolio.positions else 0.0
            sell_qty = min(current_qty, size)
            executed = portfolio.execute_sell(symbol, px, sell_qty)
            logging.info(
                "SELL symbol=%s px=%.2f qty=%.6f ok=%s vol_ratio=%.2f conf=%.2f ml_up=%.3f equity=%.2f cash=%.2f",
                symbol,
                px,
                sell_qty,
                executed,
                signal.volume_ratio,
                signal.confidence,
                up_prob if up_prob is not None else -1.0,
                equity,
                portfolio.cash_usd,
            )
        else:
            logging.info(
                "HOLD symbol=%s px=%.2f vol_ratio=%.2f conf=%.2f ml_up=%.3f equity=%.2f cash=%.2f",
                symbol,
                px,
                signal.volume_ratio,
                signal.confidence,
                up_prob if up_prob is not None else -1.0,
                equity,
                portfolio.cash_usd,
            )

    return True


def run(max_loops: int | None = None) -> None:
    config = load_config()
    if not config.paper_mode:
        raise ValueError("Live mode is intentionally disabled in this build. Use paper mode only.")

    client = KrakenClient(config.kraken_api_key, config.kraken_api_secret)
    portfolio = PaperPortfolio(cash_usd=config.starting_balance_usd, fee_rate=config.fee_rate)
    drawdown_guard = DrawdownGuard(config.starting_balance_usd, config.max_drawdown)

    ml_model = None
    if config.use_ml_filter:
        try:
            ml_model = load_model(config.ml_model_path)
            logging.info("Loaded ML model from %s", config.ml_model_path)
        except FileNotFoundError:
            logging.warning("ML filter enabled but model path missing: %s. Continuing without ML filter.", config.ml_model_path)
        except Exception as exc:  # pragma: no cover - defensive runtime guard
            logging.warning("Could not load ML model (%s). Continuing without ML filter.", exc)

    logging.info("Starting Hogan in paper mode for symbols=%s", ",".join(config.symbols))

    loops = 0
    while True:
        loops += 1
        keep_running = run_iteration(config, client, portfolio, drawdown_guard, ml_model)
        if not keep_running:
            break

        if max_loops is not None and loops >= max_loops:
            logging.info("Reached max_loops=%s; exiting.", max_loops)
            break

        time.sleep(config.sleep_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hogan BTC/ETH day trading research bot")
    parser.add_argument("--max-loops", type=int, default=None, help="Run finite cycles for testing")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(max_loops=args.max_loops)
