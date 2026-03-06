from __future__ import annotations

import argparse
import os
import logging
import time
from datetime import datetime

from hogan_bot.config import BotConfig, load_config
from hogan_bot.decision import apply_ml_filter, ml_confidence
from hogan_bot.discord_commands import make_command_listener
from hogan_bot.exchange import ExchangeClient
from hogan_bot.ml import TrainedModel, load_model, predict_up_probability
from hogan_bot.notifier import make_notifier
from hogan_bot.paper import PaperPortfolio
from hogan_bot.risk import DrawdownGuard, calculate_position_size
from hogan_bot.strategy import generate_signal
from hogan_bot.storage import get_connection, record_equity, upsert_position
from hogan_bot.execution import PaperExecution, LiveExecution

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def is_allowed_trading_time(trade_weekends: bool) -> bool:
    if trade_weekends:
        return True
    return datetime.utcnow().weekday() < 5



def run_iteration(
    config: BotConfig,
    client: ExchangeClient,
    portfolio: PaperPortfolio,
    guard: DrawdownGuard,
    ml_model: TrainedModel | None,
    notifier=None,
    conn=None,
    executor=None,
    signal_cache: dict | None = None,
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
    if conn is not None:
        dd = 0.0 if guard.peak_equity <= 0 else max(0.0, (guard.peak_equity - equity) / guard.peak_equity)
        record_equity(conn, int(time.time()*1000), portfolio.cash_usd, equity, dd)
    if not guard.update_and_check(equity):
        logging.error("Max drawdown breached. Halting bot. equity=%.2f peak=%.2f", equity, guard.peak_equity)
        if notifier:
            notifier.notify("drawdown_breach", {"equity": equity, "peak": guard.peak_equity})
        return False

    # Auto-exit positions that have hit their trailing stop or take-profit
    exits = portfolio.check_exits(mark_prices)
    for exit_symbol, reason in exits:
        pos = portfolio.positions.get(exit_symbol)
        if pos is None:
            continue
        px = mark_prices[exit_symbol]
        qty = pos.qty
        executed = portfolio.execute_sell(exit_symbol, px, qty)
        logging.info(
            "AUTO_EXIT symbol=%s reason=%s px=%.2f qty=%.6f ok=%s equity=%.2f",
            exit_symbol, reason, px, qty, executed, portfolio.total_equity(mark_prices),
        )
        if notifier and executed:
            notifier.notify("auto_exit", {"symbol": exit_symbol, "reason": reason, "price": px, "qty": qty})

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
            atr_stop_multiplier=config.atr_stop_multiplier,
            # ICT pillars — were previously missing, causing ICT to be silently disabled
            use_ict=config.use_ict,
            ict_swing_left=config.ict_swing_left,
            ict_swing_right=config.ict_swing_right,
            ict_eq_tolerance_pct=config.ict_eq_tolerance_pct,
            ict_min_displacement_pct=config.ict_min_displacement_pct,
            ict_require_time_window=config.ict_require_time_window,
            ict_time_windows=config.ict_time_windows,
            ict_require_pd=config.ict_require_pd,
            ict_ote_enabled=config.ict_ote_enabled,
            ict_ote_low=config.ict_ote_low,
            ict_ote_high=config.ict_ote_high,
        )
        px = mark_prices[symbol]

        up_prob = None
        conf_scale = 1.0
        action = signal.action
        if config.use_ml_filter and ml_model is not None:
            up_prob = predict_up_probability(candles, ml_model)
            action = apply_ml_filter(signal.action, up_prob, config.ml_buy_threshold, config.ml_sell_threshold)
            if config.ml_confidence_sizing:
                conf_scale = ml_confidence(up_prob)

        size = calculate_position_size(
            equity_usd=equity,
            price=px,
            stop_distance_pct=signal.stop_distance_pct,
            max_risk_per_trade=config.max_risk_per_trade,
            max_allocation_pct=config.aggressive_allocation,
            confidence_scale=conf_scale,
        )

        if action == "buy":
            if executor:
                res = executor.buy(symbol, px, size)
                executed = bool(res.ok)
                # Shadow-accounting so sizing/drawdown guard keep working even in live mode.
                if executed:
                    portfolio.execute_buy(symbol, px, size,
                        trailing_stop_pct=config.trailing_stop_pct,
                        take_profit_pct=config.take_profit_pct,
                    )
            else:
                executed = portfolio.execute_buy(
                    symbol, px, size,
                    trailing_stop_pct=config.trailing_stop_pct,
                    take_profit_pct=config.take_profit_pct,
                )
            if notifier and executed:
                notifier.notify("buy", {"symbol": symbol, "price": px, "qty": size, "ml_up_prob": up_prob})
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
            if executor:
                res = executor.sell(symbol, px, sell_qty)
                executed = bool(res.ok)
                if executed:
                    portfolio.execute_sell(symbol, px, sell_qty)
            else:
                executed = portfolio.execute_sell(symbol, px, sell_qty)
            if notifier and executed:
                notifier.notify("sell", {"symbol": symbol, "price": px, "qty": sell_qty, "ml_up_prob": up_prob})
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

        # Update signal cache for Discord command listener
        if signal_cache is not None:
            signal_cache[symbol] = {
                "action": action,
                "price": px,
                "ml_up": up_prob if up_prob is not None else 0.0,
                "conf": signal.confidence,
                "vol_ratio": signal.volume_ratio,
            }
            signal_cache["_mark_prices"] = mark_prices

    return True


def run(max_loops: int | None = None) -> None:
    config = load_config()

    # Persistence / journaling DB
    conn = get_connection(config.db_path)

    client = ExchangeClient(config.exchange_id, config.kraken_api_key, config.kraken_api_secret)

    # Safety latch for live mode
    live_ack = (os.getenv("HOGAN_LIVE_ACK", "") or "").strip()
    allow_live = (not config.paper_mode) and config.live_mode and (live_ack == "I_UNDERSTAND_LIVE_TRADING")

    portfolio = PaperPortfolio(cash_usd=config.starting_balance_usd, fee_rate=config.fee_rate)
    drawdown_guard = DrawdownGuard(config.starting_balance_usd, config.max_drawdown)

    if allow_live:
        logging.warning("LIVE TRADING ENABLED on exchange=%s (spot only).", config.exchange_id)
        executor = LiveExecution(client=client, conn=conn, exchange_id=config.exchange_id)
    else:
        if not config.paper_mode:
            logging.warning(
                "Paper disabled but live latch not satisfied; forcing PAPER mode. "
                "Set HOGAN_LIVE_MODE=true and HOGAN_LIVE_ACK=I_UNDERSTAND_LIVE_TRADING to allow live."
            )
        executor = PaperExecution(portfolio=portfolio, conn=conn, exchange_id="paper")

    ml_model = None
    if config.use_ml_filter:
        try:
            ml_model = load_model(config.ml_model_path)
            logging.info("Loaded ML model from %s", config.ml_model_path)
        except FileNotFoundError:
            logging.warning("ML filter enabled but model path missing: %s. Continuing without ML filter.", config.ml_model_path)
        except Exception as exc:  # pragma: no cover
            logging.warning("Could not load ML model (%s). Continuing without ML filter.", exc)

    notifier = make_notifier(config.webhook_url or None)
    mode_str = "LIVE" if allow_live else "PAPER"
    logging.info(
        "Starting Hogan mode=%s symbols=%s timeframe=%s db=%s",
        mode_str,
        ",".join(config.symbols),
        config.timeframe,
        config.db_path,
    )
    notifier.notify("info", {
        "status": f"Hogan {mode_str} started",
        "symbols": ", ".join(config.symbols),
        "timeframe": config.timeframe,
        "ml_model": "loaded" if ml_model else "no model — retraining needed",
        "ict": "enabled" if config.use_ict else "disabled",
        "equity": f"${config.starting_balance_usd:,.2f}",
    })

    # Discord command listener (runs in background thread if DISCORD_BOT_TOKEN set)
    config_summary = {
        "mode": mode_str,
        "symbols": ", ".join(config.symbols),
        "timeframe": config.timeframe,
        "model": config.ml_model_path,
        "signal_mode": config.signal_mode,
        "use_ict": config.use_ict,
        "use_ema_clouds": config.use_ema_clouds,
        "ml_buy_threshold": config.ml_buy_threshold,
    }
    signal_cache: dict = {}
    cmd_listener = make_command_listener(
        webhook_url=config.webhook_url or "",
        db_path=config.db_path,
    )
    if cmd_listener:
        cmd_listener.update_state(portfolio, {}, config_summary, signal_cache)
        cmd_listener.start()

    loops = 0
    while True:
        loops += 1
        keep_running = run_iteration(config, client, portfolio, drawdown_guard, ml_model, notifier=notifier, conn=conn, executor=executor, signal_cache=signal_cache)
        if not keep_running:
            break

        if max_loops is not None and loops >= max_loops:
            logging.info("Reached max_loops=%s; exiting.", max_loops)
            break

        # Keep Discord command listener state fresh after each loop
        if cmd_listener:
            _prices = signal_cache.get("_mark_prices", {})
            cmd_listener.update_state(portfolio, _prices, config_summary, signal_cache)

        time.sleep(config.sleep_seconds)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Hogan BTC/ETH day trading research bot")
    parser.add_argument("--max-loops", type=int, default=None, help="Run finite cycles for testing")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(max_loops=args.max_loops)
