from __future__ import annotations

import argparse
import os
import logging
import time
from datetime import datetime

from hogan_bot.config import BotConfig, load_config, symbol_config
from hogan_bot.decision import apply_ml_filter, ml_confidence
from hogan_bot.discord_commands import make_command_listener
from hogan_bot.exchange import ExchangeClient
from hogan_bot.ml import TrainedModel, load_model, predict_up_probability
from hogan_bot.mtf_ensemble import evaluate_mtf
from hogan_bot.trade_explainer import explain_trade as _explain_trade
from hogan_bot.notifier import make_notifier
from hogan_bot.paper import PaperPortfolio
from hogan_bot.regime import detect_regime, effective_thresholds, load_regime_signals
from hogan_bot.risk import DrawdownGuard, calculate_position_size
from hogan_bot.strategy import generate_signal
from hogan_bot.storage import (
    get_connection, record_equity, upsert_position,
    open_paper_trade, close_paper_trade,
)
from hogan_bot.execution import PaperExecution, LiveExecution

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def _null_regime():
    """Return a no-op RegimeState when regime detection is disabled."""
    from hogan_bot.regime import RegimeState
    return RegimeState(
        regime="ranging", adx=20.0, atr_pct_rank=0.5,
        trend_direction=0, ma_spread=0.0, confidence=0.0,
    )


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
    daily_candles_by_symbol: dict[str, object] = {}
    m30_candles_by_symbol: dict[str, object] = {}

    for symbol in config.symbols:
        candles = client.fetch_ohlcv_df(symbol, timeframe=config.timeframe, limit=config.ohlcv_limit)
        if candles.empty:
            logging.warning("No candles for %s", symbol)
            continue
        candles_by_symbol[symbol] = candles
        mark_prices[symbol] = float(candles["close"].iloc[-1])

        if config.use_mtf_ensemble:
            try:
                daily = client.fetch_ohlcv_df(symbol, timeframe=config.mtf_daily_timeframe, limit=60)
                if not daily.empty:
                    daily_candles_by_symbol[symbol] = daily
            except Exception:
                pass
            try:
                m30 = client.fetch_ohlcv_df(symbol, timeframe=config.mtf_m30_timeframe, limit=100)
                if not m30.empty:
                    m30_candles_by_symbol[symbol] = m30
            except Exception:
                pass

    if not mark_prices:
        logging.warning("No symbols had market data this cycle")
        return True

    # Load macro signals for regime detection once per cycle (cheap DB reads)
    regime_signals: dict = {}
    if config.use_regime_detection and conn is not None:
        regime_signals = load_regime_signals(conn)

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
        px = mark_prices[exit_symbol]
        now_ms = int(time.time() * 1000)
        if reason in ("trailing_stop", "take_profit"):
            pos = portfolio.positions.get(exit_symbol)
            if pos is None:
                continue
            qty = pos.qty
            executed = portfolio.execute_sell(exit_symbol, px, qty)
            if executed and conn is not None:
                fee = qty * px * config.fee_rate
                close_paper_trade(conn, exit_symbol, "long", px, fee, now_ms, close_reason=reason)
        else:  # short_trailing_stop or short_take_profit
            pos = portfolio.short_positions.get(exit_symbol)
            if pos is None:
                continue
            qty = pos.qty
            executed = portfolio.execute_cover(exit_symbol, px, qty)
            if executed and conn is not None:
                fee = qty * px * config.fee_rate
                close_paper_trade(conn, exit_symbol, "short", px, fee, now_ms, close_reason=reason)
        logging.info(
            "AUTO_EXIT symbol=%s reason=%s px=%.2f qty=%.6f ok=%s equity=%.2f",
            exit_symbol, reason, px, qty, executed, portfolio.total_equity(mark_prices),
        )
        if notifier and executed:
            notifier.notify("auto_exit", {"symbol": exit_symbol, "reason": reason, "price": px, "qty": qty})

    for symbol, candles in candles_by_symbol.items():
        # ── Per-symbol Optuna config overlay ───────────────────────────────
        cfg = symbol_config(config, symbol)

        # ── Regime detection ───────────────────────────────────────────────
        regime_state = detect_regime(
            candles,
            adx_trending_threshold=cfg.regime_adx_trending,
            adx_ranging_threshold=cfg.regime_adx_ranging,
            atr_volatile_pct=cfg.regime_atr_volatile_pct,
            btc_dominance=regime_signals.get("btc_dominance"),
            fear_greed=regime_signals.get("fear_greed"),
        ) if cfg.use_regime_detection else None

        eff = effective_thresholds(regime_state or _null_regime(), cfg)

        if regime_state is not None:
            logging.info(
                "REGIME symbol=%s regime=%s adx=%.1f atr_rank=%.2f conf=%.2f "
                "vol_thr=%.2f ml_buy=%.2f ml_sell=%.2f pos_scale=%.2f",
                symbol, regime_state.regime, regime_state.adx,
                regime_state.atr_pct_rank, regime_state.confidence,
                eff["volume_threshold"], eff["ml_buy_threshold"],
                eff["ml_sell_threshold"], eff["position_scale"],
            )

        signal = generate_signal(
            candles,
            short_window=cfg.short_ma_window,
            long_window=cfg.long_ma_window,
            volume_window=cfg.volume_window,
            volume_threshold=eff["volume_threshold"],   # regime-adjusted
            use_ema_clouds=cfg.use_ema_clouds,
            ema_fast_short=cfg.ema_fast_short,
            ema_fast_long=cfg.ema_fast_long,
            ema_slow_short=cfg.ema_slow_short,
            ema_slow_long=cfg.ema_slow_long,
            use_fvg=cfg.use_fvg,
            fvg_min_gap_pct=cfg.fvg_min_gap_pct,
            signal_mode=cfg.signal_mode,
            min_vote_margin=cfg.signal_min_vote_margin,
            atr_stop_multiplier=cfg.atr_stop_multiplier,
            use_ict=cfg.use_ict,
            ict_swing_left=cfg.ict_swing_left,
            ict_swing_right=cfg.ict_swing_right,
            ict_eq_tolerance_pct=cfg.ict_eq_tolerance_pct,
            ict_min_displacement_pct=cfg.ict_min_displacement_pct,
            ict_require_time_window=cfg.ict_require_time_window,
            ict_time_windows=cfg.ict_time_windows,
            ict_require_pd=cfg.ict_require_pd,
            ict_ote_enabled=cfg.ict_ote_enabled,
            ict_ote_low=cfg.ict_ote_low,
            ict_ote_high=cfg.ict_ote_high,
        )
        px = mark_prices[symbol]

        up_prob = None
        conf_scale = eff["position_scale"]   # start from regime position scale
        action = signal.action
        if config.use_ml_filter and ml_model is not None:
            up_prob = predict_up_probability(candles, ml_model, db_conn=conn)
            action = apply_ml_filter(
                signal.action, up_prob,
                eff["ml_buy_threshold"],
                eff["ml_sell_threshold"],
            )
            if config.ml_confidence_sizing:
                conf_scale = ml_confidence(up_prob) * eff["position_scale"]

        # ── MTF ensemble filter ────────────────────────────────────────────
        if config.use_mtf_ensemble and action != "hold":
            mtf_result = evaluate_mtf(
                daily_candles=daily_candles_by_symbol.get(symbol),
                hourly_action=action,
                m30_candles=m30_candles_by_symbol.get(symbol),
                unconfirmed_scale=config.mtf_unconfirmed_scale,
            )
            action = mtf_result.final_action
            conf_scale *= mtf_result.confidence_mult

        size = calculate_position_size(
            equity_usd=equity,
            price=px,
            stop_distance_pct=signal.stop_distance_pct,
            max_risk_per_trade=config.max_risk_per_trade,
            max_allocation_pct=config.aggressive_allocation,
            confidence_scale=conf_scale,
        )

        is_long = symbol in portfolio.positions and portfolio.positions[symbol].qty > 0
        is_short = symbol in portfolio.short_positions and portfolio.short_positions[symbol].qty > 0
        now_ms = int(time.time() * 1000)

        def _log(label, qty=0.0, ok=False):
            logging.info(
                "%s symbol=%s px=%.2f qty=%.6f ok=%s vol_ratio=%.2f conf=%.2f ml_up=%.3f equity=%.2f cash=%.2f",
                label, symbol, px, qty, ok,
                signal.volume_ratio, signal.confidence,
                up_prob if up_prob is not None else -1.0,
                portfolio.total_equity(mark_prices), portfolio.cash_usd,
            )

        def _journal_open(side: str, entry_qty: float) -> None:
            if conn is not None:
                fee = entry_qty * px * config.fee_rate
                open_paper_trade(
                    conn, symbol, side, px, entry_qty, fee, now_ms,
                    ml_up_prob=up_prob,
                    strategy_conf=signal.confidence,
                    vol_ratio=signal.volume_ratio,
                )
                # LLM explanation — fire-and-forget, never blocks the trade loop
                try:
                    fill_id = f"{symbol}_{now_ms}"
                    signal_details = {
                        "action": side,
                        "confidence": signal.confidence,
                        "explanation": getattr(signal, "explanation", ""),
                        "tech": {"action": side, "confidence": signal.confidence},
                        "macro": {"regime": regime_state.regime if regime_state else "unknown"},
                    }
                    _explain_trade(fill_id, symbol, side, px, signal_details, conn=conn)
                except Exception:
                    pass

        def _journal_close(side: str, close_qty: float, reason: str = "signal") -> None:
            if conn is not None:
                fee = close_qty * px * config.fee_rate
                close_paper_trade(conn, symbol, side, px, fee, now_ms, close_reason=reason)

        if action == "buy":
            if is_long:
                # Already long — don't pyramid, just hold
                _log("HOLD")
            elif is_short:
                # Cover the short first, then open long in the same candle
                cover_qty = portfolio.short_positions[symbol].qty
                ok = portfolio.execute_cover(symbol, px, cover_qty)
                if ok:
                    _journal_close("short", cover_qty)
                _log("COVER", cover_qty, ok)
                if notifier and ok:
                    notifier.notify("cover", {"symbol": symbol, "price": px, "qty": cover_qty, "ml_up_prob": up_prob})
                # Open long with the now-freed cash (executor owns portfolio mutation)
                if executor:
                    res = executor.buy(symbol, px, size,
                                       trailing_stop_pct=eff["trailing_stop_pct"],
                                       take_profit_pct=eff["take_profit_pct"])
                    ok = bool(res.ok)
                else:
                    ok = portfolio.execute_buy(
                        symbol, px, size,
                        trailing_stop_pct=eff["trailing_stop_pct"],
                        take_profit_pct=eff["take_profit_pct"],
                    )
                if ok:
                    _journal_open("long", size)
                if notifier and ok:
                    notifier.notify("buy", {"symbol": symbol, "price": px, "qty": size, "ml_up_prob": up_prob})
                _log("BUY", size, ok)
            else:
                # No position — open long (executor owns portfolio mutation)
                if executor:
                    res = executor.buy(symbol, px, size,
                                       trailing_stop_pct=eff["trailing_stop_pct"],
                                       take_profit_pct=eff["take_profit_pct"])
                    ok = bool(res.ok)
                else:
                    ok = portfolio.execute_buy(
                        symbol, px, size,
                        trailing_stop_pct=eff["trailing_stop_pct"],
                        take_profit_pct=eff["take_profit_pct"],
                    )
                if ok:
                    _journal_open("long", size)
                if notifier and ok:
                    notifier.notify("buy", {"symbol": symbol, "price": px, "qty": size, "ml_up_prob": up_prob})
                _log("BUY", size, ok)

        elif action == "sell":
            if is_short:
                # Already short — don't pyramid, just hold
                _log("HOLD")
            elif is_long:
                # Close the long first, then open short if allowed (executor owns portfolio mutation)
                sell_qty = portfolio.positions[symbol].qty
                if executor:
                    res = executor.sell(symbol, px, sell_qty)
                    ok = bool(res.ok)
                else:
                    ok = portfolio.execute_sell(symbol, px, sell_qty)
                if ok:
                    _journal_close("long", sell_qty)
                if notifier and ok:
                    notifier.notify("sell", {"symbol": symbol, "price": px, "qty": sell_qty, "ml_up_prob": up_prob})
                _log("SELL", sell_qty, ok)
                # Optionally flip to short after closing long
                if ok and config.allow_shorts:
                    ok_short = portfolio.execute_short(
                        symbol, px, size,
                        trailing_stop_pct=eff["trailing_stop_pct"],
                        take_profit_pct=eff["take_profit_pct"],
                    )
                    if ok_short:
                        _journal_open("short", size)
                    if notifier and ok_short:
                        notifier.notify("short", {"symbol": symbol, "price": px, "qty": size, "ml_up_prob": up_prob})
                    _log("SHORT", size, ok_short)
            elif config.allow_shorts:
                # No position — open short
                ok = portfolio.execute_short(
                    symbol, px, size,
                    trailing_stop_pct=eff["trailing_stop_pct"],
                    take_profit_pct=eff["take_profit_pct"],
                )
                if ok:
                    _journal_open("short", size)
                if notifier and ok:
                    notifier.notify("short", {"symbol": symbol, "price": px, "qty": size, "ml_up_prob": up_prob})
                _log("SHORT", size, ok)
            else:
                _log("HOLD")

        else:
            _log("HOLD")

        # Update signal cache for Discord command listener
        if signal_cache is not None:
            signal_cache[symbol] = {
                "action": action,
                "price": px,
                "ml_up": up_prob if up_prob is not None else 0.0,
                "conf": signal.confidence,
                "vol_ratio": signal.volume_ratio,
                "regime": regime_state.regime if regime_state else "unknown",
                "adx": regime_state.adx if regime_state else 0.0,
                "is_long": symbol in portfolio.positions and portfolio.positions[symbol].qty > 0,
                "is_short": symbol in portfolio.short_positions and portfolio.short_positions[symbol].qty > 0,
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
        executor = LiveExecution(client=client, conn=conn, exchange_id=config.exchange_id, portfolio=portfolio)
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
