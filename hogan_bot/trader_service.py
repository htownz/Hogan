
from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import threading
import time
from datetime import datetime

from hogan_bot.config import load_config, symbol_config, effective_hold_cooldown_bars
from hogan_bot.decision import apply_ml_filter, ml_confidence
from hogan_bot.exchange import ExchangeClient
from hogan_bot.execution import LiveExecution, PaperExecution
from hogan_bot.live_account import fetch_account_state
from hogan_bot.macro_filter import evaluate_macro
from hogan_bot.metrics import MetricsServer, LoopTimer, EQUITY, CASH, DRAWDOWN, ORDERS, ORDER_FAILS, EXCEPTIONS, SLIPPAGE_BPS, FILLS
from hogan_bot.ml import load_model as load_simple_model, predict_up_probability as predict_simple
from hogan_bot.ml_advanced import load_artifact as load_adv_artifact, predict_up_probability as predict_adv
from hogan_bot.mtf_ensemble import evaluate_mtf
from hogan_bot.notifier import make_notifier
from hogan_bot.paper import PaperPortfolio
from hogan_bot.regime import detect_regime, effective_thresholds, load_regime_signals, RegimeState
from hogan_bot.risk import DrawdownGuard, calculate_position_size
from hogan_bot.storage import (
    get_connection, record_equity, upsert_position,
    upsert_position_state, load_position_state,
    load_latest_fill_ts, record_fill,
    open_paper_trade, close_paper_trade,
)
from hogan_bot.agent_pipeline import AgentPipeline

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("hogan.trader")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _null_regime() -> RegimeState:
    return RegimeState(
        regime="ranging", adx=20.0, atr_pct_rank=0.5,
        trend_direction=0, ma_spread=0.0, confidence=0.0,
    )


def is_allowed_trading_time(trade_weekends: bool) -> bool:
    if trade_weekends:
        return True
    return datetime.utcnow().weekday() < 5


def _live_safety_latch(config) -> None:
    if not config.live_mode:
        return
    if os.getenv("HOGAN_LIVE_ACK", "") != "I_UNDERSTAND_LIVE_TRADING":
        raise RuntimeError("Refusing live trading: set HOGAN_LIVE_ACK=I_UNDERSTAND_LIVE_TRADING")


def _load_any_model(path: str):
    if not path:
        return None
    try:
        return load_adv_artifact(path)
    except Exception:
        return load_simple_model(path)


def _predict_up(model_obj, candles, db_conn=None):
    if model_obj is None:
        return None
    if getattr(model_obj, "artifact_type", "").startswith("advanced_ensemble"):
        return predict_adv(model_obj, candles)
    return predict_simple(candles, model_obj, db_conn=db_conn)


def sync_fills_deterministic(conn, client: ExchangeClient, exchange_id: str, symbols: list[str]) -> int:
    since = load_latest_fill_ts(conn, exchange_id, symbol=None)
    since = max(0, int(since) + 1)
    new = 0
    for sym in symbols:
        try:
            trades = client.fetch_my_trades(symbol=sym, since=since, limit=200)
            for t in trades:
                td = dict(t)
                td["exchange"] = exchange_id
                record_fill(conn, td)
                new += 1
        except Exception:
            continue
    if new:
        FILLS.labels(exchange=exchange_id).inc(new)
    return new


# ---------------------------------------------------------------------------
# Online learning helpers (Phase 4)
# ---------------------------------------------------------------------------

def _write_training_buffer(conn, symbol: str, ts_ms: int, features: list[float]) -> None:
    """Write a feature vector to the online_training_buffer (label pending)."""
    try:
        conn.execute(
            "INSERT INTO online_training_buffer (symbol, ts_ms, features_json, horizon_bars) VALUES (?, ?, ?, ?)",
            (symbol, ts_ms, json.dumps(features), 12),
        )
        conn.commit()
    except Exception as exc:
        logger.debug("Buffer write failed: %s", exc)


def _label_buffer_from_trade(conn, symbol: str, entry_ts_ms: int, pnl_pct: float) -> None:
    """Label the most recent pending buffer row for a symbol after trade closes."""
    try:
        label = 1 if pnl_pct > 0 else 0
        conn.execute(
            """UPDATE online_training_buffer
               SET label = ?, pnl_pct = ?, fill_ts_ms = ?
               WHERE row_id = (
                   SELECT row_id FROM online_training_buffer
                   WHERE symbol = ? AND label IS NULL
                   ORDER BY ts_ms DESC LIMIT 1
               )""",
            (label, pnl_pct, int(time.time() * 1000), symbol),
        )
        conn.commit()
    except Exception as exc:
        logger.debug("Buffer labeling failed: %s", exc)


def _start_background_retrain(
    config, conn, model_holder: dict, model_lock: threading.Lock,
) -> threading.Thread | None:
    """Start a background thread that periodically retrains the model."""
    if not config.retrain_schedule_hours or config.retrain_schedule_hours <= 0:
        return None

    def _retrain_worker():
        interval = config.retrain_schedule_hours * 3600
        while True:
            time.sleep(interval)
            try:
                logger.info("BACKGROUND_RETRAIN starting scheduled retrain...")
                from hogan_bot.storage import load_candles
                import pandas as pd
                all_candles = []
                retrain_conn = get_connection(config.db_path)
                for sym in config.training_symbols:
                    c = load_candles(retrain_conn, sym, config.timeframe, limit=50000)
                    if not c.empty:
                        all_candles.append(c)
                if not all_candles:
                    logger.warning("BACKGROUND_RETRAIN no candles available")
                    retrain_conn.close()
                    continue

                candles_concat = pd.concat(all_candles, ignore_index=True)

                import tempfile
                tmp_path = os.path.join(tempfile.gettempdir(), "hogan_bg_retrain.pkl")
                from hogan_bot.ml import train_xgboost
                result = train_xgboost(
                    candles_concat, tmp_path, horizon_bars=12,
                    db_conn=retrain_conn, prune_features=True, max_features=40,
                )
                retrain_conn.close()

                if result and os.path.exists(tmp_path):
                    new_auc = result.get("roc_auc", 0)
                    old_auc = model_holder.get("roc_auc", 0)
                    if new_auc > old_auc:
                        import shutil
                        shutil.copy2(tmp_path, config.ml_model_path)
                        new_model = _load_any_model(config.ml_model_path)
                        with model_lock:
                            model_holder["model"] = new_model
                            model_holder["roc_auc"] = new_auc
                        logger.info("BACKGROUND_RETRAIN promoted new model: auc=%.4f (was %.4f)", new_auc, old_auc)
                    else:
                        logger.info("BACKGROUND_RETRAIN candidate not better: %.4f <= %.4f", new_auc, old_auc)
            except Exception as exc:
                logger.warning("BACKGROUND_RETRAIN failed: %s", exc)

    thread = threading.Thread(target=_retrain_worker, daemon=True, name="bg-retrain")
    thread.start()
    logger.info("Background retrain thread started (every %dh)", config.retrain_schedule_hours)
    return thread


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def run_loop(max_loops: int | None = None) -> None:  # noqa: PLR0912,PLR0915
    config = load_config()
    _live_safety_latch(config)

    conn = get_connection(config.db_path)

    client = ExchangeClient(
        config.exchange_id,
        api_key=config.kraken_api_key if config.exchange_id == "kraken" else None,
        api_secret=config.kraken_api_secret if config.exchange_id == "kraken" else None,
    )

    metrics = MetricsServer(port=config.metrics_port)
    metrics.start()

    email_cfg = None
    if config.email_smtp_host and config.email_to and config.email_from:
        email_cfg = dict(
            smtp_host=config.email_smtp_host,
            smtp_port=config.email_smtp_port,
            username=config.email_username,
            password=config.email_password,
            from_addr=config.email_from,
            to_addr=config.email_to,
            use_tls=True,
        )
    notifier = make_notifier(
        discord_webhook_url=config.webhook_url or None,
        email=email_cfg,
    )

    guard = DrawdownGuard(starting_equity=config.starting_balance_usd, max_drawdown=config.max_drawdown)

    paper_port = PaperPortfolio(cash_usd=config.starting_balance_usd, fee_rate=config.fee_rate)
    mode = "paper"
    if (not config.paper_mode) and config.live_mode:
        executor = LiveExecution(client, conn=conn, exchange_id=config.exchange_id)
        mode = "live"
    else:
        executor = PaperExecution(paper_port, conn=conn)

    model_obj = _load_any_model(config.ml_model_path) if config.use_ml_filter else None

    # ── Online learning setup (Phase 4) ──────────────────────────────────
    model_lock = threading.Lock()
    model_holder: dict = {"model": model_obj, "roc_auc": 0.0}
    online_learner = None
    if config.use_online_learning:
        try:
            from hogan_bot.online_learner import OnlineLearner
            online_learner = OnlineLearner(db_path=config.db_path, min_batch=20)
            logger.info("OnlineLearner initialized (interval=%d loops)", config.online_learning_interval)
        except Exception as exc:
            logger.warning("OnlineLearner init failed: %s", exc)

    bg_retrain_thread = _start_background_retrain(config, conn, model_holder, model_lock)

    # ── Agent Pipeline (Technical + Sentiment + Macro → MetaWeigher) ─────
    pipeline = AgentPipeline(config, conn=conn)

    # ── Trade explainer (LLM, fire-and-forget) ────────────────────────────
    _explain_trade = None
    try:
        from hogan_bot.trade_explainer import explain_trade as _explain_trade
    except ImportError:
        pass

    # ── Discord command listener ──────────────────────────────────────────
    signal_cache: dict = {}
    cmd_listener = None
    mode_str = "LIVE" if mode == "live" else "PAPER"
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
    try:
        from hogan_bot.discord_commands import make_command_listener
        cmd_listener = make_command_listener(
            webhook_url=config.webhook_url or "",
            db_path=config.db_path,
        )
        if cmd_listener:
            cmd_listener.update_state(paper_port, {}, config_summary, signal_cache)
            cmd_listener.start()
    except Exception:
        pass

    # ── Dry-run validation gate ───────────────────────────────────────────
    try:
        from hogan_bot.storage import load_candles as _load_candles
        from hogan_bot.recursive_check import dry_run_validate as _dry_run

        _candles_dv = _load_candles(conn, config.symbols[0], config.timeframe, limit=200)
        if not _candles_dv.empty:
            _drv_result = _dry_run(_candles_dv, n_loops=10)
            if not _drv_result["ok"]:
                if _drv_result.get("errors"):
                    logger.error("Dry-run validation FAILED with errors: %s", _drv_result["errors"])
                    raise RuntimeError(f"Dry-run validation failed: {_drv_result['errors']}")
                elif _drv_result.get("non_hold_ratio", 0) < 0.05:
                    logger.warning(
                        "Dry-run validation: all signals are HOLD (non_hold_ratio=%.2f). "
                        "Continuing anyway — strategy may need data.",
                        _drv_result.get("non_hold_ratio", 0),
                    )
            else:
                logger.info(
                    "Dry-run validation PASSED: %d/%d non-hold signals (%.0f%%)",
                    _drv_result.get("non_hold_count", 0),
                    _drv_result.get("total_bars", 0),
                    _drv_result.get("non_hold_ratio", 0) * 100,
                )
    except ImportError:
        pass
    except Exception as exc:
        logger.warning("Dry-run validation error (non-fatal): %s", exc)

    # ── Startup notification ──────────────────────────────────────────────
    notifier.notify("info", {
        "status": f"Hogan trader_service {'LIVE' if mode == 'live' else 'PAPER'} started",
        "symbols": ", ".join(config.symbols),
        "timeframe": config.timeframe,
        "ml_model": "loaded" if model_obj else "no model",
        "regime": "enabled" if config.use_regime_detection else "disabled",
        "equity": f"${config.starting_balance_usd:,.2f}",
    })

    # ── Loss cooldown state ───────────────────────────────────────────────
    _cooldown_remaining: int = 0

    loop = 0
    while True:
        if max_loops is not None and loop >= max_loops:
            break
        loop += 1

        try:
            with LoopTimer():
                # ── Weekend gate ──────────────────────────────────────
                if not is_allowed_trading_time(config.trade_weekends):
                    logger.info("Weekend pause enabled; sleeping.")
                    time.sleep(config.sleep_seconds)
                    continue

                # Decrement cooldown each loop iteration
                if _cooldown_remaining > 0:
                    _cooldown_remaining -= 1

                # ── Live fill sync ────────────────────────────────────
                if mode == "live":
                    sync_fills_deterministic(conn, client, config.exchange_id, config.symbols)

                # ── Fetch candles ─────────────────────────────────────
                candles_by_symbol = {}
                mark_prices = {}
                daily_candles_by_symbol: dict[str, object] = {}
                m30_candles_by_symbol: dict[str, object] = {}
                for symbol in config.symbols:
                    candles = client.fetch_ohlcv_df(symbol, timeframe=config.timeframe, limit=config.ohlcv_limit)
                    if candles.empty:
                        continue
                    candles_by_symbol[symbol] = candles
                    mark_prices[symbol] = float(candles["close"].iloc[-1])

                    if config.use_mtf_ensemble:
                        if config.mtf_use_daily_filter:
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
                    time.sleep(config.sleep_seconds)
                    continue

                # ── Equity calculation ────────────────────────────────
                if mode == "live":
                    state = fetch_account_state(client, config.symbols, quote_ccy=config.quote_currency)
                    equity = state.equity_quote
                    cash = state.cash_quote
                    for sym, pos in state.positions.items():
                        upsert_position(conn, sym, pos.qty, 0.0, state.ts_ms)
                        ps = load_position_state(conn, sym)
                        px = state.marks.get(sym, mark_prices.get(sym, 0.0))
                        if ps is None:
                            upsert_position_state(conn, sym, entry_price=px, peak_price=px, updated_ms=state.ts_ms)
                        else:
                            entry, peak = ps
                            peak = max(peak, px)
                            upsert_position_state(conn, sym, entry_price=entry, peak_price=peak, updated_ms=state.ts_ms)
                else:
                    equity = paper_port.total_equity(mark_prices)
                    cash = paper_port.cash_usd

                dd = 0.0 if guard.peak_equity <= 0 else max(0.0, (guard.peak_equity - equity) / guard.peak_equity)
                record_equity(conn, int(time.time() * 1000), cash, equity, dd)
                EQUITY.set(equity)
                CASH.set(cash)
                DRAWDOWN.set(dd)

                if not guard.update_and_check(equity):
                    logger.error("Max drawdown breached. Halting. equity=%.2f peak=%.2f", equity, guard.peak_equity)
                    notifier.notify("drawdown_breach", {"equity": equity, "peak": guard.peak_equity})
                    break

                # ── Load regime signals (DB macro data) ───────────────
                regime_signals: dict = {}
                if config.use_regime_detection and conn is not None:
                    try:
                        regime_signals = load_regime_signals(conn)
                    except Exception:
                        pass

                # ── Paper exit management (trailing stop, take profit, max hold) ──
                if mode == "paper":
                    max_hold_bars, loss_cooldown_bars = effective_hold_cooldown_bars(config, config.timeframe)
                    exits = paper_port.check_exits(mark_prices, max_hold_bars=max_hold_bars)
                    for exit_symbol, reason in exits:
                        exit_px = mark_prices.get(exit_symbol, 0.0)
                        now_ms = int(time.time() * 1000)
                        if reason in ("trailing_stop", "take_profit", "max_hold_time"):
                            pos = paper_port.positions.get(exit_symbol)
                            if pos is None:
                                continue
                            qty = pos.qty
                            avg_entry = pos.avg_entry
                            ok = paper_port.execute_sell(exit_symbol, exit_px, qty)
                            if ok:
                                fee = qty * exit_px * config.fee_rate
                                close_paper_trade(conn, exit_symbol, "long", exit_px, fee, now_ms, close_reason=reason)
                                pnl_pct = (exit_px - avg_entry) / avg_entry if avg_entry > 0 else 0.0
                                _label_buffer_from_trade(conn, exit_symbol, now_ms, pnl_pct)
                                is_loss = exit_px < avg_entry
                                if is_loss and loss_cooldown_bars > 0:
                                    _cooldown_remaining = loss_cooldown_bars
                                ORDERS.labels(side="sell", mode=mode, exchange="paper").inc()
                                notifier.notify("auto_exit", {"symbol": exit_symbol, "reason": reason, "price": exit_px, "qty": qty})
                                logger.info("AUTO_EXIT symbol=%s reason=%s px=%.2f qty=%.6f equity=%.2f",
                                            exit_symbol, reason, exit_px, qty, paper_port.total_equity(mark_prices))
                        elif reason in ("short_trailing_stop", "short_take_profit"):
                            pos = paper_port.short_positions.get(exit_symbol)
                            if pos is None:
                                continue
                            qty = pos.qty
                            ok = paper_port.execute_cover(exit_symbol, exit_px, qty)
                            if ok:
                                fee = qty * exit_px * config.fee_rate
                                close_paper_trade(conn, exit_symbol, "short", exit_px, fee, now_ms, close_reason=reason)
                                ORDERS.labels(side="cover", mode=mode, exchange="paper").inc()
                                notifier.notify("auto_exit", {"symbol": exit_symbol, "reason": reason, "price": exit_px, "qty": qty})
                                logger.info("AUTO_EXIT_SHORT symbol=%s reason=%s px=%.2f qty=%.6f",
                                            exit_symbol, reason, exit_px, qty)

                # ── Live exit management (trailing stop, take profit via DB state) ──
                elif mode == "live" and (config.trailing_stop_pct > 0 or config.take_profit_pct > 0):
                    for sym in list(mark_prices.keys()):
                        px = mark_prices[sym]
                        row = conn.execute("SELECT qty FROM positions WHERE symbol=?", (sym,)).fetchone()
                        qty_live = float(row[0]) if row else 0.0
                        if qty_live <= 0:
                            continue
                        ps = load_position_state(conn, sym)
                        if ps:
                            entry, peak = ps
                            peak = max(peak, px)
                            upsert_position_state(conn, sym, entry_price=entry, peak_price=peak, updated_ms=int(time.time() * 1000))
                            if config.trailing_stop_pct > 0:
                                stop = peak * (1 - config.trailing_stop_pct)
                                if px <= stop:
                                    ORDERS.labels(side="sell", mode=mode, exchange=config.exchange_id).inc()
                                    res = executor.sell(sym, px, qty_live)
                                    if res.ok:
                                        now_ms = int(time.time() * 1000)
                                        fee = qty_live * px * config.fee_rate
                                        close_paper_trade(conn, sym, "long", px, fee, now_ms, close_reason="trailing_stop")
                                        notifier.notify("trailing_stop_exit", {"symbol": sym, "price": px, "qty": qty_live})
                                    else:
                                        ORDER_FAILS.labels(side="sell", mode=mode, exchange=config.exchange_id).inc()
                                    continue
                            if config.take_profit_pct > 0 and entry > 0:
                                tp = entry * (1 + config.take_profit_pct)
                                if px >= tp:
                                    ORDERS.labels(side="sell", mode=mode, exchange=config.exchange_id).inc()
                                    res = executor.sell(sym, px, qty_live)
                                    if res.ok:
                                        now_ms = int(time.time() * 1000)
                                        fee = qty_live * px * config.fee_rate
                                        close_paper_trade(conn, sym, "long", px, fee, now_ms, close_reason="take_profit")
                                        notifier.notify("take_profit_exit", {"symbol": sym, "price": px, "qty": qty_live})
                                    else:
                                        ORDER_FAILS.labels(side="sell", mode=mode, exchange=config.exchange_id).inc()

                # ── Main decision loop ────────────────────────────────
                for symbol, candles in candles_by_symbol.items():
                    px = mark_prices[symbol]

                    # ── Per-symbol Optuna config overlay ──────────────
                    cfg = symbol_config(config, symbol)

                    # ── Regime detection ──────────────────────────────
                    regime_state = None
                    if cfg.use_regime_detection:
                        try:
                            regime_state = detect_regime(
                                candles,
                                adx_trending_threshold=cfg.regime_adx_trending,
                                adx_ranging_threshold=cfg.regime_adx_ranging,
                                atr_volatile_pct=cfg.regime_atr_volatile_pct,
                                btc_dominance=regime_signals.get("btc_dominance"),
                                fear_greed=regime_signals.get("fear_greed"),
                            )
                        except Exception:
                            pass

                    eff = effective_thresholds(regime_state or _null_regime(), cfg)

                    if regime_state is not None:
                        logger.info(
                            "REGIME symbol=%s regime=%s adx=%.1f conf=%.2f "
                            "vol_thr=%.2f ml_buy=%.2f ml_sell=%.2f ts=%.3f tp=%.3f pos_scale=%.2f",
                            symbol, regime_state.regime, regime_state.adx,
                            regime_state.confidence,
                            eff["volume_threshold"], eff["ml_buy_threshold"],
                            eff["ml_sell_threshold"], eff["trailing_stop_pct"],
                            eff["take_profit_pct"], eff["position_scale"],
                        )

                    # ── Agent Pipeline decision ────────────────────────
                    signal = pipeline.run(candles, symbol=symbol, config_override=cfg)

                    action = signal.action
                    up_prob = None
                    conf_scale = (signal.confidence or 1.0) * eff["position_scale"]

                    if action != "hold":
                        logger.info(
                            "PIPELINE %s action=%s conf=%.2f | %s",
                            symbol, action, signal.confidence, signal.explanation,
                        )

                    # ── ML filter with regime-adjusted thresholds ─────
                    with model_lock:
                        current_model = model_holder.get("model", model_obj)
                    if config.use_ml_filter and current_model is not None:
                        up_prob = _predict_up(current_model, candles, db_conn=conn)
                        action = apply_ml_filter(
                            action, up_prob,
                            eff["ml_buy_threshold"],
                            eff["ml_sell_threshold"],
                        )
                        if config.ml_confidence_sizing and up_prob is not None:
                            conf_scale = ml_confidence(up_prob) * eff["position_scale"]

                    # ── MTF ensemble filter ────────────────────────────
                    if config.use_mtf_ensemble and action != "hold":
                        mtf_result = evaluate_mtf(
                            daily_candles=daily_candles_by_symbol.get(symbol),
                            hourly_action=action,
                            m30_candles=m30_candles_by_symbol.get(symbol),
                            unconfirmed_scale=config.mtf_unconfirmed_scale,
                        )
                        action = mtf_result.final_action
                        conf_scale *= mtf_result.confidence_mult

                    # ── Macro correlation filter ──────────────────────
                    if config.use_macro_filter and action != "hold":
                        macro_result = evaluate_macro(
                            conn, action=action,
                            ma_period=config.macro_equity_ma_period,
                            vix_caution=config.macro_vix_caution,
                            vix_block=config.macro_vix_block,
                        )
                        if macro_result.block_longs and action == "buy":
                            action = "hold"
                        conf_scale *= macro_result.confidence_mult

                    if action == "hold":
                        continue

                    # ── Position state ────────────────────────────────
                    if mode == "live":
                        row = conn.execute("SELECT qty FROM positions WHERE symbol=?", (symbol,)).fetchone()
                        cur_qty = float(row[0]) if row else 0.0
                    else:
                        cur_qty = paper_port.positions.get(symbol).qty if symbol in paper_port.positions else 0.0

                    is_long = cur_qty > 0
                    is_short = symbol in paper_port.short_positions and paper_port.short_positions[symbol].qty > 0

                    size = calculate_position_size(
                        equity_usd=equity,
                        price=px,
                        stop_distance_pct=signal.stop_distance_pct,
                        max_risk_per_trade=config.max_risk_per_trade,
                        max_allocation_pct=config.aggressive_allocation,
                        confidence_scale=conf_scale,
                    )

                    now_ms = int(time.time() * 1000)
                    ts_pct = eff["trailing_stop_pct"]
                    tp_pct = eff["take_profit_pct"]

                    # ── BUY action ────────────────────────────────────
                    if action == "buy":
                        if is_long:
                            logger.debug("HOLD %s — already long", symbol)
                        elif _cooldown_remaining > 0:
                            logger.debug("COOLDOWN %s — %d bars remaining", symbol, _cooldown_remaining)
                        elif is_short:
                            # Cover short first, then open long
                            cover_qty = paper_port.short_positions[symbol].qty
                            ok = paper_port.execute_cover(symbol, px, cover_qty)
                            if ok:
                                fee = cover_qty * px * config.fee_rate
                                close_paper_trade(conn, symbol, "short", px, fee, now_ms, close_reason="signal")
                                ORDERS.labels(side="cover", mode=mode, exchange=config.exchange_id).inc()
                                notifier.notify("cover", {"symbol": symbol, "price": px, "qty": cover_qty, "ml_up_prob": up_prob})
                                logger.info("COVER %s px=%.2f qty=%.6f", symbol, px, cover_qty)
                            # Now open long
                            if size <= 0:
                                continue
                            ORDERS.labels(side="buy", mode=mode, exchange=config.exchange_id).inc()
                            res = executor.buy(symbol, px, size, trailing_stop_pct=ts_pct, take_profit_pct=tp_pct)
                            if not res.ok:
                                ORDER_FAILS.labels(side="buy", mode=mode, exchange=config.exchange_id).inc()
                            else:
                                fee = size * px * config.fee_rate
                                open_paper_trade(conn, symbol, "long", px, size, fee, now_ms,
                                                 ml_up_prob=up_prob, strategy_conf=signal.confidence,
                                                 vol_ratio=signal.volume_ratio)
                                notifier.notify("buy", {"symbol": symbol, "price": px, "qty": size, "up_prob": up_prob})
                                logger.info("BUY %s px=%.2f qty=%.6f ml=%.3f equity=%.2f",
                                            symbol, px, size, up_prob or -1, equity)
                                if _explain_trade:
                                    try:
                                        fill_id = f"{symbol}_{now_ms}"
                                        sig_details = {
                                            "action": "long", "confidence": signal.confidence,
                                            "tech": {"action": "long", "confidence": signal.confidence},
                                            "macro": {"regime": regime_state.regime if regime_state else "unknown"},
                                        }
                                        _explain_trade(fill_id, symbol, "long", px, sig_details, conn=conn)
                                    except Exception:
                                        pass
                        else:
                            # No position — open long
                            if size <= 0:
                                continue
                            ORDERS.labels(side="buy", mode=mode, exchange=config.exchange_id).inc()
                            res = executor.buy(symbol, px, size, trailing_stop_pct=ts_pct, take_profit_pct=tp_pct)
                            if not res.ok:
                                ORDER_FAILS.labels(side="buy", mode=mode, exchange=config.exchange_id).inc()
                                notifier.notify("order_failed", {"side": "buy", "symbol": symbol, "error": res.error})
                            else:
                                fee = size * px * config.fee_rate
                                open_paper_trade(conn, symbol, "long", px, size, fee, now_ms,
                                                 ml_up_prob=up_prob, strategy_conf=signal.confidence,
                                                 vol_ratio=signal.volume_ratio)
                                notifier.notify("buy", {"symbol": symbol, "price": px, "qty": size, "up_prob": up_prob})
                                logger.info("BUY %s px=%.2f qty=%.6f ml=%.3f equity=%.2f",
                                            symbol, px, size, up_prob or -1, equity)
                                if _explain_trade:
                                    try:
                                        fill_id = f"{symbol}_{now_ms}"
                                        sig_details = {
                                            "action": "long", "confidence": signal.confidence,
                                            "tech": {"action": "long", "confidence": signal.confidence},
                                            "macro": {"regime": regime_state.regime if regime_state else "unknown"},
                                        }
                                        _explain_trade(fill_id, symbol, "long", px, sig_details, conn=conn)
                                    except Exception:
                                        pass

                    # ── SELL action ────────────────────────────────────
                    elif action == "sell":
                        if is_short:
                            logger.debug("HOLD %s — already short", symbol)
                        elif is_long:
                            # Capture entry price BEFORE the sell mutates position state
                            _pos = paper_port.positions.get(symbol)
                            _entry_px = _pos.avg_entry if _pos else px
                            sell_qty = cur_qty
                            ORDERS.labels(side="sell", mode=mode, exchange=config.exchange_id).inc()
                            res = executor.sell(symbol, px, sell_qty)
                            if not res.ok:
                                ORDER_FAILS.labels(side="sell", mode=mode, exchange=config.exchange_id).inc()
                                notifier.notify("order_failed", {"side": "sell", "symbol": symbol, "error": res.error})
                            else:
                                fee = sell_qty * px * config.fee_rate
                                close_paper_trade(conn, symbol, "long", px, fee, now_ms, close_reason="signal")
                                pnl_pct = (px - _entry_px) / _entry_px if _entry_px > 0 else 0.0
                                _label_buffer_from_trade(conn, symbol, now_ms, pnl_pct)
                                notifier.notify("sell", {"symbol": symbol, "price": px, "qty": sell_qty, "up_prob": up_prob})
                                logger.info("SELL %s px=%.2f qty=%.6f ml=%.3f equity=%.2f",
                                            symbol, px, sell_qty, up_prob or -1, equity)
                            # Optionally flip to short
                            if config.allow_shorts and size > 0:
                                ok_short = paper_port.execute_short(
                                    symbol, px, size,
                                    trailing_stop_pct=ts_pct, take_profit_pct=tp_pct,
                                )
                                if ok_short:
                                    fee = size * px * config.fee_rate
                                    open_paper_trade(conn, symbol, "short", px, size, fee, now_ms,
                                                     ml_up_prob=up_prob, strategy_conf=signal.confidence,
                                                     vol_ratio=signal.volume_ratio)
                                    ORDERS.labels(side="short", mode=mode, exchange=config.exchange_id).inc()
                                    notifier.notify("short", {"symbol": symbol, "price": px, "qty": size, "ml_up_prob": up_prob})
                                    logger.info("SHORT %s px=%.2f qty=%.6f", symbol, px, size)
                        elif config.allow_shorts:
                            # No position — open short
                            if _cooldown_remaining > 0:
                                logger.debug("COOLDOWN %s — %d bars remaining", symbol, _cooldown_remaining)
                                continue
                            ok_short = paper_port.execute_short(
                                symbol, px, size,
                                trailing_stop_pct=ts_pct, take_profit_pct=tp_pct,
                            )
                            if ok_short:
                                fee = size * px * config.fee_rate
                                open_paper_trade(conn, symbol, "short", px, size, fee, now_ms,
                                                 ml_up_prob=up_prob, strategy_conf=signal.confidence,
                                                 vol_ratio=signal.volume_ratio)
                                ORDERS.labels(side="short", mode=mode, exchange=config.exchange_id).inc()
                                notifier.notify("short", {"symbol": symbol, "price": px, "qty": size, "ml_up_prob": up_prob})
                                logger.info("SHORT %s px=%.2f qty=%.6f", symbol, px, size)
                        else:
                            logger.debug("HOLD %s — sell signal but no long position and shorts disabled", symbol)

                    # ── Online learning: buffer features at entry only (labeled at exit)
                    if action == "buy" and online_learner is not None:
                        try:
                            from hogan_bot.ml import build_feature_row
                            fv = build_feature_row(candles, db_conn=conn)
                            if fv is not None:
                                _write_training_buffer(conn, symbol, now_ms, fv)
                        except Exception:
                            pass

                    # ── Update signal cache for Discord ───────────────
                    signal_cache[symbol] = {
                        "action": action, "price": px,
                        "ml_up": up_prob if up_prob is not None else 0.0,
                        "conf": signal.confidence, "vol_ratio": signal.volume_ratio,
                        "regime": regime_state.regime if regime_state else "unknown",
                        "adx": regime_state.adx if regime_state else 0.0,
                        "is_long": symbol in paper_port.positions and paper_port.positions[symbol].qty > 0,
                        "is_short": symbol in paper_port.short_positions and paper_port.short_positions[symbol].qty > 0,
                    }
                    signal_cache["_mark_prices"] = mark_prices

                # ── Update Discord command listener state ─────────────
                if cmd_listener:
                    try:
                        _prices = signal_cache.get("_mark_prices", {})
                        cmd_listener.update_state(paper_port, _prices, config_summary, signal_cache)
                    except Exception:
                        pass

                # ── Periodic online learner update (Phase 4b) ─────────
                if online_learner is not None and loop % config.online_learning_interval == 0:
                    try:
                        for sym in config.symbols:
                            result = online_learner.update(symbol=sym)
                            if result.get("status") == "updated":
                                logger.info("ONLINE_LEARN symbol=%s rows=%d", sym, result.get("n_rows", 0))
                    except Exception as exc:
                        logger.debug("Online learner update failed: %s", exc)

            time.sleep(config.sleep_seconds)

        except Exception as exc:
            EXCEPTIONS.inc()
            logger.exception("Loop exception: %s", exc)
            notifier.notify("exception", {"error": str(exc)})
            time.sleep(min(30, config.sleep_seconds))


def main() -> None:
    ap = argparse.ArgumentParser(description="Hogan live/paper trading service")
    ap.add_argument("--max-loops", type=int, default=None)
    args = ap.parse_args()
    run_loop(max_loops=args.max_loops)


if __name__ == "__main__":
    main()
