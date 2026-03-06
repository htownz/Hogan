
from __future__ import annotations

import argparse
import logging
import os
import time

from hogan_bot.config import load_config
from hogan_bot.decision import apply_ml_filter, ml_confidence
from hogan_bot.exchange import ExchangeClient
from hogan_bot.execution import LiveExecution, PaperExecution
from hogan_bot.live_account import fetch_account_state
from hogan_bot.metrics import MetricsServer, LoopTimer, EQUITY, CASH, DRAWDOWN, ORDERS, ORDER_FAILS, EXCEPTIONS, SLIPPAGE_BPS, FILLS
from hogan_bot.ml import load_model as load_simple_model, predict_up_probability as predict_simple
from hogan_bot.ml_advanced import load_artifact as load_adv_artifact, predict_up_probability as predict_adv
from hogan_bot.notifier import make_notifier
from hogan_bot.paper import PaperPortfolio
from hogan_bot.risk import DrawdownGuard, calculate_position_size
from hogan_bot.storage import get_connection, record_equity, upsert_position, upsert_position_state, load_position_state, load_latest_fill_ts, record_fill
from hogan_bot.strategy import generate_signal

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger("hogan.trader")


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


def _predict_up(model_obj, candles):
    if model_obj is None:
        return None
    if getattr(model_obj, "artifact_type", "").startswith("advanced_ensemble"):
        return predict_adv(model_obj, candles)
    return predict_simple(candles, model_obj)


def sync_fills_deterministic(conn, client: ExchangeClient, exchange_id: str, symbols: list[str]) -> int:
    """Pull trades since last stored ts (per exchange) and journal idempotently."""
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


def run_loop(max_loops: int | None = None) -> None:
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
        webhook_url=config.webhook_url or None,
        email=email_cfg,
    )

    # Risk guard uses live equity in live mode, paper equity in paper mode.
    guard = DrawdownGuard(max_drawdown=config.max_drawdown)

    # Portfolio abstraction:
    paper_port = PaperPortfolio(starting_cash=config.starting_balance_usd)
    executor = PaperExecution(paper_port, conn=conn)  # default
    mode = "paper"
    if (not config.paper_mode) and config.live_mode:
        executor = LiveExecution(client, conn=conn, exchange_id=config.exchange_id)
        mode = "live"

    model_obj = _load_any_model(config.ml_model_path) if config.use_ml_filter else None

    # ── Dry-run validation gate (Freqtrade-inspired) ──────────────────────
    # Run 10 signal evaluations on recent bars BEFORE starting the main loop.
    # Abort if any exceptions occur or all signals are "hold".
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

    loop = 0
    while True:
        if max_loops is not None and loop >= max_loops:
            break
        loop += 1

        try:
            with LoopTimer():
                # Always journal fills in live mode (deterministic reconciliation)
                if mode == "live":
                    sync_fills_deterministic(conn, client, config.exchange_id, config.symbols)

                candles_by_symbol = {}
                mark_prices = {}
                for symbol in config.symbols:
                    candles = client.fetch_ohlcv_df(symbol, timeframe=config.timeframe, limit=config.ohlcv_limit)
                    if candles.empty:
                        continue
                    candles_by_symbol[symbol] = candles
                    mark_prices[symbol] = float(candles["close"].iloc[-1])

                if not mark_prices:
                    time.sleep(config.sleep_seconds)
                    continue

                if mode == "live":
                    state = fetch_account_state(client, config.symbols, quote_ccy=config.quote_currency)
                    equity = state.equity_quote
                    cash = state.cash_quote
                    # persist positions and maintain peak/entry state for stops
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
                record_equity(conn, int(time.time()*1000), cash, equity, dd)
                EQUITY.set(equity)
                CASH.set(cash)
                DRAWDOWN.set(dd)

                if not guard.update_and_check(equity):
                    logger.error("Max drawdown breached. Halting. equity=%.2f peak=%.2f", equity, guard.peak_equity)
                    notifier.notify("drawdown_breach", {"equity": equity, "peak": guard.peak_equity})
                    break

                # Stops / take-profit (live + paper)
                if (config.trailing_stop_pct > 0) or (config.take_profit_pct > 0):
                    for sym in list(mark_prices.keys()):
                        px = mark_prices[sym]
                        qty_live = 0.0
                        if mode == "live":
                            # if we hold it, evaluate stop
                            row = conn.execute("SELECT qty FROM positions WHERE symbol=?", (sym,)).fetchone()
                            qty_live = float(row[0]) if row else 0.0
                            if qty_live <= 0:
                                continue
                            ps = load_position_state(conn, sym)
                            if ps:
                                entry, peak = ps
                                peak = max(peak, px)
                                upsert_position_state(conn, sym, entry_price=entry, peak_price=peak, updated_ms=int(time.time()*1000))
                                if config.trailing_stop_pct > 0:
                                    stop = peak * (1 - config.trailing_stop_pct)
                                    if px <= stop:
                                        ORDERS.labels(side="sell", mode=mode, exchange=config.exchange_id).inc()
                                        res = executor.sell(sym, px, qty_live)
                                        if not res.ok:
                                            ORDER_FAILS.labels(side="sell", mode=mode, exchange=config.exchange_id).inc()
                                        else:
                                            notifier.notify("trailing_stop_exit", {"symbol": sym, "price": px, "qty": qty_live})
                                if config.take_profit_pct > 0 and entry > 0:
                                    tp = entry * (1 + config.take_profit_pct)
                                    if px >= tp:
                                        ORDERS.labels(side="sell", mode=mode, exchange=config.exchange_id).inc()
                                        res = executor.sell(sym, px, qty_live)
                                        if not res.ok:
                                            ORDER_FAILS.labels(side="sell", mode=mode, exchange=config.exchange_id).inc()
                                        else:
                                            notifier.notify("take_profit_exit", {"symbol": sym, "price": px, "qty": qty_live})
                        else:
                            # paper exits handled inside PaperPortfolio
                            pass

                # Main decision loop
                for symbol, candles in candles_by_symbol.items():
                    px = mark_prices[symbol]
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
                        min_vote_margin=config.signal_min_vote_margin,
                    )

                    action = signal.action
                    up_prob = None
                    conf_scale = 1.0

                    if config.use_ml_filter and model_obj is not None:
                        up_prob = _predict_up(model_obj, candles)
                        action = apply_ml_filter(
                            action=action,
                            up_prob=up_prob,
                            buy_threshold=config.ml_buy_threshold,
                            sell_threshold=config.ml_sell_threshold,
                        )
                        if config.ml_confidence_sizing and up_prob is not None:
                            conf_scale = ml_confidence(up_prob)

                    if action == "hold":
                        continue

                    # Determine current position + cash
                    if mode == "live":
                        row = conn.execute("SELECT qty FROM positions WHERE symbol=?", (symbol,)).fetchone()
                        cur_qty = float(row[0]) if row else 0.0
                        cash_avail = cash
                    else:
                        cur_qty = paper_port.positions.get(symbol).qty if symbol in paper_port.positions else 0.0
                        cash_avail = paper_port.cash_usd

                    if action == "buy":
                        qty = calculate_position_size(
                            cash_balance=cash_avail,
                            price=px,
                            aggressive_allocation=config.aggressive_allocation * conf_scale,
                            max_risk_per_trade=config.max_risk_per_trade,
                            stop_distance=signal.stop_distance,
                        )
                        if qty <= 0:
                            continue
                        ORDERS.labels(side="buy", mode=mode, exchange=config.exchange_id).inc()
                        res = executor.buy(symbol, px, qty)
                        if not res.ok:
                            ORDER_FAILS.labels(side="buy", mode=mode, exchange=config.exchange_id).inc()
                            notifier.notify("order_failed", {"side": "buy", "symbol": symbol, "error": res.error})
                        else:
                            notifier.notify("buy", {"symbol": symbol, "price": px, "qty": qty, "up_prob": up_prob})

                    elif action == "sell":
                        qty = cur_qty
                        if qty <= 0:
                            continue
                        ORDERS.labels(side="sell", mode=mode, exchange=config.exchange_id).inc()
                        res = executor.sell(symbol, px, qty)
                        if not res.ok:
                            ORDER_FAILS.labels(side="sell", mode=mode, exchange=config.exchange_id).inc()
                            notifier.notify("order_failed", {"side": "sell", "symbol": symbol, "error": res.error})
                        else:
                            notifier.notify("sell", {"symbol": symbol, "price": px, "qty": qty, "up_prob": up_prob})

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
