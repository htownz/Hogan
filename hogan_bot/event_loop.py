"""Async event loop — Phase 2b.

Replaces the blocking ``while True`` in ``main.py`` with an ``asyncio``
pipeline:

    LiveDataEngine (WebSocket candles)
        → CandleEvent queue
        → SignalEvaluator (strategy + ML)
        → RiskManager (DrawdownGuard + position sizing)
        → ExecutionEngine (PaperExecution | LiveExecution)
        → SQLite journal + Prometheus metrics

Each symbol's signal evaluation is wrapped in a per-symbol try/except so a
single symbol failure never aborts the whole loop (Phase 7 hardening).

Run directly::

    python -m hogan_bot.event_loop

Or import::

    from hogan_bot.event_loop import run_event_loop
    asyncio.run(run_event_loop())
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import os
import time
from collections import defaultdict, deque
from datetime import datetime

import pandas as pd

from hogan_bot.config import BotConfig, load_config
from hogan_bot.data_engine import CandleEvent, LiveDataEngine, CandleRingBuffer
from hogan_bot.decision import apply_ml_filter, ml_confidence
from hogan_bot.execution import PaperExecution, LiveExecution
from hogan_bot.ml import TrainedModel, load_model, predict_up_probability
from hogan_bot.notifier import make_notifier
from hogan_bot.paper import PaperPortfolio
from hogan_bot.risk import DrawdownGuard, calculate_position_size
from hogan_bot.storage import get_connection, record_equity, upsert_position, open_paper_trade, close_paper_trade
from hogan_bot.strategy import generate_signal

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal evaluator — stateless, one call per candle event
# ---------------------------------------------------------------------------
class SignalEvaluator:
    """Wraps strategy + ML gate; returns (action, size, up_prob)."""

    def __init__(self, config: BotConfig, ml_model: TrainedModel | None) -> None:
        self.config = config
        self.ml_model = ml_model

    def evaluate(
        self,
        symbol: str,
        candles: pd.DataFrame,
        equity: float,
    ) -> tuple[str, float, float | None]:
        cfg = self.config
        signal = generate_signal(
            candles,
            short_window=cfg.short_ma_window,
            long_window=cfg.long_ma_window,
            volume_window=cfg.volume_window,
            volume_threshold=cfg.volume_threshold,
            use_ema_clouds=cfg.use_ema_clouds,
            ema_fast_short=cfg.ema_fast_short,
            ema_fast_long=cfg.ema_fast_long,
            ema_slow_short=cfg.ema_slow_short,
            ema_slow_long=cfg.ema_slow_long,
            use_fvg=cfg.use_fvg,
            fvg_min_gap_pct=cfg.fvg_min_gap_pct,
            signal_mode=cfg.signal_mode,
            min_vote_margin=cfg.signal_min_vote_margin,
        )
        px = float(candles["close"].iloc[-1])

        up_prob = None
        conf_scale = 1.0
        action = signal.action

        if cfg.use_ml_filter and self.ml_model is not None:
            up_prob = predict_up_probability(candles, self.ml_model)
            action = apply_ml_filter(action, up_prob, cfg.ml_buy_threshold, cfg.ml_sell_threshold)
            if cfg.ml_confidence_sizing:
                conf_scale = ml_confidence(up_prob)

        size = calculate_position_size(
            equity_usd=equity,
            price=px,
            stop_distance_pct=signal.stop_distance_pct,
            max_risk_per_trade=cfg.max_risk_per_trade,
            max_allocation_pct=cfg.aggressive_allocation,
            confidence_scale=conf_scale,
        )
        return action, size, up_prob


# ---------------------------------------------------------------------------
# Main async event loop
# ---------------------------------------------------------------------------
async def run_event_loop(
    config: BotConfig | None = None,
    max_events: int | None = None,
) -> None:
    if config is None:
        config = load_config()

    conn = get_connection(config.db_path)
    notifier = make_notifier(config.webhook_url or None)

    live_ack = (os.getenv("HOGAN_LIVE_ACK", "") or "").strip()
    allow_live = (
        (not config.paper_mode)
        and config.live_mode
        and live_ack == "I_UNDERSTAND_LIVE_TRADING"
    )

    portfolio = PaperPortfolio(cash_usd=config.starting_balance_usd, fee_rate=config.fee_rate)
    guard = DrawdownGuard(config.starting_balance_usd, config.max_drawdown)

    if allow_live:
        from hogan_bot.exchange import ExchangeClient
        client = ExchangeClient(config.exchange_id, config.kraken_api_key, config.kraken_api_secret)
        executor = LiveExecution(client=client, conn=conn, exchange_id=config.exchange_id)
        logger.warning("LIVE TRADING ENABLED on exchange=%s", config.exchange_id)
    else:
        executor = PaperExecution(portfolio=portfolio, conn=conn, exchange_id="paper")

    ml_model: TrainedModel | None = None
    if config.use_ml_filter:
        try:
            ml_model = load_model(config.ml_model_path)
            logger.info("Loaded ML model from %s", config.ml_model_path)
        except FileNotFoundError:
            logger.warning("ML model not found at %s; running without ML filter.", config.ml_model_path)
        except Exception as exc:
            logger.warning("ML model load error: %s", exc)

    evaluator = SignalEvaluator(config, ml_model)
    buffer = CandleRingBuffer(maxlen=config.ohlcv_limit)

    try:
        from hogan_bot.metrics import MetricsServer, LoopTimer, EQUITY, CASH, DRAWDOWN, EXCEPTIONS
        metrics_server = MetricsServer(port=getattr(config, "metrics_port", 8000))
        metrics_server.start()
        _has_metrics = True
    except Exception:
        _has_metrics = False

    engine = LiveDataEngine(
        exchange_id=config.exchange_id,
        api_key=config.kraken_api_key or "",
        api_secret=config.kraken_api_secret or "",
        symbols=config.symbols,
        timeframes=[config.timeframe],
        ring_buffer_len=config.ohlcv_limit,
    )

    event_count = 0
    _signal_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    logger.info(
        "Event loop starting: mode=%s symbols=%s timeframe=%s",
        "LIVE" if allow_live else "PAPER",
        ",".join(config.symbols),
        config.timeframe,
    )

    async with engine:
        async for event in engine.stream():
            if max_events is not None and event_count >= max_events:
                break

            symbol = event.symbol
            tf = event.timeframe

            # Only process the primary timeframe for signals
            if tf != config.timeframe:
                buffer.push(symbol, tf, event.candle.to_dict())
                continue

            buffer.push(symbol, tf, event.candle.to_dict())
            candles = buffer.to_df(symbol, tf)

            if candles.empty or len(candles) < max(config.long_ma_window, 20):
                event_count += 1
                continue

            mark_prices = {s: float(buffer.to_df(s, tf)["close"].iloc[-1])
                           for s in config.symbols
                           if not buffer.to_df(s, tf).empty}
            if not mark_prices:
                event_count += 1
                continue

            equity = portfolio.total_equity(mark_prices)
            px = mark_prices.get(symbol, 0.0)

            # Record equity
            try:
                dd = 0.0 if guard.peak_equity <= 0 else max(
                    0.0, (guard.peak_equity - equity) / guard.peak_equity
                )
                record_equity(conn, int(time.time() * 1000), portfolio.cash_usd, equity, dd)
                if _has_metrics:
                    EQUITY.set(equity)
                    CASH.set(portfolio.cash_usd)
                    DRAWDOWN.set(dd)
            except Exception as exc:
                logger.debug("Equity record error: %s", exc)

            # DrawdownGuard
            if not guard.update_and_check(equity):
                logger.error("Drawdown limit hit: equity=%.2f peak=%.2f", equity, guard.peak_equity)
                if notifier:
                    notifier.notify("drawdown_breach", {"equity": equity, "peak": guard.peak_equity})
                break

            # Auto-exit trailing stops / take profits
            exits = portfolio.check_exits(mark_prices)
            for exit_symbol, reason in exits:
                pos = portfolio.positions.get(exit_symbol)
                if pos is None:
                    continue
                ep = mark_prices.get(exit_symbol, 0.0)
                executed = portfolio.execute_sell(exit_symbol, ep, pos.qty)
                if notifier and executed:
                    notifier.notify("auto_exit", {"symbol": exit_symbol, "reason": reason,
                                                  "price": ep, "qty": pos.qty})
                logger.info("AUTO_EXIT %s reason=%s px=%.2f", exit_symbol, reason, ep)

            # Dead-man's switch check
            stale = engine.check_dead_man()
            if stale:
                msg = f"No candles received for >15min: {stale}"
                logger.warning(msg)
                if notifier:
                    notifier.notify("dead_man_switch", {"stale_symbols": stale})
                try:
                    from hogan_bot.metrics import DEAD_MAN_ALERTS
                    DEAD_MAN_ALERTS.inc()
                except Exception:
                    pass

            # Per-symbol signal evaluation (isolated — one symbol failure won't break others)
            try:
                action, size, up_prob = evaluator.evaluate(symbol, candles, equity)
            except Exception as exc:
                logger.error("Signal eval error for %s: %s", symbol, exc)
                if _has_metrics:
                    EXCEPTIONS.inc()
                event_count += 1
                continue

            _signal_counts[symbol][action] += 1

            if action == "buy" and px > 0:
                if executor:
                    res = executor.buy(symbol, px, size)
                    executed = bool(res.ok)
                    if executed:
                        portfolio.execute_buy(
                            symbol, px, size,
                            trailing_stop_pct=config.trailing_stop_pct,
                            take_profit_pct=config.take_profit_pct,
                        )
                else:
                    executed = portfolio.execute_buy(
                        symbol, px, size,
                        trailing_stop_pct=config.trailing_stop_pct,
                        take_profit_pct=config.take_profit_pct,
                    )
                if executed:
                    now_ms = int(time.time() * 1000)
                    fee = size * px * config.fee_rate
                    open_paper_trade(conn, symbol, "buy", px, size, fee, now_ms,
                                     ml_up_prob=up_prob, strategy_conf=0.0,
                                     vol_ratio=0.0)
                    if notifier:
                        notifier.notify("buy", {"symbol": symbol, "price": px,
                                                "qty": size, "ml_up_prob": up_prob})
                logger.info("BUY %s px=%.2f qty=%.6f ml=%.3f equity=%.2f",
                            symbol, px, size, up_prob or -1, equity)

            elif action == "sell" and symbol in portfolio.positions:
                pos = portfolio.positions[symbol]
                sell_qty = min(pos.qty, size)
                if executor:
                    res = executor.sell(symbol, px, sell_qty)
                    executed = bool(res.ok)
                    if executed:
                        portfolio.execute_sell(symbol, px, sell_qty)
                else:
                    executed = portfolio.execute_sell(symbol, px, sell_qty)
                if executed:
                    now_ms = int(time.time() * 1000)
                    exit_fee = sell_qty * px * config.fee_rate
                    close_paper_trade(conn, symbol, "buy", px, exit_fee, now_ms, close_reason="signal")
                    if notifier:
                        notifier.notify("sell", {"symbol": symbol, "price": px,
                                                 "qty": sell_qty, "ml_up_prob": up_prob})
                logger.info("SELL %s px=%.2f qty=%.6f ml=%.3f equity=%.2f",
                            symbol, px, sell_qty, up_prob or -1, equity)
            else:
                logger.debug("HOLD %s px=%.2f ml=%.3f equity=%.2f",
                             symbol, px, up_prob or -1, equity)

            # Update signal quality metric
            try:
                from hogan_bot.metrics import SIGNAL_QUALITY
                counts = _signal_counts[symbol]
                total = sum(counts.values())
                non_hold = counts.get("buy", 0) + counts.get("sell", 0)
                SIGNAL_QUALITY.labels(symbol=symbol).set(non_hold / max(total, 1))
            except Exception:
                pass

            event_count += 1

    logger.info("Event loop terminated after %d events.", event_count)
    conn.close()


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------
def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Hogan async event loop")
    p.add_argument("--max-events", type=int, default=None)
    return p.parse_args()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = _parse_args()
    asyncio.run(run_event_loop(max_events=args.max_events))
