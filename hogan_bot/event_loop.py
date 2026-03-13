"""Async event loop — Phase 2b.

Replaces the blocking ``while True`` in ``main.py`` with an ``asyncio``
pipeline:

    LiveDataEngine (WebSocket candles)
        -> CandleEvent queue
        -> SignalEvaluator (AgentPipeline + ML filter)
        -> RiskManager (DrawdownGuard + position sizing)
        -> ExecutionEngine (PaperExecution | LiveExecution)
        -> SQLite journal + Prometheus metrics

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
from dataclasses import dataclass
from datetime import datetime

import pandas as pd

from hogan_bot.agent_pipeline import AgentPipeline
from hogan_bot.config import BotConfig, load_config, symbol_config, effective_hold_cooldown_bars
from hogan_bot.data_engine import CandleEvent, LiveDataEngine, CandleRingBuffer
from hogan_bot.decision import (
    apply_ml_filter, edge_gate, entry_quality_gate, ml_confidence,
    estimate_spread_from_candles,
)
from hogan_bot.execution import (
    PaperExecution, LiveExecution, SmartExecution, SmartExecConfig,
    RealisticPaperExecution, FillSimConfig,
)
from hogan_bot.ml import TrainedModel, load_model, predict_up_probability
from hogan_bot.notifier import make_notifier
from hogan_bot.paper import PaperPortfolio
from hogan_bot.risk import DrawdownGuard, calculate_position_size
from hogan_bot.expectancy import ExpectancyTracker
from hogan_bot.storage import get_connection, record_equity, upsert_position, open_paper_trade, close_paper_trade, log_decision

logger = logging.getLogger(__name__)


@dataclass
class SignalResult:
    """Rich signal evaluation output for decision logging."""
    action: str
    size: float
    up_prob: float | None
    regime: str | None
    atr_pct: float
    final_confidence: float = 0.0
    tech_confidence: float | None = None
    conf_scale: float = 1.0
    vol_ratio: float = 1.0
    explanation: str | None = None
    forecast_ret: float | None = None
    agent_weights: dict | None = None
    feature_freshness: dict | None = None


# ---------------------------------------------------------------------------
# Signal evaluator — stateless, one call per candle event
# ---------------------------------------------------------------------------
class SignalEvaluator:
    """Routes decisions through AgentPipeline — Technical + Sentiment + Macro."""

    def __init__(self, config: BotConfig, ml_model: TrainedModel | None, conn=None) -> None:
        self.config = config
        self.ml_model = ml_model
        self.pipeline = AgentPipeline(config, conn=conn)

    def evaluate(
        self,
        symbol: str,
        candles: pd.DataFrame,
        equity: float,
        recent_whipsaw_count: int = 0,
    ) -> SignalResult:
        """Returns a SignalResult with action, sizing, and rich metadata."""
        cfg = symbol_config(self.config, symbol)

        regime_name = None
        regime_conf = None
        _rstate = None
        if getattr(cfg, "use_regime_detection", True):
            try:
                from hogan_bot.regime import detect_regime
                _rstate = detect_regime(candles)
                regime_name = _rstate.regime
                regime_conf = _rstate.confidence
            except Exception:
                pass

        result = self.pipeline.run(
            candles, symbol=symbol, config_override=cfg,
            regime=regime_name, regime_state=_rstate,
        )
        px = float(candles["close"].iloc[-1])

        up_prob = None
        conf_scale = result.confidence or 1.0
        action = result.action

        if cfg.use_ml_filter and self.ml_model is not None:
            up_prob = predict_up_probability(candles, self.ml_model)
            action = apply_ml_filter(action, up_prob, cfg.ml_buy_threshold, cfg.ml_sell_threshold)
            if cfg.ml_confidence_sizing:
                conf_scale = ml_confidence(up_prob) * (result.confidence or 1.0)

        # Fee-aware edge gate
        forecast_ret = None
        if result.forecast is not None and result.forecast.confidence > 0.2:
            er = result.forecast.expected_return
            if isinstance(er, dict) and er:
                forecast_ret = max(abs(v) for v in er.values())
            elif isinstance(er, (int, float)):
                forecast_ret = abs(float(er))
        atr_pct = result.stop_distance_pct / max(getattr(cfg, "atr_stop_multiplier", 2.5), 1.0)
        spread_est = estimate_spread_from_candles(candles)
        action = edge_gate(
            action,
            atr_pct=atr_pct,
            take_profit_pct=cfg.take_profit_pct,
            fee_rate=cfg.fee_rate,
            min_edge_multiple=getattr(cfg, "min_edge_multiple", 1.5),
            forecast_expected_return=forecast_ret,
            estimated_spread=spread_est,
        )

        # Hard entry quality gate (thresholds from config)
        tech_conf = result.tech.confidence if result.tech else None
        action, quality_scale = entry_quality_gate(
            action,
            final_confidence=result.confidence,
            tech_confidence=tech_conf,
            regime=regime_name,
            regime_confidence=regime_conf,
            recent_whipsaw_count=recent_whipsaw_count,
            min_final_confidence=cfg.min_final_confidence,
            min_tech_confidence=cfg.min_tech_confidence,
            min_regime_confidence=cfg.min_regime_confidence,
            max_whipsaws=cfg.max_whipsaws,
        )

        size = calculate_position_size(
            equity_usd=equity,
            price=px,
            stop_distance_pct=result.stop_distance_pct,
            max_risk_per_trade=cfg.max_risk_per_trade,
            max_allocation_pct=cfg.aggressive_allocation,
            confidence_scale=conf_scale * quality_scale,
            fee_rate=cfg.fee_rate,
        )

        # Feature staleness check (observability, does not block trades)
        _freshness: dict | None = None
        try:
            from hogan_bot.ml import build_feature_row_checked
            feat_result = build_feature_row_checked(candles, db_conn=self.pipeline.conn)
            if feat_result is not None and feat_result.has_stale:
                _freshness = feat_result.freshness_summary
                logger.warning("STALE_FEATURES %s: %s", symbol, feat_result.stale_features)
        except Exception:
            pass

        if action != "hold":
            logger.info(
                "PIPELINE %s action=%s conf=%.2f tech_conf=%.2f regime=%s | %s",
                symbol, action, result.confidence,
                tech_conf or 0.0, regime_name or "unknown",
                result.explanation,
            )

        return SignalResult(
            action=action,
            size=size,
            up_prob=up_prob,
            regime=regime_name,
            atr_pct=atr_pct,
            final_confidence=result.confidence or 0.0,
            tech_confidence=tech_conf,
            conf_scale=conf_scale * quality_scale,
            vol_ratio=result.volume_ratio,
            explanation=result.explanation,
            forecast_ret=forecast_ret,
            agent_weights=result.agent_weights,
            feature_freshness=_freshness,
        )


# ---------------------------------------------------------------------------
# Main async event loop
# ---------------------------------------------------------------------------
async def run_event_loop(
    config: BotConfig | None = None,
    max_events: int | None = None,
) -> None:
    if config is None:
        config = load_config()

    from hogan_bot.champion import apply_champion_mode
    config = apply_champion_mode(config)

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

    use_oanda = (
        config.exchange_id.lower() == "oanda"
        or os.getenv("HOGAN_USE_OANDA", "").strip().lower() in ("1", "true", "yes")
    )
    _oanda_client = None
    if use_oanda:
        from hogan_bot.oanda_client import OandaClient
        _oanda_client = OandaClient()

    if allow_live:
        if use_oanda:
            from hogan_bot.oanda_execution import OandaExecution
            executor = OandaExecution(
                client=_oanda_client, conn=conn, portfolio=portfolio,
            )
            logger.warning("LIVE OANDA EXECUTION on account=%s env=%s", _oanda_client.account_id, _oanda_client.environment)
        else:
            from hogan_bot.exchange import ExchangeClient
            client = ExchangeClient(config.exchange_id, config.kraken_api_key, config.kraken_api_secret)
            use_smart = os.getenv("HOGAN_SMART_EXEC", "").strip().lower() in ("1", "true", "yes")
            if use_smart:
                smart_cfg = SmartExecConfig(
                    max_reprices=int(os.getenv("HOGAN_EXEC_MAX_REPRICES", "2")),
                    post_only=True,
                    chase_bps=float(os.getenv("HOGAN_EXEC_CHASE_BPS", "2.0")),
                )
                executor = SmartExecution(
                    client=client, conn=conn, exchange_id=config.exchange_id,
                    portfolio=portfolio, config=smart_cfg,
                )
                logger.warning("LIVE SMART EXECUTION on exchange=%s (post-only, %d reprices)", config.exchange_id, smart_cfg.max_reprices)
            else:
                executor = LiveExecution(client=client, conn=conn, exchange_id=config.exchange_id)
                logger.warning("LIVE TRADING ENABLED on exchange=%s", config.exchange_id)
    else:
        use_realistic = os.getenv("HOGAN_REALISTIC_PAPER", "1").strip().lower() in ("1", "true", "yes")
        if use_realistic:
            fill_cfg = FillSimConfig(
                slippage_bps=float(os.getenv("HOGAN_SLIPPAGE_BPS", "5.0")),
                spread_half_bps=float(os.getenv("HOGAN_SPREAD_HALF_BPS", "3.0")),
            )
            executor = RealisticPaperExecution(
                portfolio=portfolio, conn=conn, config=fill_cfg,
            )
            logger.info("Paper mode with realistic fills (slip=%.1fbps spread=%.1fbps)", fill_cfg.slippage_bps, fill_cfg.spread_half_bps)
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

    evaluator = SignalEvaluator(config, ml_model, conn=conn)
    from hogan_bot.exit_model import ExitEvaluator
    _exit_eval = ExitEvaluator()
    _expectancy = ExpectancyTracker()
    buffer = CandleRingBuffer(maxlen=config.ohlcv_limit)

    # Parity with backtest: max_hold_bars, loss_cooldown, slippage
    max_hold_bars, loss_cooldown_bars = effective_hold_cooldown_bars(config, config.timeframe)
    slippage_bps = float(os.getenv("HOGAN_SLIPPAGE_BPS", "5.0"))
    slip_mult = slippage_bps / 10_000.0
    # When the executor owns fill simulation (RealisticPaperExecution), skip
    # the pre-adjustment to avoid double-counting friction.
    _executor_owns_fill = isinstance(executor, RealisticPaperExecution)
    _cooldown_remaining: int = 0
    _consecutive_exit_signals: dict[str, int] = defaultdict(int)
    _whipsaw_counts: dict[str, int] = defaultdict(int)
    _last_action: dict[str, str] = {}
    min_hold_bars = getattr(config, "min_hold_bars", 3)
    exit_confirm_bars = getattr(config, "exit_confirmation_bars", 2)

    try:
        from hogan_bot.metrics import MetricsServer, LoopTimer, EQUITY, CASH, DRAWDOWN, EXCEPTIONS
        metrics_server = MetricsServer(port=getattr(config, "metrics_port", 8000))
        metrics_server.start()
        _has_metrics = True
    except Exception:
        _has_metrics = False

    if use_oanda and _oanda_client is not None:
        from hogan_bot.data_engine import OandaDataEngine
        engine = OandaDataEngine(
            client=_oanda_client,
            symbols=config.symbols,
            timeframes=[config.timeframe],
            ring_buffer_len=config.ohlcv_limit,
        )
        logger.info("Using OandaDataEngine for %s", config.symbols)
    else:
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
    _current_regime: dict[str, str] = {}
    _EXPECTANCY_LOG_INTERVAL = 50

    # FX session filter (active when any symbol looks like an FX pair)
    _fx_session_filter = None
    _has_fx_symbols = any("/" in s and not any(c in s for c in ("BTC", "ETH", "SOL", "DOGE"))
                          for s in config.symbols)
    if _has_fx_symbols:
        try:
            from hogan_bot.fx_utils import SessionFilter
            _fx_session_filter = SessionFilter()
            logger.info("FX session filter enabled for symbols: %s", config.symbols)
        except ImportError:
            pass

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

            # Decrement loss cooldown each iteration
            if _cooldown_remaining > 0:
                _cooldown_remaining -= 1
            # Decay whipsaw counts gradually (halve every 20 bars)
            if event_count > 0 and event_count % 20 == 0:
                for ws_sym in list(_whipsaw_counts):
                    _whipsaw_counts[ws_sym] = max(0, _whipsaw_counts[ws_sym] - 1)

            # Auto-exit trailing stops / take profits / max_hold_time
            # All exits go through the executor so live orders are always sent.
            exits = portfolio.check_exits(mark_prices, max_hold_bars=max_hold_bars)
            for exit_symbol, reason in exits:
                ep = mark_prices.get(exit_symbol, 0.0)
                now_ms = int(time.time() * 1000)

                if reason in ("trailing_stop", "take_profit", "max_hold_time"):
                    pos = portfolio.positions.get(exit_symbol)
                    if pos is None:
                        continue
                    qty = pos.qty
                    avg_entry = pos.avg_entry
                    bars_held = getattr(pos, "bars_held", 0)
                    mae_pct = getattr(pos, "max_adverse_pct", 0.0)
                    mfe_pct = getattr(pos, "max_favorable_pct", 0.0)
                    res = executor.exit_long(exit_symbol, ep, qty, reason=reason)
                    executed = bool(res.ok)
                    sell_px = ep
                    if executed:
                        fee = qty * sell_px * config.fee_rate
                        if not allow_live:
                            close_paper_trade(conn, exit_symbol, "long", sell_px, fee, now_ms, close_reason=reason)
                        gross_pnl_pct = (sell_px - avg_entry) / avg_entry if avg_entry else 0
                        net_pnl_pct = gross_pnl_pct - 2 * config.fee_rate
                        _expectancy.record_trade(
                            symbol=exit_symbol,
                            regime=_current_regime.get(exit_symbol, "unknown"),
                            gross_pnl_pct=gross_pnl_pct,
                            net_pnl_pct=net_pnl_pct,
                            mae_pct=mae_pct,
                            mfe_pct=mfe_pct,
                            hold_bars=bars_held,
                            close_reason=reason,
                        )
                        is_loss = sell_px < avg_entry
                        if is_loss and loss_cooldown_bars > 0:
                            _cooldown_remaining = loss_cooldown_bars
                    if notifier and executed:
                        notifier.notify("auto_exit", {"symbol": exit_symbol, "reason": reason,
                                                      "price": sell_px, "qty": qty})
                    logger.info("AUTO_EXIT %s reason=%s px=%.2f qty=%.6f", exit_symbol, reason, sell_px, qty)

                elif reason in ("short_trailing_stop", "short_take_profit"):
                    pos = portfolio.short_positions.get(exit_symbol)
                    if pos is None:
                        continue
                    qty = pos.qty
                    avg_entry = pos.avg_entry
                    bars_held = getattr(pos, "bars_held", 0)
                    mae_pct = getattr(pos, "max_adverse_pct", 0.0)
                    mfe_pct = getattr(pos, "max_favorable_pct", 0.0)
                    cover_px = ep
                    res = executor.exit_short(exit_symbol, cover_px, qty, reason=reason)
                    executed = bool(res.ok)
                    if executed:
                        fee = qty * cover_px * config.fee_rate
                        if not allow_live:
                            close_paper_trade(conn, exit_symbol, "short", cover_px, fee, now_ms, close_reason=reason)
                        gross_pnl_pct = (avg_entry - cover_px) / avg_entry if avg_entry else 0
                        net_pnl_pct = gross_pnl_pct - 2 * config.fee_rate
                        _expectancy.record_trade(
                            symbol=exit_symbol,
                            regime=_current_regime.get(exit_symbol, "unknown"),
                            gross_pnl_pct=gross_pnl_pct,
                            net_pnl_pct=net_pnl_pct,
                            mae_pct=mae_pct,
                            mfe_pct=mfe_pct,
                            hold_bars=bars_held,
                            close_reason=reason,
                        )
                    if notifier and executed:
                        notifier.notify("auto_exit", {"symbol": exit_symbol, "reason": reason,
                                                      "price": cover_px, "qty": qty})
                    logger.info("AUTO_EXIT_SHORT %s reason=%s px=%.2f qty=%.6f", exit_symbol, reason, cover_px, qty)

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

            # FX session filter: block trading during off-hours/weekends
            if _fx_session_filter is not None:
                _fx_allowed, _fx_scale, _fx_reason = _fx_session_filter.should_trade()
                if not _fx_allowed:
                    logger.debug("FX_SESSION blocked: %s", _fx_reason)
                    event_count += 1
                    continue

            # Per-symbol signal evaluation (isolated — one symbol failure won't break others)
            try:
                sig = evaluator.evaluate(symbol, candles, equity,
                                         recent_whipsaw_count=_whipsaw_counts.get(symbol, 0))
                action = sig.action
                size = sig.size
                up_prob = sig.up_prob
                _sym_regime = sig.regime
                _sym_atr_pct = sig.atr_pct
                if _sym_regime:
                    _current_regime[symbol] = _sym_regime
            except Exception as exc:
                logger.error("Signal eval error for %s: %s", symbol, exc)
                if _has_metrics:
                    EXCEPTIONS.inc()
                event_count += 1
                continue

            _signal_counts[symbol][action] += 1

            # Whipsaw tracking: count rapid direction reversals
            prev = _last_action.get(symbol, "hold")
            if action != "hold":
                if prev != "hold" and prev != action:
                    _whipsaw_counts[symbol] = _whipsaw_counts.get(symbol, 0) + 1
                _last_action[symbol] = action

            # Decision logging: persist every signal evaluation to decision_log
            try:
                log_decision(
                    conn,
                    ts_ms=int(time.time() * 1000),
                    symbol=symbol,
                    regime=_sym_regime,
                    tech_action=action,
                    tech_confidence=sig.tech_confidence,
                    final_action=action,
                    final_confidence=sig.final_confidence,
                    position_size=size,
                    ml_up_prob=up_prob,
                    conf_scale=sig.conf_scale,
                    explanation=sig.explanation,
                    meta_weights=sig.agent_weights,
                    freshness=sig.feature_freshness,
                )
            except Exception as exc:
                logger.debug("Decision log error: %s", exc)

            # Track consecutive exit signals for confirmation
            if action == "sell" and symbol in portfolio.positions:
                _consecutive_exit_signals[symbol] += 1
            else:
                _consecutive_exit_signals[symbol] = 0

            # Instrument-profile-aware stop/TP (FX uses pip-based, crypto uses %)
            _eff_stop = config.trailing_stop_pct
            _eff_tp = config.take_profit_pct
            try:
                from hogan_bot.instrument_profiles import get_profile
                _iprofile = get_profile(symbol)
                if _iprofile.use_pip_based_risk:
                    _eff_stop = _iprofile.default_stop_pct
                    _eff_tp = _iprofile.default_tp_pct
            except Exception:
                pass

            # FX session filter: block new entries during off-hours
            _fx_session_ok = True
            try:
                from hogan_bot.instrument_profiles import classify_symbol as _cls_sym
                if _cls_sym(symbol) in ("fx_major", "fx_cross"):
                    from hogan_bot.fx_utils import is_fx_weekend, session_filter
                    if is_fx_weekend():
                        _fx_session_ok = False
                    elif not session_filter():
                        _fx_session_ok = False
            except Exception:
                pass

            if action == "buy" and px > 0:
                if symbol in portfolio.positions:
                    pass  # already long
                elif _cooldown_remaining > 0:
                    logger.debug("COOLDOWN %s — %d bars remaining", symbol, _cooldown_remaining)
                elif not _fx_session_ok:
                    logger.debug("FX_SESSION_BLOCK %s — outside allowed trading hours", symbol)
                else:
                    buy_px = px if _executor_owns_fill else px * (1.0 + slip_mult)
                    res = executor.buy(symbol, buy_px, size,
                                       trailing_stop_pct=_eff_stop,
                                       take_profit_pct=_eff_tp)
                    executed = bool(res.ok)
                    if executed:
                        pos = portfolio.positions.get(symbol)
                        if pos is not None:
                            pos.entry_atr_pct = _sym_atr_pct
                        now_ms = int(time.time() * 1000)
                        fee = size * buy_px * config.fee_rate
                        if not allow_live:
                            open_paper_trade(conn, symbol, "long", buy_px, size, fee, now_ms,
                                             ml_up_prob=up_prob,
                                             strategy_conf=sig.final_confidence,
                                             vol_ratio=sig.vol_ratio)
                        if notifier:
                            notifier.notify("buy", {"symbol": symbol, "price": buy_px,
                                                    "qty": size, "ml_up_prob": up_prob})
                    logger.info("BUY %s px=%.2f qty=%.6f ml=%.3f equity=%.2f",
                                symbol, buy_px, size, up_prob or -1, equity)

            elif action == "sell" and symbol in portfolio.positions:
                pos = portfolio.positions[symbol]
                # Conviction persistence: block signal exits until min hold time
                if pos.bars_held < min_hold_bars:
                    logger.debug(
                        "HOLD %s — min hold not met (%d/%d bars)",
                        symbol, pos.bars_held, min_hold_bars,
                    )
                # Asymmetric reversal: require stronger confidence to exit than to enter
                elif sig.final_confidence < config.min_final_confidence * config.reversal_confidence_multiplier:
                    logger.debug(
                        "HOLD %s — reversal conf %.2f < asymmetric threshold %.2f",
                        symbol, sig.final_confidence,
                        config.min_final_confidence * config.reversal_confidence_multiplier,
                    )
                # Exit confirmation: require N consecutive sell signals
                elif _consecutive_exit_signals[symbol] < exit_confirm_bars:
                    logger.debug(
                        "HOLD %s — exit confirmation %d/%d",
                        symbol, _consecutive_exit_signals[symbol], exit_confirm_bars,
                    )
                else:
                    # Consult ExitEvaluator: "is the thesis broken?"
                    exit_decision = _exit_eval.should_exit(
                        candles=candles,
                        entry_price=pos.avg_entry,
                        current_price=px,
                        bars_held=pos.bars_held,
                        side="long",
                        max_hold_bars=max_hold_bars,
                        entry_atr=getattr(pos, "entry_atr_pct", None) or None,
                    )
                    if not exit_decision.should_exit:
                        logger.debug(
                            "EXIT_MODEL %s — thesis intact, holding despite sell signal",
                            symbol,
                        )
                        event_count += 1
                        continue
                    sell_qty = min(pos.qty, size)
                    sell_px = px if _executor_owns_fill else px * (1.0 - slip_mult)
                    avg_entry = pos.avg_entry
                    res = executor.exit_long(symbol, sell_px, sell_qty, reason="signal")
                    executed = bool(res.ok)
                    if executed:
                        now_ms = int(time.time() * 1000)
                        exit_fee = sell_qty * sell_px * config.fee_rate
                        if not allow_live:
                            close_paper_trade(conn, symbol, "long", sell_px, exit_fee, now_ms, close_reason="signal")
                        gross_pnl_pct = (sell_px - avg_entry) / avg_entry if avg_entry else 0
                        net_pnl_pct = gross_pnl_pct - 2 * config.fee_rate
                        bars_held = getattr(pos, "bars_held", 0)
                        _expectancy.record_trade(
                            symbol=symbol,
                            regime=_current_regime.get(symbol, "unknown"),
                            gross_pnl_pct=gross_pnl_pct,
                            net_pnl_pct=net_pnl_pct,
                            mae_pct=getattr(pos, "max_adverse_pct", 0.0),
                            mfe_pct=getattr(pos, "max_favorable_pct", 0.0),
                            hold_bars=bars_held,
                            close_reason="signal",
                        )
                        _consecutive_exit_signals[symbol] = 0
                        is_loss = sell_px < avg_entry
                        if is_loss and loss_cooldown_bars > 0:
                            _cooldown_remaining = loss_cooldown_bars
                        if notifier:
                            notifier.notify("sell", {"symbol": symbol, "price": sell_px,
                                                     "qty": sell_qty, "ml_up_prob": up_prob})
                    logger.info("SELL %s px=%.2f qty=%.6f ml=%.3f equity=%.2f",
                                symbol, sell_px, sell_qty, up_prob or -1, equity)
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

            # Periodic expectancy summary
            if (
                event_count > 0
                and event_count % _EXPECTANCY_LOG_INTERVAL == 0
                and _expectancy._trades
            ):
                report = _expectancy.summary()
                overall = report.get("overall", {})
                logger.info(
                    "EXPECTANCY [%d trades] win=%.1f%% net_edge=%.4f%% payoff=%.2f exp=%.4f%% sig_exit_loss=%.1f%%",
                    report["total_trades"],
                    overall.get("win_rate", 0) * 100,
                    overall.get("avg_net_edge_pct", 0),
                    overall.get("payoff_ratio", 0),
                    overall.get("expectancy_pct", 0),
                    overall.get("signal_exit_loss_rate", 0) * 100,
                )
                for regime, stats in report.get("by_regime", {}).items():
                    logger.info(
                        "  regime=%s n=%d win=%.1f%% exp=%.4f%%",
                        regime, stats["n"], stats["win_rate"] * 100, stats["expectancy_pct"],
                    )

            event_count += 1

    # Final expectancy report
    if _expectancy._trades:
        report = _expectancy.summary()
        logger.info("FINAL EXPECTANCY REPORT: %s", report)

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
