"""Hogan async event loop — the canonical trading runtime.

Called by ``main.py`` (the public entry point).  All trading logic lives here.

Pipeline::

    LiveDataEngine (WebSocket candles)
        -> CandleEvent queue
        -> SignalEvaluator (AgentPipeline + ML filter)
        -> RiskManager (DrawdownGuard + position sizing)
        -> ExecutionEngine (open_long / close_long / close_short / emergency_flatten)
        -> SQLite journal + Prometheus metrics

Each symbol's signal evaluation is wrapped in a per-symbol try/except so a
single symbol failure never aborts the whole loop.

Usage::

    python -m hogan_bot.main              # preferred entry point
    python -m hogan_bot.event_loop        # direct (same effect)
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import math
import os
import sqlite3
import time
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime, timezone

import pandas as pd

from hogan_bot.agent_pipeline import AgentPipeline
from hogan_bot.config import (
    BotConfig,
    effective_hold_cooldown_bars,
    effective_short_max_hold_bars,
    load_config,
    symbol_config,
)
from hogan_bot.data_engine import LiveDataEngine
from hogan_bot.decision import (
    QualityComponents,
    apply_ml_filter,
    compute_quality_components,
    edge_gate,
    entry_quality_gate,
    estimate_spread_from_candles,
    loss_streak_scale,
    ml_blind_blocks_shorts,
    ml_blind_scale,
    ml_confidence,
    ml_probability_sizer,
    pullback_gate,
    ranging_gate,
)
from hogan_bot.execution import (
    FillSimConfig,
    LiveExecution,
    PaperExecution,
    RealisticPaperExecution,
    SmartExecConfig,
    SmartExecution,
)
from hogan_bot.expectancy import ExpectancyTracker
from hogan_bot.indicators import compute_atr
from hogan_bot.ml import TrainedModel, load_model, predict_up_probability
from hogan_bot.notifier import make_notifier
from hogan_bot.paper import PaperPortfolio
from hogan_bot.risk import DrawdownGuard, calculate_position_size
from hogan_bot.storage import (
    close_paper_trade,
    get_connection,
    log_decision,
    open_paper_trade,
    record_equity,
    upsert_candles,
)

logger = logging.getLogger(__name__)


class _FailedResult:
    """Sentinel returned by _safe_exec when an executor call throws."""
    ok = False
    error: str = ""
    def __init__(self, error: str = ""):
        self.error = error


async def _safe_exec(fn, *args, **kwargs):
    """Run *fn* in a thread so blocking I/O doesn't stall the event loop."""
    try:
        return await asyncio.to_thread(fn, *args, **kwargs)
    except Exception as exc:
        logger.error("Executor call %s raised: %s", getattr(fn, "__name__", fn), exc)
        return _FailedResult(str(exc))


_DATA_AGE_TABLES: dict[str, tuple[str, str]] = {
    "onchain_metrics": ("onchain_db", "date"),
    "derivatives_metrics": ("derivatives_db", "ts_ms"),
    "sentiment_scores": ("sentiment_db", "ts_ms"),
    "macro_indicators": ("macro_db", "ts_ms"),
    "intermarket_prices": ("intermarket_db", "ts_ms"),
}


def _compute_data_ages(conn) -> dict[str, float]:
    """Compute hours since last update for each data source in the DB."""
    if conn is None:
        return {}
    ages: dict[str, float] = {}
    now_s = time.time()
    for table, (source_key, col) in _DATA_AGE_TABLES.items():
        try:
            row = conn.execute(
                f"SELECT MAX({col}) FROM {table}"
            ).fetchone()
            if row and row[0]:
                if col == "date":
                    from datetime import datetime, timezone
                    dt = datetime.fromisoformat(str(row[0]))
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    age_h = (now_s - dt.timestamp()) / 3600.0
                else:
                    age_h = (now_s - row[0] / 1000.0) / 3600.0
                ages[source_key] = max(0.0, age_h)
        except Exception as exc:
            logger.debug("_compute_data_ages %s: %s", table, exc)
    return ages


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
    direction_score: float = 0.0
    quality_score: float = 0.0
    size_score: float = 0.0
    quality_components: QualityComponents | None = None
    block_reasons: list[str] | None = None
    eff_trailing_stop_pct: float | None = None
    eff_take_profit_pct: float | None = None
    eff_allow_longs: bool = True
    eff_allow_shorts: bool = True
    eff_long_size_scale: float = 1.0
    eff_short_size_scale: float = 1.0
    swarm_decision_id: int | None = None
    raw_tech_action: str | None = None
    pipeline_action: str | None = None


# ---------------------------------------------------------------------------
# Signal evaluator — stateless, one call per candle event
# ---------------------------------------------------------------------------
class SignalEvaluator:
    """Routes decisions through AgentPipeline — Technical + Sentiment + Macro."""

    def __init__(self, config: BotConfig, ml_model: TrainedModel | None, conn=None) -> None:
        self.config = config
        self.ml_model = ml_model
        self.pipeline = AgentPipeline(config, conn=conn)
        self._ml_probs: deque[float] = deque(maxlen=50)
        self._trade_outcomes: deque[bool] = deque(maxlen=50)
        from hogan_bot.regime import RegimeTransitionTracker
        self._regime_transition = RegimeTransitionTracker(cooldown_bars=3, min_scale=0.40)
        self._use_policy_core = getattr(config, "use_policy_core", True)
        if self._use_policy_core:
            from hogan_bot.policy_core import PolicyState
            self._policy_states: dict[str, PolicyState] = {}
            self._conn = conn

    def _policy_core_evaluate(
        self,
        symbol: str,
        candles: pd.DataFrame,
        equity: float,
        recent_whipsaw_count: int = 0,
        *,
        mtf_candles: dict[str, pd.DataFrame] | None = None,
        peak_equity: float | None = None,
    ) -> SignalResult:
        """Delegate to policy_core.decide() and translate back to SignalResult."""
        from hogan_bot.policy_core import PolicyState, decide

        if symbol not in self._policy_states:
            self._policy_states[symbol] = PolicyState()
        state = self._policy_states[symbol]
        state.ml_probs = self._ml_probs
        state.trade_outcomes = self._trade_outcomes

        import time as _time_mod
        intent = decide(
            symbol=symbol,
            candles=candles,
            equity_usd=equity,
            config=self.config,
            pipeline=self.pipeline,
            ml_model=self.ml_model,
            state=state,
            conn=getattr(self, "_conn", None),
            mode="live",
            recent_whipsaw_count=recent_whipsaw_count,
            mtf_candles=mtf_candles,
            enable_pullback_gate=True,
            enable_freshness_check=True,
            peak_equity_usd=peak_equity,
            as_of_ms=int(_time_mod.time() * 1000),
        )

        px = float(candles["close"].iloc[-1])
        size_coins = intent.size_usd / px if px > 0 else 0.0

        return SignalResult(
            action=intent.action,
            size=size_coins,
            up_prob=intent.up_prob,
            regime=intent.regime,
            atr_pct=intent.atr_pct,
            final_confidence=intent.confidence,
            tech_confidence=intent.tech_confidence,
            conf_scale=intent.conf_scale * intent.quality_scale,
            vol_ratio=intent.vol_ratio,
            explanation=intent.explanation,
            forecast_ret=intent.forecast_ret,
            agent_weights=intent.agent_weights,
            feature_freshness=intent.feature_freshness,
            block_reasons=intent.block_reasons if intent.block_reasons else None,
            eff_trailing_stop_pct=intent.eff_trailing_stop_pct,
            eff_take_profit_pct=intent.eff_take_profit_pct,
            eff_allow_longs=intent.eff_allow_longs,
            eff_allow_shorts=intent.eff_allow_shorts,
            eff_long_size_scale=intent.eff_long_size_scale,
            eff_short_size_scale=intent.eff_short_size_scale,
            swarm_decision_id=intent.swarm_decision_id,
            raw_tech_action=intent.raw_tech_action,
            pipeline_action=intent.pipeline_action,
        )

    def evaluate(
        self,
        symbol: str,
        candles: pd.DataFrame,
        equity: float,
        recent_whipsaw_count: int = 0,
        *,
        mtf_candles: dict[str, pd.DataFrame] | None = None,
        peak_equity: float | None = None,
    ) -> SignalResult:
        """Returns a SignalResult with action, sizing, and rich metadata."""
        if self._use_policy_core:
            return self._policy_core_evaluate(
                symbol, candles, equity, recent_whipsaw_count,
                mtf_candles=mtf_candles,
                peak_equity=peak_equity,
            )

        cfg = symbol_config(self.config, symbol)

        regime_name = None
        regime_conf = None
        _rstate = None
        if getattr(cfg, "use_regime_detection", True):
            try:
                from hogan_bot.regime import detect_regime
                _rstate = detect_regime(candles, symbol=symbol)
                regime_name = _rstate.regime
                regime_conf = _rstate.confidence
            except Exception as exc:
                logger.warning("Regime detection failed (using fallback): %s", exc)

        # Regime-adjusted thresholds (ML gates, stop/TP, position scale)
        eff: dict[str, float] = {}
        if _rstate is not None:
            try:
                from hogan_bot.regime import effective_thresholds
                eff = effective_thresholds(_rstate, cfg)
            except Exception as exc:
                logger.warning("effective_thresholds failed (using defaults): %s", exc)
        eff_ml_buy = eff.get("ml_buy_threshold", cfg.ml_buy_threshold)
        eff_ml_sell = eff.get("ml_sell_threshold", cfg.ml_sell_threshold)
        eff_tp = eff.get("take_profit_pct", cfg.take_profit_pct)
        eff_ts = eff.get("trailing_stop_pct", cfg.trailing_stop_pct)
        eff_position_scale = eff.get("position_scale", 1.0)
        eff_allow_longs = eff.get("allow_longs", True)
        eff_allow_shorts = eff.get("allow_shorts", True)
        eff_long_size_scale = eff.get("long_size_scale", 1.0)
        eff_short_size_scale = eff.get("short_size_scale", 1.0)

        # Regime transition dampener: reduce sizing at regime boundaries
        _transition_scale = self._regime_transition.update(regime_name) if regime_name else 1.0
        eff_long_size_scale *= _transition_scale
        eff_short_size_scale *= _transition_scale

        result = self.pipeline.run(
            candles, symbol=symbol, config_override=cfg,
            regime=regime_name, regime_state=_rstate,
        )
        px = float(candles["close"].iloc[-1])

        _raw_tech_action = result.tech.action if result.tech else None
        _pipeline_action = result.action

        up_prob = None
        conf_scale = 1.0
        action = result.action

        block_reasons: list[str] = []

        if (cfg.use_ml_filter or cfg.use_ml_as_sizer) and self.ml_model is not None:
            if cfg.use_mtf_extended and mtf_candles:
                try:
                    from hogan_bot.ml import build_feature_row_extended
                    _mtf_features = build_feature_row_extended(
                        candles_5m=mtf_candles.get("5m"),
                        candles_1h=candles,
                        candles_15m=mtf_candles.get("15m"),
                        candles_10m=mtf_candles.get("10m"),
                        candles_30m=mtf_candles.get("30m"),
                        candles_3h=mtf_candles.get("3h"),
                        conn=self.pipeline.conn,
                        symbol=symbol,
                        extended_mtf=True,
                    )
                    if _mtf_features is not None:
                        logger.debug("MTF extended features computed: %d values", len(_mtf_features))
                except Exception as exc:
                    logger.debug("MTF features fallback: %s", exc)
            if hasattr(self.ml_model, "set_regime"):
                self.ml_model.set_regime(regime_name)
            up_prob = predict_up_probability(candles, self.ml_model)
            self._ml_probs.append(up_prob)
            if cfg.use_ml_as_sizer:
                conf_scale = ml_probability_sizer(action, up_prob)
            else:
                ml_gate = apply_ml_filter(action, up_prob, eff_ml_buy, eff_ml_sell)
                action = ml_gate.action
                if ml_gate.blocked_by:
                    block_reasons.append(ml_gate.blocked_by)
                if cfg.ml_confidence_sizing:
                    conf_scale = ml_confidence(up_prob)
            _blind = ml_blind_scale(self._ml_probs)
            if _blind < 1.0:
                conf_scale *= _blind
                logger.info("ML_BLIND: prob std low -> scale %.2f", _blind)

        _ls = loss_streak_scale(self._trade_outcomes)
        if _ls < 1.0:
            conf_scale *= _ls
            _streak = 0
            for _o in reversed(self._trade_outcomes):
                if not _o:
                    _streak += 1
                else:
                    break
            logger.info("LOSS_STREAK: %d consecutive losses -> scale %.2f", _streak, _ls)

        forecast_ret = None
        if result.forecast is not None and result.forecast.confidence > 0.2:
            er = result.forecast.expected_return
            if isinstance(er, dict) and er:
                forecast_ret = max(abs(v) for v in er.values())
            elif isinstance(er, (int, float)):
                forecast_ret = abs(float(er))
        _atr_s = compute_atr(candles, window=14)
        atr_pct = float(_atr_s.iloc[-1]) / max(px, 1e-9)
        spread_est = estimate_spread_from_candles(candles)
        edge_gd = edge_gate(
            action,
            atr_pct=atr_pct,
            take_profit_pct=eff_tp,
            fee_rate=cfg.fee_rate,
            min_edge_multiple=getattr(cfg, "min_edge_multiple", 1.5),
            forecast_expected_return=forecast_ret,
            estimated_spread=spread_est,
            atr_friction_multiple=getattr(cfg, "sell_atr_friction_multiple", 0.8),
            buy_atr_friction_multiple=getattr(cfg, "buy_atr_friction_multiple", 0.25),
        )
        action = edge_gd.action
        if edge_gd.blocked_by:
            block_reasons.append(edge_gd.blocked_by)

        tech_conf = result.tech.confidence if result.tech else None
        quality_gd = entry_quality_gate(
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
        action = quality_gd.action
        quality_scale = quality_gd.size_scale
        if quality_gd.blocked_by:
            block_reasons.append(quality_gd.blocked_by)

        tech_action = result.tech.action if result.tech else None
        ranging_gd = ranging_gate(
            action,
            regime=regime_name,
            tech_action=tech_action,
            up_prob=up_prob,
            recent_whipsaw_count=recent_whipsaw_count,
        )
        action = ranging_gd.action
        ranging_scale = ranging_gd.size_scale
        if ranging_gd.blocked_by:
            block_reasons.append(ranging_gd.blocked_by)

        pullback_gd = pullback_gate(action, candles, regime=regime_name)
        action = pullback_gd.action
        pullback_scale = pullback_gd.size_scale
        if pullback_gd.blocked_by:
            block_reasons.append(pullback_gd.blocked_by)

        # MTF ensemble: daily bias + 30m confirmation
        _mtf_conf_mult = 1.0
        if cfg.use_mtf_ensemble and action != "hold" and mtf_candles:
            try:
                from hogan_bot.mtf_ensemble import evaluate_mtf
                _daily_df = None
                if self.pipeline.conn is not None:
                    try:
                        _daily_df = pd.read_sql_query(
                            "SELECT * FROM candles WHERE symbol=? AND timeframe='1d' ORDER BY ts_ms",
                            self.pipeline.conn, params=(symbol,),
                        )
                    except Exception as exc:
                        logger.debug("MTF daily candle query failed: %s", exc)
                _m30_df = mtf_candles.get(cfg.mtf_m30_timeframe)
                _mtf_bias = evaluate_mtf(
                    daily_candles=_daily_df,
                    hourly_action=action,
                    m30_candles=_m30_df,
                    unconfirmed_scale=cfg.mtf_unconfirmed_scale,
                )
                action = _mtf_bias.final_action
                _mtf_conf_mult = _mtf_bias.confidence_mult
                if _mtf_bias.final_action == "hold" and _mtf_bias.hourly_action != "hold":
                    block_reasons.append(f"mtf_daily_{_mtf_bias.daily_trend}")
                logger.debug("MTF_ENSEMBLE: daily=%s m30_conf=%s action=%s mult=%.2f",
                             _mtf_bias.daily_trend, _mtf_bias.m30_confirms, action, _mtf_conf_mult)
            except Exception as exc:
                logger.debug("MTF ensemble error: %s", exc)

        # Feature staleness check with live policy
        _freshness: dict | None = None
        _freshness_scale = 1.0
        try:
            from hogan_bot.ml import build_feature_row_checked
            _data_ages = _compute_data_ages(self.pipeline.conn)
            feat_result = build_feature_row_checked(
                candles, db_conn=self.pipeline.conn, data_ages_hours=_data_ages,
            )
            if feat_result is not None:
                _freshness = feat_result.freshness_summary
                crit_stale = _freshness.get("critical_stale_count", 0)
                all_stale = _freshness.get("stale_count", 0)
                if crit_stale >= 2 and action != "hold":
                    logger.warning("FRESHNESS_BLOCK %s: %d critical stale features", symbol, crit_stale)
                    action = "hold"
                    block_reasons.append("freshness_critical_block")
                elif (crit_stale >= 1 or all_stale >= 4):
                    _freshness_scale = 0.75
                    logger.info("FRESHNESS_PENALTY %s: scale=0.75 (crit=%d all=%d)", symbol, crit_stale, all_stale)
                if feat_result.has_stale:
                    logger.warning("STALE_FEATURES %s: %s", symbol, feat_result.stale_features)
        except Exception as exc:
            logger.warning("Feature freshness check failed: %s", exc)

        _qc = compute_quality_components(
            final_confidence=result.confidence,
            tech_confidence=tech_conf,
            regime_confidence=regime_conf,
            up_prob=up_prob,
            estimated_spread=spread_est,
            atr_pct=atr_pct,
            recent_whipsaw_count=recent_whipsaw_count,
            freshness_summary=_freshness,
            ranging_scale=ranging_scale,
            pullback_scale=pullback_scale,
            quality_gate_scale=quality_scale,
        )

        _MIN_COMPOSITE_SCALE = 0.15
        _composite = max(
            _MIN_COMPOSITE_SCALE,
            conf_scale * quality_scale * ranging_scale * pullback_scale * eff_position_scale * _freshness_scale * _mtf_conf_mult,
        )
        size = calculate_position_size(
            equity_usd=equity,
            price=px,
            stop_distance_pct=result.stop_distance_pct,
            max_risk_per_trade=cfg.max_risk_per_trade,
            max_allocation_pct=cfg.aggressive_allocation,
            confidence_scale=_composite,
            fee_rate=cfg.fee_rate,
        )

        _direction_score = getattr(result, "combined_score", 0.0)
        _quality_score = _qc.overall
        _size_score = min(1.0, conf_scale * quality_scale * ranging_scale * pullback_scale * eff_position_scale)

        if action != "hold":
            logger.info(
                "PIPELINE %s action=%s dir=%.2f qual=%.2f sz=%.2f regime=%s | %s",
                symbol, action, _direction_score, _quality_score, _size_score,
                regime_name or "unknown", result.explanation,
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
            direction_score=_direction_score,
            quality_score=_quality_score,
            size_score=_size_score,
            quality_components=_qc,
            block_reasons=block_reasons if block_reasons else None,
            eff_trailing_stop_pct=eff_ts,
            eff_take_profit_pct=eff_tp,
            eff_allow_longs=eff_allow_longs,
            eff_allow_shorts=eff_allow_shorts,
            eff_long_size_scale=eff_long_size_scale,
            eff_short_size_scale=eff_short_size_scale,
            raw_tech_action=_raw_tech_action,
            pipeline_action=_pipeline_action,
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

    from hogan_bot.config import apply_sweep_results
    apply_sweep_results(config)

    # ── Startup config validation ────────────────────────────────────────
    _cfg_errors = config.validate()
    if _cfg_errors:
        for _ce in _cfg_errors:
            logger.error("CONFIG VALIDATION: %s", _ce)
        raise ValueError(f"Invalid configuration — {len(_cfg_errors)} error(s): {'; '.join(_cfg_errors)}")

    conn = get_connection(config.db_path)

    # ── Startup data refresh: populate sentiment/macro/derivatives tables ──
    _skip_fetch = os.getenv("HOGAN_SKIP_FETCH", "").strip().lower() in ("1", "true", "yes")
    if not _skip_fetch:
        try:
            from hogan_bot.fetch_orchestrator import run_all_fetchers
            _fetch_results = run_all_fetchers(
                db_path=config.db_path,
                symbols=config.symbols,
                backfill=False,
            )
            logger.info("Startup data fetch complete: %s", {k: v for k, v in _fetch_results.items() if not str(v).startswith("SKIPPED")})
        except Exception as exc:
            logger.warning("Startup data fetch failed (non-fatal): %s", exc)
    else:
        logger.info("Startup data fetch skipped (HOGAN_SKIP_FETCH=true)")

    try:
        await _run_event_loop_inner(config, conn, max_events)
    finally:
        conn.close()
        logger.info("Database connection closed.")


async def _run_event_loop_inner(
    config: BotConfig,
    conn,
    max_events: int | None,
) -> None:
    notifier = make_notifier(config.webhook_url or None)

    live_ack = (os.getenv("HOGAN_LIVE_ACK", "") or "").strip()
    allow_live = (
        (not config.paper_mode)
        and config.live_mode
        and live_ack == "I_UNDERSTAND_LIVE_TRADING"
    )

    # Restore cash balance from the last equity snapshot if available,
    # so restarts don't reset P&L to starting_balance_usd.
    _initial_cash = config.starting_balance_usd
    try:
        _eq_row = conn.execute(
            "SELECT cash_usd FROM equity_snapshots ORDER BY ts_ms DESC LIMIT 1"
        ).fetchone()
        if _eq_row and _eq_row[0] is not None and float(_eq_row[0]) > 0:
            _initial_cash = float(_eq_row[0])
            logger.info("RESTORE: loaded last cash balance %.2f from DB (seed was %.2f)",
                        _initial_cash, config.starting_balance_usd)
    except Exception as exc:
        logger.debug("Could not restore cash from DB (using seed balance): %s", exc)

    portfolio = PaperPortfolio(
        cash_usd=_initial_cash,
        fee_rate=config.fee_rate,
        short_max_loss_pct=getattr(config, "short_max_loss_pct", 0.10),
    )
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

    # Reconcile portfolio with DB: restore open paper trades from prior session
    try:
        _open_trades = conn.execute(
            "SELECT symbol, side, entry_price, qty FROM paper_trades WHERE exit_price IS NULL"
        ).fetchall()
        for _ot_symbol, _ot_side, _ot_entry, _ot_qty in _open_trades:
            _ot_entry = float(_ot_entry)
            _ot_qty = float(_ot_qty)
            if _ot_side == "long" and _ot_symbol not in portfolio.positions:
                portfolio.execute_buy(_ot_symbol, _ot_entry, _ot_qty)
                logger.info("RECONCILE: restored open long %s qty=%.6f entry=%.2f", _ot_symbol, _ot_qty, _ot_entry)
            elif _ot_side == "short" and _ot_symbol not in portfolio.short_positions:
                portfolio.execute_short(_ot_symbol, _ot_entry, _ot_qty)
                logger.info("RECONCILE: restored open short %s qty=%.6f entry=%.2f", _ot_symbol, _ot_qty, _ot_entry)
        if _open_trades:
            logger.info("RECONCILE: restored %d open position(s) from DB", len(_open_trades))
    except Exception as exc:
        logger.warning("Position reconciliation failed: %s", exc)

    ml_model: TrainedModel | None = None
    if config.use_ml_filter or config.use_ml_as_sizer:
        try:
            ml_model = load_model(config.ml_model_path)
            logger.info("Loaded ML model from %s", config.ml_model_path)
        except FileNotFoundError:
            logger.warning(
                "ML model not found at %s — running in DEGRADED mode (no ML filter/sizer). "
                "Train a model or set HOGAN_USE_ML_FILTER=false and HOGAN_ML_AS_SIZER=false.",
                config.ml_model_path,
            )
        except Exception as exc:
            logger.warning("ML model load error: %s — running without ML.", exc)

    evaluator = SignalEvaluator(config, ml_model, conn=conn)

    _macro_sitout = None
    _use_macro_sitout = os.getenv("HOGAN_MACRO_SITOUT", "true").strip().lower() in ("1", "true", "yes")
    if _use_macro_sitout:
        try:
            from hogan_bot.macro_sitout import MacroSitout
            _macro_sitout = MacroSitout.from_db(conn)
            logger.info("Macro sitout filter enabled (greed-only scaling)")
        except Exception as exc:
            logger.warning("Macro sitout init failed: %s", exc)

    _funding_overlay = None
    _use_funding = os.getenv("HOGAN_USE_FUNDING_OVERLAY", "true").strip().lower() in ("1", "true", "yes")
    if _use_funding:
        try:
            from hogan_bot.funding_overlay import FundingOverlay
            _funding_overlay = FundingOverlay.from_db(conn)
            logger.info("Funding rate overlay enabled")
        except Exception as exc:
            logger.warning("Funding overlay init failed: %s", exc)

    from hogan_bot.exit_model import ExitEvaluator
    _exit_eval = ExitEvaluator(
        drawdown_panic_pct=config.exit_drawdown_pct,
        time_decay_threshold=config.exit_time_decay,
        volatility_expansion_threshold=config.exit_vol_expansion,
        max_consolidation_bars=config.exit_stagnation_bars,
    )
    _expectancy = ExpectancyTracker()

    # Online learner (behind config flag)
    _online_learner = None
    _online_learner_bar_count = 0
    _online_drift_scale = 1.0  # reduced when feature drift detected
    if getattr(config, "use_online_learning", False):
        try:
            from hogan_bot.online_learner import OnlineLearner
            _online_learner = OnlineLearner(db_path=config.db_path)
            logger.info("OnlineLearner initialized (interval=%d bars)",
                        getattr(config, "online_learning_interval", 50))
        except Exception as exc:
            logger.warning("OnlineLearner unavailable: %s", exc)

    _perf_tracker = None
    try:
        from hogan_bot.performance_tracker import PerformanceTracker
        _perf_tracker = PerformanceTracker(db_path=config.db_path)
        if _perf_tracker.restore_from_db():
            logger.info("PerformanceTracker restored state from DB snapshot")
        else:
            logger.info("PerformanceTracker initialized (no prior snapshot)")
    except Exception as exc:
        logger.warning("PerformanceTracker unavailable: %s", exc)
    _perf_trade_count = 0
    _PERF_PROPOSAL_INTERVAL = 50

    # Walk-forward failure response: compute sizing penalty at startup
    _wf_scale = 1.0
    try:
        from hogan_bot.decision import wf_failure_scale
        _wf_scale = wf_failure_scale()
        if _wf_scale < 1.0:
            logger.info("WF failure scale applied: %.2f", _wf_scale)
    except Exception as exc:
        logger.debug("wf_failure_scale error: %s", exc)

    # Auto-train forecast models if missing and flag is set
    if getattr(config, "auto_train_forecast", False):
        import glob as _glob
        _forecast_files = _glob.glob("models/forecast_*.pkl")
        if not _forecast_files:
            logger.info("AUTO_TRAIN_FORECAST: no forecast models found — training from historical data")
            try:
                from hogan_bot.forecast import train_forecast_models
                with sqlite3.connect(config.db_path) as _train_conn:
                    _hist_candles = pd.read_sql_query(
                        "SELECT * FROM candles WHERE symbol = ? ORDER BY ts_ms",
                        _train_conn,
                        params=[config.symbols[0] if config.symbols else "BTC/USD"],
                    )
                    if len(_hist_candles) >= 500:
                        _fc_result = train_forecast_models(_hist_candles, db_conn=_train_conn)
                        logger.info("AUTO_TRAIN_FORECAST: trained %d models: %s",
                                    len(_fc_result), list(_fc_result.keys()))
                    else:
                        logger.warning("AUTO_TRAIN_FORECAST: insufficient historical candles (%d < 500)", len(_hist_candles))
            except Exception as exc:
                logger.warning("AUTO_TRAIN_FORECAST failed: %s", exc)
        else:
            logger.debug("AUTO_TRAIN_FORECAST: %d forecast models already exist", len(_forecast_files))

    # Parity with backtest: max_hold_bars, loss_cooldown, slippage
    max_hold_bars, loss_cooldown_bars = effective_hold_cooldown_bars(config, config.timeframe)
    short_max_hold_bars = effective_short_max_hold_bars(config, config.timeframe)
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
    enable_shorts = os.getenv("HOGAN_ENABLE_SHORTS", "").strip().lower() in ("1", "true", "yes")
    enable_close_and_reverse = os.getenv("HOGAN_CLOSE_AND_REVERSE", "").strip().lower() in ("1", "true", "yes")
    _consecutive_short_exit_signals: dict[str, int] = defaultdict(int)
    _consecutive_eval_failures: dict[str, int] = {}
    _heartbeat_count = 0
    _HEARTBEAT_INTERVAL = 100  # log a heartbeat every N events

    try:
        from hogan_bot.metrics import (
            CASH,
            DRAWDOWN,
            EQUITY,
            EXCEPTIONS,
            MetricsServer,
        )
        metrics_server = MetricsServer(port=getattr(config, "metrics_port", 8000))
        metrics_server.start()
        _has_metrics = True
    except Exception as exc:
        logger.warning("Metrics server failed to start (Prometheus unavailable): %s", exc)
        _has_metrics = False

    _mtf_timeframes = config.mtf_timeframes or []
    _all_timeframes = [config.timeframe] + [tf for tf in _mtf_timeframes if tf != config.timeframe]

    if use_oanda and _oanda_client is not None:
        from hogan_bot.data_engine import OandaDataEngine
        engine = OandaDataEngine(
            client=_oanda_client,
            symbols=config.symbols,
            timeframes=_all_timeframes,
            ring_buffer_len=config.ohlcv_limit,
        )
        logger.info("Using OandaDataEngine for %s", config.symbols)
    else:
        engine = LiveDataEngine(
            exchange_id=config.exchange_id,
            api_key=config.kraken_api_key or "",
            api_secret=config.kraken_api_secret or "",
            symbols=config.symbols,
            timeframes=_all_timeframes,
            ring_buffer_len=config.ohlcv_limit,
        )
    buffer = engine.buffer

    if _mtf_timeframes:
        logger.info("MTF candle subscriptions: %s (primary=%s)", _all_timeframes, config.timeframe)

    event_count = 0
    _signal_counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    _current_regime: dict[str, str] = {}
    _entry_regime: dict[str, str] = {}
    _entry_regime_conf: dict[str, float] = {}
    _EXPECTANCY_LOG_INTERVAL = 50
    _entry_up_prob: dict[str, float | None] = {}

    def _record_outcome(sym: str, regime: str, pnl: float, up_prob: float | None = None) -> None:
        """Feed closed-trade results back to AgentPipeline for learned weights."""
        last_sig = evaluator.pipeline._last_signal.get(sym)
        if last_sig is not None:
            evaluator.pipeline.record_trade_outcome(
                symbol=sym,
                regime=regime,
                signal=last_sig,
                realized_pnl=pnl,
                ml_up_prob=up_prob,
            )

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
    logger.info(
        "STARTUP SUMMARY: balance=%.2f max_dd=%.1f%% risk/trade=%.1f%% "
        "trailing_stop=%.1f%% take_profit=%.1f%% ML=%s swarm=%s regime=%s "
        "policy_core=%s online_learn=%s",
        config.starting_balance_usd,
        config.max_drawdown * 100,
        config.max_risk_per_trade * 100,
        config.trailing_stop_pct * 100,
        config.take_profit_pct * 100,
        "ON" if (config.use_ml_filter or config.use_ml_as_sizer) else "OFF",
        "ON" if config.swarm_enabled else "OFF",
        "ON" if config.use_regime_detection else "OFF",
        "ON" if config.use_policy_core else "OFF",
        "ON" if getattr(config, "use_online_learning", False) else "OFF",
    )

    async with engine:
        async for event in engine.stream():
            if max_events is not None and event_count >= max_events:
                break

            symbol = event.symbol
            tf = event.timeframe

            # Non-primary timeframes: buffer + persist
            if tf != config.timeframe:
                buffer.push(symbol, tf, event.candle.to_dict())
                _mtf_cd = event.candle.to_dict()
                try:
                    _mtf_row = {
                        "timestamp": _mtf_cd.get("ts_ms", 0),
                        "open": _mtf_cd.get("open", 0),
                        "high": _mtf_cd.get("high", 0),
                        "low": _mtf_cd.get("low", 0),
                        "close": _mtf_cd.get("close", 0),
                        "volume": _mtf_cd.get("volume", 0),
                    }
                    upsert_candles(conn, symbol, tf, pd.DataFrame([_mtf_row]))
                except Exception:
                    pass
                # Sub-hourly candles trigger a re-evaluation using the
                # buffered 1h data so the bot can react faster than once/hour
                _EVAL_TIMEFRAMES = ("1m", "5m", "15m", "30m")
                if tf not in _EVAL_TIMEFRAMES:
                    continue
                _primary_df = buffer.to_df(symbol, config.timeframe)
                if _primary_df.empty or len(_primary_df) < max(config.long_ma_window, 20):
                    continue
                candles = _primary_df
                _is_mtf_eval = True
                # fall through to the signal evaluation below
            else:
                _is_mtf_eval = False

            if not _is_mtf_eval:
                buffer.push(symbol, tf, event.candle.to_dict())

                _candle_dict = event.candle.to_dict()
                try:
                    _persist_row = {
                        "timestamp": _candle_dict.get("ts_ms", 0),
                        "open": _candle_dict.get("open", 0),
                        "high": _candle_dict.get("high", 0),
                        "low": _candle_dict.get("low", 0),
                        "close": _candle_dict.get("close", 0),
                        "volume": _candle_dict.get("volume", 0),
                    }
                    upsert_candles(conn, symbol, tf, pd.DataFrame([_persist_row]))
                except Exception as _persist_err:
                    logger.debug("candle persist %s/%s: %s", symbol, tf, _persist_err)

                candles = buffer.to_df(symbol, tf)
            _min_candles = max(config.long_ma_window, 20)

            if candles.empty or len(candles) < _min_candles:
                event_count += 1
                continue

            # Gap staleness guard: skip trading if most recent candle is too old
            # (e.g., after weekend gaps or exchange outages).  Indicators need
            # fresh data to produce reliable signals.
            if "ts_ms" in candles.columns:
                _latest_candle_ms = int(candles["ts_ms"].iloc[-1])
                _candle_age_h = (time.time() * 1000 - _latest_candle_ms) / 3_600_000
                if _candle_age_h > 4.0:
                    logger.warning("GAP_GUARD %s — latest candle is %.1fh old, skipping trading", symbol, _candle_age_h)
                    if notifier and _candle_age_h > 8.0:
                        notifier.notify("warning", {"msg": f"GAP_GUARD {symbol} — candle {_candle_age_h:.1f}h stale"})
                    event_count += 1
                    continue

            mark_prices = {}
            for s in config.symbols:
                _s_df = buffer.to_df(s, tf)
                if _s_df.empty:
                    continue
                _s_px = float(_s_df["close"].iloc[-1])
                if not math.isfinite(_s_px) or _s_px <= 0:
                    logger.error("PRICE_INVALID %s — close=%.6f (NaN/Inf/zero), skipping", s, _s_px)
                    continue
                mark_prices[s] = _s_px
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
                logger.warning("Equity record error: %s", exc)

            # DrawdownGuard
            if not guard.update_and_check(equity):
                logger.error("Drawdown limit hit: equity=%.2f peak=%.2f", equity, guard.peak_equity)
                if notifier:
                    notifier.notify("drawdown_breach", {"equity": equity, "peak": guard.peak_equity})
                # Flatten all positions before stopping — use emergency_flatten
                # which routes through the proper exit_long / exit_short methods
                _flat_syms = set(portfolio.positions) | set(portfolio.short_positions)
                for _flat_sym in _flat_syms:
                    try:
                        _flat_px = mark_prices.get(_flat_sym, 0.0)
                        # Fallback: use position entry price if mark price unavailable
                        if _flat_px <= 0:
                            _flat_pos = portfolio.positions.get(_flat_sym) or portfolio.short_positions.get(_flat_sym)
                            if _flat_pos is not None and _flat_pos.avg_entry > 0:
                                _flat_px = _flat_pos.avg_entry
                                logger.warning("FLATTEN %s — no mark price, using entry price %.2f as fallback", _flat_sym, _flat_px)
                        if _flat_px <= 0:
                            logger.error("FLATTEN_FAILED %s — no price available, cannot flatten!", _flat_sym)
                            if notifier:
                                notifier.notify("error", {"msg": f"Cannot flatten {_flat_sym} — no price"})
                            continue
                        res = await _safe_exec(executor.emergency_flatten, _flat_sym, _flat_px)
                        logger.warning("FLATTEN %s px=%.2f ok=%s", _flat_sym, _flat_px, res.ok)
                    except Exception as _flat_exc:
                        logger.error("Failed to flatten %s: %s", _flat_sym, _flat_exc, exc_info=True)
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
            exits = portfolio.check_exits(mark_prices, max_hold_bars=max_hold_bars, short_max_hold_bars=short_max_hold_bars)
            for exit_symbol, reason in exits:
                ep = mark_prices.get(exit_symbol, 0.0)
                now_ms = int(time.time() * 1000)

                if reason in ("trailing_stop", "take_profit", "max_hold_time", "breakeven_stop"):
                    pos = portfolio.positions.get(exit_symbol)
                    if pos is None:
                        continue
                    qty = pos.qty
                    avg_entry = pos.avg_entry
                    bars_held = getattr(pos, "bars_held", 0)
                    mae_pct = getattr(pos, "max_adverse_pct", 0.0)
                    mfe_pct = getattr(pos, "max_favorable_pct", 0.0)
                    res = await _safe_exec(executor.close_long, exit_symbol, ep, qty, reason=reason)
                    executed = bool(res.ok)
                    sell_px = ep
                    _exit_entry_regime = _entry_regime.pop(exit_symbol, _current_regime.get(exit_symbol, "unknown"))
                    if executed:
                        fee = qty * sell_px * config.fee_rate
                        if not allow_live:
                            close_paper_trade(
                                conn, exit_symbol, "long", sell_px, fee, now_ms, close_reason=reason,
                                max_adverse_pct=mae_pct, max_favorable_pct=mfe_pct,
                                bars_held=bars_held,
                                exit_regime=_exit_entry_regime,
                                entry_atr_pct=getattr(pos, "entry_atr_pct", None),
                            )
                        evaluator._trade_outcomes.append(sell_px > avg_entry)
                        gross_pnl_pct = (sell_px - avg_entry) / avg_entry if avg_entry else 0
                        net_pnl_pct = gross_pnl_pct - 2 * config.fee_rate
                        _expectancy.record_trade(
                            symbol=exit_symbol,
                            regime=_exit_entry_regime,
                            gross_pnl_pct=gross_pnl_pct,
                            net_pnl_pct=net_pnl_pct,
                            mae_pct=mae_pct,
                            mfe_pct=mfe_pct,
                            hold_bars=bars_held,
                            close_reason=reason,
                        )
                        _record_outcome(exit_symbol, _exit_entry_regime, gross_pnl_pct, _entry_up_prob.pop(exit_symbol, None))
                        if _perf_tracker:
                            try:
                                _perf_tracker.record_trade_outcome(
                                    symbol=exit_symbol,
                                    regime=_exit_entry_regime,
                                    tech_action="sell", tech_confidence=0.0,
                                    sent_bias="neutral", sent_strength=0.0,
                                    macro_regime="unknown",
                                    realized_pnl=gross_pnl_pct,
                                )
                                _perf_trade_count += 1
                            except Exception as exc:
                                logger.debug("PerfTracker record error: %s", exc)
                        is_loss = sell_px < avg_entry
                        if is_loss and loss_cooldown_bars > 0:
                            _cooldown_remaining = loss_cooldown_bars
                    if not executed:
                        logger.error("AUTO_EXIT_FAILED %s reason=%s px=%.2f qty=%.6f — %s",
                                     exit_symbol, reason, ep, qty, getattr(res, "error", "unknown"))
                    if notifier and executed:
                        notifier.notify("auto_exit", {"symbol": exit_symbol, "reason": reason,
                                                      "price": sell_px, "qty": qty})
                    logger.info("AUTO_EXIT %s reason=%s px=%.2f qty=%.6f", exit_symbol, reason, sell_px, qty)

                elif reason in ("short_trailing_stop", "short_take_profit", "short_max_hold_time", "short_max_loss", "short_breakeven_stop"):
                    pos = portfolio.short_positions.get(exit_symbol)
                    if pos is None:
                        continue
                    qty = pos.qty
                    avg_entry = pos.avg_entry
                    bars_held = getattr(pos, "bars_held", 0)
                    mae_pct = getattr(pos, "max_adverse_pct", 0.0)
                    mfe_pct = getattr(pos, "max_favorable_pct", 0.0)
                    cover_px = ep
                    res = await _safe_exec(executor.close_short, exit_symbol, cover_px, qty, reason=reason)
                    executed = bool(res.ok)
                    _exit_s_entry_regime = _entry_regime.pop(exit_symbol, _current_regime.get(exit_symbol, "unknown"))
                    if executed:
                        fee = qty * cover_px * config.fee_rate
                        if not allow_live:
                            close_paper_trade(
                                conn, exit_symbol, "short", cover_px, fee, now_ms, close_reason=reason,
                                max_adverse_pct=mae_pct, max_favorable_pct=mfe_pct,
                                bars_held=bars_held,
                                exit_regime=_exit_s_entry_regime,
                                entry_atr_pct=getattr(pos, "entry_atr_pct", None),
                            )
                        evaluator._trade_outcomes.append(cover_px < avg_entry)
                        gross_pnl_pct = (avg_entry - cover_px) / avg_entry if avg_entry else 0
                        net_pnl_pct = gross_pnl_pct - 2 * config.fee_rate
                        _expectancy.record_trade(
                            symbol=exit_symbol,
                            regime=_exit_s_entry_regime,
                            gross_pnl_pct=gross_pnl_pct,
                            net_pnl_pct=net_pnl_pct,
                            mae_pct=mae_pct,
                            mfe_pct=mfe_pct,
                            hold_bars=bars_held,
                            close_reason=reason,
                        )
                        _record_outcome(exit_symbol, _exit_s_entry_regime, gross_pnl_pct, _entry_up_prob.pop(exit_symbol, None))
                        if _perf_tracker:
                            try:
                                _perf_tracker.record_trade_outcome(
                                    symbol=exit_symbol,
                                    regime=_exit_s_entry_regime,
                                    tech_action="buy", tech_confidence=0.0,
                                    sent_bias="neutral", sent_strength=0.0,
                                    macro_regime="unknown",
                                    realized_pnl=gross_pnl_pct,
                                )
                                _perf_trade_count += 1
                            except Exception as exc:
                                logger.debug("PerfTracker record error: %s", exc)
                    if not executed:
                        logger.error("AUTO_EXIT_SHORT_FAILED %s reason=%s px=%.2f qty=%.6f — %s",
                                     exit_symbol, reason, cover_px, qty, getattr(res, "error", "unknown"))
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
                except Exception as exc:
                    logger.debug("DEAD_MAN_ALERTS metric error: %s", exc)

            # FX session filter: block trading during off-hours/weekends
            if _fx_session_filter is not None:
                _fx_allowed, _fx_scale, _fx_reason = _fx_session_filter.should_trade()
                if not _fx_allowed:
                    logger.debug("FX_SESSION blocked: %s", _fx_reason)
                    event_count += 1
                    continue

            # Per-symbol signal evaluation (isolated — one symbol failure won't break others)
            try:
                _mtf_data: dict[str, pd.DataFrame] = {}
                for _mtf_tf in _mtf_timeframes:
                    _mtf_df = buffer.to_df(symbol, _mtf_tf)
                    if not _mtf_df.empty and len(_mtf_df) >= 10:
                        _mtf_data[_mtf_tf] = _mtf_df
                sig = evaluator.evaluate(
                    symbol, candles, equity,
                    recent_whipsaw_count=_whipsaw_counts.get(symbol, 0),
                    mtf_candles=_mtf_data if _mtf_data else None,
                    peak_equity=guard.peak_equity,
                )
                action = sig.action
                size = sig.size * _online_drift_scale
                up_prob = sig.up_prob
                _sym_regime = sig.regime
                _sym_atr_pct = sig.atr_pct
                if _is_mtf_eval:
                    _MTF_SIZE_SCALE = {"1m": 0.30, "5m": 0.40, "15m": 0.60, "30m": 0.70}
                    _mtf_scale = _MTF_SIZE_SCALE.get(tf, 0.50)
                    size *= _mtf_scale
                    logger.info("MTF_EVAL %s/%s → %s (size scaled %.2fx for sub-hour eval)", symbol, tf, action, _mtf_scale)
                if _sym_regime:
                    _current_regime[symbol] = _sym_regime
            except Exception as exc:
                logger.error("Signal eval error for %s: %s", symbol, exc, exc_info=True)
                _consecutive_eval_failures[symbol] = _consecutive_eval_failures.get(symbol, 0) + 1
                if _consecutive_eval_failures[symbol] >= 5 and notifier:
                    notifier.notify("error", {"msg": f"Signal eval failing repeatedly for {symbol} ({_consecutive_eval_failures[symbol]}x)"})
                if _has_metrics:
                    EXCEPTIONS.inc()
                event_count += 1
                continue

            _consecutive_eval_failures.pop(symbol, None)  # clear on success
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
                    tech_action=sig.raw_tech_action or action,
                    tech_confidence=sig.tech_confidence,
                    pipeline_action=sig.pipeline_action or action,
                    final_action=action,
                    final_confidence=sig.final_confidence,
                    position_size=size,
                    ml_up_prob=up_prob,
                    conf_scale=sig.conf_scale,
                    explanation=sig.explanation,
                    meta_weights=sig.agent_weights,
                    freshness=sig.feature_freshness,
                    direction_score=sig.direction_score,
                    quality_score=sig.quality_score,
                    size_score=sig.size_score,
                    quality_components_json=sig.quality_components.to_json() if sig.quality_components else None,
                    block_reasons=sig.block_reasons,
                    swarm_decision_id=sig.swarm_decision_id,
                )
            except Exception as exc:
                logger.warning("Decision log error: %s", exc)

            # Track consecutive exit signals for confirmation
            if action == "sell" and symbol in portfolio.positions:
                _consecutive_exit_signals[symbol] += 1
            else:
                _consecutive_exit_signals[symbol] = 0
            if action == "buy" and symbol in portfolio.short_positions:
                _consecutive_short_exit_signals[symbol] += 1
            else:
                _consecutive_short_exit_signals[symbol] = 0

            # Regime-adjusted stop/TP from three-tier confidence gate,
            # with instrument-profile override for FX pip-based risk.
            _eff_stop = sig.eff_trailing_stop_pct or config.trailing_stop_pct
            _eff_tp = sig.eff_take_profit_pct or config.take_profit_pct
            try:
                from hogan_bot.instrument_profiles import get_profile
                _iprofile = get_profile(symbol)
                if _iprofile.use_pip_based_risk:
                    _eff_stop = _iprofile.default_stop_pct
                    _eff_tp = _iprofile.default_tp_pct
            except Exception as exc:
                logger.debug("Instrument profile lookup failed for %s: %s", symbol, exc)

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
            except Exception as exc:
                logger.debug("FX session check error for %s: %s", symbol, exc)

            # ── Macro sitout filter (parity with backtest) ────────────
            _macro_scale = 1.0
            if _macro_sitout is not None and action != "hold":
                try:
                    from datetime import timezone as _tz
                    _bar_dt = datetime.now(tz=_tz.utc)
                    _sitout_r = _macro_sitout.check(_bar_dt)
                    if _sitout_r.should_sitout:
                        logger.debug("MACRO_SITOUT %s — %s", symbol, _sitout_r.summary)
                        action = "hold"
                    elif _sitout_r.size_scale < 1.0:
                        _macro_scale = _sitout_r.size_scale
                        logger.debug("MACRO_SCALE %s — %.2fx — %s", symbol, _macro_scale, _sitout_r.summary)
                except Exception as _mse:
                    logger.warning("Macro sitout check error: %s", _mse)

            if action == "buy" and px > 0:
                # Cover existing short first (with ExitEvaluator confirmation — parity with backtest)
                if enable_shorts and symbol in portfolio.short_positions:
                    spos = portfolio.short_positions[symbol]
                    if spos.bars_held < min_hold_bars:
                        logger.debug(
                            "HOLD_SHORT %s — min hold not met (%d/%d bars)",
                            symbol, spos.bars_held, min_hold_bars,
                        )
                    elif _consecutive_short_exit_signals[symbol] < exit_confirm_bars:
                        logger.debug(
                            "HOLD_SHORT %s — exit confirmation %d/%d",
                            symbol, _consecutive_short_exit_signals[symbol], exit_confirm_bars,
                        )
                    else:
                        _short_exit_dec = _exit_eval.should_exit(
                            candles=candles,
                            entry_price=spos.avg_entry,
                            current_price=px,
                            bars_held=spos.bars_held,
                            side="short",
                            max_hold_bars=short_max_hold_bars if short_max_hold_bars > 0 else max_hold_bars,
                            entry_atr=getattr(spos, "entry_atr_pct", None) or None,
                            vol_ratio=sig.vol_ratio,
                            regime=_sym_regime,
                            max_favorable_pct=getattr(spos, "max_favorable_pct", 0.0),
                        )
                        if not _short_exit_dec.should_exit:
                            logger.debug(
                                "EXIT_MODEL_SHORT %s — thesis intact, holding despite buy signal",
                                symbol,
                            )
                        else:
                            _consecutive_short_exit_signals[symbol] = 0
                            cover_qty = spos.qty
                            s_avg_entry = spos.avg_entry
                            cover_px = px if _executor_owns_fill else px * (1.0 + slip_mult)
                            res = await _safe_exec(executor.close_short, symbol, cover_px, cover_qty, reason="buy_signal")
                            if res.ok:
                                _exit_s_regime = _entry_regime.pop(symbol, _current_regime.get(symbol, "unknown"))
                                now_ms = int(time.time() * 1000)
                                fee = cover_qty * cover_px * config.fee_rate
                                if not allow_live:
                                    s_bars = getattr(spos, "bars_held", 0)
                                    close_paper_trade(
                                        conn, symbol, "short", cover_px, fee, now_ms,
                                        close_reason="buy_signal",
                                        max_adverse_pct=getattr(spos, "max_adverse_pct", 0.0),
                                        max_favorable_pct=getattr(spos, "max_favorable_pct", 0.0),
                                        bars_held=s_bars,
                                        exit_regime=_exit_s_regime,
                                        entry_atr_pct=getattr(spos, "entry_atr_pct", None),
                                    )
                                evaluator._trade_outcomes.append(cover_px < s_avg_entry)
                                gross_pnl_pct = (s_avg_entry - cover_px) / s_avg_entry if s_avg_entry else 0
                                net_pnl_pct = gross_pnl_pct - 2 * config.fee_rate
                                _expectancy.record_trade(
                                    symbol=symbol,
                                    regime=_exit_s_regime,
                                    gross_pnl_pct=gross_pnl_pct,
                                    net_pnl_pct=net_pnl_pct,
                                    hold_bars=getattr(spos, "bars_held", 0),
                                    close_reason="buy_signal",
                                )
                                _record_outcome(symbol, _exit_s_regime, gross_pnl_pct, _entry_up_prob.pop(symbol, None))
                                is_loss = cover_px > s_avg_entry
                                if is_loss and loss_cooldown_bars > 0:
                                    _cooldown_remaining = loss_cooldown_bars
                                if notifier:
                                    notifier.notify("cover", {"symbol": symbol, "price": cover_px,
                                                              "qty": cover_qty, "reason": "buy_signal"})
                            if not res.ok:
                                logger.error("COVER_SHORT_FAILED %s px=%.2f qty=%.6f — %s",
                                             symbol, cover_px, cover_qty, getattr(res, "error", "unknown"))
                            logger.info("COVER_SHORT %s px=%.2f qty=%.6f equity=%.2f reason=buy_signal ok=%s",
                                        symbol, cover_px, cover_qty, equity, res.ok)

                if not sig.eff_allow_longs:
                    logger.info("BUY_BLOCK %s — regime %s disallows longs (eff_allow_longs=False)", symbol, _sym_regime)
                elif symbol in portfolio.positions:
                    pass  # already long
                elif symbol in portfolio.short_positions:
                    pass  # short still held (min_hold_bars not met); block long to prevent simultaneous long+short
                elif _cooldown_remaining > 0:
                    logger.debug("COOLDOWN %s — %d bars remaining", symbol, _cooldown_remaining)
                elif not _fx_session_ok:
                    logger.debug("FX_SESSION_BLOCK %s — outside allowed trading hours", symbol)
                else:
                    _momentum_scale = 1.0
                    if not evaluator._use_policy_core:
                        if len(candles) >= 8 and _sym_regime != "ranging":
                            _ema8 = candles["close"].ewm(span=8, min_periods=8).mean().iloc[-1]
                            if _ema8 > 0 and px < _ema8:
                                _pct_below = (_ema8 - px) / _ema8
                                _momentum_scale = max(0.40, 1.0 - _pct_below * 20.0)
                    _funding_scale = 1.0
                    if _funding_overlay is not None:
                        _funding_scale = _funding_overlay.position_scale("buy", datetime.now(tz=timezone.utc))
                    _exp_scale = _expectancy.expectancy_size_scale(regime=_sym_regime or "unknown")
                    # Portfolio correlation: reduce size if already holding another symbol
                    _corr_scale = 1.0
                    _other_syms = set(portfolio.positions) | set(portfolio.short_positions)
                    _other_syms.discard(symbol)
                    if _other_syms and config.portfolio_correlation_scale < 1.0:
                        _corr_scale = config.portfolio_correlation_scale
                        logger.debug("CORR_SCALE %s — other positions %s, scale=%.2f", symbol, _other_syms, _corr_scale)
                    _long_size = size * sig.eff_long_size_scale * _macro_scale * _momentum_scale * _funding_scale * _wf_scale * _exp_scale * _corr_scale
                    if _long_size <= 0:
                        logger.info("BUY_BLOCK %s — size scaled to zero (base=%.6f eff_scale=%.2f macro=%.2f mom=%.2f fund=%.2f)",
                                    symbol, size, sig.eff_long_size_scale, _macro_scale, _momentum_scale, _funding_scale)
                    else:
                        buy_px = px if _executor_owns_fill else px * (1.0 + slip_mult)
                        res = await _safe_exec(executor.open_long, symbol, buy_px, _long_size,
                                               trailing_stop_pct=_eff_stop,
                                               take_profit_pct=_eff_tp,
                                               trail_activation_pct=config.trail_activation_pct)
                        executed = bool(res.ok)
                        if executed:
                            _entry_regime[symbol] = _sym_regime or "unknown"
                            _entry_regime_conf[symbol] = getattr(sig, "final_confidence", 0.0)
                            _entry_up_prob[symbol] = up_prob
                            pos = portfolio.positions.get(symbol)
                            if pos is not None:
                                pos.entry_atr_pct = _sym_atr_pct
                                pos.breakeven_pct = getattr(config, "breakeven_stop_pct", 0.0)
                            now_ms = int(time.time() * 1000)
                            fee = _long_size * buy_px * config.fee_rate
                            if not allow_live:
                                open_paper_trade(conn, symbol, "long", buy_px, _long_size, fee, now_ms,
                                                 ml_up_prob=up_prob,
                                                 strategy_conf=sig.final_confidence,
                                                 vol_ratio=sig.vol_ratio)
                            if notifier:
                                notifier.notify("buy", {"symbol": symbol, "price": buy_px,
                                                        "qty": _long_size, "ml_up_prob": up_prob})
                            logger.info("BUY %s px=%.2f qty=%.6f ml=%.3f equity=%.2f regime=%s long_scale=%.2f",
                                        symbol, buy_px, _long_size, up_prob or -1, equity, _sym_regime, sig.eff_long_size_scale)
                        else:
                            _fail_detail = getattr(res, "error", "") or "unknown"
                            logger.warning("BUY_FAILED %s px=%.2f qty=%.6f — %s",
                                           symbol, buy_px, _long_size, _fail_detail)

            elif action == "sell" and px > 0:
                _closed_long_this_bar = False

                # ── Close existing long ───────────────────────────────
                if symbol in portfolio.positions:
                    pos = portfolio.positions[symbol]
                    if pos.bars_held < min_hold_bars:
                        logger.debug(
                            "HOLD %s — min hold not met (%d/%d bars)",
                            symbol, pos.bars_held, min_hold_bars,
                        )
                    elif sig.final_confidence < config.min_final_confidence * config.reversal_confidence_multiplier:
                        logger.debug(
                            "HOLD %s — reversal conf %.2f < asymmetric threshold %.2f",
                            symbol, sig.final_confidence,
                            config.min_final_confidence * config.reversal_confidence_multiplier,
                        )
                    elif _consecutive_exit_signals[symbol] < exit_confirm_bars:
                        logger.debug(
                            "HOLD %s — exit confirmation %d/%d",
                            symbol, _consecutive_exit_signals[symbol], exit_confirm_bars,
                        )
                    else:
                        exit_decision = _exit_eval.should_exit(
                            candles=candles,
                            entry_price=pos.avg_entry,
                            current_price=px,
                            bars_held=pos.bars_held,
                            side="long",
                            max_hold_bars=max_hold_bars,
                            entry_atr=getattr(pos, "entry_atr_pct", None) or None,
                            vol_ratio=sig.vol_ratio,
                            regime=_sym_regime,
                            max_favorable_pct=getattr(pos, "max_favorable_pct", 0.0),
                        )
                        if not exit_decision.should_exit:
                            logger.debug(
                                "EXIT_MODEL %s — thesis intact, holding despite sell signal",
                                symbol,
                            )
                        else:
                            sell_qty = pos.qty if size <= 0 else min(pos.qty, size)
                            sell_px = px if _executor_owns_fill else px * (1.0 - slip_mult)
                            avg_entry = pos.avg_entry
                            res = await _safe_exec(executor.close_long, symbol, sell_px, sell_qty, reason="signal")
                            executed = bool(res.ok)
                            _sig_entry_regime = _entry_regime.pop(symbol, _current_regime.get(symbol, "unknown"))
                            if executed:
                                _closed_long_this_bar = True
                                now_ms = int(time.time() * 1000)
                                exit_fee = sell_qty * sell_px * config.fee_rate
                                if not allow_live:
                                    _sig_bars = getattr(pos, "bars_held", 0)
                                    close_paper_trade(
                                        conn, symbol, "long", sell_px, exit_fee, now_ms, close_reason="signal",
                                        max_adverse_pct=getattr(pos, "max_adverse_pct", None),
                                        max_favorable_pct=getattr(pos, "max_favorable_pct", None),
                                        bars_held=_sig_bars,
                                        exit_regime=_sig_entry_regime,
                                        entry_atr_pct=getattr(pos, "entry_atr_pct", None),
                                    )
                                evaluator._trade_outcomes.append(sell_px > avg_entry)
                                gross_pnl_pct = (sell_px - avg_entry) / avg_entry if avg_entry else 0
                                net_pnl_pct = gross_pnl_pct - 2 * config.fee_rate
                                bars_held = getattr(pos, "bars_held", 0)
                                _expectancy.record_trade(
                                    symbol=symbol,
                                    regime=_sig_entry_regime,
                                    gross_pnl_pct=gross_pnl_pct,
                                    net_pnl_pct=net_pnl_pct,
                                    mae_pct=getattr(pos, "max_adverse_pct", 0.0),
                                    mfe_pct=getattr(pos, "max_favorable_pct", 0.0),
                                    hold_bars=bars_held,
                                    close_reason="signal",
                                )
                                _record_outcome(symbol, _sig_entry_regime, gross_pnl_pct, _entry_up_prob.pop(symbol, None))
                                if _perf_tracker:
                                    try:
                                        _perf_tracker.record_trade_outcome(
                                            symbol=symbol,
                                            regime=_sig_entry_regime,
                                            tech_action=action, tech_confidence=sig.tech_confidence or 0.0,
                                            sent_bias="neutral", sent_strength=0.0,
                                            macro_regime="unknown",
                                            realized_pnl=gross_pnl_pct,
                                        )
                                        _perf_trade_count += 1
                                    except Exception as exc:
                                        logger.debug("PerfTracker record error: %s", exc)
                                _consecutive_exit_signals[symbol] = 0
                                is_loss = sell_px < avg_entry
                                if is_loss and loss_cooldown_bars > 0:
                                    _cooldown_remaining = loss_cooldown_bars
                                if notifier:
                                    notifier.notify("sell", {"symbol": symbol, "price": sell_px,
                                                             "qty": sell_qty, "ml_up_prob": up_prob})
                            if not executed:
                                logger.error("SELL_FAILED %s px=%.2f qty=%.6f — %s",
                                             symbol, sell_px, sell_qty, getattr(res, "error", "unknown"))
                            logger.info("SELL %s px=%.2f qty=%.6f ml=%.3f equity=%.2f ok=%s",
                                        symbol, sell_px, sell_qty, up_prob or -1, equity, executed)

                # ── Open short when flat and shorts enabled ───────────
                _allow_short_entry = (
                    enable_shorts
                    and symbol not in portfolio.positions
                    and symbol not in portfolio.short_positions
                    and _fx_session_ok
                )
                if _closed_long_this_bar and not enable_close_and_reverse:
                    _allow_short_entry = False
                if _allow_short_entry:
                    if _closed_long_this_bar:
                        _cooldown_remaining = 0
                    _funding_short_scale = 1.0
                    if _funding_overlay is not None:
                        _funding_short_scale = _funding_overlay.position_scale("sell", datetime.now(tz=timezone.utc))
                    _exp_short_scale = _expectancy.expectancy_size_scale(regime=_sym_regime or "unknown")
                    # Portfolio correlation: reduce size if already holding another symbol
                    _corr_short_scale = 1.0
                    _other_short_syms = set(portfolio.positions) | set(portfolio.short_positions)
                    _other_short_syms.discard(symbol)
                    if _other_short_syms and config.portfolio_correlation_scale < 1.0:
                        _corr_short_scale = config.portfolio_correlation_scale
                        logger.debug("CORR_SCALE_SHORT %s — other positions %s, scale=%.2f", symbol, _other_short_syms, _corr_short_scale)
                    _short_size = size * sig.eff_short_size_scale * _macro_scale * _funding_short_scale * _wf_scale * _exp_short_scale * _corr_short_scale
                    if not sig.eff_allow_shorts:
                        logger.info("SHORT_BLOCK %s — regime %s disallows shorts (eff_allow_shorts=False)", symbol, _sym_regime)
                    elif ml_blind_blocks_shorts(evaluator._ml_probs):
                        logger.info("SHORT_BLOCK %s — ML blind (prob std too low for shorts)", symbol)
                    elif _cooldown_remaining > 0:
                        logger.info("SHORT_BLOCK %s — cooldown %d bars remaining", symbol, _cooldown_remaining)
                    elif _short_size <= 0:
                        logger.info("SHORT_BLOCK %s — size scaled to zero (base=%.6f eff_scale=%.2f macro=%.2f)",
                                    symbol, size, sig.eff_short_size_scale, _macro_scale)
                    else:
                        short_px = px if _executor_owns_fill else px * (1.0 - slip_mult)
                        res = await _safe_exec(
                            executor.open_short,
                            symbol, short_px, _short_size,
                            trailing_stop_pct=_eff_stop,
                            take_profit_pct=_eff_tp,
                            trail_activation_pct=config.trail_activation_pct,
                        )
                        if res.ok:
                            _entry_regime[symbol] = _sym_regime or "unknown"
                            _entry_regime_conf[symbol] = getattr(sig, "final_confidence", 0.0)
                            _entry_up_prob[symbol] = up_prob
                            spos = portfolio.short_positions.get(symbol)
                            if spos is not None:
                                spos.entry_atr_pct = _sym_atr_pct
                                spos.breakeven_pct = getattr(config, "breakeven_stop_pct", 0.0)
                            now_ms = int(time.time() * 1000)
                            fee = _short_size * short_px * config.fee_rate
                            if not allow_live:
                                open_paper_trade(conn, symbol, "short", short_px, _short_size, fee, now_ms,
                                                 ml_up_prob=up_prob,
                                                 strategy_conf=sig.final_confidence,
                                                 vol_ratio=sig.vol_ratio)
                            if notifier:
                                notifier.notify("short", {"symbol": symbol, "price": short_px,
                                                          "qty": _short_size, "ml_up_prob": up_prob})
                            logger.info("SHORT %s px=%.2f qty=%.6f ml=%.3f equity=%.2f regime=%s short_scale=%.2f",
                                        symbol, short_px, _short_size, up_prob or -1, equity, _sym_regime, sig.eff_short_size_scale)
                        else:
                            _fail_detail = getattr(res, "error", "") or "unknown"
                            logger.warning("SHORT_FAILED %s px=%.2f qty=%.6f — %s",
                                           symbol, short_px, _short_size, _fail_detail)
                elif not _allow_short_entry:
                    logger.info("SHORT_SKIP %s — entry preconditions not met (shorts=%s in_long=%s in_short=%s fx_ok=%s close_rev=%s)",
                                symbol, enable_shorts, symbol in portfolio.positions,
                                symbol in portfolio.short_positions, _fx_session_ok,
                                _closed_long_this_bar and not enable_close_and_reverse)
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
            except Exception as exc:
                logger.debug("SIGNAL_QUALITY metric error: %s", exc)

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

            # Weight proposals from PerformanceTracker — auto-apply or shadow
            if (
                _perf_tracker
                and _perf_trade_count > 0
                and _perf_trade_count % _PERF_PROPOSAL_INTERVAL == 0
            ):
                try:
                    proposal = _perf_tracker.propose_weight_update()
                    if proposal is not None:
                        _auto_apply = getattr(config, "auto_apply_weights", False)
                        _min_trades = getattr(config, "auto_apply_min_trades", 30)
                        _should_apply = (
                            _auto_apply
                            and hasattr(proposal, "should_update")
                            and proposal.should_update
                            and hasattr(proposal, "trade_count")
                            and proposal.trade_count >= _min_trades
                        )
                        if _should_apply and getattr(proposal, "regime_weights", None) is not None:
                            # Apply proposed weights to MetaWeigher, bounded ±0.05 per agent
                            _old_w = dict(evaluator.pipeline.meta_weigher._weights)
                            for regime, rw in proposal.regime_weights.items():
                                for agent_key in ("technical", "sentiment", "macro"):
                                    if agent_key in rw and agent_key in _old_w:
                                        delta = rw[agent_key] - _old_w[agent_key]
                                        delta = max(-0.05, min(0.05, delta))
                                        rw[agent_key] = _old_w[agent_key] + delta
                            # Apply the default (non-regime) weights
                            _new_w = proposal.regime_weights.get("default", proposal.regime_weights.get(
                                next(iter(proposal.regime_weights), "default"), {}
                            ))
                            if _new_w:
                                _bounded = {}
                                for k in ("technical", "sentiment", "macro"):
                                    if k in _new_w:
                                        _bounded[k] = max(0.10, _new_w[k])
                                if _bounded:
                                    evaluator.pipeline.meta_weigher.update_weights(_bounded)
                                    logger.info(
                                        "AUTO_APPLY_WEIGHTS: applied proposal (%d trades evidence) "
                                        "old=%s new=%s",
                                        getattr(proposal, "trade_count", 0),
                                        {k: round(v, 3) for k, v in _old_w.items()},
                                        {k: round(v, 3) for k, v in _bounded.items()},
                                    )
                        else:
                            logger.info(
                                "PERF_TRACKER weight proposal (shadow): %s",
                                {r: {k: round(v, 3) for k, v in w.items()} for r, w in proposal.regime_weights.items()}
                                if getattr(proposal, "regime_weights", None) is not None else str(proposal),
                            )
                    _perf_tracker.save_to_db()
                except Exception as exc:
                    logger.debug("PerformanceTracker proposal error: %s", exc)

            # Online learner periodic update
            if _online_learner is not None:
                _online_learner_bar_count += 1
                _ol_interval = getattr(config, "online_learning_interval", 50)
                if _online_learner_bar_count % _ol_interval == 0:
                    try:
                        _ol_result = _online_learner.update()
                        _drift_features = _ol_result.get("drift_features", [])
                        if len(_drift_features) >= 5:
                            _online_drift_scale = 0.70
                            logger.warning(
                                "ONLINE_DRIFT: %d features drifted — reducing ML confidence to 0.70x",
                                len(_drift_features),
                            )
                        elif len(_drift_features) >= 2:
                            _online_drift_scale = 0.85
                            logger.info(
                                "ONLINE_DRIFT: %d features drifted — reducing ML confidence to 0.85x",
                                len(_drift_features),
                            )
                        else:
                            _online_drift_scale = 1.0
                        logger.debug("ONLINE_LEARNER update: %s", _ol_result)
                    except Exception as exc:
                        logger.debug("OnlineLearner update error: %s", exc)

            # Phase 5E: Walk-forward auto-retrain trigger
            # Check every 240 events (~4h of 1h bars) whether model is stale
            if (
                getattr(config, "auto_retrain", False)
                and event_count > 0
                and event_count % 240 == 0
            ):
                try:
                    import glob as _retrain_glob
                    _model_files = _retrain_glob.glob(config.ml_model_path)
                    _model_age_h = 999.0
                    if _model_files:
                        _model_mtime = os.path.getmtime(_model_files[0])
                        _model_age_h = (time.time() - _model_mtime) / 3600.0
                    _schedule_h = getattr(config, "retrain_schedule_hours", 24.0)
                    if _model_age_h > _schedule_h:
                        with sqlite3.connect(config.db_path) as _retrain_conn:
                            _candle_count = _retrain_conn.execute(
                                "SELECT COUNT(*) FROM candles WHERE symbol = ?",
                                (config.symbols[0] if config.symbols else "BTC/USD",),
                            ).fetchone()[0]
                        _min_candles = getattr(config, "auto_retrain_min_candles", 1000)
                        if _candle_count >= _min_candles:
                            logger.info(
                                "AUTO_RETRAIN: model age %.1fh > %.1fh schedule, %d candles available — triggering retrain",
                                _model_age_h, _schedule_h, _candle_count,
                            )
                            from hogan_bot.retrain import retrain_once
                            _retrain_result = retrain_once(
                                db_path=config.db_path,
                                symbol=config.symbols[0] if config.symbols else "BTC/USD",
                                model_type=getattr(config, "retrain_model_type", "logreg"),
                                window_bars=getattr(config, "retrain_window_bars", 50000),
                                min_improvement=getattr(config, "retrain_min_improvement", 0.005),
                                promotion_metric=getattr(config, "retrain_promotion_metric", "roc_auc"),
                            )
                            logger.info("AUTO_RETRAIN result: %s", _retrain_result)
                        else:
                            logger.debug(
                                "AUTO_RETRAIN: insufficient candles (%d < %d), skipping",
                                _candle_count, _min_candles,
                            )
                except Exception as _retrain_exc:
                    logger.warning("AUTO_RETRAIN error: %s", _retrain_exc)

            # Heartbeat: periodic health log
            event_count += 1
            _heartbeat_count += 1
            if _heartbeat_count >= _HEARTBEAT_INTERVAL:
                _heartbeat_count = 0
                _n_longs = len(portfolio.positions)
                _n_shorts = len(portfolio.short_positions)
                logger.info(
                    "HEARTBEAT [%d events] equity=%.2f cash=%.2f longs=%d shorts=%d regime=%s",
                    event_count, equity, portfolio.cash_usd, _n_longs, _n_shorts,
                    _current_regime.get(symbol, "unknown"),
                )

    # Final expectancy report
    if _expectancy._trades:
        report = _expectancy.summary()
        logger.info("FINAL EXPECTANCY REPORT: %s", report)

    # Persist tracker state for restart survival
    if _perf_tracker:
        try:
            _perf_tracker.save_to_db()
            logger.info("PerformanceTracker state saved on shutdown")
        except Exception as exc:
            logger.debug("PerformanceTracker save on shutdown failed: %s", exc)

    logger.info("Event loop terminated after %d events.", event_count)


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
