"""Canonical decision core — single codepath for live and backtest.

Every trade decision in Hogan flows through ``decide()``.  Both
``event_loop.py`` (live/paper) and ``backtest.py`` call this function so
that the decision logic is provably identical in both contexts.

The decision pipeline:

    1. Regime detection + transition dampener
    2. AgentPipeline.run() → signal
    3. ML processing (filter / sizer / blind detection)
    4. Loss-streak dampener
    5. Gate chain: edge → quality → ranging → pullback
    6. Feature freshness check (live-only by default)
    7. Position sizing
    8. Macro sitout
    9. → DecisionIntent

Swarm integration:  When ``config.swarm_enabled`` is True, the swarm
controller runs in parallel (shadow) or in-line (active) and its output
is attached to the DecisionIntent for logging.  In shadow mode the
baseline decision is executed; in active mode the swarm decision replaces
it.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

import pandas as pd

from hogan_bot.swarm_decision.types import DecisionIntent

logger = logging.getLogger(__name__)


def _compute_data_ages(conn) -> dict[str, float]:
    """Hours since last update for each data source (PIT-safe utility)."""
    if conn is None:
        return {}
    ages: dict[str, float] = {}
    now_s = time.time()
    for table, source_key in (
        ("onchain_metrics", "onchain_db"),
        ("derivatives_metrics", "derivatives_db"),
        ("sentiment_scores", "sentiment_db"),
        ("macro_indicators", "macro_db"),
        ("intermarket_prices", "intermarket_db"),
    ):
        try:
            row = conn.execute(f"SELECT MAX(ts_ms) FROM {table}").fetchone()
            if row and row[0]:
                age_h = (now_s - row[0] / 1000.0) / 3600.0
                ages[source_key] = max(0.0, age_h)
        except Exception:
            pass
    return ages


# ---------------------------------------------------------------------------
# Pipeline state that persists across bars (per-symbol)
# ---------------------------------------------------------------------------

@dataclass
class PolicyState:
    """Rolling state consumed by the decision pipeline.

    Create one per symbol and feed it into ``decide()`` on each bar.
    The pipeline updates it in-place (ML prob history, trade outcomes, …).
    """

    ml_probs: list[float] = field(default_factory=list)
    trade_outcomes: list[bool] = field(default_factory=list)
    swarm_weights_logged: bool = False

    # Regime transition tracker (created lazily)
    _regime_transition: object | None = None

    @property
    def regime_transition(self):
        if self._regime_transition is None:
            from hogan_bot.regime import RegimeTransitionTracker
            self._regime_transition = RegimeTransitionTracker(
                cooldown_bars=3, min_scale=0.40,
            )
        return self._regime_transition

    def record_outcome(self, won: bool) -> None:
        self.trade_outcomes.append(won)


# ---------------------------------------------------------------------------
# Swarm weight resolution
# ---------------------------------------------------------------------------

def _parse_weight_string(weight_str: str) -> dict[str, float]:
    """Parse ``"agent_a:0.4,agent_b:0.3"`` into a dict."""
    weights: dict[str, float] = {}
    for pair in weight_str.split(","):
        pair = pair.strip()
        if ":" not in pair:
            continue
        agent_id, val = pair.split(":", 1)
        try:
            weights[agent_id.strip()] = float(val.strip())
        except ValueError:
            continue
    return weights


def _resolve_swarm_weights(
    config,
    conn,
    symbol: str,
    regime: str | None,
) -> dict[str, float] | None:
    """Resolve swarm weights in priority order.

    1. Regime-aware promoted weights from DB (if ``swarm_use_regime_weights``)
    2. Explicit config string (``HOGAN_SWARM_WEIGHTS``)
    3. None (uniform weights — SwarmController default)
    """
    # 1. Regime-aware lookup
    if getattr(config, "swarm_use_regime_weights", False) and conn is not None and regime:
        try:
            row = conn.execute(
                """SELECT weights_json FROM swarm_weight_snapshots
                   WHERE symbol = ? AND regime = ? AND source = 'promoted'
                   ORDER BY ts_ms DESC LIMIT 1""",
                (symbol, regime),
            ).fetchone()
            if row:
                import json
                return json.loads(row[0])
        except Exception:
            pass

    # 2. Explicit config string
    _w_str = getattr(config, "swarm_weights", "")
    if _w_str:
        parsed = _parse_weight_string(_w_str)
        if parsed:
            return parsed

    # 3. Uniform (let controller default)
    return None


# ---------------------------------------------------------------------------
# decide()
# ---------------------------------------------------------------------------

def decide(
    *,
    symbol: str,
    candles: pd.DataFrame,
    equity_usd: float,
    config,
    pipeline,
    ml_model=None,
    state: PolicyState,
    conn=None,
    as_of_ms: int | None = None,
    mode: str = "backtest",
    recent_whipsaw_count: int = 0,
    macro_sitout=None,
    funding_overlay=None,
    mtf_candles: dict[str, pd.DataFrame] | None = None,
    enable_pullback_gate: bool = True,
    enable_freshness_check: bool = False,
    peak_equity_usd: float | None = None,
) -> DecisionIntent:
    """Single canonical decision path used by both live and backtest.

    Parameters
    ----------
    symbol : str
        Trading pair (e.g. ``"BTC/USD"``).
    candles : pd.DataFrame
        OHLCV window up to (and including) the current bar.
    equity_usd : float
        Current portfolio equity in quote currency.
    config : BotConfig
        Fully resolved config (with champion mode applied if applicable).
    pipeline : AgentPipeline
        Pre-initialised agent pipeline.
    ml_model : TrainedModel | None
        Champion ML model (or RegimeModelRouter).
    state : PolicyState
        Per-symbol rolling state (ML probs, outcomes, transition tracker).
    conn : sqlite3.Connection | None
        DB connection for PIT queries (live) or ``None`` (backtest).
    as_of_ms : int | None
        Point-in-time timestamp for PIT-safe DB queries.
    mode : str
        ``"live"`` or ``"backtest"``.
    recent_whipsaw_count : int
        Number of recent whipsaw reversals for gate thresholds.
    macro_sitout : MacroSitout | None
        Macro event sit-out filter instance.
    funding_overlay : FundingOverlay | None
        BTC funding rate overlay instance.
    mtf_candles : dict | None
        Multi-timeframe candle data keyed by timeframe string.
    enable_pullback_gate : bool
        Whether to apply the pullback anti-chase filter.
    enable_freshness_check : bool
        Whether to run feature staleness checks (typically live-only).
    peak_equity_usd : float | None
        Portfolio high-water mark.  Used by RiskSteward to detect drawdown.
        When ``None``, defaults to ``equity_usd`` (no drawdown signal).
    """
    from hogan_bot.config import symbol_config
    from hogan_bot.indicators import compute_atr
    from hogan_bot.risk import calculate_position_size
    from hogan_bot.decision import (
        apply_ml_filter,
        edge_gate,
        entry_quality_gate,
        ranging_gate,
        pullback_gate,
        ml_confidence,
        ml_probability_sizer,
        ml_blind_scale,
        loss_streak_scale,
        estimate_spread_from_candles,
        compute_quality_components,
    )
    from hogan_bot.ml import predict_up_probability

    cfg = symbol_config(config, symbol)
    px = float(candles["close"].iloc[-1])
    block_reasons: list[str] = []

    # ------------------------------------------------------------------
    # 1. Regime detection + transition dampener
    # ------------------------------------------------------------------
    regime_name: str | None = None
    regime_conf: float | None = None
    _rstate = None

    if getattr(cfg, "use_regime_detection", True) and len(candles) >= 80:
        try:
            from hogan_bot.regime import detect_regime
            _rstate = detect_regime(candles)
            regime_name = _rstate.regime
            regime_conf = _rstate.confidence
        except Exception:
            pass

    eff: dict[str, float] = {}
    if _rstate is not None:
        try:
            from hogan_bot.regime import effective_thresholds
            eff = effective_thresholds(_rstate, cfg)
        except Exception:
            pass

    eff_ml_buy = eff.get("ml_buy_threshold", cfg.ml_buy_threshold)
    eff_ml_sell = eff.get("ml_sell_threshold", cfg.ml_sell_threshold)
    eff_tp = eff.get("take_profit_pct", cfg.take_profit_pct)
    eff_ts = eff.get("trailing_stop_pct", cfg.trailing_stop_pct)
    eff_position_scale = eff.get("position_scale", 1.0)
    eff_allow_longs = eff.get("allow_longs", True)
    eff_allow_shorts = eff.get("allow_shorts", True)
    eff_long_size_scale = eff.get("long_size_scale", 1.0)
    eff_short_size_scale = eff.get("short_size_scale", 1.0)

    _transition_scale = (
        state.regime_transition.update(regime_name) if regime_name else 1.0
    )
    eff_long_size_scale *= _transition_scale
    eff_short_size_scale *= _transition_scale
    eff_position_scale *= _transition_scale

    # ------------------------------------------------------------------
    # 2. AgentPipeline → signal
    # ------------------------------------------------------------------
    signal = pipeline.run(
        candles,
        symbol=symbol,
        config_override=cfg,
        regime=regime_name,
        regime_state=_rstate,
        as_of_ms=as_of_ms,
    )

    action = signal.action
    conf_scale = signal.confidence or 1.0

    # ------------------------------------------------------------------
    # 3. ML processing
    # ------------------------------------------------------------------
    up_prob: float | None = None

    if (cfg.use_ml_filter or cfg.use_ml_as_sizer) and ml_model is not None:
        if hasattr(ml_model, "set_regime"):
            ml_model.set_regime(regime_name)
        up_prob = predict_up_probability(candles, ml_model)
        state.ml_probs.append(up_prob)

        if cfg.use_ml_as_sizer:
            conf_scale *= ml_probability_sizer(action, up_prob)
        else:
            ml_gate = apply_ml_filter(action, up_prob, eff_ml_buy, eff_ml_sell)
            action = ml_gate.action
            if ml_gate.blocked_by:
                block_reasons.append(ml_gate.blocked_by)
            if cfg.ml_confidence_sizing:
                conf_scale *= ml_confidence(up_prob)

        _blind = ml_blind_scale(state.ml_probs)
        if _blind < 1.0:
            conf_scale *= _blind

    # ------------------------------------------------------------------
    # 4. Loss-streak dampener
    # ------------------------------------------------------------------
    _ls_scale = loss_streak_scale(state.trade_outcomes)
    if _ls_scale < 1.0:
        conf_scale *= _ls_scale

    # ------------------------------------------------------------------
    # 5. Gate chain: edge → quality → ranging → pullback
    # ------------------------------------------------------------------
    _atr_series = compute_atr(candles, window=14)
    atr_pct = float(_atr_series.iloc[-1]) / max(px, 1e-9)
    spread_est = estimate_spread_from_candles(candles)

    forecast_ret: float | None = None
    if signal.forecast is not None and getattr(signal.forecast, "confidence", 0) > 0.2:
        _er = signal.forecast.expected_return
        if isinstance(_er, dict) and _er:
            forecast_ret = max(abs(v) for v in _er.values())
        elif isinstance(_er, (int, float)):
            forecast_ret = abs(float(_er))

    # Edge gate
    _edge_gd = edge_gate(
        action,
        atr_pct=atr_pct,
        take_profit_pct=eff_tp,
        fee_rate=cfg.fee_rate,
        min_edge_multiple=getattr(cfg, "min_edge_multiple", 1.5),
        forecast_expected_return=forecast_ret,
        estimated_spread=spread_est,
    )
    action = _edge_gd.action
    if _edge_gd.blocked_by:
        block_reasons.append(_edge_gd.blocked_by)

    # Quality gate
    _tech_conf = signal.tech.confidence if signal.tech else None
    _quality_gd = entry_quality_gate(
        action,
        final_confidence=signal.confidence,
        tech_confidence=_tech_conf,
        regime=regime_name,
        regime_confidence=regime_conf,
        recent_whipsaw_count=recent_whipsaw_count,
        min_final_confidence=cfg.min_final_confidence,
        min_tech_confidence=cfg.min_tech_confidence,
        min_regime_confidence=cfg.min_regime_confidence,
        max_whipsaws=cfg.max_whipsaws,
    )
    action = _quality_gd.action
    quality_scale = _quality_gd.size_scale
    if _quality_gd.blocked_by:
        block_reasons.append(_quality_gd.blocked_by)

    # Ranging gate
    _tech_action = signal.tech.action if signal.tech else None
    _ranging_gd = ranging_gate(
        action,
        regime=regime_name,
        tech_action=_tech_action,
        up_prob=up_prob if ml_model is not None else None,
        recent_whipsaw_count=recent_whipsaw_count,
    )
    action = _ranging_gd.action
    ranging_scale = _ranging_gd.size_scale
    if _ranging_gd.blocked_by:
        block_reasons.append(_ranging_gd.blocked_by)

    # Pullback gate
    pullback_scale = 1.0
    if enable_pullback_gate:
        _pullback_gd = pullback_gate(action, candles, regime=regime_name)
        action = _pullback_gd.action
        pullback_scale = _pullback_gd.size_scale
        if _pullback_gd.blocked_by:
            block_reasons.append(_pullback_gd.blocked_by)

    # ------------------------------------------------------------------
    # 6. Feature freshness (live-only by default)
    # ------------------------------------------------------------------
    freshness_scale = 1.0
    _freshness: dict | None = None

    if enable_freshness_check and conn is not None:
        try:
            from hogan_bot.ml import build_feature_row_checked

            _data_ages = _compute_data_ages(conn)
            feat_result = build_feature_row_checked(
                candles, db_conn=conn, data_ages_hours=_data_ages,
            )
            if feat_result is not None:
                _freshness = feat_result.freshness_summary
                crit_stale = _freshness.get("critical_stale_count", 0)
                all_stale = _freshness.get("stale_count", 0)
                if crit_stale >= 2 and action != "hold":
                    action = "hold"
                    block_reasons.append("freshness_critical_block")
                elif crit_stale >= 1 or all_stale >= 4:
                    freshness_scale = 0.75
        except Exception:
            pass

    # ------------------------------------------------------------------
    # 7. Momentum confirmation (long-only)
    # ------------------------------------------------------------------
    momentum_scale = 1.0
    if action == "buy" and len(candles) >= 8:
        _ema8 = candles["close"].ewm(span=8, min_periods=8).mean().iloc[-1]
        if px < _ema8:
            momentum_scale = 0.3

    # ------------------------------------------------------------------
    # 8. Position sizing
    # ------------------------------------------------------------------
    composite_scale = (
        conf_scale
        * quality_scale
        * ranging_scale
        * pullback_scale
        * eff_position_scale
        * freshness_scale
        * momentum_scale
    )

    size = calculate_position_size(
        equity_usd=equity_usd,
        price=px,
        stop_distance_pct=signal.stop_distance_pct,
        max_risk_per_trade=cfg.max_risk_per_trade,
        max_allocation_pct=cfg.aggressive_allocation,
        confidence_scale=composite_scale,
        fee_rate=cfg.fee_rate,
    )

    # ------------------------------------------------------------------
    # 9. Macro sitout
    # ------------------------------------------------------------------
    macro_scale = 1.0
    if macro_sitout is not None and action != "hold":
        _bar_ts = candles.iloc[-1].get("timestamp") if "timestamp" in candles.columns else None
        _sitout = macro_sitout.check(_bar_ts)
        if _sitout.should_sitout:
            action = "hold"
            block_reasons.append("macro_sitout")
        elif _sitout.size_scale < 1.0:
            macro_scale = _sitout.size_scale
            size *= macro_scale

    # ------------------------------------------------------------------
    # Swarm Decision Layer (shadow / active)
    # ------------------------------------------------------------------
    swarm_decision = None
    _swarm_decision_id = None

    if getattr(config, "swarm_enabled", False):
        try:
            from hogan_bot.swarm_decision.controller import SwarmController
            from hogan_bot.swarm_decision.agents.pipeline_agent import PipelineAgent
            from hogan_bot.swarm_decision.agents.risk_steward import RiskStewardAgent
            from hogan_bot.swarm_decision.agents.data_guardian import DataGuardianAgent
            from hogan_bot.swarm_decision.agents.execution_cost import ExecutionCostAgent

            _agents_str = getattr(config, "swarm_agents", "")
            _agent_ids = [a.strip() for a in _agents_str.split(",") if a.strip()]

            agents = []
            for aid in _agent_ids:
                if aid == "pipeline_v1":
                    agents.append(PipelineAgent(pipeline))
                elif aid == "risk_steward_v1":
                    agents.append(RiskStewardAgent(
                        max_drawdown_pct=getattr(config, "swarm_risk_max_drawdown_pct", 0.10),
                        vol_scale_threshold=getattr(config, "swarm_risk_vol_scale_threshold", 2.5),
                        vol_veto_threshold=getattr(config, "swarm_risk_vol_veto_threshold", 4.0),
                    ))
                elif aid == "data_guardian_v1":
                    agents.append(DataGuardianAgent(
                        max_gap_bars=getattr(config, "swarm_data_max_gap_bars", 3),
                        max_stale_hours=getattr(config, "swarm_data_max_stale_hours", 2.0),
                        min_bars_required=getattr(config, "swarm_data_min_bars", 50),
                    ))
                elif aid == "execution_cost_v1":
                    agents.append(ExecutionCostAgent(
                        fee_rate=getattr(config, "swarm_exec_fee_rate", cfg.fee_rate),
                        min_edge_over_cost=getattr(config, "swarm_exec_min_edge_ratio", 1.5),
                    ))

            if agents:
                import math as _math_sw
                from hogan_bot.indicators import compute_atr as _sw_atr
                _hist_vol = float(candles["close"].pct_change().rolling(20).std().iloc[-1]) if len(candles) >= 21 else 0.0
                if _math_sw.isnan(_hist_vol) or _math_sw.isinf(_hist_vol):
                    _hist_vol = 0.0
                _shared_ctx = {
                    "regime": regime_name,
                    "regime_state": _rstate,
                    "equity_usd": equity_usd,
                    "peak_equity_usd": peak_equity_usd if peak_equity_usd is not None else equity_usd,
                    "atr_pct": atr_pct,
                    "hist_vol_20": _hist_vol,
                    "take_profit_pct": eff_tp,
                    "pipeline_signal": signal,
                    "up_prob": up_prob,
                }

                # Resolve weights: config string → regime DB lookup → uniform
                _sw_weights = _resolve_swarm_weights(config, conn, symbol, regime_name)

                controller = SwarmController(
                    agents=agents,
                    weights=_sw_weights,
                    config=config,
                )

                if conn is not None and not state.swarm_weights_logged:
                    try:
                        from hogan_bot.swarm_decision.logging import log_weight_snapshot
                        _bar_ts_ms_sw = int(candles["ts_ms"].iloc[-1]) if "ts_ms" in candles.columns else 0
                        log_weight_snapshot(
                            conn, _bar_ts_ms_sw, symbol,
                            getattr(config, "timeframe", "1h"),
                            controller.weights,
                            regime=regime_name,
                            source="static_init",
                        )
                        state.swarm_weights_logged = True
                    except Exception:
                        pass

                # Load agent modes from DB for quarantine/advisory enforcement
                _agent_modes: dict[str, str] = {}
                if conn is not None:
                    try:
                        from hogan_bot.agent_quarantine import get_all_agent_modes
                        _am = get_all_agent_modes(conn)
                        _agent_modes = {aid: st.mode for aid, st in _am.items()}
                    except Exception:
                        pass

                swarm_decision = controller.decide(
                    symbol=symbol,
                    candles=candles,
                    as_of_ms=as_of_ms,
                    shared_context=_shared_ctx,
                    agent_modes=_agent_modes,
                )

                _swarm_mode = getattr(config, "swarm_mode", "shadow")
                if _swarm_mode == "active" and swarm_decision is not None:
                    action = swarm_decision.final_action
                    size *= swarm_decision.final_size_scale

                if conn is not None:
                    try:
                        _bar_ts_ms = int(candles["ts_ms"].iloc[-1]) if "ts_ms" in candles.columns else 0
                        from hogan_bot.swarm_decision.logging import (
                            log_swarm_decision,
                            log_agent_votes,
                        )
                        _swarm_decision_id = log_swarm_decision(
                            conn, _bar_ts_ms, symbol,
                            getattr(config, "timeframe", "1h"),
                            swarm_decision, _swarm_mode,
                            as_of_ms=as_of_ms,
                            regime=regime_name,
                        )
                        if getattr(config, "swarm_log_full_votes", True):
                            log_agent_votes(
                                conn, _bar_ts_ms, symbol,
                                getattr(config, "timeframe", "1h"),
                                swarm_decision.votes,
                                as_of_ms=as_of_ms,
                                decision_id=_swarm_decision_id,
                            )
                        # Backfill outcomes for prior decisions
                        try:
                            from hogan_bot.swarm_decision.outcome_writer import backfill_outcomes
                            backfill_outcomes(conn, symbol=symbol, lookback_hours=72)
                        except Exception:
                            pass
                    except Exception:
                        pass
        except Exception as exc:
            logger.warning("Swarm layer error: %s", exc)

    # ------------------------------------------------------------------
    # Build and return DecisionIntent
    # ------------------------------------------------------------------
    return DecisionIntent(
        action=action,
        confidence=signal.confidence or 0.0,
        size_usd=size * px,
        stop_distance_pct=signal.stop_distance_pct,
        up_prob=up_prob,
        regime=regime_name,
        regime_confidence=regime_conf,
        atr_pct=atr_pct,
        eff_trailing_stop_pct=eff_ts,
        eff_take_profit_pct=eff_tp,
        eff_allow_longs=eff_allow_longs,
        eff_allow_shorts=eff_allow_shorts,
        eff_long_size_scale=eff_long_size_scale,
        eff_short_size_scale=eff_short_size_scale,
        conf_scale=conf_scale,
        quality_scale=quality_scale,
        ranging_scale=ranging_scale,
        pullback_scale=pullback_scale,
        momentum_scale=momentum_scale,
        freshness_scale=freshness_scale,
        macro_scale=macro_scale,
        explanation=signal.explanation,
        forecast_ret=forecast_ret,
        agent_weights=signal.agent_weights,
        feature_freshness=_freshness,
        vol_ratio=signal.volume_ratio,
        tech_confidence=_tech_conf,
        block_reasons=block_reasons,
        swarm=swarm_decision,
        swarm_decision_id=_swarm_decision_id,
    )
