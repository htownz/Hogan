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
from collections import deque
from dataclasses import dataclass, field

import pandas as pd

from hogan_bot.swarm_decision.types import DecisionIntent

logger = logging.getLogger(__name__)


_DATA_AGE_TABLES: dict[str, tuple[str, str]] = {
    "onchain_metrics": ("onchain_db", "date"),
    "derivatives_metrics": ("derivatives_db", "ts_ms"),
    "sentiment_scores": ("sentiment_db", "ts_ms"),
    "macro_indicators": ("macro_db", "ts_ms"),
    "intermarket_prices": ("intermarket_db", "ts_ms"),
}

def _compute_data_ages(conn) -> dict[str, float]:
    """Hours since last update for each data source (PIT-safe utility)."""
    if conn is None:
        return {}
    ages: dict[str, float] = {}
    now_s = time.time()
    for table, (source_key, col) in _DATA_AGE_TABLES.items():
        try:
            row = conn.execute(
                f"SELECT MAX({col}) FROM {table}"  # noqa: S608 — table/col from hardcoded whitelist
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
            if "no such table" in str(exc).lower():
                logger.debug("_compute_data_ages: table %s not created yet", table)
            else:
                logger.warning("_compute_data_ages %s error: %s", table, exc)
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

    ml_probs: deque[float] = field(default_factory=lambda: deque(maxlen=50))
    trade_outcomes: deque[bool] = field(default_factory=lambda: deque(maxlen=50))
    swarm_weights_logged: bool = False
    swarm_bars_since_weight_learn: int = 0

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
        except Exception as exc:
            logger.debug("Swarm regime weight DB lookup failed: %s", exc)

    # 1b. Global promoted weights (fallback when regime-specific not available)
    if conn is not None:
        try:
            row = conn.execute(
                """SELECT weights_json FROM swarm_weight_snapshots
                   WHERE symbol = ? AND regime IS NULL AND source = 'promoted'
                   ORDER BY ts_ms DESC LIMIT 1""",
                (symbol,),
            ).fetchone()
            if row:
                import json
                return json.loads(row[0])
        except Exception as exc:
            logger.debug("Swarm global weight DB lookup failed: %s", exc)

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
    from hogan_bot.decision import (
        apply_ml_filter,
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
        sell_pullback_gate,
    )
    from hogan_bot.indicators import compute_atr
    from hogan_bot.ml import predict_up_probability
    from hogan_bot.risk import calculate_position_size

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
            _rstate = detect_regime(candles, symbol=symbol)
            regime_name = _rstate.regime
            regime_conf = _rstate.confidence
        except Exception as exc:
            logger.warning("Regime detection failed (using fallback): %s", exc)

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

    _transition_scale = (
        state.regime_transition.update(regime_name) if regime_name else 1.0
    )
    eff_long_size_scale *= _transition_scale
    eff_short_size_scale *= _transition_scale
    # NOTE: Do NOT apply _transition_scale to eff_position_scale here.
    # eff_position_scale flows into composite_scale → size, and then size is
    # multiplied by eff_long/short_size_scale in the execution layer.  Applying
    # transition dampening to both would square the effect (0.40 → 0.16).

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
        mtf_candles=mtf_candles,
    )

    _raw_tech_action = signal.tech.action if signal.tech else None
    _pipeline_action = signal.action

    action = signal.action
    conf_scale = 1.0

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

        if action == "sell" and ml_blind_blocks_shorts(state.ml_probs):
            action = "hold"
            block_reasons.append("ml_blind_blocks_shorts")

    # ------------------------------------------------------------------
    # 3b. Optional regime-ensemble blend
    # ------------------------------------------------------------------
    if up_prob is not None and getattr(cfg, "use_regime_ensemble", False):
        try:
            from hogan_bot.ml_advanced import load_artifact
            from hogan_bot.ml_advanced import predict_up_probability as regime_predict
            _ensemble_path = getattr(cfg, "regime_ensemble_path", "models/advanced_ensemble.pkl")
            _artifact = load_artifact(_ensemble_path)
            if _artifact is not None:
                _regime_prob = regime_predict(_artifact, candles)
                _blend = getattr(cfg, "regime_ensemble_blend", 0.30)
                up_prob = up_prob * (1.0 - _blend) + _regime_prob * _blend
                logger.debug(
                    "REGIME_ENSEMBLE: blended prob=%.4f (standard=%.4f, regime=%.4f, blend=%.2f)",
                    up_prob, up_prob, _regime_prob, _blend,
                )
        except Exception as exc:
            logger.debug("Regime ensemble unavailable: %s", exc)

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
    # ATR-adaptive trailing stop floor: never tighter than 1.5× current ATR
    eff_ts = max(eff_ts, atr_pct * 1.5)

    # Adaptive stop widening via RiskHead MAE: if the model predicts a high
    # probability of hitting the stop, widen by the MAE ratio (capped at 1.5×).
    _risk_est = signal.risk_estimate
    if _risk_est is not None and _risk_est.stop_hit_prob > 0.65:
        _stop_pct_bps = signal.stop_distance_pct * 100  # convert to bps-like
        if _stop_pct_bps > 0:
            _mae_scale = max(1.0, _risk_est.max_adverse_pct / _stop_pct_bps)
            signal.stop_distance_pct *= min(1.5, _mae_scale)
            logger.debug(
                "POLICY: adaptive stop widened (stop_hit_prob=%.2f, mae_scale=%.2f, new_stop=%.4f)",
                _risk_est.stop_hit_prob, _mae_scale, signal.stop_distance_pct,
            )

    spread_est = estimate_spread_from_candles(candles)

    forecast_ret: float | None = None
    if signal.forecast is not None and getattr(signal.forecast, "confidence", 0) > 0.2:
        _er = signal.forecast.expected_return
        if isinstance(_er, dict) and _er:
            forecast_ret = max(abs(v) for v in _er.values())
        elif isinstance(_er, (int, float)):
            forecast_ret = abs(float(_er))

    # Paper-mode gate relaxation: lower thresholds by 20% to generate more
    # trades for learning.  Only active when paper_mode AND paper_relaxed_gates.
    _paper_relax = getattr(cfg, "paper_relaxed_gates", False) and getattr(cfg, "paper_mode", True)

    # Edge gate
    _edge_relax = 0.70 if _paper_relax else 1.0  # 30% wider edge gate in paper mode
    _edge_gd = edge_gate(
        action,
        atr_pct=atr_pct,
        take_profit_pct=eff_tp,
        fee_rate=cfg.fee_rate,
        min_edge_multiple=getattr(cfg, "min_edge_multiple", 1.5) * _edge_relax,
        forecast_expected_return=forecast_ret,
        estimated_spread=spread_est,
        atr_friction_multiple=getattr(cfg, "sell_atr_friction_multiple", 0.8) * _edge_relax,
        buy_atr_friction_multiple=getattr(cfg, "buy_atr_friction_multiple", 0.25) * _edge_relax,
    )
    action = _edge_gd.action
    if _edge_gd.blocked_by:
        block_reasons.append(_edge_gd.blocked_by)

    # Quality gate
    _tech_conf = signal.tech.confidence if signal.tech else None
    _relax_mult = 0.80 if _paper_relax else 1.0
    _tech_source = signal.tech.details.get("source") if signal.tech and signal.tech.details else None
    _quality_gd = entry_quality_gate(
        action,
        final_confidence=signal.confidence,
        tech_confidence=_tech_conf,
        regime=regime_name,
        regime_confidence=regime_conf,
        recent_whipsaw_count=recent_whipsaw_count,
        min_final_confidence=cfg.min_final_confidence * _relax_mult,
        min_tech_confidence=cfg.min_tech_confidence * _relax_mult,
        min_regime_confidence=cfg.min_regime_confidence * _relax_mult,
        max_whipsaws=cfg.max_whipsaws,
        tech_source=_tech_source,
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

        _sell_pullback_gd = sell_pullback_gate(action, candles, regime=regime_name)
        action = _sell_pullback_gd.action
        pullback_scale = min(pullback_scale, _sell_pullback_gd.size_scale)
        if _sell_pullback_gd.blocked_by:
            block_reasons.append(_sell_pullback_gd.blocked_by)

    # ------------------------------------------------------------------
    # 5b. MTF signal confirmation (soft adjustments, not hard blocks)
    # ------------------------------------------------------------------
    mtf_conf_scale = 1.0
    mtf_size_scale = 1.0
    if mtf_candles and action != "hold":
        # 3h trend alignment: penalise signals that disagree with 3h trend
        _candles_3h = mtf_candles.get("3h")
        if _candles_3h is not None and len(_candles_3h) >= 20:
            try:
                _h3_close = _candles_3h["close"].astype(float)
                _h3_ma20 = float(_h3_close.rolling(20).mean().iloc[-1])
                _h3_trend_up = float(_h3_close.iloc[-1]) > _h3_ma20
                if action == "buy" and not _h3_trend_up:
                    mtf_conf_scale *= 0.70
                    logger.debug("MTF: 3h trend DOWN disagrees with BUY, conf -30%%")
                elif action == "sell" and _h3_trend_up:
                    mtf_conf_scale *= 0.70
                    logger.debug("MTF: 3h trend UP disagrees with SELL, conf -30%%")
                else:
                    logger.debug("MTF: 3h trend aligned with %s", action)
            except Exception as _h3_exc:
                logger.debug("MTF 3h trend check error: %s", _h3_exc)

        # 15m momentum: penalise when MACD histogram opposes signal direction
        _candles_15m = mtf_candles.get("15m")
        if _candles_15m is not None and len(_candles_15m) >= 26:
            try:
                _m15_close = _candles_15m["close"].astype(float)
                _ema12 = _m15_close.ewm(span=12, adjust=False).mean()
                _ema26 = _m15_close.ewm(span=26, adjust=False).mean()
                _macd_line = _ema12 - _ema26
                _signal_line = _macd_line.ewm(span=9, adjust=False).mean()
                _m15_macd_hist = float((_macd_line - _signal_line).iloc[-1])
                if action == "buy" and _m15_macd_hist < 0:
                    mtf_size_scale *= 0.85
                    logger.debug("MTF: 15m MACD hist negative (%.6f), size -15%%", _m15_macd_hist)
                elif action == "sell" and _m15_macd_hist > 0:
                    mtf_size_scale *= 0.85
                    logger.debug("MTF: 15m MACD hist positive (%.6f), size -15%%", _m15_macd_hist)
                else:
                    logger.debug("MTF: 15m MACD hist aligned with %s (%.6f)", action, _m15_macd_hist)
            except Exception as _m15_exc:
                logger.debug("MTF 15m momentum check error: %s", _m15_exc)

    # Apply MTF confirmation scale to conf_scale
    conf_scale *= mtf_conf_scale

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
        except Exception as exc:
            logger.warning("Feature freshness check failed: %s", exc)

    # ------------------------------------------------------------------
    # 6b. Macro correlation filter (SPY/QQQ/DXY/VIX/GLD)
    # ------------------------------------------------------------------
    macro_filter_scale = 1.0
    if getattr(cfg, "use_macro_filter", False) and conn is not None and action != "hold":
        try:
            from hogan_bot.macro_filter import evaluate_macro
            _mf = evaluate_macro(
                conn,
                action=action,
                vix_caution=getattr(cfg, "macro_vix_caution", 25.0),
                vix_block=getattr(cfg, "macro_vix_block", 35.0),
            )
            if _mf.block_longs and action == "buy":
                action = "hold"
                block_reasons.append("macro_filter_block_longs")
            elif _mf.confidence_mult < 1.0:
                macro_filter_scale = _mf.confidence_mult
        except Exception as _mf_exc:
            logger.warning("Macro filter error: %s", _mf_exc)

    # ------------------------------------------------------------------
    # 7. Momentum confirmation (long-only, graduated)
    # ------------------------------------------------------------------
    momentum_scale = 1.0
    if action == "buy" and len(candles) >= 8 and regime_name != "ranging":
        _ema8 = candles["close"].ewm(span=8, min_periods=8).mean().iloc[-1]
        if _ema8 > 0 and px < _ema8:
            _pct_below = (_ema8 - px) / _ema8
            momentum_scale = max(0.40, 1.0 - _pct_below * 20.0)

    # ------------------------------------------------------------------
    # 7b. Forecast-driven position sizing (Phase 5D)
    # ------------------------------------------------------------------
    forecast_size_scale = 1.0
    if (
        getattr(cfg, "forecast_driven_sizing", False)
        and action != "hold"
        and signal.forecast is not None
        and getattr(signal.forecast, "confidence", 0) > 0.2
    ):
        try:
            _fc = signal.forecast
            _fc_er = _fc.expected_return
            # Resolve expected_return (may be dict or scalar)
            if isinstance(_fc_er, dict) and _fc_er:
                _fc_ret = max(_fc_er.values()) if action == "buy" else min(_fc_er.values())
            elif isinstance(_fc_er, (int, float)):
                _fc_ret = float(_fc_er)
            else:
                _fc_ret = None

            if _fc_ret is not None and eff_tp > 0:
                _fc_ratio = abs(_fc_ret) / eff_tp
                # Direction check: forecast should agree with action
                _fc_agrees = (action == "buy" and _fc_ret > 0) or (action == "sell" and _fc_ret < 0)
                if not _fc_agrees:
                    forecast_size_scale = 0.50
                    logger.debug("FORECAST_SIZING: direction conflict (fc=%.4f, action=%s) → 0.50×", _fc_ret, action)
                elif _fc_ratio > 2.0:
                    forecast_size_scale = 1.20
                    logger.debug("FORECAST_SIZING: strong conviction (ratio=%.2f) → 1.20×", _fc_ratio)
                elif _fc_ratio < 0.5:
                    forecast_size_scale = 0.70
                    logger.debug("FORECAST_SIZING: weak conviction (ratio=%.2f) → 0.70×", _fc_ratio)
        except Exception as _fc_exc:
            logger.debug("Forecast-driven sizing error: %s", _fc_exc)

    # ------------------------------------------------------------------
    # 8. Position sizing
    # ------------------------------------------------------------------
    _MIN_COMPOSITE_SCALE = 0.15
    composite_scale = max(
        _MIN_COMPOSITE_SCALE,
        conf_scale
        * quality_scale
        * ranging_scale
        * pullback_scale
        * eff_position_scale
        * freshness_scale
        * momentum_scale
        * macro_filter_scale
        * mtf_size_scale
        * forecast_size_scale,
    )

    # Compute average ATR for volatility-adjusted sizing
    _avg_atr_pct = 0.0
    if len(_atr_series) >= 50:
        _avg_atr_pct = float(_atr_series.iloc[-50:].mean()) / max(px, 1e-9)

    size = calculate_position_size(
        equity_usd=equity_usd,
        price=px,
        stop_distance_pct=signal.stop_distance_pct,
        max_risk_per_trade=cfg.max_risk_per_trade,
        max_allocation_pct=cfg.aggressive_allocation,
        confidence_scale=composite_scale,
        fee_rate=cfg.fee_rate,
        atr_pct=atr_pct,
        avg_atr_pct=_avg_atr_pct,
    )

    # ------------------------------------------------------------------
    # 9. Macro sitout
    # ------------------------------------------------------------------
    macro_scale = 1.0
    if macro_sitout is not None and action != "hold":
        _bar_ts = candles.iloc[-1]["timestamp"] if "timestamp" in candles.columns else None
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
            from hogan_bot.swarm_decision.agents.data_guardian import DataGuardianAgent
            from hogan_bot.swarm_decision.agents.execution_cost import (
                ExecutionCostAgent,
            )
            from hogan_bot.swarm_decision.agents.pipeline_agent import PipelineAgent
            from hogan_bot.swarm_decision.agents.risk_steward import RiskStewardAgent
            from hogan_bot.swarm_decision.agents.volatility_regime import (
                VolatilityRegimeAgent,
            )
            from hogan_bot.swarm_decision.controller import SwarmController

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
                elif aid == "volatility_regime_v1":
                    agents.append(VolatilityRegimeAgent())

            if agents:
                import math as _math_sw

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
                    "trade_outcomes": list(state.trade_outcomes),
                    "data_ages": _data_ages if "_data_ages" in locals() else {},
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
                    except Exception as exc:
                        logger.debug("Swarm weight snapshot log error: %s", exc)

                # Load agent modes from DB for quarantine/advisory enforcement
                _agent_modes: dict[str, str] = {}
                if conn is not None:
                    try:
                        from hogan_bot.agent_quarantine import get_all_agent_modes
                        _am = get_all_agent_modes(conn)
                        _agent_modes = {aid: st.mode for aid, st in _am.items()}
                    except Exception as exc:
                        logger.debug("Agent modes load error: %s", exc)

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
                elif _swarm_mode == "conditional_active" and swarm_decision is not None:
                    _ca_min_agreement = getattr(config, "swarm_conditional_min_agreement", 0.70)
                    _ca_min_confidence = getattr(config, "swarm_conditional_min_confidence", 0.60)
                    if (
                        swarm_decision.agreement >= _ca_min_agreement
                        and swarm_decision.final_confidence >= _ca_min_confidence
                        and not swarm_decision.vetoed
                    ):
                        action = swarm_decision.final_action
                        size *= swarm_decision.final_size_scale
                        logger.debug(
                            "SWARM conditional_active OVERRIDE: action=%s agreement=%.2f conf=%.2f",
                            action, swarm_decision.agreement, swarm_decision.final_confidence,
                        )

                if conn is not None:
                    try:
                        _bar_ts_ms = int(candles["ts_ms"].iloc[-1]) if "ts_ms" in candles.columns else 0
                        from hogan_bot.swarm_decision.logging import (
                            log_agent_votes,
                            log_swarm_decision,
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
                            from hogan_bot.swarm_decision.outcome_writer import (
                                backfill_outcomes,
                            )
                            backfill_outcomes(conn, symbol=symbol, lookback_hours=72)
                        except Exception as exc:
                            logger.debug("Swarm outcome backfill error: %s", exc)

                        # Periodic weight learning loop
                        state.swarm_bars_since_weight_learn += 1
                        _wl_interval = getattr(config, "swarm_weight_learning_interval_bars", 24)
                        if (
                            getattr(config, "swarm_weight_learning_enabled", False)
                            and state.swarm_bars_since_weight_learn >= _wl_interval
                        ):
                            state.swarm_bars_since_weight_learn = 0
                            try:
                                from hogan_bot.swarm_decision.weight_learner import (
                                    WeightProposal,
                                )
                                from hogan_bot.swarm_decision.weight_learner import (
                                    log_weight_proposal as _log_w_proposal,
                                )
                                from hogan_bot.swarm_decision.weight_learner import (
                                    promote_weights as _promote_w,
                                )
                                from hogan_bot.swarm_decision.weight_learner import (
                                    propose_weights as _propose_w,
                                )
                                _tf = getattr(config, "timeframe", "1h")
                                _min_t = getattr(config, "swarm_weight_min_trades", 50)
                                _max_s = getattr(config, "swarm_weight_max_daily_shift", 0.05)
                                _auto = getattr(config, "swarm_weight_auto_promote", False)

                                # Regime-specific proposal (for regime-aware weights)
                                if regime_name:
                                    _wl_regime = _propose_w(
                                        conn,
                                        current_weights=controller.weights,
                                        symbol=symbol,
                                        min_trades=_min_t,
                                        max_daily_shift=_max_s,
                                        regime=regime_name,
                                    )
                                    _log_w_proposal(conn, symbol, _tf, _wl_regime)
                                    if _wl_regime.min_trades_met and not _wl_regime.stable and _auto:
                                        _promote_w(conn, symbol, _tf, _wl_regime)
                                        if getattr(config, "swarm_use_regime_weights", False):
                                            controller.set_weights(_wl_regime.proposed_weights)
                                            logger.info(
                                                "SWARM regime weight promoted (%s): %s",
                                                regime_name,
                                                {k: round(v, 4) for k, v in _wl_regime.proposed_weights.items()},
                                            )

                                # Global proposal (all-regime fallback)
                                _wl_global = _propose_w(
                                    conn,
                                    current_weights=controller.weights,
                                    symbol=symbol,
                                    min_trades=_min_t,
                                    max_daily_shift=_max_s,
                                    regime=None,
                                )
                                _wl_global_copy = WeightProposal(
                                    current_weights=_wl_global.current_weights,
                                    proposed_weights=_wl_global.proposed_weights,
                                    deltas=_wl_global.deltas,
                                    evidence=_wl_global.evidence,
                                    min_trades_met=_wl_global.min_trades_met,
                                    regime=None,
                                    stable=_wl_global.stable,
                                    notes=_wl_global.notes,
                                )
                                _log_w_proposal(conn, symbol, _tf, _wl_global_copy)
                                if _wl_global.min_trades_met and not _wl_global.stable and _auto:
                                    _promote_w(conn, symbol, _tf, _wl_global_copy)
                                    if not getattr(config, "swarm_use_regime_weights", False):
                                        controller.set_weights(_wl_global.proposed_weights)
                                        logger.info(
                                            "SWARM global weight promoted: %s",
                                            {k: round(v, 4) for k, v in _wl_global.proposed_weights.items()},
                                        )
                                # Auto-quarantine underperformers
                                try:
                                    from hogan_bot.agent_quarantine import (
                                        auto_quarantine_check,
                                    )
                                    _quarantined = auto_quarantine_check(conn, symbol=symbol)
                                    if _quarantined:
                                        logger.info("SWARM auto-quarantined: %s", _quarantined)
                                except Exception as _aq_exc:
                                    logger.debug("Auto-quarantine error: %s", _aq_exc)
                            except Exception as exc:
                                logger.debug("Swarm weight learning error: %s", exc)
                    except Exception as exc:
                        logger.warning("Swarm decision logging error: %s", exc)
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
        raw_tech_action=_raw_tech_action,
        pipeline_action=_pipeline_action,
    )
