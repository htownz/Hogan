"""Multi-Agent Signal Pipeline — Hogan Market OS.

Four-plane decision architecture:

    TechnicalAgent  --+
    SentimentAgent  --+--> MetaWeigher --> AgentSignal
    MacroAgent      --+
    ForecastHead    ------> direction_prob, expected_return, trend_persistence
    RiskHead        ------> expected_vol, MAE, stop_hit_prob, hold_time

The MetaWeigher combines agent votes into an action. The forecast and risk
heads produce independent estimates that the policy layer uses to validate
*whether there is edge* and *whether size/exits are acceptable*.

Weights are adaptive — updated by the online learner on regime shifts.

Usage::

    pipeline = AgentPipeline(config, conn=conn)
    result = pipeline.run(candles, symbol="BTC/USD")
    # result.action, result.confidence, result.forecast, result.risk_estimate
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from hogan_bot.forecast import ForecastResult, compute_forecast
from hogan_bot.risk_head import RiskEstimate, compute_risk

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Signal data classes
# ---------------------------------------------------------------------------

@dataclass
class TechSignal:
    action: str          # "buy" | "sell" | "hold"
    confidence: float    # [0.0, 1.0]
    stop_distance_pct: float = 0.02
    volume_ratio: float = 1.0
    details: dict = field(default_factory=dict)


@dataclass
class SentimentSignal:
    bias: str            # "bullish" | "bearish" | "neutral"
    strength: float      # [0.0, 1.0]
    details: dict = field(default_factory=dict)


@dataclass
class MacroSignal:
    regime: str          # "risk_on" | "risk_off" | "neutral"
    risk_on: bool = True
    details: dict = field(default_factory=dict)


@dataclass
class AgentSignal:
    """Final combined signal from the MetaWeigher + forecast + risk heads."""
    action: str           # "buy" | "sell" | "hold"
    confidence: float     # [0.0, 1.0]
    stop_distance_pct: float = 0.02
    volume_ratio: float = 1.0
    tech: TechSignal | None = None
    sentiment: SentimentSignal | None = None
    macro: MacroSignal | None = None
    forecast: ForecastResult | None = None
    risk_estimate: RiskEstimate | None = None
    explanation: str = ""
    agent_weights: dict = field(default_factory=dict)
    combined_score: float = 0.0   # raw directional score from MetaWeigher [-1, 1]
    timestamp: float = field(default_factory=time.time)

    @property
    def action_str(self) -> str:
        return self.action


# ---------------------------------------------------------------------------
# Technical Agent
# ---------------------------------------------------------------------------

class TechnicalAgent:
    """Wraps generate_signal() as one sub-brain — MA, EMA clouds, FVG, ICT, RL.

    When ``use_strategy_router=True`` (default), signal generation is
    delegated to the :class:`~hogan_bot.strategy_router.StrategyRouter`
    which selects a regime-appropriate strategy family.  When the router
    is disabled or no regime is available, falls back to the classic
    ``generate_signal()`` path.
    """

    def __init__(self, config) -> None:
        self.config = config
        self._router = None
        if getattr(config, "use_strategy_router", True):
            try:
                from hogan_bot.strategy_router import StrategyRouter
                self._router = StrategyRouter(config)
            except Exception:
                pass

    def analyze(self, candles: pd.DataFrame, **runtime_state) -> TechSignal:
        """Produce a TechSignal. ``runtime_state`` passes dynamic per-bar RL
        fields (rl_in_position, rl_unrealized_pnl, rl_bars_in_trade) and
        optionally ``regime_state`` for strategy routing."""
        cfg = self.config
        if candles.empty or len(candles) < max(cfg.long_ma_window, 20):
            return TechSignal(action="hold", confidence=0.0)

        regime_state = runtime_state.pop("regime_state", None)

        # Strategy-router path: regime-aware family selection
        if self._router is not None and regime_state is not None:
            try:
                raw = self._router.route(candles, cfg, regime_state)
                conf = float(raw.confidence)
                if raw.action != "hold" and conf <= 0.0:
                    conf = 1.0
                family_name = self._router.families.get(
                    regime_state.regime, self._router.families.get("trending_up")
                )
                src = f"router/{getattr(family_name, 'name', 'unknown')}"
                return TechSignal(
                    action=raw.action,
                    confidence=conf,
                    stop_distance_pct=float(raw.stop_distance_pct),
                    volume_ratio=float(raw.volume_ratio),
                    details={"source": src, "regime": regime_state.regime},
                )
            except Exception as exc:
                logger.warning("StrategyRouter failed, falling back: %s", exc)

        # Fallback: classic generate_signal() path
        try:
            from hogan_bot.strategy import generate_signal

            kwargs: dict[str, Any] = dict(
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
                atr_stop_multiplier=getattr(cfg, "atr_stop_multiplier", 1.5),
                use_ict=getattr(cfg, "use_ict", False),
                ict_swing_left=getattr(cfg, "ict_swing_left", 2),
                ict_swing_right=getattr(cfg, "ict_swing_right", 2),
                ict_eq_tolerance_pct=getattr(cfg, "ict_eq_tolerance_pct", 0.0008),
                ict_min_displacement_pct=getattr(cfg, "ict_min_displacement_pct", 0.003),
                ict_require_time_window=getattr(cfg, "ict_require_time_window", True),
                ict_time_windows=getattr(cfg, "ict_time_windows", "03:00-04:00,10:00-11:00,14:00-15:00"),
                ict_require_pd=getattr(cfg, "ict_require_pd", True),
                ict_ote_enabled=getattr(cfg, "ict_ote_enabled", False),
                ict_ote_low=getattr(cfg, "ict_ote_low", 0.62),
                ict_ote_high=getattr(cfg, "ict_ote_high", 0.79),
                use_rl_agent=getattr(cfg, "use_rl_agent", False),
                rl_policy=getattr(cfg, "rl_policy", None),
            )
            kwargs.update(runtime_state)

            raw = generate_signal(candles, **kwargs)
            conf = float(raw.confidence)
            if raw.action != "hold" and conf <= 0.0:
                conf = 1.0
            return TechSignal(
                action=raw.action,
                confidence=conf,
                stop_distance_pct=float(raw.stop_distance_pct),
                volume_ratio=float(raw.volume_ratio),
                details={"source": "strategy.generate_signal"},
            )
        except Exception as exc:
            logger.warning("TechnicalAgent failed: %s", exc, exc_info=True)
            return TechSignal(action="hold", confidence=0.0)


# ---------------------------------------------------------------------------
# Sentiment Agent
# ---------------------------------------------------------------------------

class SentimentAgent:
    """Weighs fear/greed, news sentiment, funding rate, and social volume."""

    def __init__(self, conn=None, symbol: str = "BTC/USD") -> None:
        self.conn = conn
        self.symbol = symbol

    def analyze(self, as_of_ms: int | None = None) -> SentimentSignal:
        """Produce a sentiment signal.

        Parameters
        ----------
        as_of_ms
            Point-in-time cutoff (epoch ms).  When set, all DB queries
            restrict to data available *at or before* this timestamp.
            ``None`` means "now" (live mode).
        """
        if self.conn is None:
            return SentimentSignal(bias="neutral", strength=0.0)

        if as_of_ms is not None:
            cutoff_date = pd.Timestamp(as_of_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d")
            cutoff_ts = int(as_of_ms)
            now_ts = int(as_of_ms)
        else:
            cutoff_date = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
            cutoff_ts = int(pd.Timestamp.utcnow().timestamp() * 1000)
            now_ts = cutoff_ts

        scores: dict[str, float] = {}
        _sent_ages: list[float] = []
        try:
            # Fear & Greed (0=extreme fear, 100=extreme greed)
            row = self.conn.execute(
                "SELECT value, date FROM onchain_metrics WHERE symbol=? AND metric='fear_greed_value' "
                "AND date<=? ORDER BY date DESC LIMIT 1",
                (self.symbol, cutoff_date),
            ).fetchone()
            if row:
                scores["fear_greed"] = float(row[0]) / 100.0
                try:
                    _d = pd.Timestamp(row[1], tz="UTC")
                    _age = (pd.Timestamp(cutoff_date, tz="UTC") - _d).total_seconds() / 3600.0
                    _sent_ages.append(_age)
                except Exception:
                    pass

            # News sentiment [-1, 1]
            row = self.conn.execute(
                "SELECT value FROM onchain_metrics WHERE symbol=? AND metric='news_sentiment_score' "
                "AND date<=? ORDER BY date DESC LIMIT 1",
                (self.symbol, cutoff_date),
            ).fetchone()
            if row:
                scores["news_sentiment"] = float(row[0])

            # Social volume change
            row = self.conn.execute(
                "SELECT value FROM onchain_metrics WHERE symbol=? AND metric='santiment_social_vol_chg' "
                "AND date<=? ORDER BY date DESC LIMIT 1",
                (self.symbol, cutoff_date),
            ).fetchone()
            if row:
                scores["social_vol"] = float(row[0])

            # Funding rate (positive = longs paying, bearish signal)
            row = self.conn.execute(
                "SELECT value FROM derivatives_metrics WHERE symbol=? AND metric='funding_rate' "
                "AND ts_ms<=? ORDER BY ts_ms DESC LIMIT 1",
                (self.symbol, cutoff_ts),
            ).fetchone()
            if row:
                raw_funding = float(row[0])
                scores["funding"] = max(-1.0, min(1.0, -raw_funding * 0.5))

        except Exception as exc:
            logger.debug("SentimentAgent data lookup failed: %s", exc)

        if not scores:
            return SentimentSignal(bias="neutral", strength=0.0)

        all_keys = ["fear_greed", "news_sentiment", "social_vol", "funding"]
        weights = {"fear_greed": 0.35, "news_sentiment": 0.30,
                   "social_vol": 0.20, "funding": 0.15}
        composite = sum(
            scores.get(k, 0.5 if k == "fear_greed" else 0.0) * w
            for k, w in weights.items()
        )

        coverage = sum(1 for k in all_keys if k in scores) / len(all_keys)

        data_age_hours = max(_sent_ages) if _sent_ages else 999.0

        freshness = 1.0
        if data_age_hours > 48:
            freshness = 0.1
        elif data_age_hours > 24:
            freshness = 0.3
        elif data_age_hours > 12:
            freshness = 0.5

        if composite > 0.55:
            bias, strength = "bullish", min(1.0, (composite - 0.55) * 4) * coverage * freshness
        elif composite < 0.40:
            bias, strength = "bearish", min(1.0, (0.40 - composite) * 4) * coverage * freshness
        else:
            bias, strength = "neutral", 0.0

        if freshness < 1.0:
            scores["_freshness"] = freshness
            scores["_age_h"] = data_age_hours

        return SentimentSignal(bias=bias, strength=strength, details=scores)


# ---------------------------------------------------------------------------
# Macro Agent
# ---------------------------------------------------------------------------

class MacroAgent:
    """Reads GPR, VIX, DXY, Fed calendar, SPY return to determine regime."""

    def __init__(self, conn=None, symbol: str = "BTC/USD") -> None:
        self.conn = conn
        self.symbol = symbol

    def analyze(self, as_of_ms: int | None = None) -> MacroSignal:
        """Produce a macro signal.

        Parameters
        ----------
        as_of_ms
            Point-in-time cutoff (epoch ms).  ``None`` = now.
        """
        if self.conn is None:
            return MacroSignal(regime="neutral", risk_on=True)

        if as_of_ms is not None:
            cutoff_date = pd.Timestamp(as_of_ms, unit="ms", tz="UTC").strftime("%Y-%m-%d")
        else:
            cutoff_date = pd.Timestamp.utcnow().strftime("%Y-%m-%d")

        indicators: dict[str, float] = {}
        _metric_ages: list[float] = []
        try:
            for metric in ["vix_close", "dxy_close", "gpr_index", "fomc_proximity",
                           "spy_return_pct"]:
                row = self.conn.execute(
                    "SELECT value, date FROM onchain_metrics WHERE symbol=? AND metric=? "
                    "AND date<=? ORDER BY date DESC LIMIT 1",
                    (self.symbol, metric, cutoff_date),
                ).fetchone()
                if row:
                    indicators[metric] = float(row[0])
                    try:
                        _d = pd.Timestamp(row[1], tz="UTC")
                        _age = (pd.Timestamp(cutoff_date, tz="UTC") - _d).total_seconds() / 3600.0
                        _metric_ages.append(_age)
                    except Exception:
                        pass
        except Exception as exc:
            logger.debug("MacroAgent data lookup failed: %s", exc)

        data_age_hours = max(_metric_ages) if _metric_ages else 999.0

        if not indicators:
            return MacroSignal(regime="neutral", risk_on=True)

        risk_score = 0.0

        vix = indicators.get("vix_close", 20.0)
        risk_score += 1.0 if vix < 15 else (-1.0 if vix > 25 else 0.0)

        dxy = indicators.get("dxy_close", 100.0)
        risk_score += -0.5 if dxy > 105 else (0.5 if dxy < 95 else 0.0)

        gpr = indicators.get("gpr_index", 0.0)
        risk_score += -0.5 if gpr > 1.5 else (0.3 if gpr < -0.5 else 0.0)

        fomc = indicators.get("fomc_proximity", 0.0)
        risk_score += -0.3 if fomc > 0.5 else 0.0

        spy_ret = indicators.get("spy_return_pct", 0.0)
        risk_score += 0.5 * np.sign(spy_ret) * min(abs(spy_ret) / 2.0, 1.0)

        # Freshness discount: pull score toward neutral when data is stale
        if data_age_hours > 48:
            risk_score *= 0.1
        elif data_age_hours > 24:
            risk_score *= 0.3
        elif data_age_hours > 12:
            risk_score *= 0.5

        risk_on = risk_score >= 0
        if abs(risk_score) < 0.5:
            regime = "neutral"
        elif risk_on:
            regime = "risk_on"
        else:
            regime = "risk_off"

        if data_age_hours < 999:
            indicators["_age_h"] = data_age_hours

        return MacroSignal(regime=regime, risk_on=risk_on, details=indicators)


# ---------------------------------------------------------------------------
# Meta-Weigher
# ---------------------------------------------------------------------------

_DEFAULT_WEIGHTS = {
    "technical": 0.55,
    "sentiment": 0.25,
    "macro": 0.20,
}


class MetaWeigher:
    """Combine TechSignal + SentimentSignal + MacroSignal into AgentSignal.

    Weights and action thresholds are configurable via constructor and can
    also be updated at runtime by the OnlineLearner.  Per-regime adjustments
    are driven by ``RegimeConfig`` when provided.
    """

    def __init__(
        self,
        weights: dict[str, float] | None = None,
        buy_threshold: float = 0.25,
        sell_threshold: float = -0.25,
        regime_configs: dict | None = None,
        learned_regime_weights: dict[str, dict[str, float]] | None = None,
        learned_blend_ratio: float = 0.30,
        min_learned_samples: int = 30,
    ) -> None:
        self._weights = weights or _DEFAULT_WEIGHTS.copy()
        self._buy_threshold = buy_threshold
        self._sell_threshold = sell_threshold
        self._regime_configs = regime_configs
        self._learned_regime_weights = learned_regime_weights or {}
        self._learned_blend_ratio = learned_blend_ratio
        self._min_learned_samples = min_learned_samples

    def _regime_cfg(self, regime: str | None):
        """Return the RegimeConfig for *regime*, or None."""
        if regime is None or self._regime_configs is None:
            return None
        return self._regime_configs.get(regime)

    def combine(
        self,
        tech: TechSignal,
        sentiment: SentimentSignal,
        macro: MacroSignal,
        rag_context: dict | None = None,
        regime: str | None = None,
    ) -> AgentSignal:
        """Produce a final signal by combining three agent outputs.

        Parameters
        ----------
        regime : str or None
            Market microstructure regime from ``detect_regime()``:
            ``"trending_up"`` / ``"trending_down"`` / ``"ranging"`` / ``"volatile"``.
            Adjusts agent weights and action thresholds via RegimeConfig.
        """

        rc = self._regime_cfg(regime)

        w = dict(self._weights)
        if rc is not None:
            w["technical"] = max(0.10, w["technical"] + rc.meta_tech_delta)
            w["sentiment"] = max(0.10, w["sentiment"] + rc.meta_sent_delta)
            w["macro"] = max(0.10, w["macro"] + rc.meta_macro_delta)

        # Blend learned weights if available for this regime
        if regime and regime in self._learned_regime_weights:
            lw = self._learned_regime_weights[regime]
            sample_count = lw.get("_sample_count", 0)
            if sample_count >= self._min_learned_samples:
                blend = self._learned_blend_ratio
                for k in ("technical", "sentiment", "macro"):
                    if k in lw:
                        manual_v = w.get(k, 0.0)
                        learned_v = lw[k]
                        w[k] = (1.0 - blend) * manual_v + blend * learned_v
                        if abs(learned_v - manual_v) > 0.15:
                            logger.debug(
                                "LEARNED_WEIGHTS %s: %s manual=%.3f learned=%.3f blended=%.3f",
                                regime, k, manual_v, learned_v, w[k],
                            )

        total = sum(w.values())
        if total > 0:
            w = {k: v / total for k, v in w.items()}
        else:
            n = len(w)
            w = {k: 1.0 / n for k in w} if n > 0 else w

        # Convert tech signal to a numeric vote [-1, 0, 1]
        tech_vote = {"buy": 1.0, "sell": -1.0, "hold": 0.0}.get(tech.action, 0.0)
        tech_score = tech_vote * tech.confidence

        # Sentiment -> [-1, 1]
        sent_vote = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}.get(sentiment.bias, 0.0)
        sent_score = sent_vote * sentiment.strength

        # Macro as directional voter: risk_off pushes bearish, risk_on pushes bullish
        macro_vote = {"risk_on": 0.5, "risk_off": -0.5, "neutral": 0.0}.get(
            macro.regime, 0.0
        )
        macro_score = macro_vote

        # RAG context boost (optional)
        rag_boost = 0.0
        if rag_context:
            rag_win_rate = float(rag_context.get("similar_win_rate", 0.5))
            rag_boost = (rag_win_rate - 0.5) * 0.2

        raw_score = (
            w["technical"] * tech_score
            + w["sentiment"] * sent_score
            + w["macro"] * macro_score
            + rag_boost
        )

        # When the technical agent returns "hold" with high confidence, it
        # means the price action does NOT support any trade.  This should
        # dampen how much sentiment/macro alone can drive a trade.
        # Raise the floor when sentiment+macro agree directionally so that
        # strong non-tech consensus can still produce entries.
        if tech.action == "hold" and tech.confidence > 0.3:
            hold_dampen = 1.0 - (tech.confidence * w["technical"] * 2.0)
            _non_tech_agree = (sent_score > 0 and macro_score > 0) or (sent_score < 0 and macro_score < 0)
            _dampen_floor = 0.4 if _non_tech_agree else 0.2
            hold_dampen = max(_dampen_floor, hold_dampen)
            raw_score *= hold_dampen

        combined_score = raw_score

        # Action thresholds: use regime-specific overrides when available
        buy_threshold = self._buy_threshold
        sell_threshold = self._sell_threshold
        if rc is not None:
            if rc.meta_buy_threshold is not None:
                buy_threshold = rc.meta_buy_threshold
            if rc.meta_sell_threshold is not None:
                sell_threshold = rc.meta_sell_threshold

        if combined_score >= buy_threshold:
            action = "buy"
        elif combined_score <= sell_threshold:
            action = "sell"
        else:
            action = "hold"

        confidence = min(1.0, abs(combined_score))

        # Regime-specific sub-strategy gates — light-touch; the downstream ML
        # filter, edge gate, quality gate, and ranging gate handle fine-grained
        # quality filtering.  These thresholds only catch very-low-conviction noise.
        if regime == "ranging" and action != "hold":
            if tech.action == action and tech.confidence < 0.15:
                action = "hold"
                confidence *= 0.5
        elif regime == "volatile" and action != "hold":
            if tech.confidence < 0.15:
                action = "hold"
                confidence *= 0.6
        elif regime not in ("trending_up", "trending_down", "volatile", "ranging", None):
            action = "hold"
            confidence *= 0.3

        # Build explanation
        explanation_parts = [f"Tech: {tech.action}({tech.confidence:.2f})"]
        if sentiment.bias != "neutral":
            explanation_parts.append(f"Sentiment: {sentiment.bias}({sentiment.strength:.2f})")
        if macro.regime != "neutral":
            explanation_parts.append(f"Macro: {macro.regime}")
        if regime:
            explanation_parts.append(f"Regime: {regime}")
        if rag_boost != 0:
            explanation_parts.append(f"RAG: {rag_boost:+.3f}")
        explanation = " | ".join(explanation_parts)
        explanation += f" -> {action.upper()} (conf={confidence:.2f})"

        return AgentSignal(
            action=action,
            confidence=confidence,
            stop_distance_pct=tech.stop_distance_pct,
            volume_ratio=tech.volume_ratio,
            tech=tech,
            sentiment=sentiment,
            macro=macro,
            explanation=explanation,
            agent_weights=dict(w),
            combined_score=combined_score,
        )

    def update_weights(self, new_weights: dict[str, float]) -> None:
        """Update meta-weights (called by the online learner on regime shifts)."""
        total = sum(new_weights.values())
        self._weights = {k: v / total for k, v in new_weights.items()}
        logger.info("MetaWeigher weights updated: %s", self._weights)


# ---------------------------------------------------------------------------
# Agent Pipeline — orchestrates all three agents + meta-weigher
# ---------------------------------------------------------------------------

class AgentPipeline:
    """Drop-in replacement for ``generate_signal()``.

    Usage::

        pipeline = AgentPipeline(config, conn=conn)
        result = pipeline.run(candles_5m, symbol="BTC/USD")
        # result is an AgentSignal compatible with existing code
    """

    def __init__(
        self,
        config,
        conn=None,
        weights: dict[str, float] | None = None,
        rag_retriever=None,
        performance_tracker=None,
        adaptive_confidence=None,
    ) -> None:
        self.config = config
        self.conn = conn
        self.tech_agent = TechnicalAgent(config)
        self.sent_agent = SentimentAgent(conn=conn, symbol=config.symbols[0] if config.symbols else "BTC/USD")
        self.macro_agent = MacroAgent(conn=conn, symbol=config.symbols[0] if config.symbols else "BTC/USD")

        if weights is None:
            weights = {
                "technical": getattr(config, "meta_weight_technical", 0.55),
                "sentiment": getattr(config, "meta_weight_sentiment", 0.25),
                "macro": getattr(config, "meta_weight_macro", 0.20),
            }

        from hogan_bot.config import DEFAULT_REGIME_CONFIGS
        _learned_weights: dict[str, dict[str, float]] = {}
        if getattr(config, "use_learned_weights", False) and performance_tracker is not None:
            try:
                for regime_name, rp in getattr(performance_tracker, "_regime_data", {}).items():
                    if hasattr(rp, "agent_scores") and rp.trade_count >= 30:
                        _learned_weights[regime_name] = {
                            **rp.agent_scores,
                            "_sample_count": rp.trade_count,
                        }
            except Exception:
                pass
        self.meta = MetaWeigher(
            weights=weights,
            buy_threshold=getattr(config, "meta_buy_threshold", 0.25),
            sell_threshold=getattr(config, "meta_sell_threshold", -0.25),
            regime_configs=DEFAULT_REGIME_CONFIGS,
            learned_regime_weights=_learned_weights if _learned_weights else None,
        )
        self.rag_retriever = rag_retriever
        self.perf_tracker = performance_tracker
        self.adaptive_conf = adaptive_confidence

    def run(
        self,
        candles: pd.DataFrame,
        symbol: str = "BTC/USD",
        features: list[float] | None = None,
        config_override=None,
        regime: str | None = None,
        regime_state=None,
        as_of_ms: int | None = None,
        **runtime_state,
    ) -> AgentSignal:
        """Run all agents and return a combined AgentSignal.

        Parameters
        ----------
        config_override
            Per-symbol config (from ``symbol_config()``) that overrides the
            base config for this run.  When provided, TechnicalAgent uses
            these params instead of the pipeline's base config.
        regime
            Market microstructure regime (from ``detect_regime()``).
        regime_state
            Full ``RegimeState`` object from ``detect_regime()``.  When
            provided, the ``StrategyRouter`` inside ``TechnicalAgent``
            uses it to select a regime-appropriate strategy family.
        as_of_ms
            Point-in-time cutoff (epoch ms) for all DB lookups.
            Used by backtest to prevent future data leakage.
            ``None`` means "now" (live mode).
        """
        agent = self.tech_agent
        _effective_cfg = config_override
        if _effective_cfg is not None and regime is not None:
            from hogan_bot.regime import _REGIME_OVERRIDES
            _overrides = _REGIME_OVERRIDES.get(regime, {})
            vol_mult = _overrides.get("volume_threshold_mult")
            if vol_mult is not None and hasattr(_effective_cfg, "volume_threshold"):
                _effective_cfg.volume_threshold = _effective_cfg.volume_threshold * vol_mult
        if _effective_cfg is not None:
            agent = TechnicalAgent(_effective_cfg)
        tech = agent.analyze(candles, regime_state=regime_state, **runtime_state)
        sent = self.sent_agent.analyze(as_of_ms=as_of_ms)
        macro = self.macro_agent.analyze(as_of_ms=as_of_ms)

        # RAG context (optional)
        rag_context = None
        if self.rag_retriever is not None and features is not None:
            try:
                rag_context = self.rag_retriever.retrieve_relevant_context(features, k=5)
            except Exception as exc:
                logger.debug("RAG retrieval failed (non-fatal): %s", exc)

        signal = self.meta.combine(tech, sent, macro, rag_context=rag_context, regime=regime)

        # Forecast and risk heads run independently of the vote
        try:
            forecast = compute_forecast(candles)
        except Exception as exc:
            logger.debug("Forecast head failed (non-fatal): %s", exc)
            forecast = ForecastResult(confidence=0.0)

        try:
            stop = signal.stop_distance_pct
            tp = getattr(self.config, "take_profit_pct", 0.05)
            mhb = getattr(self.config, "max_hold_bars", 24)
            risk_est = compute_risk(candles, stop_pct=stop, tp_pct=tp, max_hold_bars=mhb)
        except Exception as exc:
            logger.debug("Risk head failed (non-fatal): %s", exc)
            risk_est = RiskEstimate()

        signal.forecast = forecast
        signal.risk_estimate = risk_est

        # Forecast-driven action validation:
        # The forecast head is an independent directional estimate.  If it
        # strongly disagrees with the MetaWeigher action, override or dampen.
        # If it agrees, boost.  This prevents stale sentiment/macro from
        # overriding what the price data is actually showing.
        if forecast.confidence > 0.2:
            fc_4h = forecast.bullish_4h
            fc_12h = forecast.bullish_12h

            fc_bias = (fc_4h - 0.5) * 0.6 + (fc_12h - 0.5) * 0.4

            if signal.action == "sell" and fc_bias > 0.03:
                signal.explanation += f" | Forecast disagrees ({fc_4h:.0%}up@4h, bias={fc_bias:+.3f})"
                if fc_bias > 0.08:
                    signal.action = "hold"
                    signal.confidence *= 0.3
                    signal.explanation += " -> VETO to hold"
                else:
                    signal.confidence *= 0.5
            elif signal.action == "buy" and fc_bias < -0.03:
                signal.explanation += f" | Forecast disagrees ({fc_4h:.0%}up@4h, bias={fc_bias:+.3f})"
                if fc_bias < -0.08:
                    signal.action = "hold"
                    signal.confidence *= 0.3
                    signal.explanation += " -> VETO to hold"
                else:
                    signal.confidence *= 0.5
            elif signal.action == "sell" and fc_bias < -0.03:
                signal.confidence = min(1.0, signal.confidence * 1.2)
                signal.explanation += f" | Forecast confirms ({fc_4h:.0%}up@4h)"
            elif signal.action == "buy" and fc_bias > 0.03:
                signal.confidence = min(1.0, signal.confidence * 1.2)
                signal.explanation += f" | Forecast confirms ({fc_4h:.0%}up@4h)"

        # Risk-based position scaling
        if risk_est.regime_risk == "high":
            signal.confidence *= risk_est.position_scale

        # Adaptive confidence scaling: uses recency-weighted model accuracy,
        # forecast agreement, and calibration quality to modulate position size.
        if self.adaptive_conf is not None and signal.action != "hold":
            ml_prob = getattr(signal, '_ml_up_prob', None)
            if ml_prob is not None:
                fc_bias_val = None
                if forecast.confidence > 0.2:
                    fc_bias_val = (forecast.bullish_4h - 0.5) * 0.6 + (forecast.bullish_12h - 0.5) * 0.4
                adaptive_scale = self.adaptive_conf.compute(
                    up_prob=ml_prob,
                    forecast_bias=fc_bias_val,
                    regime=regime,
                )
                signal.confidence *= adaptive_scale

        logger.debug(
            "AgentPipeline: %s | tech=%s sent=%s macro=%s conf=%.2f | %s | %s",
            symbol, tech.action, sent.bias, macro.regime, signal.confidence,
            forecast.summary(), risk_est.summary(),
        )
        return signal

    def record_trade_outcome(
        self,
        symbol: str,
        regime: str,
        signal: AgentSignal,
        realized_pnl: float,
        ml_up_prob: float | None = None,
    ) -> None:
        """Record a completed trade outcome for performance tracking and
        adaptive confidence learning.

        Should be called after each trade closes (by the event loop or
        execution layer).
        """
        if self.perf_tracker is not None:
            try:
                self.perf_tracker.record_trade_outcome(
                    symbol=symbol,
                    regime=regime or "unknown",
                    tech_action=signal.tech.action if signal.tech else "hold",
                    tech_confidence=signal.tech.confidence if signal.tech else 0.5,
                    sent_bias=signal.sentiment.bias if signal.sentiment else "neutral",
                    sent_strength=signal.sentiment.strength if signal.sentiment else 0.0,
                    macro_regime=signal.macro.regime if signal.macro else "neutral",
                    realized_pnl=realized_pnl,
                )
            except Exception as exc:
                logger.debug("Performance tracker record failed: %s", exc)

        if self.adaptive_conf is not None and ml_up_prob is not None:
            try:
                actual_label = 1 if realized_pnl > 0 else 0
                self.adaptive_conf.record_outcome(ml_up_prob, actual_label)
            except Exception as exc:
                logger.debug("Adaptive confidence record failed: %s", exc)
