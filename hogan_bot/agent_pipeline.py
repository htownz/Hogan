"""Multi-Agent Signal Pipeline — Phase 8a.

Replaces the monolithic ``generate_signal()`` with three specialized analysis
agents whose outputs are combined by a learned-weight MetaWeigher.

Architecture (Awesome LLM Apps multi-agent pattern):

    TechnicalAgent  ─┐
    SentimentAgent  ─┤──► MetaWeigher ──► AgentSignal(action, confidence, ...)
    MacroAgent      ─┘

Each agent produces a typed signal object.  The MetaWeigher combines them
using weights that can be updated by the online learner, making the pipeline
adaptive to changing market regimes.

This is more interpretable than a single ``generate_signal()`` vote count
and naturally extensible — adding a new agent is a new class, not a
modification to existing logic.

Usage::

    from hogan_bot.agent_pipeline import AgentPipeline

    pipeline = AgentPipeline(config, ml_model=..., conn=conn)
    result = pipeline.run(candles_5m, candles_1h=..., candles_15m=..., symbol="BTC/USD")
    # result.action in ("buy", "sell", "hold")
    # result.confidence in [0.0, 1.0]
    # result.explanation — human-readable summary
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

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
    """Final combined signal from the MetaWeigher."""
    action: str           # "buy" | "sell" | "hold"
    confidence: float     # [0.0, 1.0]
    stop_distance_pct: float = 0.02
    volume_ratio: float = 1.0
    tech: TechSignal | None = None
    sentiment: SentimentSignal | None = None
    macro: MacroSignal | None = None
    explanation: str = ""
    agent_weights: dict = field(default_factory=dict)
    timestamp: float = field(default_factory=time.time)

    # Compatibility with existing generate_signal() callers
    @property
    def action_str(self) -> str:
        return self.action


# ---------------------------------------------------------------------------
# Technical Agent
# ---------------------------------------------------------------------------

class TechnicalAgent:
    """Wraps generate_signal() as one sub-brain — MA, EMA clouds, FVG, ICT, RL."""

    def __init__(self, config) -> None:
        self.config = config

    def analyze(self, candles: pd.DataFrame, **runtime_state) -> TechSignal:
        """Produce a TechSignal. ``runtime_state`` passes dynamic per-bar RL
        fields (rl_in_position, rl_unrealized_pnl, rl_bars_in_trade)."""
        cfg = self.config
        if candles.empty or len(candles) < max(cfg.long_ma_window, 20):
            return TechSignal(action="hold", confidence=0.0)

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
            logger.warning("TechnicalAgent failed: %s", exc)
            return TechSignal(action="hold", confidence=0.0)


# ---------------------------------------------------------------------------
# Sentiment Agent
# ---------------------------------------------------------------------------

class SentimentAgent:
    """Weighs fear/greed, news sentiment, funding rate, and social volume."""

    def __init__(self, conn=None, symbol: str = "BTC/USD") -> None:
        self.conn = conn
        self.symbol = symbol

    def analyze(self) -> SentimentSignal:
        if self.conn is None:
            return SentimentSignal(bias="neutral", strength=0.0)

        scores: dict[str, float] = {}
        try:
            today = pd.Timestamp.utcnow().strftime("%Y-%m-%d")

            # Fear & Greed (0=extreme fear, 100=extreme greed)
            row = self.conn.execute(
                "SELECT value FROM onchain_metrics WHERE symbol=? AND metric='fear_greed_value' "
                "AND date<=? ORDER BY date DESC LIMIT 1",
                (self.symbol, today),
            ).fetchone()
            if row:
                scores["fear_greed"] = float(row[0]) / 100.0

            # News sentiment [-1, 1]
            row = self.conn.execute(
                "SELECT value FROM onchain_metrics WHERE symbol=? AND metric='news_sentiment_score' "
                "AND date<=? ORDER BY date DESC LIMIT 1",
                (self.symbol, today),
            ).fetchone()
            if row:
                scores["news_sentiment"] = float(row[0])

            # Social volume change ÷100
            row = self.conn.execute(
                "SELECT value FROM onchain_metrics WHERE symbol=? AND metric='santiment_social_vol_chg' "
                "AND date<=? ORDER BY date DESC LIMIT 1",
                (self.symbol, today),
            ).fetchone()
            if row:
                scores["social_vol"] = float(row[0])

            # Funding rate (positive = longs paying, bearish signal)
            row = self.conn.execute(
                "SELECT value FROM derivatives_metrics WHERE symbol=? AND metric='funding_rate' "
                "ORDER BY ts_ms DESC LIMIT 1",
                (self.symbol,),
            ).fetchone()
            if row:
                raw_funding = float(row[0])
                scores["funding"] = max(-1.0, min(1.0, -raw_funding * 0.5))

        except Exception as exc:
            logger.debug("SentimentAgent data lookup failed: %s", exc)

        if not scores:
            return SentimentSignal(bias="neutral", strength=0.0)

        # Weighted composite sentiment score
        all_keys = ["fear_greed", "news_sentiment", "social_vol", "funding"]
        weights = {"fear_greed": 0.35, "news_sentiment": 0.30,
                   "social_vol": 0.20, "funding": 0.15}
        composite = sum(
            scores.get(k, 0.5 if k == "fear_greed" else 0.0) * w
            for k, w in weights.items()
        )

        # Scale strength by data coverage — don't claim full confidence
        # when half the inputs are missing
        coverage = sum(1 for k in all_keys if k in scores) / len(all_keys)

        if composite > 0.55:
            bias, strength = "bullish", min(1.0, (composite - 0.55) * 4) * coverage
        elif composite < 0.40:
            bias, strength = "bearish", min(1.0, (0.40 - composite) * 4) * coverage
        else:
            bias, strength = "neutral", 0.0

        return SentimentSignal(bias=bias, strength=strength, details=scores)


# ---------------------------------------------------------------------------
# Macro Agent
# ---------------------------------------------------------------------------

class MacroAgent:
    """Reads GPR, VIX, DXY, Fed calendar, SPY return to determine regime."""

    def __init__(self, conn=None, symbol: str = "BTC/USD") -> None:
        self.conn = conn
        self.symbol = symbol

    def analyze(self) -> MacroSignal:
        if self.conn is None:
            return MacroSignal(regime="neutral", risk_on=True)

        indicators: dict[str, float] = {}
        try:
            today = pd.Timestamp.utcnow().strftime("%Y-%m-%d")
            for metric in ["vix_close", "dxy_close", "gpr_index", "fomc_proximity",
                           "spy_return_pct"]:
                row = self.conn.execute(
                    "SELECT value FROM onchain_metrics WHERE symbol=? AND metric=? "
                    "AND date<=? ORDER BY date DESC LIMIT 1",
                    (self.symbol, metric, today),
                ).fetchone()
                if row:
                    indicators[metric] = float(row[0])
        except Exception as exc:
            logger.debug("MacroAgent data lookup failed: %s", exc)

        if not indicators:
            return MacroSignal(regime="neutral", risk_on=True)

        risk_score = 0.0   # positive = risk-on, negative = risk-off

        # VIX: above 25 = elevated fear (risk-off), below 15 = complacency
        vix = indicators.get("vix_close", 20.0)
        risk_score += 1.0 if vix < 15 else (-1.0 if vix > 25 else 0.0)

        # DXY: rising dollar = risk-off for crypto
        dxy = indicators.get("dxy_close", 100.0)
        risk_score += -0.5 if dxy > 105 else (0.5 if dxy < 95 else 0.0)

        # GPR: high geopolitical risk = risk-off
        gpr = indicators.get("gpr_index", 0.0)
        risk_score += -0.5 if gpr > 1.5 else (0.3 if gpr < -0.5 else 0.0)

        # FOMC proximity: slight uncertainty near Fed meetings
        fomc = indicators.get("fomc_proximity", 0.0)
        risk_score += -0.3 if fomc > 0.5 else 0.0

        # SPY return: positive = risk-on
        spy_ret = indicators.get("spy_return_pct", 0.0)
        risk_score += 0.5 * np.sign(spy_ret) * min(abs(spy_ret) / 2.0, 1.0)

        risk_on = risk_score >= 0
        if abs(risk_score) < 0.5:
            regime = "neutral"
        elif risk_on:
            regime = "risk_on"
        else:
            regime = "risk_off"

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

    Weights are adaptive and can be updated by the OnlineLearner.
    """

    def __init__(self, weights: dict[str, float] | None = None) -> None:
        self._weights = weights or _DEFAULT_WEIGHTS.copy()

    def combine(
        self,
        tech: TechSignal,
        sentiment: SentimentSignal,
        macro: MacroSignal,
        rag_context: dict | None = None,
    ) -> AgentSignal:
        """Produce a final signal by combining three agent outputs."""

        # Convert tech signal to a numeric vote [-1, 0, 1]
        tech_vote = {"buy": 1.0, "sell": -1.0, "hold": 0.0}.get(tech.action, 0.0)
        tech_score = tech_vote * tech.confidence

        # Sentiment → [-1, 1]
        sent_vote = {"bullish": 1.0, "bearish": -1.0, "neutral": 0.0}.get(sentiment.bias, 0.0)
        sent_score = sent_vote * sentiment.strength

        # Macro modifier: risk-off halves all bullish components independently
        if not macro.risk_on:
            macro_mult = 0.5
            if tech_score > 0:
                tech_score *= macro_mult
            if sent_score > 0:
                sent_score *= macro_mult

        # RAG context boost (optional — from Phase 8b)
        rag_boost = 0.0
        if rag_context:
            rag_win_rate = float(rag_context.get("similar_win_rate", 0.5))
            rag_boost = (rag_win_rate - 0.5) * 0.2  # ±0.1 max

        w = self._weights
        combined_score = (
            w["technical"] * tech_score
            + w["sentiment"] * sent_score
            + rag_boost
        )

        # Threshold requires at least mild agreement across agents —
        # no single agent at its default weight can trigger alone.
        buy_threshold = 0.30
        sell_threshold = -0.30

        if combined_score >= buy_threshold:
            action = "buy"
        elif combined_score <= sell_threshold:
            action = "sell"
        else:
            action = "hold"

        confidence = min(1.0, abs(combined_score))

        # Build explanation
        explanation_parts = [f"Tech: {tech.action}({tech.confidence:.2f})"]
        if sentiment.bias != "neutral":
            explanation_parts.append(f"Sentiment: {sentiment.bias}({sentiment.strength:.2f})")
        if macro.regime != "neutral":
            explanation_parts.append(f"Macro: {macro.regime}")
        if rag_boost != 0:
            explanation_parts.append(f"RAG: {rag_boost:+.3f}")
        explanation = " | ".join(explanation_parts)
        explanation += f" → {action.upper()} (conf={confidence:.2f})"

        return AgentSignal(
            action=action,
            confidence=confidence,
            stop_distance_pct=tech.stop_distance_pct,
            volume_ratio=tech.volume_ratio,
            tech=tech,
            sentiment=sentiment,
            macro=macro,
            explanation=explanation,
            agent_weights=dict(self._weights),
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
    ) -> None:
        self.config = config
        self.conn = conn
        self.tech_agent = TechnicalAgent(config)
        self.sent_agent = SentimentAgent(conn=conn, symbol=config.symbols[0] if config.symbols else "BTC/USD")
        self.macro_agent = MacroAgent(conn=conn, symbol=config.symbols[0] if config.symbols else "BTC/USD")
        self.meta = MetaWeigher(weights=weights)
        self.rag_retriever = rag_retriever

    def run(
        self,
        candles: pd.DataFrame,
        symbol: str = "BTC/USD",
        features: list[float] | None = None,
        config_override=None,
        **runtime_state,
    ) -> AgentSignal:
        """Run all agents and return a combined AgentSignal.

        Parameters
        ----------
        config_override
            Per-symbol config (from ``symbol_config()``) that overrides the
            base config for this run.  When provided, TechnicalAgent uses
            these params instead of the pipeline's base config.
        """
        agent = self.tech_agent
        if config_override is not None:
            agent = TechnicalAgent(config_override)
        tech = agent.analyze(candles, **runtime_state)
        sent = self.sent_agent.analyze()
        macro = self.macro_agent.analyze()

        # RAG context (optional)
        rag_context = None
        if self.rag_retriever is not None and features is not None:
            try:
                rag_context = self.rag_retriever.retrieve_relevant_context(features, k=5)
            except Exception as exc:
                logger.debug("RAG retrieval failed (non-fatal): %s", exc)

        signal = self.meta.combine(tech, sent, macro, rag_context=rag_context)
        logger.debug(
            "AgentPipeline: %s | tech=%s sent=%s macro=%s conf=%.2f",
            symbol, tech.action, sent.bias, macro.regime, signal.confidence,
        )
        return signal
