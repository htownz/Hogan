"""Dedicated tests for hogan_bot.agent_pipeline — MetaWeigher, agents, regime weights."""
from __future__ import annotations

import pytest

from hogan_bot.agent_pipeline import (
    AgentSignal,
    MacroAgent,
    MacroSignal,
    MetaWeigher,
    SentimentAgent,
    SentimentSignal,
    TechSignal,
)
from hogan_bot.config import DEFAULT_REGIME_CONFIGS

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tech(action="buy", conf=0.8, stop=0.02, vol=1.0):
    return TechSignal(action=action, confidence=conf, stop_distance_pct=stop, volume_ratio=vol)

def _sent(bias="neutral", strength=0.0):
    return SentimentSignal(bias=bias, strength=strength)

def _macro(regime="neutral", risk_on=True):
    return MacroSignal(regime=regime, risk_on=risk_on)


# ---------------------------------------------------------------------------
# MetaWeigher — basic combining
# ---------------------------------------------------------------------------

class TestMetaWeigherBasics:
    def test_strong_buy_produces_buy(self):
        mw = MetaWeigher()
        sig = mw.combine(_tech("buy", 0.9), _sent("bullish", 0.7), _macro("risk_on"))
        assert sig.action == "buy"
        assert sig.confidence > 0

    def test_strong_sell_produces_sell(self):
        mw = MetaWeigher()
        sig = mw.combine(_tech("sell", 0.9), _sent("bearish", 0.7), _macro("risk_off"))
        assert sig.action == "sell"
        assert sig.confidence > 0

    def test_conflicting_signals_may_hold(self):
        mw = MetaWeigher()
        sig = mw.combine(_tech("buy", 0.3), _sent("bearish", 0.9), _macro("risk_off"))
        # With low-conf tech buy and strong bearish, should be hold or sell
        assert sig.action in ("hold", "sell")

    def test_all_neutral_gives_hold(self):
        mw = MetaWeigher()
        sig = mw.combine(_tech("hold", 0.5), _sent("neutral", 0.0), _macro("neutral"))
        assert sig.action == "hold"

    def test_output_type(self):
        mw = MetaWeigher()
        sig = mw.combine(_tech(), _sent(), _macro())
        assert isinstance(sig, AgentSignal)
        assert sig.tech is not None
        assert sig.sentiment is not None
        assert sig.macro is not None

    def test_explanation_populated(self):
        mw = MetaWeigher()
        sig = mw.combine(_tech("buy", 0.9), _sent(), _macro())
        assert "Tech:" in sig.explanation

    def test_agent_weights_stored(self):
        mw = MetaWeigher()
        sig = mw.combine(_tech(), _sent(), _macro())
        assert "technical" in sig.agent_weights
        assert sum(sig.agent_weights.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# MetaWeigher — regime-adaptive weights
# ---------------------------------------------------------------------------

class TestMetaWeigherRegime:
    def test_trending_boosts_tech(self):
        mw = MetaWeigher(regime_configs=DEFAULT_REGIME_CONFIGS)
        sig = mw.combine(_tech("buy", 0.7), _sent(), _macro(), regime="trending_up")
        assert sig.agent_weights["technical"] > 0.55

    def test_volatile_reduces_tech(self):
        mw = MetaWeigher(regime_configs=DEFAULT_REGIME_CONFIGS)
        sig = mw.combine(_tech("buy", 0.7), _sent(), _macro(), regime="volatile")
        assert sig.agent_weights["technical"] < 0.55

    def test_ranging_reduces_tech_most(self):
        mw = MetaWeigher(regime_configs=DEFAULT_REGIME_CONFIGS)
        sig = mw.combine(_tech("buy", 0.7), _sent(), _macro(), regime="ranging")
        assert sig.agent_weights["technical"] <= 0.55

    def test_volatile_gate_blocks_very_low_conf(self):
        mw = MetaWeigher(regime_configs=DEFAULT_REGIME_CONFIGS)
        sig = mw.combine(_tech("buy", 0.1), _sent("bullish", 0.9), _macro("risk_on"), regime="volatile")
        assert sig.action == "hold"

    def test_ranging_gate_blocks_very_low_conf(self):
        mw = MetaWeigher(regime_configs=DEFAULT_REGIME_CONFIGS)
        sig = mw.combine(_tech("buy", 0.1), _sent("bullish", 0.9), _macro("risk_on"), regime="ranging")
        assert sig.action == "hold"

    def test_unknown_regime_blocks(self):
        mw = MetaWeigher(regime_configs=DEFAULT_REGIME_CONFIGS)
        sig = mw.combine(_tech("buy", 0.9), _sent("bullish", 0.9), _macro("risk_on"), regime="alien")
        assert sig.action == "hold"

    def test_no_regime_configs_uses_base_weights(self):
        mw = MetaWeigher()
        sig = mw.combine(_tech("buy", 0.7), _sent(), _macro(), regime="trending_up")
        assert sig.agent_weights["technical"] == pytest.approx(0.55, abs=0.01)


# ---------------------------------------------------------------------------
# MetaWeigher — weight normalization
# ---------------------------------------------------------------------------

class TestMetaWeigherWeights:
    def test_custom_weights(self):
        mw = MetaWeigher(weights={"technical": 0.8, "sentiment": 0.1, "macro": 0.1})
        sig = mw.combine(_tech("buy", 0.9), _sent("bearish", 0.2), _macro("neutral"))
        assert sig.agent_weights["technical"] > 0.7

    def test_update_weights(self):
        mw = MetaWeigher()
        mw.update_weights({"technical": 0.3, "sentiment": 0.3, "macro": 0.4})
        assert mw._weights["macro"] == pytest.approx(0.4)

    def test_weights_always_sum_to_one(self):
        mw = MetaWeigher(weights={"technical": 1, "sentiment": 2, "macro": 3})
        sig = mw.combine(_tech(), _sent(), _macro())
        assert sum(sig.agent_weights.values()) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# MetaWeigher — hold dampening
# ---------------------------------------------------------------------------

class TestMetaWeigherHoldDampen:
    def test_tech_hold_dampens_sentiment(self):
        mw = MetaWeigher()
        # Tech says hold with high conf; sentiment is bullish
        sig_dampened = mw.combine(_tech("hold", 0.8), _sent("bullish", 0.9), _macro("risk_on"))
        sig_normal = mw.combine(_tech("buy", 0.8), _sent("bullish", 0.9), _macro("risk_on"))
        assert sig_dampened.confidence <= sig_normal.confidence


# ---------------------------------------------------------------------------
# SentimentAgent / MacroAgent without DB
# ---------------------------------------------------------------------------

class TestAgentsNoDb:
    def test_sentiment_no_conn(self):
        sa = SentimentAgent(conn=None)
        sig = sa.analyze()
        assert sig.bias == "neutral"
        assert sig.strength == 0.0

    def test_macro_no_conn(self):
        ma = MacroAgent(conn=None)
        sig = ma.analyze()
        assert sig.regime == "neutral"
        assert sig.risk_on is True


# ---------------------------------------------------------------------------
# AgentSignal dataclass
# ---------------------------------------------------------------------------

class TestAgentSignal:
    def test_action_str_property(self):
        sig = AgentSignal(action="buy", confidence=0.8)
        assert sig.action_str == "buy"

    def test_defaults(self):
        sig = AgentSignal(action="hold", confidence=0.0)
        assert sig.forecast is None
        assert sig.risk_estimate is None
        assert sig.agent_weights == {}


# ---------------------------------------------------------------------------
# RAG context boost
# ---------------------------------------------------------------------------

class TestRAGBoost:
    def test_positive_rag_boosts(self):
        mw = MetaWeigher()
        sig_no_rag = mw.combine(_tech("buy", 0.5), _sent(), _macro())
        sig_rag = mw.combine(_tech("buy", 0.5), _sent(), _macro(),
                             rag_context={"similar_win_rate": 0.8})
        # RAG with high win rate should give higher or equal confidence
        assert sig_rag.confidence >= sig_no_rag.confidence or sig_rag.action != "hold"
