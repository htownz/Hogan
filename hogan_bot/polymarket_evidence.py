"""Structured evidence adapters for Polymarket category expansion."""
from __future__ import annotations

from dataclasses import dataclass

from hogan_bot.fetch_polymarket import PolymarketMarketSnapshot


@dataclass(frozen=True)
class PolymarketEvidence:
    category_id: str
    source: str
    confidence: float
    summary: str
    flags: list[str]

    def to_dict(self) -> dict:
        return {
            "category_id": self.category_id,
            "source": self.source,
            "confidence": round(self.confidence, 4),
            "summary": self.summary,
            "flags": list(self.flags),
        }


def assess_polymarket_evidence(snapshot: PolymarketMarketSnapshot) -> PolymarketEvidence:
    """Return first-pass structured evidence for a normalized market."""
    if snapshot.category_id in ("crypto_price_target", "crypto_directional"):
        return PolymarketEvidence(
            category_id=snapshot.category_id,
            source="crypto_price_history",
            confidence=0.65,
            summary="Uses Hogan crypto candles, latest ML probability, and long-horizon price model when applicable.",
            flags=[],
        )
    if snapshot.category_id == "crypto_treasury":
        return PolymarketEvidence(
            category_id=snapshot.category_id,
            source="crypto_treasury_public_context",
            confidence=0.30,
            summary="Treasury/holdings market detected; needs structured holdings or filing evidence before fair-value modeling.",
            flags=["structured_holdings_data_missing"],
        )
    if snapshot.category_id == "crypto_policy":
        return PolymarketEvidence(
            category_id=snapshot.category_id,
            source="crypto_policy_public_context",
            confidence=0.25,
            summary="Policy market detected; needs structured event calendar/news markers before fair-value modeling.",
            flags=["structured_policy_model_missing"],
        )
    if snapshot.category_id == "macro_event":
        return PolymarketEvidence(
            category_id=snapshot.category_id,
            source="macro_calendar_and_market_context",
            confidence=0.40,
            summary="Macro event market detected; can be connected to CPI/FOMC/NFP and macro-market context.",
            flags=["macro_event_model_missing"],
        )
    if snapshot.category_id == "equity_index_or_single_name":
        return PolymarketEvidence(
            category_id=snapshot.category_id,
            source="equity_candles",
            confidence=0.35,
            summary="Equity market detected; needs equity candle baseline and sector/index context.",
            flags=["equity_fair_value_model_missing"],
        )
    return PolymarketEvidence(
        category_id=snapshot.category_id,
        source="unsupported",
        confidence=0.0,
        summary="Unsupported category; research only until evidence adapter is added.",
        flags=["unsupported_category"],
    )
