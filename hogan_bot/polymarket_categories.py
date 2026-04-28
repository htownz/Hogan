"""Category registry for Polymarket markets.

The registry keeps category expansion explicit: each supported class declares
its keywords, evidence requirement, allowed fair-value sources, and shadow
policy. Unsupported categories remain research-only.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class CategoryDefinition:
    category_id: str
    asset_category: str
    market_type: str
    keywords: tuple[str, ...]
    exclusions: tuple[str, ...]
    required_evidence_source: str
    allowed_fair_value_sources: tuple[str, ...]
    shadow_policy: str


@dataclass(frozen=True)
class CategoryMatch:
    category_id: str
    asset_category: str
    market_type: str
    required_evidence_source: str
    allowed_fair_value_sources: tuple[str, ...]
    shadow_policy: str


def _contains_term(text: str, term: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(_contains_term(text, term) for term in terms)


_BTC_TERMS = ("bitcoin", "btc")
_ETH_TERMS = ("ethereum", "ether", "eth")

POLYMARKET_CATEGORY_REGISTRY: tuple[CategoryDefinition, ...] = (
    CategoryDefinition(
        category_id="crypto_treasury",
        asset_category="btc",
        market_type="crypto_treasury",
        keywords=("microstrategy", "strategy", "el salvador", "strategic reserve", "hold", "holds", "sell any bitcoin"),
        exclusions=(),
        required_evidence_source="crypto_treasury_public_context",
        allowed_fair_value_sources=("market_implied_only",),
        shadow_policy="research_only",
    ),
    CategoryDefinition(
        category_id="crypto_policy",
        asset_category="btc",
        market_type="crypto_policy",
        keywords=("capital gains", "crypto tax", "sec", "etf", "tax on crypto", "strategic reserve", "crypto reserve", "regulation", "crypto bill"),
        exclusions=("microstrategy", "el salvador"),
        required_evidence_source="crypto_policy_public_context",
        allowed_fair_value_sources=("market_implied_only",),
        shadow_policy="research_only",
    ),
    CategoryDefinition(
        category_id="macro_event",
        asset_category="macro_risk",
        market_type="macro_event",
        keywords=("cpi", "fed", "fomc", "inflation", "recession", "unemployment", "rate hike", "bank failure", "default", "war"),
        exclusions=(),
        required_evidence_source="macro_calendar_and_market_context",
        allowed_fair_value_sources=("market_implied_only",),
        shadow_policy="research_only",
    ),
    CategoryDefinition(
        category_id="equity_index_or_single_name",
        asset_category="equity",
        market_type="equity_index_or_single_name",
        keywords=("spy", "qqq", "nasdaq", "s&p", "nvidia", "nvda", "tesla", "tsla", "mstr", "stock"),
        exclusions=(),
        required_evidence_source="equity_candles",
        allowed_fair_value_sources=("market_implied_only",),
        shadow_policy="research_only",
    ),
    CategoryDefinition(
        category_id="crypto_price_target",
        asset_category="crypto",
        market_type="price_target",
        keywords=_BTC_TERMS + _ETH_TERMS,
        exclusions=("microstrategy", "strategy", "el salvador", "capital gains", "sec", "tax", "reserve"),
        required_evidence_source="crypto_price_history",
        allowed_fair_value_sources=("hogan_short_term_ml", "calibrated_long_horizon"),
        shadow_policy="fair_value_required",
    ),
    CategoryDefinition(
        category_id="crypto_directional",
        asset_category="crypto",
        market_type="directional",
        keywords=_BTC_TERMS + _ETH_TERMS,
        exclusions=("microstrategy", "strategy", "el salvador", "capital gains", "sec", "tax", "reserve"),
        required_evidence_source="crypto_price_history",
        allowed_fair_value_sources=("hogan_short_term_ml",),
        shadow_policy="fair_value_required",
    ),
)


def classify_polymarket_category(text: str, *, target_price: float | None) -> CategoryMatch:
    normalized = text.lower()
    for definition in POLYMARKET_CATEGORY_REGISTRY:
        if definition.exclusions and _contains_any(normalized, definition.exclusions):
            continue
        if not _contains_any(normalized, definition.keywords):
            continue
        if definition.category_id == "crypto_price_target" and target_price is None:
            continue
        if definition.category_id == "crypto_directional" and target_price is not None:
            continue
        asset_category = definition.asset_category
        if asset_category == "crypto":
            asset_category = "eth" if _contains_any(normalized, _ETH_TERMS) else "btc"
        return CategoryMatch(
            category_id=definition.category_id,
            asset_category=asset_category,
            market_type=definition.market_type,
            required_evidence_source=definition.required_evidence_source,
            allowed_fair_value_sources=definition.allowed_fair_value_sources,
            shadow_policy=definition.shadow_policy,
        )
    return CategoryMatch(
        category_id="unsupported",
        asset_category="other",
        market_type="other",
        required_evidence_source="unsupported",
        allowed_fair_value_sources=("market_implied_only",),
        shadow_policy="research_only",
    )
