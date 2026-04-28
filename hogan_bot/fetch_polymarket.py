"""Fetch public Polymarket prediction-market signals for Hogan.

Phase one is analysis-only: no wallet keys, no authenticated CLOB trading, and
no order placement. We use public Gamma market discovery data to derive compact
daily metrics for BTC/ETH and broader macro risk sentiment.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from datetime import date
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_GAMMA_BASE = "https://gamma-api.polymarket.com"
_CLOB_BASE = "https://clob.polymarket.com"
_TIMEOUT = 20
_SLEEP = 0.25
_HEADERS = {
    "Accept": "application/json",
    "User-Agent": "HoganBot/1.0 (+https://github.com/htownz/Hogan; public market data research)",
}

_BTC_TERMS = ("bitcoin", "btc")
_ETH_TERMS = ("ethereum", "ether", "eth")
_BULLISH_TERMS = (
    "above",
    "all-time high",
    "ath",
    "break",
    "exceed",
    "hit",
    "new high",
    "over",
    "reach",
    "rise",
    "up",
)
_BEARISH_TERMS = (
    "below",
    "crash",
    "dip",
    "down",
    "drop",
    "fall",
    "less than",
    "lower than",
    "under",
)
_MACRO_RISK_TERMS = (
    "bank failure",
    "cpi",
    "default",
    "fed",
    "inflation",
    "recession",
    "rate hike",
    "unemployment",
    "war",
)
_CRYPTO_MACRO_TERMS = (
    "capital gains",
    "crypto",
    "el salvador",
    "etf",
    "microstrategy",
    "sec",
    "strategic reserve",
    "strategy",
    "trump",
)
_PRICE_RE = re.compile(
    r"(?:\$\s*([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s*(k|m|thousand|million)?)"
    r"|(?:\b([0-9]+(?:,[0-9]{3})*(?:\.[0-9]+)?)\s*(k|m|thousand|million)\b)",
    re.I,
)
_LONG_HORIZON_TERMS = (
    "2026",
    "2027",
    "2028",
    "next year",
    "this year",
    "december",
    "november",
    "october",
    "september",
    "june 30",
    "december 31",
)
_SHORT_HORIZON_TERMS = (
    "today",
    "tomorrow",
    "this week",
    "this month",
    "week",
    "month",
)


def _contains_term(text: str, term: str) -> bool:
    return re.search(rf"(?<![a-z0-9]){re.escape(term)}(?![a-z0-9])", text) is not None


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    return any(_contains_term(text, term) for term in terms)


@dataclass(frozen=True)
class PolymarketMarketSnapshot:
    """Normalized public Polymarket market view with data-quality metadata."""

    market_id: str
    slug: str
    question: str
    event_slug: str
    category: str
    market_type: str
    horizon: str
    target_price: float | None
    yes_probability: float | None
    probability_source: str
    spread: float | None
    clob_status: str
    clob_reason: str | None
    clob_token_id: str | None
    liquidity: float
    volume: float
    liquidity_score: float
    spread_score: float
    data_quality_score: float
    eligibility: str
    quality_flags: list[str]

    def to_dict(self) -> dict:
        payload = {
            "market_id": self.market_id,
            "slug": self.slug,
            "question": self.question,
            "event_slug": self.event_slug,
            "category": self.category,
            "market_type": self.market_type,
            "horizon": self.horizon,
            "yes_probability": round(self.yes_probability, 6) if self.yes_probability is not None else None,
            "probability_source": self.probability_source,
            "spread": round(self.spread, 6) if self.spread is not None else None,
            "clob_status": self.clob_status,
            "clob_reason": self.clob_reason,
            "clob_token_id": self.clob_token_id,
            "liquidity": round(self.liquidity, 2),
            "volume": round(self.volume, 2),
            "liquidity_score": round(self.liquidity_score, 4),
            "spread_score": round(self.spread_score, 4),
            "data_quality_score": round(self.data_quality_score, 4),
            "eligibility": self.eligibility,
            "quality_flags": list(self.quality_flags),
        }
        if self.target_price is not None:
            payload["target_price"] = round(self.target_price, 2)
        return payload


@dataclass(frozen=True)
class PolymarketOpportunity:
    """Ranked analysis candidate from a public Polymarket market."""

    market_id: str
    slug: str
    question: str
    category: str
    candidate_side: str
    crowd_prob: float
    hogan_prob: float | None
    edge_score: float
    liquidity_score: float
    spread_score: float
    catalyst_score: float
    total_score: float
    rationale: str
    market_type: str = "unknown"
    horizon: str = "unknown"
    target_price: float | None = None
    safety_note: str | None = None

    def to_dict(self) -> dict:
        payload = {
            "market_id": self.market_id,
            "slug": self.slug,
            "question": self.question,
            "category": self.category,
            "candidate_side": self.candidate_side,
            "crowd_prob": round(self.crowd_prob, 4),
            "hogan_prob": round(self.hogan_prob, 4) if self.hogan_prob is not None else None,
            "edge_score": round(self.edge_score, 4),
            "liquidity_score": round(self.liquidity_score, 4),
            "spread_score": round(self.spread_score, 4),
            "catalyst_score": round(self.catalyst_score, 4),
            "total_score": round(self.total_score, 4),
            "rationale": self.rationale,
            "market_type": self.market_type,
            "horizon": self.horizon,
        }
        if self.target_price is not None:
            payload["target_price"] = round(self.target_price, 2)
        if self.safety_note:
            payload["safety_note"] = self.safety_note
        return payload


def _get_json(path: str, params: dict | None = None) -> object:
    query = f"?{urlencode(params)}" if params else ""
    url = f"{_GAMMA_BASE}{path}{query}"
    req = Request(url, headers=_HEADERS)
    try:
        with urlopen(req, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"Polymarket Gamma HTTP {exc.code} for {path}") from exc
    except URLError as exc:
        raise RuntimeError(f"Polymarket Gamma request failed for {path}: {exc}") from exc


def _get_clob_json(path: str, params: dict | None = None) -> object:
    query = f"?{urlencode(params)}" if params else ""
    url = f"{_CLOB_BASE}{path}{query}"
    req = Request(url, headers=_HEADERS)
    try:
        with urlopen(req, timeout=_TIMEOUT) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except HTTPError as exc:
        raise RuntimeError(f"Polymarket CLOB HTTP {exc.code} for {path}") from exc
    except URLError as exc:
        raise RuntimeError(f"Polymarket CLOB request failed for {path}: {exc}") from exc


def _json_list(value) -> list:
    if value is None:
        return []
    if isinstance(value, list):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return []
        return parsed if isinstance(parsed, list) else []
    return []


def _to_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _clamp_prob(value: float) -> float:
    return max(0.0, min(1.0, value))


def _market_text(market: dict) -> str:
    fields = [
        market.get("question"),
        market.get("title"),
        market.get("slug"),
        market.get("description"),
    ]
    return " ".join(str(v or "") for v in fields).lower()


def _market_id(market: dict) -> str:
    return str(
        market.get("id")
        or market.get("conditionId")
        or market.get("condition_id")
        or market.get("slug")
        or ""
    )


def _yes_probability(market: dict) -> float | None:
    if "poly_clob_midpoint" in market:
        return _clamp_prob(_to_float(market.get("poly_clob_midpoint"), default=0.5))
    outcomes = [str(v).strip().lower() for v in _json_list(market.get("outcomes"))]
    prices = [_to_float(v, default=-1.0) for v in _json_list(market.get("outcomePrices"))]
    if not outcomes or not prices or len(outcomes) != len(prices):
        return None
    try:
        idx = outcomes.index("yes")
    except ValueError:
        return None
    price = prices[idx]
    if price < 0:
        return None
    return _clamp_prob(price)


def _probability_source(market: dict) -> str:
    if "poly_clob_midpoint" in market:
        return "clob_midpoint"
    if _yes_probability(market) is not None:
        return "gamma_outcome_price"
    return "unavailable"


def _market_weight(market: dict) -> float:
    liquidity = _to_float(market.get("liquidity"), default=0.0)
    volume = _to_float(
        market.get("volume24hr", market.get("volume_24hr", market.get("volume"))),
        default=0.0,
    )
    return max(1.0, liquidity + volume)


def _market_liquidity(market: dict) -> float:
    return max(0.0, _to_float(market.get("liquidity"), default=0.0))


def _market_volume(market: dict) -> float:
    return max(
        0.0,
        _to_float(
            market.get("volume24hr", market.get("volume_24hr", market.get("volume"))),
            default=0.0,
        ),
    )


def _liquidity_score(market: dict) -> float:
    # Log scale: Polymarket liquidity varies by orders of magnitude.
    return max(0.0, min(1.0, math.log10(_market_weight(market) + 1.0) / 6.0))


def _spread_score(market: dict) -> float:
    spread = _to_float(market.get("poly_clob_spread"), default=0.05)
    return max(0.0, min(1.0, 1.0 - spread / 0.10))


def _data_quality_score(market: dict) -> tuple[float, list[str]]:
    flags: list[str] = []
    source = _probability_source(market)
    source_score = 0.0
    if source == "clob_midpoint":
        source_score = 1.0
    elif source == "gamma_outcome_price":
        source_score = 0.65
        flags.append("gamma_price_only")
    else:
        flags.append("missing_probability")

    spread_score = _spread_score(market)
    liquidity_score = _liquidity_score(market)
    if "poly_clob_spread" not in market:
        flags.append("missing_clob_spread")
    clob_status = str(market.get("poly_clob_status") or "not_enriched")
    if source != "clob_midpoint":
        flags.append(f"clob_{clob_status}")
    if spread_score < 0.35:
        flags.append("wide_spread")
    if liquidity_score < 0.20:
        flags.append("low_liquidity")

    score = source_score * 0.45 + spread_score * 0.30 + liquidity_score * 0.25
    return max(0.0, min(1.0, score)), flags


def _market_category(market: dict) -> str:
    text = _market_text(market)
    if _contains_any(text, _BTC_TERMS):
        return "btc"
    if _contains_any(text, _ETH_TERMS):
        return "eth"
    if _contains_any(text, _MACRO_RISK_TERMS):
        return "macro_risk"
    return "other"


def _extract_price_target(text: str) -> float | None:
    match = _PRICE_RE.search(text)
    if not match:
        return None
    raw = (match.group(1) or match.group(3) or "").replace(",", "")
    if not raw:
        return None
    value = float(raw)
    suffix = (match.group(2) or match.group(4) or "").lower()
    if suffix in ("k", "thousand"):
        value *= 1_000
    elif suffix in ("m", "million"):
        value *= 1_000_000
    return value


def _market_horizon(market: dict) -> str:
    text = _market_text(market)
    if _contains_any(text, _LONG_HORIZON_TERMS):
        return "long_term"
    if _contains_any(text, _SHORT_HORIZON_TERMS):
        return "short_term"
    return "unknown"


def _market_type(market: dict, category: str) -> tuple[str, str, float | None]:
    text = _market_text(market)
    target = _extract_price_target(text)
    horizon = _market_horizon(market)
    if category in ("btc", "eth") and target is not None:
        return "price_target", horizon, target
    if category in ("btc", "eth") and _contains_any(text, _CRYPTO_MACRO_TERMS):
        return "macro_crypto", horizon, target
    if category == "macro_risk":
        return "macro_risk", horizon, target
    if category in ("btc", "eth"):
        return "directional", horizon, target
    return "other", horizon, target


def normalize_market_snapshot(market: dict) -> PolymarketMarketSnapshot:
    """Normalize one public Polymarket payload for scoring and reporting."""
    category = _market_category(market)
    market_type, horizon, target_price = _market_type(market, category)
    yes_prob = _yes_probability(market)
    quality_score, quality_flags = _data_quality_score(market)
    eligibility = "research"
    if yes_prob is None or category == "other":
        eligibility = "blocked"
    elif quality_score >= 0.55 and not {"wide_spread", "low_liquidity"} & set(quality_flags):
        eligibility = "shadow_candidate"
    return PolymarketMarketSnapshot(
        market_id=_market_id(market),
        slug=str(market.get("slug") or market.get("eventSlug") or ""),
        question=str(market.get("question") or market.get("title") or ""),
        event_slug=str(market.get("eventSlug") or ""),
        category=category,
        market_type=market_type,
        horizon=horizon,
        target_price=target_price,
        yes_probability=yes_prob,
        probability_source=_probability_source(market),
        spread=_to_float(market.get("poly_clob_spread"), default=float("nan"))
        if "poly_clob_spread" in market
        else None,
        clob_status=str(market.get("poly_clob_status") or "not_enriched"),
        clob_reason=(
            str(market.get("poly_clob_reason"))
            if market.get("poly_clob_reason") is not None
            else None
        ),
        clob_token_id=(
            str(market.get("poly_clob_token_id"))
            if market.get("poly_clob_token_id") is not None
            else None
        ),
        liquidity=_market_liquidity(market),
        volume=_market_volume(market),
        liquidity_score=_liquidity_score(market),
        spread_score=_spread_score(market),
        data_quality_score=quality_score,
        eligibility=eligibility,
        quality_flags=quality_flags,
    )


def _yes_direction(market: dict) -> int:
    """Return +1 if YES is bullish, -1 if YES is bearish, 0 if unknown."""
    text = _market_text(market)
    bullish_hits = sum(1 for term in _BULLISH_TERMS if _contains_term(text, term))
    bearish_hits = sum(1 for term in _BEARISH_TERMS if _contains_term(text, term))
    if bullish_hits > bearish_hits:
        return 1
    if bearish_hits > bullish_hits:
        return -1
    return 0


def _catalyst_score(market: dict) -> float:
    text = _market_text(market)
    score = 0.0
    if _contains_any(text, _BTC_TERMS + _ETH_TERMS):
        score += 0.35
    if _contains_any(text, _MACRO_RISK_TERMS):
        score += 0.25
    if _contains_any(text, ("today", "tomorrow", "week", "month", "2026", "etf")):
        score += 0.20
    if _contains_any(text, ("above", "below", "reach", "hit", "under", "over")):
        score += 0.20
    return max(0.0, min(1.0, score))


def _yes_token_id(market: dict) -> str | None:
    outcomes = [str(v).strip().lower() for v in _json_list(market.get("outcomes"))]
    token_ids = _json_list(market.get("clobTokenIds") or market.get("clob_token_ids"))
    if outcomes and token_ids and len(outcomes) == len(token_ids):
        try:
            idx = outcomes.index("yes")
        except ValueError:
            idx = 0
        token_id = str(token_ids[idx]).strip()
        return token_id or None
    for token in market.get("tokens") or []:
        if not isinstance(token, dict):
            continue
        outcome = str(token.get("outcome") or token.get("o") or "").strip().lower()
        if outcome == "yes":
            token_id = str(token.get("token_id") or token.get("asset_id") or token.get("t") or "").strip()
            return token_id or None
    return None


def _extract_response_float(payload: object, *keys: str) -> float | None:
    if isinstance(payload, dict):
        for key in keys:
            if key in payload:
                return _to_float(payload[key], default=float("nan"))
        if len(payload) == 1:
            return _to_float(next(iter(payload.values())), default=float("nan"))
    if isinstance(payload, (int, float, str)):
        return _to_float(payload, default=float("nan"))
    return None


def fetch_clob_token_snapshot(token_id: str) -> dict[str, float]:
    """Fetch public CLOB midpoint/spread for a token ID."""
    snapshot: dict[str, float] = {}
    midpoint = _extract_response_float(
        _get_clob_json("/midpoint", {"token_id": token_id}),
        "mid",
        "midpoint",
        "price",
    )
    if midpoint is not None and midpoint == midpoint:
        snapshot["midpoint"] = _clamp_prob(midpoint)
    spread = _extract_response_float(
        _get_clob_json("/spread", {"token_id": token_id}),
        "spread",
    )
    if spread is not None and spread == spread:
        snapshot["spread"] = max(0.0, float(spread))
    return snapshot


def enrich_clob_snapshots(markets: list[dict], max_markets: int = 12) -> list[dict]:
    """Attach best-effort public CLOB midpoint/spread snapshots to markets."""
    enriched: list[dict] = []
    candidates = sorted(markets, key=_market_weight, reverse=True)
    fetched = 0
    for market in candidates:
        out = dict(market)
        token_id = _yes_token_id(out)
        if token_id:
            out["poly_clob_token_id"] = token_id
        if not token_id:
            out["poly_clob_status"] = "no_token_id"
            out["poly_clob_reason"] = "Gamma payload did not include a usable YES CLOB token ID"
        elif fetched >= max_markets:
            out["poly_clob_status"] = "skipped_limit"
            out["poly_clob_reason"] = f"CLOB enrichment limit reached ({max_markets})"
        else:
            try:
                snapshot = fetch_clob_token_snapshot(token_id)
                if "midpoint" in snapshot:
                    out["poly_clob_midpoint"] = snapshot["midpoint"]
                if "spread" in snapshot:
                    out["poly_clob_spread"] = snapshot["spread"]
                if "midpoint" in snapshot and "spread" in snapshot:
                    out["poly_clob_status"] = "ok"
                    out["poly_clob_reason"] = "CLOB midpoint and spread fetched"
                elif snapshot:
                    out["poly_clob_status"] = "partial"
                    out["poly_clob_reason"] = "CLOB returned only partial quote data"
                else:
                    out["poly_clob_status"] = "empty"
                    out["poly_clob_reason"] = "CLOB returned no usable midpoint or spread"
                fetched += 1
                time.sleep(_SLEEP)
            except Exception as exc:
                logger.debug("Polymarket CLOB snapshot failed for token %s: %s", token_id, exc)
                out["poly_clob_status"] = "failed"
                out["poly_clob_reason"] = str(exc)
        enriched.append(out)
    return enriched


def _directional_probability(market: dict, terms: tuple[str, ...]) -> float | None:
    text = _market_text(market)
    if not _contains_any(text, terms):
        return None
    yes_prob = _yes_probability(market)
    if yes_prob is None:
        return None
    bullish_hits = sum(1 for term in _BULLISH_TERMS if _contains_term(text, term))
    bearish_hits = sum(1 for term in _BEARISH_TERMS if _contains_term(text, term))
    if bullish_hits > bearish_hits:
        return yes_prob
    if bearish_hits > bullish_hits:
        return 1.0 - yes_prob
    return None


def _risk_probability(market: dict) -> float | None:
    text = _market_text(market)
    if not _contains_any(text, _MACRO_RISK_TERMS):
        return None
    return _yes_probability(market)


def score_polymarket_opportunities(
    markets: list[dict],
    *,
    hogan_btc_bull_prob: float | None = None,
    hogan_eth_bull_prob: float | None = None,
    hogan_btc_long_horizon_prob: float | None = None,
    hogan_eth_long_horizon_prob: float | None = None,
    hogan_btc_long_horizon_probs: dict[str, float] | None = None,
    hogan_eth_long_horizon_probs: dict[str, float] | None = None,
    limit: int = 10,
) -> list[PolymarketOpportunity]:
    """Rank individual Polymarket markets for research/trading review.

    If Hogan model probabilities are provided, the primary edge is crowd/model
    disagreement. Without model probabilities, this still ranks liquid,
    tight-spread, high-relevance markets as intelligence targets.
    """
    opportunities: list[PolymarketOpportunity] = []
    for market in markets:
        if str(market.get("closed", "")).lower() == "true":
            continue
        category = _market_category(market)
        if category not in ("btc", "eth", "macro_risk"):
            continue
        yes_prob = _yes_probability(market)
        if yes_prob is None:
            continue

        direction = _yes_direction(market)
        market_type, horizon, target_price = _market_type(market, category)
        safety_note = None
        if category == "btc" and direction != 0:
            crowd_bull = yes_prob if direction > 0 else 1.0 - yes_prob
            if market_type == "price_target" and horizon == "long_term":
                hogan_prob = (
                    (hogan_btc_long_horizon_probs or {}).get(_market_id(market))
                    if hogan_btc_long_horizon_probs
                    else hogan_btc_long_horizon_prob
                )
                if hogan_prob is None:
                    safety_note = "long_horizon_price_target_requires_calibrated_fair_value"
            else:
                hogan_prob = hogan_btc_bull_prob
        elif category == "eth" and direction != 0:
            crowd_bull = yes_prob if direction > 0 else 1.0 - yes_prob
            if market_type == "price_target" and horizon == "long_term":
                hogan_prob = (
                    (hogan_eth_long_horizon_probs or {}).get(_market_id(market))
                    if hogan_eth_long_horizon_probs
                    else hogan_eth_long_horizon_prob
                )
                if hogan_prob is None:
                    safety_note = "long_horizon_price_target_requires_calibrated_fair_value"
            else:
                hogan_prob = hogan_eth_bull_prob
        else:
            crowd_bull = 1.0 - yes_prob if category == "macro_risk" else yes_prob
            hogan_prob = None

        if safety_note:
            edge_score = min(1.0, abs(crowd_bull - 0.5) / 0.35)
            candidate_side = "research"
            rationale = f"{safety_note}; crowd probability {crowd_bull:.2f}"
        elif hogan_prob is not None:
            hogan_prob = _clamp_prob(float(hogan_prob))
            disagreement = hogan_prob - crowd_bull
            edge_score = min(1.0, abs(disagreement) / 0.30)
            bullish_candidate = disagreement > 0
            if direction < 0:
                bullish_candidate = not bullish_candidate
            candidate_side = "buy_yes" if bullish_candidate else "buy_no"
            rationale = f"Hogan {hogan_prob:.2f} vs crowd {crowd_bull:.2f}"
        else:
            # No model comparator: favor informative non-consensus, tradable markets.
            edge_score = min(1.0, abs(crowd_bull - 0.5) / 0.35)
            candidate_side = "research"
            rationale = f"prediction-market crowd probability {crowd_bull:.2f}"

        liq = _liquidity_score(market)
        spread = _spread_score(market)
        catalyst = _catalyst_score(market)
        total = edge_score * 0.45 + spread * 0.25 + liq * 0.20 + catalyst * 0.10
        opportunities.append(PolymarketOpportunity(
            market_id=_market_id(market),
            slug=str(market.get("slug") or market.get("eventSlug") or ""),
            question=str(market.get("question") or market.get("title") or ""),
            category=category,
            candidate_side=candidate_side,
            crowd_prob=_clamp_prob(crowd_bull),
            hogan_prob=hogan_prob,
            edge_score=edge_score,
            liquidity_score=liq,
            spread_score=spread,
            catalyst_score=catalyst,
            total_score=max(0.0, min(1.0, total)),
            rationale=rationale,
            market_type=market_type,
            horizon=horizon,
            target_price=target_price,
            safety_note=safety_note,
        ))

    return sorted(opportunities, key=lambda opp: opp.total_score, reverse=True)[:limit]


def _weighted_average(items: list[tuple[float, float]]) -> float | None:
    if not items:
        return None
    total_w = sum(weight for _value, weight in items)
    if total_w <= 0:
        return None
    return sum(value * weight for value, weight in items) / total_w


def _dispersion(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    var = sum((v - mean) ** 2 for v in values) / len(values)
    return var ** 0.5


def extract_polymarket_metrics(markets: list[dict]) -> dict[str, float]:
    """Return compact Hogan metrics from Polymarket market payloads."""
    btc: list[tuple[float, float]] = []
    eth: list[tuple[float, float]] = []
    macro_risk: list[tuple[float, float]] = []
    crypto_risk: list[tuple[float, float]] = []
    all_directional: list[float] = []
    clob_midpoints: list[float] = []
    clob_spreads: list[float] = []
    total_weight = 0.0

    for market in markets:
        if str(market.get("closed", "")).lower() == "true":
            continue
        weight = _market_weight(market)
        total_weight += weight
        if "poly_clob_midpoint" in market:
            clob_midpoints.append(_clamp_prob(_to_float(market["poly_clob_midpoint"], 0.5)))
        if "poly_clob_spread" in market:
            clob_spreads.append(max(0.0, _to_float(market["poly_clob_spread"], 0.0)))

        btc_prob = _directional_probability(market, _BTC_TERMS)
        if btc_prob is not None:
            btc.append((btc_prob, weight))
            crypto_risk.append((1.0 - btc_prob, weight))
            all_directional.append(btc_prob)

        eth_prob = _directional_probability(market, _ETH_TERMS)
        if eth_prob is not None:
            eth.append((eth_prob, weight))
            crypto_risk.append((1.0 - eth_prob, weight))
            all_directional.append(eth_prob)

        risk_prob = _risk_probability(market)
        if risk_prob is not None:
            macro_risk.append((risk_prob, weight))

    metrics: dict[str, float] = {
        "poly_market_count": float(len(markets)),
        "poly_liquidity_score": min(1.0, total_weight / 1_000_000.0),
        "poly_signal_dispersion": _dispersion(all_directional),
    }
    if clob_midpoints:
        metrics["poly_orderbook_midpoint_avg"] = round(sum(clob_midpoints) / len(clob_midpoints), 6)
    if clob_spreads:
        metrics["poly_orderbook_spread_avg"] = round(sum(clob_spreads) / len(clob_spreads), 6)
    top_opps = score_polymarket_opportunities(markets, limit=5)
    if top_opps:
        metrics["poly_top_opportunity_score"] = round(top_opps[0].total_score, 6)
        metrics["poly_opportunity_count"] = float(len(top_opps))
    for name, values in (
        ("poly_btc_bull_prob", btc),
        ("poly_eth_bull_prob", eth),
        ("poly_crypto_risk_prob", crypto_risk),
        ("poly_macro_risk_prob", macro_risk),
    ):
        avg = _weighted_average(values)
        if avg is not None:
            metrics[name] = round(_clamp_prob(avg), 6)
    return metrics


def fetch_active_markets(limit: int = 100) -> list[dict]:
    """Fetch active markets via Gamma events and flatten their market lists."""
    payload = _get_json(
        "/events",
        {
            "active": "true",
            "closed": "false",
            "order": "volume_24hr",
            "ascending": "false",
            "limit": max(1, int(limit)),
        },
    )
    events = payload if isinstance(payload, list) else []
    markets: list[dict] = []
    for event in events:
        if not isinstance(event, dict):
            continue
        event_slug = event.get("slug")
        for market in event.get("markets") or []:
            if not isinstance(market, dict):
                continue
            market.setdefault("eventSlug", event_slug)
            markets.append(market)
    return markets


def fetch_and_store(
    symbol: str = "BTC/USD",
    db_path: str = "data/hogan.db",
    limit: int = 100,
    include_clob: bool = True,
    clob_limit: int = 12,
    hogan_btc_bull_prob: float | None = None,
    hogan_eth_bull_prob: float | None = None,
) -> int:
    """Fetch active Polymarket markets and store compact daily metrics."""
    from hogan_bot.storage import (
        get_connection,
        insert_polymarket_opportunities,
        upsert_onchain,
    )

    logger.info("Fetching Polymarket public markets (limit=%d)", limit)
    markets = fetch_active_markets(limit=limit)
    if include_clob:
        markets = enrich_clob_snapshots(markets, max_markets=max(0, int(clob_limit)))
    metrics = extract_polymarket_metrics(markets)
    top_opps = score_polymarket_opportunities(
        markets,
        hogan_btc_bull_prob=hogan_btc_bull_prob,
        hogan_eth_bull_prob=hogan_eth_bull_prob,
        limit=5,
    )
    if top_opps:
        metrics["poly_top_edge_score"] = round(top_opps[0].edge_score, 6)
        metrics["poly_top_tradability_score"] = round(
            (top_opps[0].liquidity_score + top_opps[0].spread_score) / 2.0,
            6,
        )
    if not metrics:
        return 0
    today = date.today().isoformat()
    records = [(today, metric, value) for metric, value in sorted(metrics.items())]
    conn = get_connection(db_path)
    try:
        written = upsert_onchain(conn, symbol, records)
        if top_opps:
            insert_polymarket_opportunities(
                conn,
                symbol,
                int(time.time() * 1000),
                [opp.to_dict() for opp in top_opps],
            )
        return written
    finally:
        conn.close()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fetch public Polymarket prediction-market signals",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="BTC/USD", help="Symbol for DB storage")
    p.add_argument("--db", default=os.getenv("HOGAN_DB_PATH", "data/hogan.db"))
    p.add_argument("--limit", type=int, default=100, help="Active events to inspect")
    p.add_argument("--no-clob", action="store_true", help="Skip public CLOB midpoint/spread snapshots")
    p.add_argument("--clob-limit", type=int, default=12, help="Maximum markets to enrich via CLOB")
    p.add_argument("--scan", action="store_true", help="Print ranked Polymarket opportunity candidates")
    p.add_argument("--btc-prob", type=float, default=None, help="Optional Hogan BTC bull probability for edge scoring")
    p.add_argument("--eth-prob", type=float, default=None, help="Optional Hogan ETH bull probability for edge scoring")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    written = fetch_and_store(
        symbol=args.symbol,
        db_path=args.db,
        limit=args.limit,
        include_clob=not args.no_clob,
        clob_limit=args.clob_limit,
        hogan_btc_bull_prob=args.btc_prob,
        hogan_eth_bull_prob=args.eth_prob,
    )
    print(f"Polymarket metrics written: {written}")
    if args.scan:
        markets = fetch_active_markets(limit=args.limit)
        if not args.no_clob:
            markets = enrich_clob_snapshots(markets, max_markets=args.clob_limit)
        opportunities = score_polymarket_opportunities(
            markets,
            hogan_btc_bull_prob=args.btc_prob,
            hogan_eth_bull_prob=args.eth_prob,
        )
        for idx, opp in enumerate(opportunities, start=1):
            print(f"{idx}. {opp.total_score:.3f} {opp.candidate_side} {opp.question}")
            print(f"   {opp.rationale}; spread={opp.spread_score:.2f} liquidity={opp.liquidity_score:.2f}")


if __name__ == "__main__":
    main()
