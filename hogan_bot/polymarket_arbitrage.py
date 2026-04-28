"""Logical inconsistency detectors for Polymarket analysis.

The output is alert-only. These helpers do not place orders.
"""
from __future__ import annotations

from dataclasses import dataclass

from hogan_bot.fetch_polymarket import (
    _extract_price_target,
    _json_list,
    _yes_probability,
)


@dataclass(frozen=True)
class ArbitrageAlert:
    kind: str
    market_ids: list[str]
    severity: float
    message: str


def _market_id(market: dict) -> str:
    return str(market.get("id") or market.get("conditionId") or market.get("slug") or "")


def _text(market: dict) -> str:
    return " ".join(str(market.get(k) or "") for k in ("question", "title", "slug")).lower()


def detect_crypto_ladder_violations(markets: list[dict], asset_terms: tuple[str, ...] = ("bitcoin", "btc")) -> list[ArbitrageAlert]:
    """Detect monotonicity breaks in 'above/reach target' crypto ladders."""
    ladder: list[tuple[float, float, str]] = []
    for market in markets:
        text = _text(market)
        if not any(term in text for term in asset_terms):
            continue
        if not any(term in text for term in ("above", "over", "reach", "hit")):
            continue
        target = _extract_price_target(text)
        prob = _yes_probability(market)
        if target is None or prob is None:
            continue
        ladder.append((target, prob, _market_id(market)))
    ladder.sort(key=lambda row: row[0])
    alerts: list[ArbitrageAlert] = []
    for lower, higher in zip(ladder, ladder[1:]):
        low_target, low_prob, low_id = lower
        high_target, high_prob, high_id = higher
        if high_prob > low_prob + 0.02:
            alerts.append(ArbitrageAlert(
                kind="crypto_ladder_monotonicity",
                market_ids=[low_id, high_id],
                severity=min(1.0, high_prob - low_prob),
                message=(
                    f"Higher target ${high_target:,.0f} priced above lower target "
                    f"${low_target:,.0f}: {high_prob:.2f} > {low_prob:.2f}"
                ),
            ))
    return alerts


def detect_mutually_exclusive_overpricing(markets: list[dict], group_key: str = "eventSlug") -> list[ArbitrageAlert]:
    """Detect event groups whose YES prices sum materially above 1."""
    groups: dict[str, list[tuple[str, float]]] = {}
    for market in markets:
        group = str(market.get(group_key) or "")
        outcomes = [str(v).lower() for v in _json_list(market.get("outcomes"))]
        if group and "yes" in outcomes and "no" in outcomes:
            prob = _yes_probability(market)
            if prob is not None:
                groups.setdefault(group, []).append((_market_id(market), prob))
    alerts: list[ArbitrageAlert] = []
    for group, rows in groups.items():
        if len(rows) < 2:
            continue
        total = sum(prob for _market_id_value, prob in rows)
        if total > 1.03:
            alerts.append(ArbitrageAlert(
                kind="mutually_exclusive_overpricing",
                market_ids=[market_id for market_id, _prob in rows],
                severity=min(1.0, total - 1.0),
                message=f"{group} YES probabilities sum to {total:.2f}",
            ))
    return alerts


def detect_arbitrage_alerts(markets: list[dict]) -> list[ArbitrageAlert]:
    """Run all lightweight alert-only inconsistency detectors."""
    return [
        *detect_crypto_ladder_violations(markets, ("bitcoin", "btc")),
        *detect_crypto_ladder_violations(markets, ("ethereum", "eth")),
        *detect_mutually_exclusive_overpricing(markets),
    ]
