"""Fetch public Polymarket prediction-market signals for Hogan.

Phase one is analysis-only: no wallet keys, no authenticated CLOB trading, and
no order placement. We use public Gamma market discovery data to derive compact
daily metrics for BTC/ETH and broader macro risk sentiment.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import time
from datetime import date
from urllib.parse import urlencode
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_GAMMA_BASE = "https://gamma-api.polymarket.com"
_CLOB_BASE = "https://clob.polymarket.com"
_TIMEOUT = 20
_SLEEP = 0.25

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


def _get_json(path: str, params: dict | None = None) -> object:
    query = f"?{urlencode(params)}" if params else ""
    url = f"{_GAMMA_BASE}{path}{query}"
    req = Request(url, headers={"Accept": "application/json"})
    with urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


def _get_clob_json(path: str, params: dict | None = None) -> object:
    query = f"?{urlencode(params)}" if params else ""
    url = f"{_CLOB_BASE}{path}{query}"
    req = Request(url, headers={"Accept": "application/json"})
    with urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode("utf-8"))


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


def _market_weight(market: dict) -> float:
    liquidity = _to_float(market.get("liquidity"), default=0.0)
    volume = _to_float(
        market.get("volume24hr", market.get("volume_24hr", market.get("volume"))),
        default=0.0,
    )
    return max(1.0, liquidity + volume)


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
        if token_id and fetched < max_markets:
            try:
                snapshot = fetch_clob_token_snapshot(token_id)
                if "midpoint" in snapshot:
                    out["poly_clob_midpoint"] = snapshot["midpoint"]
                if "spread" in snapshot:
                    out["poly_clob_spread"] = snapshot["spread"]
                fetched += 1
                time.sleep(_SLEEP)
            except Exception as exc:
                logger.debug("Polymarket CLOB snapshot failed for token %s: %s", token_id, exc)
        enriched.append(out)
    return enriched


def _directional_probability(market: dict, terms: tuple[str, ...]) -> float | None:
    text = _market_text(market)
    if not any(term in text for term in terms):
        return None
    yes_prob = _yes_probability(market)
    if yes_prob is None:
        return None
    bullish_hits = sum(1 for term in _BULLISH_TERMS if term in text)
    bearish_hits = sum(1 for term in _BEARISH_TERMS if term in text)
    if bullish_hits > bearish_hits:
        return yes_prob
    if bearish_hits > bullish_hits:
        return 1.0 - yes_prob
    return None


def _risk_probability(market: dict) -> float | None:
    text = _market_text(market)
    if not any(term in text for term in _MACRO_RISK_TERMS):
        return None
    return _yes_probability(market)


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
) -> int:
    """Fetch active Polymarket markets and store compact daily metrics."""
    from hogan_bot.storage import get_connection, upsert_onchain

    logger.info("Fetching Polymarket public markets (limit=%d)", limit)
    markets = fetch_active_markets(limit=limit)
    if include_clob:
        markets = enrich_clob_snapshots(markets, max_markets=max(0, int(clob_limit)))
    metrics = extract_polymarket_metrics(markets)
    if not metrics:
        return 0
    today = date.today().isoformat()
    records = [(today, metric, value) for metric, value in sorted(metrics.items())]
    conn = get_connection(db_path)
    try:
        return upsert_onchain(conn, symbol, records)
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
    )
    print(f"Polymarket metrics written: {written}")


if __name__ == "__main__":
    main()
