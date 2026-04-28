"""Polymarket Alpha Lab runner.

This module runs live public-data analysis and optional shadow-trade tracking.
It never authenticates to Polymarket and never places real orders.
"""
from __future__ import annotations

import argparse
import logging
import time
from dataclasses import dataclass
from pathlib import Path

from hogan_bot.fetch_polymarket import (
    PolymarketMarketSnapshot,
    PolymarketOpportunity,
    _market_id,
    _yes_probability,
    enrich_clob_snapshots,
    fetch_active_markets,
    normalize_market_snapshot,
    score_polymarket_opportunities,
)
from hogan_bot.polymarket_arbitrage import ArbitrageAlert, detect_arbitrage_alerts
from hogan_bot.polymarket_edge import EdgeAssessment, assess_opportunity_edge
from hogan_bot.polymarket_intelligence import (
    IntelligenceAssessment,
    assess_intelligence,
)
from hogan_bot.polymarket_long_horizon import estimate_btc_long_horizon_probability
from hogan_bot.polymarket_promotion import evaluate_shadow_ledger
from hogan_bot.storage import (
    close_polymarket_shadow_trade,
    get_connection,
    insert_polymarket_opportunities,
    open_polymarket_shadow_trade,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class AlphaCandidate:
    opportunity: PolymarketOpportunity
    edge: EdgeAssessment
    snapshot: PolymarketMarketSnapshot
    assessment: IntelligenceAssessment
    shadow_trade_id: int | None = None


@dataclass(frozen=True)
class ShadowLedgerUpdate:
    marked_open: int
    closed: int
    open_count: int
    open_exposure_usd: float
    unrealized_pnl: float


@dataclass(frozen=True)
class ShadowPositionView:
    trade_id: int
    status: str
    slug: str
    side: str
    entry_prob: float
    current_prob: float | None
    size_usd: float
    pnl: float | None


@dataclass(frozen=True)
class AlphaRunResult:
    ts_ms: int
    report_path: str
    opportunities: list[PolymarketOpportunity]
    candidates: list[AlphaCandidate]
    arbitrage_alerts: int
    shadow_opened: int
    shadow_ledger: ShadowLedgerUpdate
    authority_mode: str
    btc_prob: float | None
    eth_prob: float | None
    promotion_approved: bool
    promotion_reasons: list[str]


@dataclass(frozen=True)
class RecommendationRunResult:
    opportunities: list[PolymarketOpportunity]
    candidates: list[AlphaCandidate]
    arbitrage_alerts: int
    btc_prob: float | None
    eth_prob: float | None


def _has_open_shadow(conn, market_id: str, side: str) -> bool:
    row = conn.execute(
        """
        SELECT 1
        FROM polymarket_shadow_trades
        WHERE status = 'open' AND market_id = ? AND side = ?
        LIMIT 1
        """,
        (market_id, side),
    ).fetchone()
    return row is not None


def _open_shadow_if_new(
    conn,
    *,
    symbol: str,
    ts_ms: int,
    opportunity: PolymarketOpportunity,
    edge: EdgeAssessment,
    assessment: IntelligenceAssessment,
    max_open_trades: int,
    max_open_exposure_usd: float,
) -> int | None:
    if not assessment.shadow_eligible:
        return None
    if opportunity.candidate_side not in ("buy_yes", "buy_no"):
        return None
    if _has_open_shadow(conn, opportunity.market_id, opportunity.candidate_side):
        return None
    open_count, open_exposure = _open_shadow_budget(conn)
    if open_count >= max_open_trades or open_exposure >= max_open_exposure_usd:
        return None
    remaining_exposure = max(0.0, max_open_exposure_usd - open_exposure)
    if remaining_exposure < 1.0:
        return None
    size_usd = max(1.0, min(assessment.recommended_size_usd, edge.max_size_usd, remaining_exposure))
    return open_polymarket_shadow_trade(
        conn,
        opened_ts_ms=ts_ms,
        symbol=symbol,
        market_id=opportunity.market_id,
        slug=opportunity.slug,
        side=opportunity.candidate_side,
        entry_prob=opportunity.crowd_prob,
        size_usd=size_usd,
        rationale=f"{opportunity.rationale}; after_cost_ev={edge.after_cost_ev:.4f}",
        raw={
            "opportunity": opportunity.to_dict(),
            "assessment": assessment.to_dict(),
            "edge": {
                "after_cost_ev": edge.after_cost_ev,
                "expected_value": edge.expected_value,
                "decision": edge.decision,
                "reject_reasons": edge.reject_reasons,
            },
        },
    )


def _open_shadow_budget(conn) -> tuple[int, float]:
    row = conn.execute(
        """
        SELECT COUNT(*), COALESCE(SUM(size_usd), 0.0)
        FROM polymarket_shadow_trades
        WHERE status = 'open'
        """
    ).fetchone()
    if row is None:
        return 0, 0.0
    return int(row[0] or 0), float(row[1] or 0.0)


def _authority_allows_shadow(authority_mode: str, promotion_approved: bool) -> bool:
    if authority_mode == "shadow":
        return True
    if authority_mode == "conditional":
        return promotion_approved
    return False


def _position_delta(side: str, entry_yes_prob: float, current_yes_prob: float) -> float:
    delta = current_yes_prob - entry_yes_prob
    if side == "buy_no":
        return -delta
    return delta


def _shadow_pnl(side: str, entry_prob: float, current_prob: float, size_usd: float) -> float:
    return float(size_usd) * _position_delta(side, entry_prob, current_prob)


def _market_price_index(markets: list[dict]) -> dict[str, float]:
    prices: dict[str, float] = {}
    for market in markets:
        market_id = _market_id(market)
        prob = _yes_probability(market)
        if market_id and prob is not None:
            prices[market_id] = prob
    return prices


def _update_shadow_ledger(
    conn,
    *,
    markets: list[dict],
    ts_ms: int,
    skip_trade_ids: set[int] | None = None,
    take_profit_delta: float = 0.12,
    stop_loss_delta: float = -0.08,
) -> ShadowLedgerUpdate:
    """Mark open shadow trades to latest public YES probability and close exits."""
    prices = _market_price_index(markets)
    rows = conn.execute(
        """
        SELECT id, market_id, side, entry_prob, size_usd
        FROM polymarket_shadow_trades
        WHERE status = 'open'
        """,
    ).fetchall()
    skip_trade_ids = skip_trade_ids or set()
    marked = 0
    closed = 0
    open_count = 0
    exposure = 0.0
    unrealized = 0.0

    for trade_id, market_id, side, entry_prob, size_usd in rows:
        if int(trade_id) in skip_trade_ids:
            open_count += 1
            exposure += float(size_usd)
            continue
        current_prob = prices.get(str(market_id))
        if current_prob is None:
            open_count += 1
            exposure += float(size_usd)
            continue
        marked += 1
        delta = _position_delta(str(side), float(entry_prob), float(current_prob))
        pnl = float(size_usd) * delta
        if delta >= take_profit_delta or delta <= stop_loss_delta:
            close_polymarket_shadow_trade(
                conn,
                int(trade_id),
                closed_ts_ms=ts_ms,
                exit_prob=float(current_prob),
            )
            closed += 1
        else:
            open_count += 1
            exposure += float(size_usd)
            unrealized += pnl

    return ShadowLedgerUpdate(
        marked_open=marked,
        closed=closed,
        open_count=open_count,
        open_exposure_usd=exposure,
        unrealized_pnl=unrealized,
    )


def _load_shadow_position_views(conn, prices: dict[str, float], *, limit: int = 10) -> list[ShadowPositionView]:
    rows = conn.execute(
        """
        SELECT id, status, market_id, slug, side, entry_prob, exit_prob, size_usd, realized_pnl
        FROM polymarket_shadow_trades
        ORDER BY opened_ts_ms DESC
        LIMIT ?
        """,
        (int(limit),),
    ).fetchall()
    views: list[ShadowPositionView] = []
    for trade_id, status, market_id, slug, side, entry_prob, exit_prob, size_usd, realized_pnl in rows:
        current_prob = float(exit_prob) if exit_prob is not None else prices.get(str(market_id))
        pnl = (
            float(realized_pnl)
            if realized_pnl is not None
            else (
                _shadow_pnl(str(side), float(entry_prob), float(current_prob), float(size_usd))
                if current_prob is not None
                else None
            )
        )
        views.append(ShadowPositionView(
            trade_id=int(trade_id),
            status=str(status),
            slug=str(slug or market_id),
            side=str(side),
            entry_prob=float(entry_prob),
            current_prob=current_prob,
            size_usd=float(size_usd),
            pnl=pnl,
        ))
    return views


def _latest_ml_probability(conn, symbol: str) -> float | None:
    row = conn.execute(
        """
        SELECT ml_up_prob
        FROM decision_log
        WHERE symbol = ? AND ml_up_prob IS NOT NULL
        ORDER BY ts_ms DESC
        LIMIT 1
        """,
        (symbol,),
    ).fetchone()
    if row is None:
        return None
    try:
        return max(0.0, min(1.0, float(row[0])))
    except (TypeError, ValueError):
        return None


def _resolve_model_probability(
    conn,
    *,
    explicit_prob: float | None,
    preferred_symbol: str,
    fallback_symbol: str,
) -> float | None:
    if explicit_prob is not None:
        return max(0.0, min(1.0, float(explicit_prob)))
    return _latest_ml_probability(conn, preferred_symbol) or _latest_ml_probability(conn, fallback_symbol)


def _resolve_probabilities(
    conn,
    *,
    symbol: str,
    btc_prob: float | None,
    eth_prob: float | None,
) -> tuple[float | None, float | None]:
    symbol_upper = symbol.upper()
    btc_symbol = symbol if symbol_upper.startswith("BTC/") else "BTC/USD"
    eth_symbol = symbol if symbol_upper.startswith("ETH/") else "ETH/USD"
    resolved_btc = _resolve_model_probability(
        conn,
        explicit_prob=btc_prob,
        preferred_symbol=btc_symbol,
        fallback_symbol="BTC/USD",
    )
    resolved_eth = _resolve_model_probability(
        conn,
        explicit_prob=eth_prob,
        preferred_symbol=eth_symbol,
        fallback_symbol="ETH/USD",
    )
    return resolved_btc, resolved_eth


def _build_candidates(
    *,
    markets: list[dict],
    btc_prob: float | None,
    eth_prob: float | None,
    btc_long_prob: float | None,
    eth_long_prob: float | None,
    btc_long_probs: dict[str, float] | None = None,
) -> tuple[list[PolymarketOpportunity], list[AlphaCandidate], list[ArbitrageAlert]]:
    snapshots = {snapshot.market_id: snapshot for snapshot in map(normalize_market_snapshot, markets)}
    opportunities = score_polymarket_opportunities(
        markets,
        hogan_btc_bull_prob=btc_prob,
        hogan_eth_bull_prob=eth_prob,
        hogan_btc_long_horizon_prob=btc_long_prob,
        hogan_eth_long_horizon_prob=eth_long_prob,
        hogan_btc_long_horizon_probs=btc_long_probs,
        limit=25,
    )
    candidates: list[AlphaCandidate] = []
    for opp in opportunities:
        edge = assess_opportunity_edge(opp)
        snapshot = snapshots.get(opp.market_id)
        if snapshot is None:
            snapshot = PolymarketMarketSnapshot(
                market_id=opp.market_id,
                slug=opp.slug,
                question=opp.question,
                event_slug="",
                category=opp.category,
                market_type=opp.market_type,
                horizon=opp.horizon,
                target_price=opp.target_price,
                yes_probability=opp.crowd_prob,
                probability_source="opportunity_fallback",
                spread=None,
                clob_status="snapshot_missing",
                clob_reason="No normalized snapshot was available for this opportunity",
                clob_token_id=None,
                liquidity=0.0,
                volume=0.0,
                liquidity_score=opp.liquidity_score,
                spread_score=opp.spread_score,
                data_quality_score=0.25,
                eligibility="research",
                quality_flags=["snapshot_missing"],
            )
        assessment = assess_intelligence(opp, edge, snapshot)
        candidates.append(AlphaCandidate(opp, edge, snapshot, assessment))
    return opportunities, candidates, detect_arbitrage_alerts(markets)


def run_recommendations_only(
    *,
    db_path: str = "data/hogan.db",
    symbol: str = "BTC/USD",
    limit: int = 100,
    include_clob: bool = True,
    clob_limit: int = 12,
    btc_prob: float | None = None,
    eth_prob: float | None = None,
    btc_long_prob: float | None = None,
    eth_long_prob: float | None = None,
    use_long_horizon_model: bool = True,
) -> RecommendationRunResult:
    """Scan and assess markets without writing reports or shadow ledger rows."""
    markets = fetch_active_markets(limit=limit)
    if include_clob:
        markets = enrich_clob_snapshots(markets, max_markets=clob_limit)
    conn = get_connection(db_path)
    try:
        btc_prob, eth_prob = _resolve_probabilities(
            conn,
            symbol=symbol,
            btc_prob=btc_prob,
            eth_prob=eth_prob,
        )
        btc_long_probs = _estimate_btc_long_horizon_probs(
            conn,
            markets=markets,
            symbol=symbol,
            enabled=use_long_horizon_model and btc_long_prob is None,
        )
    finally:
        conn.close()
    opportunities, candidates, alerts = _build_candidates(
        markets=markets,
        btc_prob=btc_prob,
        eth_prob=eth_prob,
        btc_long_prob=btc_long_prob,
        eth_long_prob=eth_long_prob,
        btc_long_probs=btc_long_probs,
    )
    return RecommendationRunResult(
        opportunities=opportunities,
        candidates=candidates,
        arbitrage_alerts=len(alerts),
        btc_prob=btc_prob,
        eth_prob=eth_prob,
    )


def print_recommendations(result: RecommendationRunResult, *, limit: int = 10) -> None:
    print(f"Opportunities: {len(result.opportunities)}")
    print(f"Arbitrage alerts: {result.arbitrage_alerts}")
    print(f"Hogan BTC probability: {result.btc_prob:.4f}" if result.btc_prob is not None else "Hogan BTC probability: n/a")
    print(f"Hogan ETH probability: {result.eth_prob:.4f}" if result.eth_prob is not None else "Hogan ETH probability: n/a")
    for idx, candidate in enumerate(
        sorted(result.candidates, key=lambda c: c.assessment.evidence_score, reverse=True)[:limit],
        start=1,
    ):
        assessment = candidate.assessment
        opp = candidate.opportunity
        flags = ", ".join(assessment.risk_flags) if assessment.risk_flags else "none"
        print(f"{idx}. {assessment.recommendation} {opp.question}")
        print(
            f"   evidence={assessment.evidence_score:.3f} "
            f"confidence={assessment.confidence:.3f} "
            f"size=${assessment.recommended_size_usd:.2f} "
            f"fair={assessment.fair_value_source}"
        )
        print(f"   flags={flags}")
        print(f"   clob={candidate.snapshot.clob_status}: {candidate.snapshot.clob_reason or 'n/a'}")
        print(f"   thesis={assessment.thesis}")


def _estimate_btc_long_horizon_probs(
    conn,
    *,
    markets: list[dict],
    symbol: str,
    enabled: bool,
) -> dict[str, float]:
    if not enabled:
        return {}
    btc_symbol = symbol if symbol.upper().startswith("BTC/") else "BTC/USD"
    probs: dict[str, float] = {}
    for market in markets:
        snapshot = normalize_market_snapshot(market)
        if (
            snapshot.category != "btc"
            or snapshot.market_type != "price_target"
            or snapshot.horizon != "long_term"
            or snapshot.target_price is None
        ):
            continue
        estimate = estimate_btc_long_horizon_probability(
            conn,
            target_price=snapshot.target_price,
            question=snapshot.question,
            symbol=btc_symbol,
        )
        if estimate is not None:
            probs[snapshot.market_id] = estimate.probability
    return probs


def _write_report(
    *,
    report_dir: str,
    ts_ms: int,
    candidates: list[AlphaCandidate],
    arbitrage_alerts: list[ArbitrageAlert],
    shadow_ledger: ShadowLedgerUpdate,
    shadow_positions: list[ShadowPositionView],
    promotion_metrics: dict[str, float],
    authority_mode: str,
    btc_prob: float | None,
    eth_prob: float | None,
    promotion_approved: bool,
    promotion_reasons: list[str],
) -> str:
    out_dir = Path(report_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"polymarket_alpha_{ts_ms}.md"
    lines = [
        "# Polymarket Alpha Lab Report",
        "",
        f"- Timestamp ms: `{ts_ms}`",
        f"- Candidates reviewed: `{len(candidates)}`",
        f"- Arbitrage alerts: `{len(arbitrage_alerts)}`",
        f"- Shadow marked open: `{shadow_ledger.marked_open}`",
        f"- Shadow closed: `{shadow_ledger.closed}`",
        f"- Shadow open exposure: `${shadow_ledger.open_exposure_usd:.2f}`",
        f"- Shadow unrealized PnL: `${shadow_ledger.unrealized_pnl:.2f}`",
        f"- Closed shadow PnL: `${promotion_metrics.get('total_pnl', 0.0):.2f}`",
        f"- Closed shadow win rate: `{promotion_metrics.get('win_rate', 0.0):.2%}`",
        f"- Max drawdown: `${promotion_metrics.get('max_drawdown', 0.0):.2f}`",
        f"- Hogan BTC probability: `{btc_prob:.4f}`" if btc_prob is not None else "- Hogan BTC probability: `n/a`",
        f"- Hogan ETH probability: `{eth_prob:.4f}`" if eth_prob is not None else "- Hogan ETH probability: `n/a`",
        f"- Promotion approved: `{promotion_approved}`",
        f"- Authority mode: `{authority_mode}`",
    ]
    if promotion_reasons:
        lines.append(f"- Promotion blockers: `{', '.join(promotion_reasons)}`")
    avg_quality = (
        sum(candidate.assessment.data_quality_score for candidate in candidates) / len(candidates)
        if candidates
        else 0.0
    )
    shadow_candidates = sum(1 for candidate in candidates if candidate.assessment.recommendation == "shadow_candidate")
    research_only = sum(1 for candidate in candidates if candidate.assessment.recommendation == "research")
    avoid = sum(1 for candidate in candidates if candidate.assessment.recommendation == "avoid")
    lines.extend([
        "",
        "## Data Quality",
        "",
        f"- Average data quality: `{avg_quality:.4f}`",
        f"- Shadow candidates: `{shadow_candidates}`",
        f"- Research-only: `{research_only}`",
        f"- Avoid: `{avoid}`",
        "",
        "## Machine Recommendations",
        "",
    ])
    for idx, candidate in enumerate(sorted(candidates, key=lambda c: c.assessment.evidence_score, reverse=True)[:10], start=1):
        assessment = candidate.assessment
        opp = candidate.opportunity
        flags = ", ".join(assessment.risk_flags) if assessment.risk_flags else "none"
        lines.extend([
            f"### {idx}. {opp.question}",
            f"- Recommendation: `{assessment.recommendation}`",
            f"- Evidence score: `{assessment.evidence_score:.4f}`",
            f"- Confidence: `{assessment.confidence:.4f}`",
            f"- Recommended size: `${assessment.recommended_size_usd:.2f}`",
            f"- Fair-value source: `{assessment.fair_value_source}`",
            f"- CLOB diagnostic: `{candidate.snapshot.clob_status}` - {candidate.snapshot.clob_reason or 'n/a'}",
            f"- Risk flags: `{flags}`",
            f"- Thesis: {assessment.thesis}",
            "",
        ])
    lines.extend(["", "## Shadow Positions", ""])
    if not shadow_positions:
        lines.append("No shadow positions found.")
    for pos in shadow_positions:
        current = f"{pos.current_prob:.4f}" if pos.current_prob is not None else "n/a"
        pnl = f"${pos.pnl:.2f}" if pos.pnl is not None else "n/a"
        lines.append(
            f"- `#{pos.trade_id}` `{pos.status}` `{pos.side}` "
            f"entry=`{pos.entry_prob:.4f}` current=`{current}` size=`${pos.size_usd:.2f}` pnl=`{pnl}` {pos.slug}"
        )
    lines.extend(["", "## Top Candidates", ""])
    for idx, candidate in enumerate(candidates[:10], start=1):
        opp = candidate.opportunity
        edge = candidate.edge
        assessment = candidate.assessment
        shadow = f" shadow_id={candidate.shadow_trade_id}" if candidate.shadow_trade_id else ""
        lines.extend([
            f"### {idx}. {opp.question}",
            f"- Side: `{opp.candidate_side}`{shadow}",
            f"- Decision: `{edge.decision}`",
            f"- Recommendation: `{assessment.recommendation}`",
            f"- Market type: `{opp.market_type}` / `{opp.horizon}`",
            f"- Data quality: `{assessment.data_quality_score:.4f}`",
            f"- CLOB diagnostic: `{candidate.snapshot.clob_status}` - {candidate.snapshot.clob_reason or 'n/a'}",
            f"- Confidence: `{assessment.confidence:.4f}`",
            f"- Recommended size: `${assessment.recommended_size_usd:.2f}`",
            f"- Total score: `{opp.total_score:.4f}`",
            f"- After-cost EV: `{edge.after_cost_ev:.4f}`",
            f"- Crowd probability: `{opp.crowd_prob:.4f}`",
            f"- Hogan probability: `{opp.hogan_prob:.4f}`" if opp.hogan_prob is not None else "- Hogan probability: `n/a`",
            f"- Rationale: {opp.rationale}",
        ])
        if opp.target_price is not None:
            lines.append(f"- Target price: `${opp.target_price:,.0f}`")
        if opp.safety_note:
            lines.append(f"- Safety note: `{opp.safety_note}`")
        if edge.reject_reasons:
            lines.append(f"- Reject reasons: `{', '.join(edge.reject_reasons)}`")
        lines.append("")
    lines.extend(["", "## Arbitrage Alerts", ""])
    if not arbitrage_alerts:
        lines.append("No alert-only inconsistencies detected.")
    for idx, alert in enumerate(sorted(arbitrage_alerts, key=lambda a: a.severity, reverse=True)[:15], start=1):
        ids = ", ".join(alert.market_ids[:5])
        lines.extend([
            f"### {idx}. {alert.kind}",
            f"- Severity: `{alert.severity:.4f}`",
            f"- Market IDs: `{ids}`",
            f"- Message: {alert.message}",
            "",
        ])
    next_action = "Keep collecting shadow evidence."
    if authority_mode in ("research", "advisory"):
        next_action = f"Authority is `{authority_mode}`; review recommendations without opening new shadow trades."
    elif shadow_candidates:
        next_action = "Review shadow candidates and let auto-shadow open only eligible hypothetical positions."
    if promotion_approved:
        next_action = "Promotion gate passed; review evidence before any authority increase."
    lines.extend(["", "## Next Action", "", next_action])
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def run_alpha_lab(
    *,
    db_path: str = "data/hogan.db",
    symbol: str = "BTC/USD",
    limit: int = 100,
    include_clob: bool = True,
    clob_limit: int = 12,
    btc_prob: float | None = None,
    eth_prob: float | None = None,
    btc_long_prob: float | None = None,
    eth_long_prob: float | None = None,
    auto_shadow: bool = True,
    use_long_horizon_model: bool = True,
    authority_mode: str = "shadow",
    max_open_shadow_trades: int = 10,
    max_open_shadow_exposure_usd: float = 250.0,
    report_dir: str = "reports/polymarket",
) -> AlphaRunResult:
    """Run public-data scan, edge assessment, shadow tracking, and report."""
    authority_mode = authority_mode.strip().lower()
    if authority_mode not in ("research", "shadow", "advisory", "conditional"):
        raise ValueError(f"unknown Polymarket authority mode: {authority_mode}")
    ts_ms = int(time.time() * 1000)
    markets = fetch_active_markets(limit=limit)
    if include_clob:
        markets = enrich_clob_snapshots(markets, max_markets=clob_limit)
    conn = get_connection(db_path)
    btc_prob, eth_prob = _resolve_probabilities(
        conn,
        symbol=symbol,
        btc_prob=btc_prob,
        eth_prob=eth_prob,
    )
    btc_long_probs = _estimate_btc_long_horizon_probs(
        conn,
        markets=markets,
        symbol=symbol,
        enabled=use_long_horizon_model and btc_long_prob is None,
    )
    opportunities, built_candidates, alerts = _build_candidates(
        markets=markets,
        btc_prob=btc_prob,
        eth_prob=eth_prob,
        btc_long_prob=btc_long_prob,
        eth_long_prob=eth_long_prob,
        btc_long_probs=btc_long_probs,
    )

    candidates: list[AlphaCandidate] = []
    shadow_opened = 0
    opened_shadow_ids: set[int] = set()
    shadow_positions: list[ShadowPositionView] = []
    try:
        promotion = evaluate_shadow_ledger(conn)
        allow_shadow = auto_shadow and _authority_allows_shadow(authority_mode, promotion.approved)
        insert_polymarket_opportunities(
            conn,
            symbol,
            ts_ms,
            [opp.to_dict() for opp in opportunities],
        )
        for candidate in built_candidates:
            opp = candidate.opportunity
            edge = candidate.edge
            assessment = candidate.assessment
            shadow_id = None
            if allow_shadow:
                shadow_id = _open_shadow_if_new(
                    conn,
                    symbol=symbol,
                    ts_ms=ts_ms,
                    opportunity=opp,
                    edge=edge,
                    assessment=assessment,
                    max_open_trades=max(0, int(max_open_shadow_trades)),
                    max_open_exposure_usd=max(0.0, float(max_open_shadow_exposure_usd)),
                )
                if shadow_id is not None:
                    shadow_opened += 1
                    opened_shadow_ids.add(shadow_id)
            candidates.append(AlphaCandidate(opp, edge, candidate.snapshot, assessment, shadow_id))
        shadow_ledger = _update_shadow_ledger(
            conn,
            markets=markets,
            ts_ms=ts_ms,
            skip_trade_ids=opened_shadow_ids,
        )
        promotion = evaluate_shadow_ledger(conn)
        shadow_positions = _load_shadow_position_views(conn, _market_price_index(markets))
    finally:
        conn.close()

    report_path = _write_report(
        report_dir=report_dir,
        ts_ms=ts_ms,
        candidates=candidates,
        arbitrage_alerts=alerts,
        shadow_ledger=shadow_ledger,
        shadow_positions=shadow_positions,
        promotion_metrics=promotion.metrics,
        authority_mode=authority_mode,
        btc_prob=btc_prob,
        eth_prob=eth_prob,
        promotion_approved=promotion.approved,
        promotion_reasons=promotion.reasons,
    )
    return AlphaRunResult(
        ts_ms=ts_ms,
        report_path=report_path,
        opportunities=opportunities,
        candidates=candidates,
        arbitrage_alerts=len(alerts),
        shadow_opened=shadow_opened,
        shadow_ledger=shadow_ledger,
        authority_mode=authority_mode,
        btc_prob=btc_prob,
        eth_prob=eth_prob,
        promotion_approved=promotion.approved,
        promotion_reasons=promotion.reasons,
    )


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the analysis-only Polymarket Alpha Lab",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--db", default="data/hogan.db")
    p.add_argument("--symbol", default="BTC/USD")
    p.add_argument("--limit", type=int, default=100)
    p.add_argument("--no-clob", action="store_true")
    p.add_argument("--clob-limit", type=int, default=12)
    p.add_argument("--btc-prob", type=float, default=None)
    p.add_argument("--eth-prob", type=float, default=None)
    p.add_argument("--btc-long-prob", type=float, default=None, help="Optional calibrated long-horizon BTC fair probability")
    p.add_argument("--eth-long-prob", type=float, default=None, help="Optional calibrated long-horizon ETH fair probability")
    p.add_argument("--no-long-horizon-model", action="store_true", help="Disable automatic BTC long-horizon fair-value estimates")
    p.add_argument("--authority-mode", choices=("research", "shadow", "advisory", "conditional"), default="shadow")
    p.add_argument("--max-open-shadow-trades", type=int, default=10)
    p.add_argument("--max-open-shadow-exposure", type=float, default=250.0)
    p.add_argument("--recommendations-only", action="store_true", help="Print machine recommendations without writing reports or shadow rows")
    p.add_argument("--recommendation-limit", type=int, default=10)
    p.add_argument("--no-auto-shadow", action="store_true")
    p.add_argument("--report-dir", default="reports/polymarket")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    if args.recommendations_only:
        result = run_recommendations_only(
            db_path=args.db,
            symbol=args.symbol,
            limit=args.limit,
            include_clob=not args.no_clob,
            clob_limit=args.clob_limit,
            btc_prob=args.btc_prob,
            eth_prob=args.eth_prob,
            btc_long_prob=args.btc_long_prob,
            eth_long_prob=args.eth_long_prob,
            use_long_horizon_model=not args.no_long_horizon_model,
        )
        print_recommendations(result, limit=args.recommendation_limit)
        return
    result = run_alpha_lab(
        db_path=args.db,
        symbol=args.symbol,
        limit=args.limit,
        include_clob=not args.no_clob,
        clob_limit=args.clob_limit,
        btc_prob=args.btc_prob,
        eth_prob=args.eth_prob,
        btc_long_prob=args.btc_long_prob,
        eth_long_prob=args.eth_long_prob,
        use_long_horizon_model=not args.no_long_horizon_model,
        auto_shadow=not args.no_auto_shadow,
        authority_mode=args.authority_mode,
        max_open_shadow_trades=args.max_open_shadow_trades,
        max_open_shadow_exposure_usd=args.max_open_shadow_exposure,
        report_dir=args.report_dir,
    )
    print(f"Report: {result.report_path}")
    print(f"Opportunities: {len(result.opportunities)}")
    print(f"Shadow trades opened: {result.shadow_opened}")
    print(f"Shadow trades closed: {result.shadow_ledger.closed}")
    print(f"Shadow unrealized PnL: {result.shadow_ledger.unrealized_pnl:.2f}")
    print(f"Hogan BTC probability: {result.btc_prob:.4f}" if result.btc_prob is not None else "Hogan BTC probability: n/a")
    print(f"Hogan ETH probability: {result.eth_prob:.4f}" if result.eth_prob is not None else "Hogan ETH probability: n/a")
    print(f"Arbitrage alerts: {result.arbitrage_alerts}")
    print(f"Promotion approved: {result.promotion_approved}")
    print(f"Authority mode: {result.authority_mode}")
    if result.promotion_reasons:
        print("Promotion blockers:", ", ".join(result.promotion_reasons))


if __name__ == "__main__":
    main()
