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
    PolymarketOpportunity,
    _market_id,
    _yes_probability,
    enrich_clob_snapshots,
    fetch_active_markets,
    score_polymarket_opportunities,
)
from hogan_bot.polymarket_arbitrage import detect_arbitrage_alerts
from hogan_bot.polymarket_edge import EdgeAssessment, assess_opportunity_edge
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
    shadow_trade_id: int | None = None


@dataclass(frozen=True)
class ShadowLedgerUpdate:
    marked_open: int
    closed: int
    open_count: int
    open_exposure_usd: float
    unrealized_pnl: float


@dataclass(frozen=True)
class AlphaRunResult:
    ts_ms: int
    report_path: str
    opportunities: list[PolymarketOpportunity]
    candidates: list[AlphaCandidate]
    arbitrage_alerts: int
    shadow_opened: int
    shadow_ledger: ShadowLedgerUpdate
    promotion_approved: bool
    promotion_reasons: list[str]


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
) -> int | None:
    if edge.decision != "shadow_trade":
        return None
    if opportunity.candidate_side not in ("buy_yes", "buy_no"):
        return None
    if _has_open_shadow(conn, opportunity.market_id, opportunity.candidate_side):
        return None
    size_usd = max(1.0, min(25.0, edge.max_size_usd))
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
            "edge": {
                "after_cost_ev": edge.after_cost_ev,
                "expected_value": edge.expected_value,
                "decision": edge.decision,
                "reject_reasons": edge.reject_reasons,
            },
        },
    )


def _position_delta(side: str, entry_yes_prob: float, current_yes_prob: float) -> float:
    delta = current_yes_prob - entry_yes_prob
    if side == "buy_no":
        return -delta
    return delta


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


def _write_report(
    *,
    report_dir: str,
    ts_ms: int,
    candidates: list[AlphaCandidate],
    arbitrage_alert_count: int,
    shadow_ledger: ShadowLedgerUpdate,
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
        f"- Arbitrage alerts: `{arbitrage_alert_count}`",
        f"- Shadow marked open: `{shadow_ledger.marked_open}`",
        f"- Shadow closed: `{shadow_ledger.closed}`",
        f"- Shadow open exposure: `${shadow_ledger.open_exposure_usd:.2f}`",
        f"- Shadow unrealized PnL: `${shadow_ledger.unrealized_pnl:.2f}`",
        f"- Promotion approved: `{promotion_approved}`",
    ]
    if promotion_reasons:
        lines.append(f"- Promotion blockers: `{', '.join(promotion_reasons)}`")
    lines.extend(["", "## Top Candidates", ""])
    for idx, candidate in enumerate(candidates[:10], start=1):
        opp = candidate.opportunity
        edge = candidate.edge
        shadow = f" shadow_id={candidate.shadow_trade_id}" if candidate.shadow_trade_id else ""
        lines.extend([
            f"### {idx}. {opp.question}",
            f"- Side: `{opp.candidate_side}`{shadow}",
            f"- Decision: `{edge.decision}`",
            f"- Total score: `{opp.total_score:.4f}`",
            f"- After-cost EV: `{edge.after_cost_ev:.4f}`",
            f"- Crowd probability: `{opp.crowd_prob:.4f}`",
            f"- Hogan probability: `{opp.hogan_prob:.4f}`" if opp.hogan_prob is not None else "- Hogan probability: `n/a`",
            f"- Rationale: {opp.rationale}",
        ])
        if edge.reject_reasons:
            lines.append(f"- Reject reasons: `{', '.join(edge.reject_reasons)}`")
        lines.append("")
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
    auto_shadow: bool = True,
    report_dir: str = "reports/polymarket",
) -> AlphaRunResult:
    """Run public-data scan, edge assessment, shadow tracking, and report."""
    ts_ms = int(time.time() * 1000)
    markets = fetch_active_markets(limit=limit)
    if include_clob:
        markets = enrich_clob_snapshots(markets, max_markets=clob_limit)
    opportunities = score_polymarket_opportunities(
        markets,
        hogan_btc_bull_prob=btc_prob,
        hogan_eth_bull_prob=eth_prob,
        limit=25,
    )
    alerts = detect_arbitrage_alerts(markets)

    conn = get_connection(db_path)
    candidates: list[AlphaCandidate] = []
    shadow_opened = 0
    opened_shadow_ids: set[int] = set()
    try:
        insert_polymarket_opportunities(
            conn,
            symbol,
            ts_ms,
            [opp.to_dict() for opp in opportunities],
        )
        for opp in opportunities:
            edge = assess_opportunity_edge(opp)
            shadow_id = None
            if auto_shadow:
                shadow_id = _open_shadow_if_new(
                    conn,
                    symbol=symbol,
                    ts_ms=ts_ms,
                    opportunity=opp,
                    edge=edge,
                )
                if shadow_id is not None:
                    shadow_opened += 1
                    opened_shadow_ids.add(shadow_id)
            candidates.append(AlphaCandidate(opp, edge, shadow_id))
        shadow_ledger = _update_shadow_ledger(
            conn,
            markets=markets,
            ts_ms=ts_ms,
            skip_trade_ids=opened_shadow_ids,
        )
        promotion = evaluate_shadow_ledger(conn)
    finally:
        conn.close()

    report_path = _write_report(
        report_dir=report_dir,
        ts_ms=ts_ms,
        candidates=candidates,
        arbitrage_alert_count=len(alerts),
        shadow_ledger=shadow_ledger,
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
    p.add_argument("--no-auto-shadow", action="store_true")
    p.add_argument("--report-dir", default="reports/polymarket")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    result = run_alpha_lab(
        db_path=args.db,
        symbol=args.symbol,
        limit=args.limit,
        include_clob=not args.no_clob,
        clob_limit=args.clob_limit,
        btc_prob=args.btc_prob,
        eth_prob=args.eth_prob,
        auto_shadow=not args.no_auto_shadow,
        report_dir=args.report_dir,
    )
    print(f"Report: {result.report_path}")
    print(f"Opportunities: {len(result.opportunities)}")
    print(f"Shadow trades opened: {result.shadow_opened}")
    print(f"Shadow trades closed: {result.shadow_ledger.closed}")
    print(f"Shadow unrealized PnL: {result.shadow_ledger.unrealized_pnl:.2f}")
    print(f"Arbitrage alerts: {result.arbitrage_alerts}")
    print(f"Promotion approved: {result.promotion_approved}")
    if result.promotion_reasons:
        print("Promotion blockers:", ", ".join(result.promotion_reasons))


if __name__ == "__main__":
    main()
