"""Shadow-trading helpers for Polymarket analysis.

This module only records hypothetical positions. It has no authenticated API
calls and cannot place real Polymarket orders.
"""
from __future__ import annotations

import time


def open_shadow_from_opportunity(
    conn,
    opportunity,
    *,
    symbol: str = "BTC/USD",
    size_usd: float = 10.0,
) -> int:
    """Open a hypothetical trade from a ranked opportunity object."""
    from hogan_bot.storage import open_polymarket_shadow_trade

    return open_polymarket_shadow_trade(
        conn,
        opened_ts_ms=int(time.time() * 1000),
        symbol=symbol,
        market_id=opportunity.market_id,
        slug=opportunity.slug,
        side=opportunity.candidate_side,
        entry_prob=opportunity.crowd_prob,
        size_usd=size_usd,
        rationale=opportunity.rationale,
        raw=opportunity.to_dict(),
    )


def close_shadow(conn, trade_id: int, exit_prob: float) -> float:
    """Close a hypothetical trade at the latest observed probability."""
    from hogan_bot.storage import close_polymarket_shadow_trade

    return close_polymarket_shadow_trade(
        conn,
        trade_id,
        closed_ts_ms=int(time.time() * 1000),
        exit_prob=exit_prob,
    )


def promotion_snapshot(conn) -> dict[str, float]:
    """Return compact shadow-trading promotion metrics."""
    from hogan_bot.storage import summarize_polymarket_shadow_trades

    return summarize_polymarket_shadow_trades(conn)
