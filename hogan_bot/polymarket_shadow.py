"""Shadow-trading helpers for Polymarket analysis.

This module only records hypothetical positions. It has no authenticated API
calls and cannot place real Polymarket orders.
"""
from __future__ import annotations

import argparse
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


def load_shadow_ledger(conn, *, status: str = "all", limit: int = 20) -> list[dict]:
    """Load recent Polymarket shadow trades for operator inspection."""
    clauses: list[str] = []
    params: list[object] = []
    if status != "all":
        clauses.append("status = ?")
        params.append(status)
    where = f"WHERE {' AND '.join(clauses)}" if clauses else ""
    params.append(int(limit))
    rows = conn.execute(
        f"""
        SELECT id, opened_ts_ms, closed_ts_ms, symbol, market_id, slug, side,
               entry_prob, exit_prob, size_usd, status, realized_pnl, rationale
        FROM polymarket_shadow_trades
        {where}
        ORDER BY opened_ts_ms DESC
        LIMIT ?
        """,
        params,
    ).fetchall()
    columns = [
        "id",
        "opened_ts_ms",
        "closed_ts_ms",
        "symbol",
        "market_id",
        "slug",
        "side",
        "entry_prob",
        "exit_prob",
        "size_usd",
        "status",
        "realized_pnl",
        "rationale",
    ]
    return [dict(zip(columns, row)) for row in rows]


def _fmt_prob(value) -> str:
    return "n/a" if value is None else f"{float(value):.4f}"


def _fmt_money(value) -> str:
    return "n/a" if value is None else f"${float(value):.2f}"


def print_shadow_ledger(conn, *, status: str = "all", limit: int = 20) -> None:
    """Print compact shadow ledger rows and promotion summary."""
    summary = promotion_snapshot(conn)
    print(
        "Summary: "
        f"closed={summary['trades']:.0f} "
        f"total_pnl={_fmt_money(summary['total_pnl'])} "
        f"avg_pnl={_fmt_money(summary['avg_pnl'])} "
        f"win_rate={summary['win_rate']:.2%}"
    )
    rows = load_shadow_ledger(conn, status=status, limit=limit)
    if not rows:
        print("No Polymarket shadow trades found.")
        return
    for row in rows:
        print(
            f"#{row['id']} {row['status']} {row['side']} "
            f"entry={_fmt_prob(row['entry_prob'])} exit={_fmt_prob(row['exit_prob'])} "
            f"size={_fmt_money(row['size_usd'])} pnl={_fmt_money(row['realized_pnl'])} "
            f"{row['slug'] or row['market_id']}"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Inspect analysis-only Polymarket shadow trades",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--db", default="data/hogan.db")
    parser.add_argument("--status", choices=("all", "open", "closed"), default="all")
    parser.add_argument("--limit", type=int, default=20)
    return parser.parse_args()


def main() -> None:
    from hogan_bot.storage import get_connection

    args = parse_args()
    conn = get_connection(args.db)
    try:
        print_shadow_ledger(conn, status=args.status, limit=args.limit)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
