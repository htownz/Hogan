"""Close all stale (unclosed) paper trade journal entries from past sessions.

Uses the proper close_paper_trade() function instead of a raw SQL UPDATE,
so ML labeling and decision-outcome linking are preserved.

Usage:
    python scripts/python/cleanup_stale_trades.py
    python scripts/python/cleanup_stale_trades.py --exclude-from-analytics
"""
import argparse
import os
import sys
import time

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

DB = os.path.join(PROJECT_ROOT, "data", "hogan.db")

from hogan_bot.storage import get_connection, close_paper_trade  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description="Clean up stale open paper trades.")
    parser.add_argument(
        "--exclude-from-analytics",
        action="store_true",
        help="Mark these closes with 'stale_cleanup_excluded' so analytics can filter them.",
    )
    parser.add_argument("--db", default=DB, help="Path to hogan.db")
    args = parser.parse_args()

    conn = get_connection(args.db)

    stale = conn.execute("""
        SELECT trade_id, symbol, side, entry_price, qty,
               datetime(open_ts_ms/1000, 'unixepoch') as opened
        FROM paper_trades WHERE close_ts_ms IS NULL
    """).fetchall()

    print(f"Found {len(stale)} stale open positions:")
    for r in stale:
        print(f"  [{r[5]}] {r[1]} {r[2]} entry={r[3]:.2f} qty={r[4]:.5f}")

    if not stale:
        print("Nothing to clean up.")
        conn.close()
        return

    reason = "stale_cleanup_excluded" if args.exclude_from_analytics else "stale_cleanup"
    now_ms = int(time.time() * 1000)
    closed_count = 0

    for trade_id, symbol, side, entry_price, qty, _ in stale:
        norm_side = "long" if side in ("buy", "long") else "short"
        result = close_paper_trade(
            conn,
            symbol=symbol,
            side=norm_side,
            exit_price=entry_price,
            exit_fee=0.0,
            close_ts_ms=now_ms,
            close_reason=reason,
        )
        if result is not None:
            closed_count += 1
        else:
            conn.execute(
                """UPDATE paper_trades
                   SET close_ts_ms=?, exit_price=entry_price, exit_fee=0,
                       realized_pnl=0.0, pnl_pct=0.0, close_reason=?
                   WHERE trade_id=?""",
                (now_ms, reason, trade_id),
            )
            conn.commit()
            closed_count += 1

    remaining = conn.execute(
        "SELECT COUNT(*) FROM paper_trades WHERE close_ts_ms IS NULL"
    ).fetchone()[0]
    print(f"\nClosed {closed_count} stale trades (reason='{reason}'). Open positions remaining: {remaining}")
    conn.close()


if __name__ == "__main__":
    main()
