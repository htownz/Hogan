"""Close all stale (unclosed) paper trade journal entries from past sessions."""
import sqlite3
import time
import os

DB = os.path.join(os.path.dirname(__file__), "..", "data", "hogan.db")
conn = sqlite3.connect(DB)

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
    raise SystemExit(0)

now_ms = int(time.time() * 1000)
conn.execute("""
    UPDATE paper_trades
    SET close_ts_ms   = ?,
        exit_price    = entry_price,
        exit_fee      = 0,
        realized_pnl  = 0.0,
        pnl_pct       = 0.0,
        close_reason  = 'stale_cleanup'
    WHERE close_ts_ms IS NULL
""", (now_ms,))
conn.commit()

remaining = conn.execute(
    "SELECT COUNT(*) FROM paper_trades WHERE close_ts_ms IS NULL"
).fetchone()[0]
print(f"\nDone. Open positions remaining: {remaining}")
conn.close()
