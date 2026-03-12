"""Close all zombie open positions so the pipeline can trade fresh."""
import sqlite3
import time

DB = "data/hogan.db"
conn = sqlite3.connect(DB)
conn.row_factory = sqlite3.Row

now_ms = int(time.time() * 1000)

# Get all open positions
open_trades = conn.execute(
    "SELECT trade_id, symbol, side, entry_price, qty, open_ts_ms "
    "FROM paper_trades WHERE exit_price IS NULL"
).fetchall()

print(f"Found {len(open_trades)} open zombie positions")

for t in open_trades:
    d = dict(t)
    print(f"  Closing: {d['symbol']} {d['side']} entry=${d['entry_price']:.2f} qty={d['qty']:.6f}")
    conn.execute(
        "UPDATE paper_trades SET exit_price=?, realized_pnl=0, pnl_pct=0, "
        "close_ts_ms=?, close_reason='cleanup_stale' "
        "WHERE trade_id=? AND exit_price IS NULL",
        (d["entry_price"], now_ms, d["trade_id"])
    )

conn.commit()

# Verify
remaining = conn.execute(
    "SELECT COUNT(*) FROM paper_trades WHERE exit_price IS NULL"
).fetchone()[0]
print(f"\nRemaining open positions: {remaining}")
print("Done. Restart the bot for a clean slate.")
conn.close()
