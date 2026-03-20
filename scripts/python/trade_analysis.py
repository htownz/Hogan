"""Quick analysis of paper trade history and signal behavior."""
import sqlite3
import sys
sys.path.insert(0, ".")

conn = sqlite3.connect("data/hogan.db")
cur = conn.cursor()

print("=== CLOSED TRADES BY SIDE ===")
cur.execute("""
    SELECT side, COUNT(*) as trades,
           ROUND(AVG(pnl_pct)*100,3) as avg_pnl_pct,
           ROUND(SUM(realized_pnl),4) as total_pnl,
           SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins
    FROM paper_trades WHERE exit_price IS NOT NULL GROUP BY side
""")
for r in cur.fetchall():
    print(f"  {r}")

print("\n=== OPEN POSITIONS ===")
cur.execute("""
    SELECT symbol, side, entry_price, qty, ml_up_prob, strategy_conf,
           datetime(open_ts_ms/1000, 'unixepoch') as opened
    FROM paper_trades WHERE exit_price IS NULL
""")
for r in cur.fetchall():
    print(f"  {r}")

print("\n=== LAST 15 TRADE ENTRIES ===")
cur.execute("""
    SELECT symbol, side, entry_price, exit_price, realized_pnl, close_reason,
           ml_up_prob, strategy_conf, vol_ratio,
           datetime(open_ts_ms/1000,'unixepoch') as opened
    FROM paper_trades ORDER BY open_ts_ms DESC LIMIT 15
""")
for r in cur.fetchall():
    print(f"  {r}")

print("\n=== EQUITY RANGE ===")
cur.execute("SELECT MIN(equity_usd), MAX(equity_usd), COUNT(*), MAX(equity_usd)-MIN(equity_usd) FROM equity_snapshots")
mn, mx, cnt, rng = cur.fetchone()
print(f"  min=${mn:.2f}  max=${mx:.2f}  snapshots={cnt}  range=${rng:.4f}")

print("\n=== SIGNAL CONF DISTRIBUTION (last 500 equity snaps ~4h) ===")
print("  (conf=0 means strategy produced no directional signal this tick)")

print("\n=== TRADES PER DAY ===")
cur.execute("""
    SELECT date(open_ts_ms/1000,'unixepoch') as day, COUNT(*) as trades
    FROM paper_trades GROUP BY day ORDER BY day DESC LIMIT 10
""")
for r in cur.fetchall():
    print(f"  {r}")

conn.close()
