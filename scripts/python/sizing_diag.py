"""Diagnose position sizing bug and trade frequency."""
import sqlite3
conn = sqlite3.connect("data/hogan.db")
cur = conn.cursor()

cur.execute("SELECT symbol, qty, entry_price, qty*entry_price, ml_up_prob FROM paper_trades WHERE exit_price IS NULL")
rows = cur.fetchall()
for r in rows:
    notional = r[3]
    ml_conf_scale = abs(r[4] - 0.5) * 2
    print(f"Open {r[0]}: qty={r[1]:.8f}  entry=${r[2]:.2f}  notional=${notional:.2f}")
    print(f"  ml_up_prob={r[4]:.4f}  => ml_confidence_scale={ml_conf_scale:.4f} ({ml_conf_scale*100:.1f}% of full size)")
    full_size = 5000 / r[2]
    print(f"  Expected full-size (50% of 10k equity): {full_size:.6f} BTC = ${full_size*r[2]:.2f}")
    print(f"  Actual / expected: {r[1]/full_size*100:.2f}%")

print()
cur.execute("SELECT COUNT(*) FROM equity_snapshots WHERE equity_usd < 9900")
print(f"Snapshots below $9900 (pre-fix losses): {cur.fetchone()[0]}")

print()
print("=== SIGNAL CONF STATS (all paper_trades) ===")
cur.execute("SELECT AVG(strategy_conf), MIN(strategy_conf), MAX(strategy_conf) FROM paper_trades")
avg, mn, mx = cur.fetchone()
print(f"  avg={avg}  min={mn}  max={mx}")

conn.close()
