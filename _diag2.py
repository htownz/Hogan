"""Deeper diagnostic — schema + bot restart status."""
import sqlite3
import time

conn = sqlite3.connect("data/hogan.db")
conn.row_factory = sqlite3.Row
now_ms = int(time.time() * 1000)

# Decision log schema
print("=== DECISION_LOG COLUMNS ===")
row = conn.execute("SELECT * FROM decision_log LIMIT 1").fetchone()
if row:
    cols = row.keys()
    print(f"  {cols}")
    d = dict(row)
    for k, v in d.items():
        print(f"  {k} = {repr(v)[:120]}")

# Candles schema
print("\n=== CANDLES COLUMNS ===")
row = conn.execute("SELECT * FROM candles LIMIT 1").fetchone()
if row:
    cols = row.keys()
    print(f"  {cols}")
    # Find the timestamp column name
    ts_col = None
    for c in cols:
        if "time" in c.lower() or "ts" in c.lower():
            ts_col = c
            break
    print(f"  Timestamp column: {ts_col}")

    # Freshness per timeframe
    print("\n=== CANDLE FRESHNESS ===")
    for tf in ["1h", "5m", "15m", "30m", "3h", "4h"]:
        try:
            r = conn.execute(
                f"SELECT MAX({ts_col}) as mx, COUNT(*) as cnt FROM candles WHERE timeframe=?",
                (tf,),
            ).fetchone()
            mx = r["mx"]
            cnt = r["cnt"]
            if mx:
                age_h = (now_ms - mx) / 3600000
                print(f"  {tf:>4s}: {cnt:>6d} candles, age = {age_h:.1f}h")
            else:
                print(f"  {tf:>4s}: NO DATA")
        except Exception as e:
            print(f"  {tf:>4s}: ERROR {e}")

# Recent decisions with ALL columns
print("\n=== LAST 5 DECISIONS (full) ===")
rows = conn.execute("SELECT * FROM decision_log ORDER BY rowid DESC LIMIT 5").fetchall()
for i, r in enumerate(rows):
    d = dict(r)
    print(f"\n  --- Decision {i+1} ---")
    for k, v in d.items():
        if v is not None and v != "" and v != 0:
            print(f"    {k} = {repr(v)[:200]}")

# Check if the most recent decision has our new signal types
print("\n=== CHECKING FOR NEW SIGNAL TYPES ===")
rows = conn.execute(
    "SELECT tech_action, tech_confidence, rowid FROM decision_log "
    "WHERE tech_action != 'hold' ORDER BY rowid DESC LIMIT 10"
).fetchall()
if not rows:
    print("  ** No non-hold tech signals found at all **")
else:
    for r in rows:
        print(f"  rowid={r['rowid']} tech={r['tech_action']} conf={r['tech_confidence']}")

# Total decision count
total = conn.execute("SELECT COUNT(*) FROM decision_log").fetchone()[0]
holds = conn.execute("SELECT COUNT(*) FROM decision_log WHERE tech_action='hold'").fetchone()[0]
print(f"\n=== SUMMARY ===")
print(f"  Total decisions: {total}")
print(f"  Tech=hold: {holds} ({100*holds/max(total,1):.1f}%)")
print(f"  Tech!=hold: {total-holds} ({100*(total-holds)/max(total,1):.1f}%)")

conn.close()
