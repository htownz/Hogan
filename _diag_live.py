"""Quick diagnostic: what is the bot actually doing right now?"""
import sqlite3
import time

conn = sqlite3.connect("data/hogan.db")
conn.row_factory = sqlite3.Row
now_ms = int(time.time() * 1000)

print("=" * 60)
print("TRADES (all time)")
print("=" * 60)
rows = conn.execute("SELECT * FROM paper_trades ORDER BY rowid DESC LIMIT 15").fetchall()
if not rows:
    print("  ** NO TRADES AT ALL **")
for r in rows:
    d = dict(r)
    print(f"  side={d.get('side','?')} entry={d.get('entry_price','?')} "
          f"exit={d.get('exit_price','?')} pnl={d.get('realized_pnl','?')} "
          f"reason={d.get('exit_reason','?')} ts={d.get('entry_ts','?')}")

print()
print("=" * 60)
print("DECISION LOG (last 30 entries)")
print("=" * 60)
rows = conn.execute("SELECT * FROM decision_log ORDER BY rowid DESC LIMIT 30").fetchall()
if not rows:
    print("  ** NO DECISION LOG ENTRIES **")
for r in rows:
    d = dict(r)
    ts = d.get("timestamp", 0)
    age_h = (now_ms - ts) / 3600000 if ts else -1
    blocked = d.get("blocked_by", "")
    print(f"  age={age_h:.1f}h action={d.get('action','?')} "
          f"tech={d.get('tech_action','?')}(conf={d.get('tech_confidence','?')}) "
          f"regime={d.get('regime','?')} blocked={blocked or 'none'}")

print()
print("=" * 60)
print("CANDLE FRESHNESS BY TIMEFRAME")
print("=" * 60)
for tf in ["1h", "5m", "15m", "30m", "3h", "4h"]:
    row = conn.execute(
        "SELECT MAX(timestamp) as mx, COUNT(*) as cnt FROM candles WHERE timeframe=?",
        (tf,),
    ).fetchone()
    mx = row["mx"]
    cnt = row["cnt"]
    if mx:
        age_h = (now_ms - mx) / 3600000
        print(f"  {tf:>4s}: {cnt:>6d} candles, newest age = {age_h:.1f}h ago")
    else:
        print(f"  {tf:>4s}: NO DATA")

print()
print("=" * 60)
print("BLOCKED-BY BREAKDOWN (last 200 decisions)")
print("=" * 60)
rows = conn.execute(
    "SELECT blocked_by, COUNT(*) as cnt FROM decision_log "
    "WHERE blocked_by IS NOT NULL AND blocked_by != '' "
    "GROUP BY blocked_by ORDER BY cnt DESC"
).fetchall()
if not rows:
    print("  No blocks recorded")
for r in rows:
    print(f"  {r['blocked_by']}: {r['cnt']} times")

print()
print("=" * 60)
print("ACTION DISTRIBUTION (last 200 decisions)")
print("=" * 60)
rows = conn.execute(
    "SELECT action, COUNT(*) as cnt FROM decision_log "
    "GROUP BY action ORDER BY cnt DESC"
).fetchall()
for r in rows:
    print(f"  {r['action']}: {r['cnt']} times")

print()
print("=" * 60)
print("TECH ACTION DISTRIBUTION (last 200 decisions)")
print("=" * 60)
rows = conn.execute(
    "SELECT tech_action, ROUND(AVG(tech_confidence),4) as avg_conf, COUNT(*) as cnt "
    "FROM decision_log GROUP BY tech_action ORDER BY cnt DESC"
).fetchall()
for r in rows:
    print(f"  {r['tech_action']}: {r['cnt']} times (avg_conf={r['avg_conf']})")

print()
print("=" * 60)
print("IS THE BOT RUNNING? (check last decision timestamp)")
print("=" * 60)
row = conn.execute("SELECT MAX(timestamp) as mx FROM decision_log").fetchone()
if row["mx"]:
    age_min = (now_ms - row["mx"]) / 60000
    print(f"  Last decision: {age_min:.1f} minutes ago")
    if age_min > 120:
        print("  ** WARNING: Bot appears INACTIVE (>2h since last decision) **")
    elif age_min > 70:
        print("  ** Bot is slow (>70min since last decision, expected ~60min for 1h candles) **")
    else:
        print("  Bot appears active.")
else:
    print("  ** No decisions recorded at all **")

conn.close()
