"""Deep diagnostic: why is TechnicalAgent always returning hold?"""
import sqlite3

conn = sqlite3.connect("data/hogan.db")
conn.row_factory = sqlite3.Row

# 1. Any non-hold entries?
non_hold_tech = conn.execute(
    "SELECT tech_action, COUNT(*) as cnt FROM decision_log "
    "WHERE tech_action != 'hold' GROUP BY tech_action"
).fetchall()
print("Non-hold tech_action:", [(r["tech_action"], r["cnt"]) for r in non_hold_tech] or "NONE")

non_hold_final = conn.execute(
    "SELECT final_action, COUNT(*) as cnt FROM decision_log "
    "WHERE final_action != 'hold' GROUP BY final_action"
).fetchall()
print("Non-hold final_action:", [(r["final_action"], r["cnt"]) for r in non_hold_final] or "NONE")

# 2. Regime distribution
regimes = conn.execute(
    "SELECT regime, COUNT(*) as cnt FROM decision_log GROUP BY regime ORDER BY cnt DESC"
).fetchall()
print("\nRegime distribution:")
for r in regimes:
    print(f"  {r['regime']}: {r['cnt']}")

# 3. ML probs
ml = conn.execute(
    "SELECT AVG(ml_up_prob), MIN(ml_up_prob), MAX(ml_up_prob), COUNT(*) "
    "FROM decision_log WHERE ml_up_prob IS NOT NULL"
).fetchone()
if ml[3]:
    print(f"\nML probs: n={ml[3]}, avg={ml[0]:.4f}, min={ml[1]:.4f}, max={ml[2]:.4f}")
else:
    print("\nNo ML probs recorded")

# 4. Confidence stats
tc = conn.execute(
    "SELECT AVG(tech_confidence), MIN(tech_confidence), MAX(tech_confidence) "
    "FROM decision_log WHERE tech_confidence IS NOT NULL"
).fetchone()
if tc[0] is not None:
    print(f"Tech confidence: avg={tc[0]:.4f} min={tc[1]:.4f} max={tc[2]:.4f}")
else:
    print("No tech confidence data")

# 5. Sent/macro
sent = conn.execute(
    "SELECT AVG(sent_bias), AVG(sent_strength) FROM decision_log "
    "WHERE sent_bias IS NOT NULL"
).fetchone()
if sent[0] is not None:
    print(f"Sentiment: avg_bias={sent[0]:.4f} avg_strength={sent[1]:.4f}")

macro = conn.execute(
    "SELECT AVG(macro_score) FROM decision_log WHERE macro_score IS NOT NULL"
).fetchone()
if macro[0] is not None:
    print(f"Macro: avg_score={macro[0]:.4f}")

# 6. Detailed look at last 3 entries
cols = [r[1] for r in conn.execute("PRAGMA table_info(decision_log)").fetchall()]
rows = conn.execute("SELECT * FROM decision_log ORDER BY id DESC LIMIT 3").fetchall()
print(f"\nLast 3 entries (all columns):")
for r in rows:
    print("---")
    for c in cols:
        v = r[c]
        if v is not None:
            print(f"  {c}: {v}")

# 7. Pipeline_action distribution
pipe = conn.execute(
    "SELECT pipeline_action, COUNT(*) as cnt FROM decision_log "
    "GROUP BY pipeline_action ORDER BY cnt DESC"
).fetchall()
print(f"\nPipeline action distribution:")
for r in pipe:
    print(f"  {r['pipeline_action']}: {r['cnt']}")

# 8. Time range - are we getting new entries?
from datetime import datetime, timezone
tr = conn.execute(
    "SELECT MIN(ts_ms), MAX(ts_ms), COUNT(*) FROM decision_log"
).fetchone()
if tr[2]:
    t0 = datetime.fromtimestamp(tr[0]/1000, tz=timezone.utc)
    t1 = datetime.fromtimestamp(tr[1]/1000, tz=timezone.utc)
    hours = (tr[1] - tr[0]) / 1000 / 3600
    print(f"\nTime range: {t0} -> {t1}")
    print(f"Span: {hours:.1f}h ({hours/24:.1f}d), {tr[2]/hours:.1f} decisions/hour")

# 9. Entries in the last hour
import time
one_hour_ago = (time.time() - 3600) * 1000
recent = conn.execute(
    "SELECT COUNT(*) FROM decision_log WHERE ts_ms > ?", (one_hour_ago,)
).fetchone()[0]
print(f"Entries in last hour: {recent}")

# 10. Check candles
candle_count = conn.execute(
    "SELECT COUNT(*) FROM candles WHERE symbol='BTC/USD' AND timeframe='1h'"
).fetchone()[0]
latest_candle = conn.execute(
    "SELECT ts_ms FROM candles WHERE symbol='BTC/USD' AND timeframe='1h' ORDER BY ts_ms DESC LIMIT 1"
).fetchone()
if latest_candle:
    lc_time = datetime.fromtimestamp(latest_candle[0]/1000, tz=timezone.utc)
    print(f"\nCandles: {candle_count} total, latest: {lc_time}")

conn.close()
