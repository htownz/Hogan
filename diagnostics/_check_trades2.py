"""Deeper diagnostic: pipeline signal vs final action, block reasons."""
import sqlite3
import json
from collections import Counter

conn = sqlite3.connect("data/hogan.db")
conn.row_factory = sqlite3.Row

# 1. tech_action vs final_action distribution
print("=== Pipeline Signal vs Final Action ===")
rows = conn.execute(
    "SELECT tech_action, final_action, COUNT(*) as cnt "
    "FROM decision_log GROUP BY tech_action, final_action ORDER BY cnt DESC"
).fetchall()
for r in rows:
    print(f"  tech={r['tech_action']:>6} -> final={r['final_action']:>6}: {r['cnt']}")

# 2. How many tech signals are generated?
print("\n=== Tech Action Distribution ===")
rows = conn.execute(
    "SELECT tech_action, COUNT(*) as cnt, AVG(tech_confidence) as avg_conf "
    "FROM decision_log GROUP BY tech_action ORDER BY cnt DESC"
).fetchall()
for r in rows:
    print(f"  {r['tech_action']:>6}: {r['cnt']:>6} (avg conf={r['avg_conf']:.3f})")

# 3. Block reasons for non-hold tech signals that became hold
print("\n=== Block Reasons (tech=buy/sell -> final=hold) ===")
blocked = conn.execute(
    "SELECT block_reasons_json, tech_action, tech_confidence, ml_up_prob "
    "FROM decision_log "
    "WHERE tech_action IN ('buy','sell') AND final_action = 'hold' "
    "ORDER BY rowid DESC LIMIT 50"
).fetchall()
reason_counter = Counter()
for r in blocked:
    reasons = json.loads(r["block_reasons_json"]) if r["block_reasons_json"] else []
    for reason in reasons:
        reason_counter[reason] += 1
    if not reasons:
        reason_counter["(no block reasons logged)"] += 1

print(f"  Total blocked signals (last 50 of {len(blocked)}):")
for reason, cnt in reason_counter.most_common(20):
    print(f"    {reason}: {cnt}")

# 4. The 13 sell signals that got through
print("\n=== Successful Signals (final != hold) ===")
success = conn.execute(
    "SELECT final_action, final_confidence, tech_action, tech_confidence, "
    "ml_up_prob, conf_scale, position_size, regime, block_reasons_json "
    "FROM decision_log WHERE final_action != 'hold' ORDER BY rowid DESC LIMIT 20"
).fetchall()
for r in success:
    print(f"  final={r['final_action']} conf={r['final_confidence']:.3f} "
          f"tech={r['tech_action']} tconf={r['tech_confidence']:.3f} "
          f"ml_prob={r['ml_up_prob']} scale={r['conf_scale']} "
          f"size={r['position_size']} regime={r['regime']}")

# 5. Regime distribution in decision_log
print("\n=== Regime Distribution ===")
regimes = conn.execute(
    "SELECT regime, COUNT(*) as cnt FROM decision_log GROUP BY regime ORDER BY cnt DESC"
).fetchall()
for r in regimes:
    print(f"  {r['regime']}: {r['cnt']}")

# 6. ML probabilities
print("\n=== ML Probability Stats ===")
ml = conn.execute(
    "SELECT AVG(ml_up_prob), MIN(ml_up_prob), MAX(ml_up_prob), COUNT(*) "
    "FROM decision_log WHERE ml_up_prob IS NOT NULL"
).fetchone()
print(f"  n={ml[3]} avg={ml[0]:.4f}" if ml[3] else "  No ML probs recorded")
if ml[3]:
    print(f"  min={ml[1]:.4f} max={ml[2]:.4f}")

# 7. Confidence distribution for tech signals
print("\n=== Tech Confidence for buy/sell signals ===")
tc = conn.execute(
    "SELECT tech_action, AVG(tech_confidence) as avg, MIN(tech_confidence) as mn, "
    "MAX(tech_confidence) as mx, COUNT(*) as cnt "
    "FROM decision_log WHERE tech_action IN ('buy','sell') "
    "GROUP BY tech_action"
).fetchall()
for r in tc:
    print(f"  {r['tech_action']}: n={r['cnt']} avg={r['avg']:.3f} min={r['mn']:.3f} max={r['mx']:.3f}")

# 8. Check conf_scale and position_size
print("\n=== Conf Scale for non-hold entries ===")
cs = conn.execute(
    "SELECT AVG(conf_scale), MIN(conf_scale), MAX(conf_scale), COUNT(*) "
    "FROM decision_log WHERE final_action != 'hold' AND conf_scale IS NOT NULL"
).fetchone()
if cs[3]:
    print(f"  n={cs[3]} avg={cs[0]:.4f} min={cs[1]:.4f} max={cs[2]:.4f}")
else:
    print("  No non-hold entries with conf_scale")

# 9. Position size for signals that passed
print("\n=== Position Size Stats (final != hold) ===")
ps = conn.execute(
    "SELECT AVG(position_size), MIN(position_size), MAX(position_size), COUNT(*) "
    "FROM decision_log WHERE final_action != 'hold' AND position_size IS NOT NULL"
).fetchone()
if ps[3]:
    print(f"  n={ps[3]} avg={ps[0]:.6f} min={ps[1]:.6f} max={ps[2]:.6f}")

# 10. Check if there are entries where swarm_decision_id is populated
print("\n=== Policy Core / Swarm Integration ===")
try:
    swarm_linked = conn.execute(
        "SELECT COUNT(*) FROM decision_log WHERE swarm_decision_id IS NOT NULL"
    ).fetchone()[0]
    print(f"  Decision log entries linked to swarm: {swarm_linked}")
except Exception:
    print("  swarm_decision_id column not present")

# 11. Time range of data
print("\n=== Data Time Range ===")
tr = conn.execute(
    "SELECT MIN(ts_ms), MAX(ts_ms), COUNT(*) FROM decision_log"
).fetchone()
if tr[2]:
    from datetime import datetime, timezone
    t0 = datetime.fromtimestamp(tr[0]/1000, tz=timezone.utc)
    t1 = datetime.fromtimestamp(tr[1]/1000, tz=timezone.utc)
    hours = (tr[1] - tr[0]) / 1000 / 3600
    print(f"  From: {t0}")
    print(f"  To:   {t1}")
    print(f"  Span: {hours:.1f} hours ({hours/24:.1f} days)")
    print(f"  Decisions per hour: {tr[2] / hours:.1f}")

conn.close()
