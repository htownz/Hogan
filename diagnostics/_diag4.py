"""What's killing the buy signals? And why aren't sells becoming trades?"""
import sqlite3
import json
from collections import Counter

conn = sqlite3.connect("data/hogan.db")
conn.row_factory = sqlite3.Row

# 1. Buy signals that became hold
print("=== Buy signals killed (tech=buy -> final=hold) ===")
blocked_buys = conn.execute(
    "SELECT tech_confidence, ml_up_prob, conf_scale, position_size, regime, "
    "block_reasons_json, pipeline_action, final_confidence "
    "FROM decision_log WHERE tech_action = 'buy' AND final_action = 'hold' "
    "ORDER BY id DESC LIMIT 50"
).fetchall()
print(f"Total: {len(blocked_buys)}")
reason_counter = Counter()
for r in blocked_buys:
    reasons = json.loads(r["block_reasons_json"]) if r["block_reasons_json"] else []
    for reason in reasons:
        reason_counter[reason] += 1
    if not reasons:
        reason_counter["(no block reasons logged)"] += 1
print("Block reasons:")
for reason, cnt in reason_counter.most_common(20):
    print(f"  {reason}: {cnt}")
print("\nSample blocked buys:")
for r in list(blocked_buys)[:5]:
    print(f"  pipe={r['pipeline_action']} tconf={r['tech_confidence']:.3f} "
          f"ml={r['ml_up_prob']} scale={r['conf_scale']} "
          f"size={r['position_size']} regime={r['regime']} "
          f"fconf={r['final_confidence']} reasons={r['block_reasons_json']}")

# 2. Buy signals that survived (should be 0)
print("\n=== Buy signals that survived (tech=buy -> final=buy) ===")
survived_buys = conn.execute(
    "SELECT COUNT(*) FROM decision_log WHERE tech_action = 'buy' AND final_action = 'buy'"
).fetchone()[0]
print(f"Count: {survived_buys}")

# 3. Pipeline action for buy tech signals
print("\n=== Pipeline action for tech=buy signals ===")
pipe_for_buys = conn.execute(
    "SELECT pipeline_action, COUNT(*) as cnt FROM decision_log "
    "WHERE tech_action = 'buy' GROUP BY pipeline_action ORDER BY cnt DESC"
).fetchall()
for r in pipe_for_buys:
    print(f"  pipeline_action={r['pipeline_action']}: {r['cnt']}")

# 4. Sells that made it to final_action
print("\n=== Sell signals that survived (final=sell) ===")
sells = conn.execute(
    "SELECT tech_action, pipeline_action, tech_confidence, final_confidence, "
    "ml_up_prob, conf_scale, position_size, regime, block_reasons_json "
    "FROM decision_log WHERE final_action = 'sell' ORDER BY id DESC LIMIT 15"
).fetchall()
for r in sells:
    print(f"  tech={r['tech_action']} pipe={r['pipeline_action']} "
          f"tconf={r['tech_confidence']:.3f} fconf={r['final_confidence']:.4f} "
          f"ml={r['ml_up_prob']} scale={r['conf_scale']} "
          f"size={r['position_size']} regime={r['regime']}")

# 5. Paper trades - any at all?
pt = conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
print(f"\nPaper trades: {pt}")

# 6. Check what the conf_scale and position_size are for the sells
print("\n=== Sell position sizes ===")
sell_sizes = conn.execute(
    "SELECT AVG(position_size), MIN(position_size), MAX(position_size) "
    "FROM decision_log WHERE final_action = 'sell' AND position_size IS NOT NULL"
).fetchone()
if sell_sizes[0] is not None:
    print(f"  avg={sell_sizes[0]:.6f} min={sell_sizes[1]:.6f} max={sell_sizes[2]:.6f}")

# 7. Check what regime the buy signals come from
print("\n=== Regime distribution for tech=buy ===")
buy_regimes = conn.execute(
    "SELECT regime, COUNT(*) as cnt FROM decision_log "
    "WHERE tech_action = 'buy' GROUP BY regime ORDER BY cnt DESC"
).fetchall()
for r in buy_regimes:
    print(f"  {r['regime']}: {r['cnt']}")

# 8. Check min_final_confidence vs actual final_confidence for blocked buys
print("\n=== Final confidence for blocked buys ===")
fc = conn.execute(
    "SELECT AVG(final_confidence), MIN(final_confidence), MAX(final_confidence) "
    "FROM decision_log WHERE tech_action = 'buy' AND final_action = 'hold'"
).fetchone()
if fc[0] is not None:
    print(f"  avg={fc[0]:.4f} min={fc[1]:.4f} max={fc[2]:.4f}")

# 9. Check entry in swarm table for these periods
print("\n=== Swarm decisions breakdown ===")
sd = conn.execute(
    "SELECT final_action, vetoed, mode, COUNT(*) as cnt "
    "FROM swarm_decisions GROUP BY final_action, vetoed, mode ORDER BY cnt DESC"
).fetchall()
for r in sd:
    print(f"  action={r['final_action']} vetoed={r['vetoed']} mode={r['mode']}: {r['cnt']}")

# 10. What is the actual config at runtime?
print("\n=== Key .env values ===")
try:
    from dotenv import dotenv_values
    env = dotenv_values(".env")
    keys = [
        "HOGAN_USE_ML_FILTER", "HOGAN_ML_AS_SIZER", "HOGAN_ML_CONFIDENCE_SIZING",
        "HOGAN_ENABLE_SHORTS", "HOGAN_CHAMPION_MODE", "HOGAN_USE_POLICY_CORE",
        "HOGAN_SWARM_MODE", "HOGAN_MIN_FINAL_CONFIDENCE", "HOGAN_MACRO_SITOUT",
        "HOGAN_USE_REGIME_DETECTION", "HOGAN_TRAILING_STOP_PCT",
        "HOGAN_TAKE_PROFIT_PCT", "HOGAN_ML_BUY_THRESHOLD",
    ]
    for k in keys:
        print(f"  {k}={env.get(k, '(not set)')}")
except Exception as e:
    print(f"  Error: {e}")

conn.close()
