import sqlite3

c = sqlite3.connect("data/hogan.db")
c.row_factory = sqlite3.Row

# Where did the 52 closed trades come from? Check if they have entry_decision_id
trades = c.execute("""
    SELECT trade_id, symbol, side, entry_price, exit_price, realized_pnl,
           entry_decision_id, open_ts_ms, close_ts_ms, close_reason
    FROM paper_trades WHERE exit_price IS NOT NULL
    ORDER BY close_ts_ms DESC LIMIT 10
""").fetchall()
print("=== LAST 10 CLOSED TRADES ===")
for t in trades:
    d = dict(t)
    print(f"  {d['symbol']} {d['side']} pnl={d['realized_pnl']:.2f} reason={d['close_reason']} dec_id={d['entry_decision_id']}")

# Check if any decisions ever had action != hold
print("\n=== DISTINCT ACTIONS IN DECISION LOG ===")
acts = c.execute("SELECT DISTINCT final_action, COUNT(*) as n FROM decision_log GROUP BY final_action").fetchall()
for a in acts:
    print(f"  {dict(a)}")

# What's the conf_scale doing? Look at pipeline
print("\n=== CONF_SCALE VALUES (last 20) ===")
cs = c.execute("SELECT conf_scale FROM decision_log WHERE final_action != 'shadow_weight_update' ORDER BY ts_ms DESC LIMIT 20").fetchall()
vals = [r['conf_scale'] for r in cs]
print(f"  All values: {set(vals)}")

# Check open positions — what opened them?
print("\n=== OPEN POSITIONS ===")
open_trades = c.execute("""
    SELECT trade_id, symbol, side, entry_price, open_ts_ms, entry_decision_id
    FROM paper_trades WHERE exit_price IS NULL
""").fetchall()
for t in open_trades:
    print(f"  {dict(t)}")
