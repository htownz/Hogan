import sqlite3, json

c = sqlite3.connect("data/hogan.db")
c.row_factory = sqlite3.Row

rows = c.execute("""
    SELECT final_action, final_confidence, tech_action, tech_confidence,
           sent_bias, sent_strength, macro_regime, macro_risk_on,
           ml_up_prob, forecast_4h, forecast_12h, forecast_24h,
           risk_vol_pct, risk_stop_hit, conf_scale, position_size,
           meta_weights_json, explanation
    FROM decision_log
    WHERE final_action != 'shadow_weight_update'
    ORDER BY ts_ms DESC LIMIT 5
""").fetchall()

for r in rows:
    d = dict(r)
    print(f"ACTION={d['final_action']}  CONF={d['final_confidence']}")
    print(f"  tech={d['tech_action']}({d['tech_confidence']})  sent={d['sent_bias']}({d['sent_strength']})  macro={d['macro_regime']}(risk_on={d['macro_risk_on']})")
    print(f"  ml_up={d['ml_up_prob']}  f4h={d['forecast_4h']}  f12h={d['forecast_12h']}  f24h={d['forecast_24h']}")
    print(f"  risk_vol={d['risk_vol_pct']}  stop_hit={d['risk_stop_hit']}  conf_scale={d['conf_scale']}  pos_sz={d['position_size']}")
    w = d['meta_weights_json']
    if w:
        try:
            print(f"  weights={json.loads(w)}")
        except:
            print(f"  weights={w[:80]}")
    print(f"  explain={d['explanation'][:250] if d['explanation'] else 'None'}")
    print()

# Count action distribution
print("=== ACTION DISTRIBUTION (last 200) ===")
dist = c.execute("""
    SELECT final_action, COUNT(*) as n, AVG(final_confidence) as avg_conf
    FROM decision_log
    WHERE final_action != 'shadow_weight_update'
    GROUP BY final_action
    ORDER BY n DESC
""").fetchall()
for r in dist:
    print(f"  {dict(r)}")

# Check if there are ANY buy/sell in the whole history
print("\n=== BUY/SELL DECISIONS EVER ===")
buysell = c.execute("""
    SELECT final_action, COUNT(*) as n
    FROM decision_log
    WHERE final_action IN ('buy','sell')
    GROUP BY final_action
""").fetchall()
for r in buysell:
    print(f"  {dict(r)}")
if not buysell:
    print("  NONE - Hogan has NEVER issued a buy or sell from the pipeline!")

# Check trade count
print(f"\n=== TRADES: {c.execute('SELECT COUNT(*) FROM paper_trades').fetchone()[0]} total ===")
print(f"    Open: {c.execute('SELECT COUNT(*) FROM paper_trades WHERE exit_price IS NULL').fetchone()[0]}")
print(f"    Closed: {c.execute('SELECT COUNT(*) FROM paper_trades WHERE exit_price IS NOT NULL').fetchone()[0]}")
