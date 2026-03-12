"""Simulate what the new MetaWeigher math produces with current data."""

# Current live signals
tech_action, tech_conf = "hold", 0.0
sent_bias, sent_strength = "bearish", 0.5
macro_regime, macro_risk_on = "risk_off", False
regime = "ranging"

# Old weights (no ranging adjustment)
w_old = {"technical": 0.55, "sentiment": 0.25, "macro": 0.20}
tech_score = 0  # hold * 0
sent_score = -1 * 0.5  # bearish * 0.5
old_combined = w_old["technical"] * tech_score + w_old["sentiment"] * sent_score
print(f"OLD: combined={old_combined:.4f}  threshold=-0.15  => {'SELL' if old_combined <= -0.15 else 'HOLD'}")

# New weights (ranging adjustment: tech-0.20, sent+0.10, macro+0.10)
w_new = {"technical": 0.55 - 0.20, "sentiment": 0.25 + 0.10, "macro": 0.20 + 0.10}
total = sum(w_new.values())
w_new = {k: v/total for k, v in w_new.items()}
print(f"\nNew weights (ranging): {', '.join(f'{k}={v:.2f}' for k,v in w_new.items())}")

# Macro as directional voter
macro_score = -0.5  # risk_off
new_combined = (
    w_new["technical"] * tech_score
    + w_new["sentiment"] * sent_score
    + w_new["macro"] * macro_score
)
new_threshold = -0.12
print(f"NEW: combined={new_combined:.4f}  threshold={new_threshold}  => {'SELL' if new_combined <= new_threshold else 'HOLD'}")
print(f"\n  tech: {w_new['technical']:.2f} * 0 = 0")
print(f"  sent: {w_new['sentiment']:.2f} * -0.50 = {w_new['sentiment'] * -0.5:.4f}")
print(f"  macro: {w_new['macro']:.2f} * -0.50 = {w_new['macro'] * -0.5:.4f}")
print(f"  total = {new_combined:.4f}")

print("\n=== Other scenarios ===")
for sc in [
    ("ranging", "hold", 0.0, "bullish", 0.6, "risk_on"),
    ("ranging", "hold", 0.0, "neutral", 0.0, "risk_off"),
    ("ranging", "buy", 0.7, "bullish", 0.5, "risk_on"),
    ("trending_up", "buy", 0.8, "bullish", 0.5, "risk_on"),
    ("volatile", "hold", 0.0, "bearish", 0.8, "risk_off"),
]:
    reg, ta, tc, sb, ss, mr = sc
    w = {"technical": 0.55, "sentiment": 0.25, "macro": 0.20}
    if reg in ("trending_up", "trending_down"):
        w["technical"] = min(0.75, w["technical"] + 0.10)
        w["sentiment"] = max(0.10, w["sentiment"] - 0.05)
        w["macro"] = max(0.10, w["macro"] - 0.05)
    elif reg == "volatile":
        w["technical"] = max(0.35, w["technical"] - 0.10)
        w["sentiment"] += 0.05; w["macro"] += 0.05
    elif reg == "ranging":
        w["technical"] = max(0.25, w["technical"] - 0.20)
        w["sentiment"] += 0.10; w["macro"] += 0.10
    t = sum(w.values()); w = {k:v/t for k,v in w.items()}
    ts2 = {"buy":1,"sell":-1,"hold":0}[ta] * tc
    ss2 = {"bullish":1,"bearish":-1,"neutral":0}[sb] * ss
    ms2 = {"risk_on":0.5,"risk_off":-0.5,"neutral":0}[mr]
    cs = w["technical"]*ts2 + w["sentiment"]*ss2 + w["macro"]*ms2
    bt = {"volatile":0.25,"trending_up":0.10,"trending_down":0.10,"ranging":0.12}.get(reg, 0.15)
    st = -bt
    act = "buy" if cs >= bt else ("sell" if cs <= st else "hold")
    print(f"  {reg}: tech={ta}({tc}) sent={sb}({ss}) macro={mr} => score={cs:.3f} => {act}")
