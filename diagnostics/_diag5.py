"""Final diagnostic: sell timing, edge gate values, and execution blockers."""
import sqlite3
import time
from datetime import datetime, timezone

conn = sqlite3.connect("data/hogan.db")
conn.row_factory = sqlite3.Row

# 1. When were the 15 sell signals logged?
sells = conn.execute(
    "SELECT id, ts_ms, tech_action, pipeline_action, final_action, regime, "
    "tech_confidence, final_confidence, ml_up_prob, conf_scale, position_size "
    "FROM decision_log WHERE final_action = 'sell' ORDER BY ts_ms DESC"
).fetchall()
print(f"=== All {len(sells)} sell signals with timestamps ===")
for r in sells:
    ts = datetime.fromtimestamp(r["ts_ms"] / 1000, tz=timezone.utc)
    print(f"  id={r['id']} ts={ts} regime={r['regime']} "
          f"tech={r['tech_action']} pipe={r['pipeline_action']} "
          f"tconf={r['tech_confidence']:.3f} fconf={r['final_confidence']:.4f} "
          f"ml={r['ml_up_prob']:.4f} scale={r['conf_scale']:.4f} "
          f"size={r['position_size']:.6f}")

# Bot restart was ~2026-03-17 23:24 UTC -> ts_ms ~1742256240000
# Actually let me compute it properly
restart_ts = datetime(2026, 3, 18, 4, 24, 0, tzinfo=timezone.utc)  # from terminal log
restart_ms = restart_ts.timestamp() * 1000
print(f"\nBot restart at: {restart_ts} ({restart_ms:.0f})")
post_restart_sells = [r for r in sells if r["ts_ms"] > restart_ms]
print(f"Sells AFTER restart: {len(post_restart_sells)}")
pre_restart_sells = [r for r in sells if r["ts_ms"] <= restart_ms]
print(f"Sells BEFORE restart: {len(pre_restart_sells)}")

# 2. Edge gate: what ATR values are buy signals seeing?
cols = [r[1] for r in conn.execute("PRAGMA table_info(decision_log)").fetchall()]
print(f"\n=== Decision log columns (checking for ATR/edge data) ===")
atr_cols = [c for c in cols if "atr" in c.lower() or "edge" in c.lower() or "friction" in c.lower()]
print(f"ATR-related columns: {atr_cols}")

# Check block_reasons_json for edge_gate details
print("\n=== Edge gate block details (buy signals) ===")
edge_blocked = conn.execute(
    "SELECT block_reasons_json FROM decision_log "
    "WHERE tech_action = 'buy' AND final_action = 'hold' "
    "AND block_reasons_json LIKE '%edge_gate%' "
    "ORDER BY id DESC LIMIT 10"
).fetchall()
for r in edge_blocked:
    print(f"  {r['block_reasons_json']}")

# 3. What fee_rate is being used?
print("\n=== Config from .env ===")
try:
    from dotenv import dotenv_values
    env = dotenv_values(".env")
    for k in ["HOGAN_FEE_RATE", "HOGAN_SLIPPAGE_BPS", "HOGAN_SPREAD_BPS",
              "HOGAN_USE_ML_FILTER", "HOGAN_ML_AS_SIZER",
              "HOGAN_MACRO_SITOUT", "HOGAN_USE_FUNDING_OVERLAY",
              "HOGAN_SWARM_MODE", "HOGAN_USE_POLICY_CORE",
              "HOGAN_ENABLE_SHORTS", "HOGAN_MIN_FINAL_CONFIDENCE"]:
        print(f"  {k}={env.get(k, '(not set)')}")
except Exception as e:
    print(f"  Error: {e}")

# 4. Check what BotConfig.fee_rate defaults to
print("\n=== BotConfig defaults ===")
from hogan_bot.config import load_config, apply_champion_mode
cfg = load_config()
cfg = apply_champion_mode(cfg)
print(f"  fee_rate={cfg.fee_rate}")
print(f"  slippage_bps={cfg.slippage_bps}")
print(f"  enable_shorts={cfg.enable_shorts}")
print(f"  use_ml_filter={cfg.use_ml_filter}")
print(f"  use_ml_as_sizer={cfg.use_ml_as_sizer}")
print(f"  ml_confidence_sizing={cfg.ml_confidence_sizing}")
print(f"  min_final_confidence={cfg.min_final_confidence}")
print(f"  ml_buy_threshold={cfg.ml_buy_threshold}")
print(f"  trailing_stop_pct={cfg.trailing_stop_pct}")
print(f"  take_profit_pct={cfg.take_profit_pct}")
print(f"  use_policy_core={cfg.use_policy_core}")
print(f"  macro_sitout={getattr(cfg, 'macro_sitout', 'N/A')}")

# 5. Compute typical ATR from recent candles
import pandas as pd
candles = pd.read_sql_query(
    "SELECT ts_ms, open, high, low, close FROM candles "
    "WHERE symbol = 'BTC/USD' AND timeframe = '1h' ORDER BY ts_ms DESC LIMIT 50",
    conn,
)
if len(candles) > 14:
    tr = candles["high"] - candles["low"]
    atr_14 = tr.rolling(14).mean().iloc[-1]
    close = candles["close"].iloc[-1]
    atr_pct = atr_14 / close
    print(f"\n=== Current BTC/USD ATR (14-bar) ===")
    print(f"  ATR: {atr_14:.2f}")
    print(f"  Close: {close:.2f}")
    print(f"  ATR%: {atr_pct:.6f} ({atr_pct*100:.4f}%)")
    
    friction = 2 * cfg.fee_rate
    buy_threshold = friction * 0.5
    sell_threshold = friction * 0.8
    print(f"\n  Fee rate: {cfg.fee_rate}")
    print(f"  Total friction (no spread): {friction:.6f}")
    print(f"  Buy ATR threshold (0.5x friction): {buy_threshold:.6f} ({buy_threshold*100:.4f}%)")
    print(f"  Sell ATR threshold (0.8x friction): {sell_threshold:.6f} ({sell_threshold*100:.4f}%)")
    print(f"  ATR passes buy gate: {atr_pct >= buy_threshold}")
    print(f"  ATR passes sell gate: {atr_pct >= sell_threshold}")

# 6. Check recent entries after restart more carefully
print("\n=== Post-restart entry sample (last 5) ===")
recent = conn.execute(
    "SELECT id, ts_ms, tech_action, pipeline_action, final_action, "
    "regime, tech_confidence, final_confidence, ml_up_prob, conf_scale, "
    "position_size, block_reasons_json "
    "FROM decision_log WHERE ts_ms > ? ORDER BY id DESC LIMIT 5",
    (restart_ms,),
).fetchall()
for r in recent:
    ts = datetime.fromtimestamp(r["ts_ms"] / 1000, tz=timezone.utc)
    print(f"  id={r['id']} ts={ts} tech={r['tech_action']} pipe={r['pipeline_action']} "
          f"final={r['final_action']} regime={r['regime']} "
          f"tconf={r['tech_confidence']:.3f} fconf={r['final_confidence']:.4f} "
          f"ml={r['ml_up_prob']:.4f} scale={r['conf_scale']:.4f} "
          f"size={r['position_size']:.6f} blocks={r['block_reasons_json']}")

# Any non-hold post-restart?
post_restart_non_hold = conn.execute(
    "SELECT final_action, COUNT(*) as cnt FROM decision_log "
    "WHERE ts_ms > ? GROUP BY final_action",
    (restart_ms,),
).fetchall()
print(f"\nPost-restart action distribution:")
for r in post_restart_non_hold:
    print(f"  {r['final_action']}: {r['cnt']}")

conn.close()
