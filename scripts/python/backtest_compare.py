"""Compare BEFORE vs AFTER settings on local DB candles.

Before: volume_threshold=0.9, ml_confidence_sizing=True
After:  volume_threshold=0.3, ml_confidence_sizing=False, EMA cloud fix
"""
from __future__ import annotations

import os
import sys

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from hogan_bot.backtest import run_backtest_on_candles
from hogan_bot.config import load_config
from hogan_bot.storage import get_connection, load_candles

cfg = load_config()
conn = get_connection("data/hogan.db")
candles = load_candles(conn, "BTC/USD", "1h", limit=2000)
conn.close()

print(f"Loaded {len(candles)} candles from local DB (BTC/USD / 1h)\n")

SHARED = dict(
    candles=candles,
    symbol="BTC/USD",
    starting_balance_usd=10_000.0,
    aggressive_allocation=cfg.aggressive_allocation,
    max_risk_per_trade=cfg.max_risk_per_trade,
    max_drawdown=cfg.max_drawdown,
    short_ma_window=cfg.short_ma_window,
    long_ma_window=cfg.long_ma_window,
    volume_window=cfg.volume_window,
    fee_rate=cfg.fee_rate,
    use_ema_clouds=cfg.use_ema_clouds,
    signal_mode=cfg.signal_mode,
    min_vote_margin=cfg.signal_min_vote_margin,
    trailing_stop_pct=cfg.trailing_stop_pct,
    take_profit_pct=cfg.take_profit_pct,
)

# ── BEFORE: old settings ──────────────────────────────────────────────────────
r_before = run_backtest_on_candles(
    **SHARED,
    timeframe="1h",
    volume_threshold=0.9,      # old: too restrictive
    ml_confidence_sizing=True, # old: shrunk positions to ~1%
)

# ── AFTER: current settings ───────────────────────────────────────────────────
r_after = run_backtest_on_candles(
    **SHARED,
    timeframe="1h",
    volume_threshold=0.3,       # new: allows more trades
    ml_confidence_sizing=False, # new: full position sizing
)

b = r_before.summary_dict()
a = r_after.summary_dict()

metrics = [
    ("total_return_pct", "%"),
    ("max_drawdown_pct", "%"),
    ("win_rate",         ""),
    ("trades",           ""),
    ("sharpe_ratio",     ""),
]

print(f"{'Metric':<26} {'BEFORE (old)':>12} {'AFTER (new)':>12}")
print("-" * 52)
for k, unit in metrics:
    bv = b.get(k, 0)
    av = a.get(k, 0)
    arrow = "^" if av > bv else ("=" if av == bv else "v")
    print(f"{k:<26} {bv:>11.3f}{unit}  {av:>11.3f}{unit}  {arrow}")

print()
print("Note: last 7 days of BTC data includes a significant downtrend.")
print("Losses are expected for long-only — regime detection should tighten in these conditions.")
print("Key change: AFTER fires", int(a.get('trades',0)), "trades vs BEFORE", int(b.get('trades',0)),
      "— more data collection for ML feedback loop.")
