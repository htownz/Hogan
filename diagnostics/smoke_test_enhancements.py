"""Quick smoke test for entry context, proactive exits, and spread_est feature."""
from hogan_bot.storage import get_connection, load_candles
from hogan_bot.backtest import run_backtest_on_candles, enrich_trades_with_entry_context
from hogan_bot.config import load_config
from hogan_bot.profiles import CANONICAL_PROFILE, apply_profile
from hogan_bot.champion import apply_champion_mode

config = load_config()
config, cli_ov = apply_profile(config, CANONICAL_PROFILE)
config = apply_champion_mode(config)

conn = get_connection("data/hogan.db")
candles = load_candles(conn, "BTC/USD", "1h", limit=3000)
conn.close()
print(f"Candles: {len(candles)}")

result = run_backtest_on_candles(
    candles, symbol="BTC/USD",
    starting_balance_usd=config.starting_balance_usd,
    aggressive_allocation=config.aggressive_allocation,
    max_risk_per_trade=config.max_risk_per_trade,
    max_drawdown=config.max_drawdown,
    short_ma_window=config.short_ma_window,
    long_ma_window=config.long_ma_window,
    volume_window=config.volume_window,
    volume_threshold=config.volume_threshold,
    fee_rate=config.fee_rate,
    timeframe="1h",
    trailing_stop_pct=config.trailing_stop_pct,
    take_profit_pct=config.take_profit_pct,
    max_hold_hours=config.max_hold_hours,
    enable_shorts=True,
    short_max_hold_hours=config.short_max_hold_hours,
    slippage_bps=5.0,
    execution_mode="next_open",
    use_policy_core=True,
    trail_activation_pct=config.trail_activation_pct,
    breakeven_stop_pct=0.015,
    enable_pullback_gate=True,
)

trades = result.closed_trades
print(f"Total trades: {len(trades)}")

with_ctx = [t for t in trades if t.get("entry_context")]
print(f"Trades with entry_context: {len(with_ctx)}")

if with_ctx:
    ctx = with_ctx[0]["entry_context"]
    print(f"Entry context keys: {sorted(ctx.keys())}")
    print(f"  Has spread_est: {'spread_est' in ctx}")
    print(f"  spread_est value: {ctx.get('spread_est', 'MISSING')}")

enrich_trades_with_entry_context(candles, trades)
enriched = [t for t in trades if "local_range_position" in t]
print(f"Enriched trades (with local_range_position): {len(enriched)}")

proactive = [t for t in trades if t.get("close_reason", "").startswith("proactive_")]
print(f"\nProactive exits: {len(proactive)}")
for t in proactive[:8]:
    print(f"  {t['close_reason']:35s} pnl={t['pnl_pct']:+.2f}%")

reasons = {}
for t in trades:
    r = t.get("close_reason", "unknown")
    reasons[r] = reasons.get(r, 0) + 1
print(f"\nClose reason breakdown:")
for r, n in sorted(reasons.items(), key=lambda x: -x[1]):
    print(f"  {r:35s} {n}")

# Test trade quality feature building
from hogan_bot.trade_quality import _build_feature_row_from_context, FEATURE_COLUMNS
if with_ctx:
    t0 = with_ctx[0]
    feats = _build_feature_row_from_context(t0["entry_context"], t0)
    print(f"\nFeature row keys ({len(feats)}): {sorted(feats.keys())}")
    print(f"FEATURE_COLUMNS ({len(FEATURE_COLUMNS)}): matches = {sorted(feats.keys()) == sorted(FEATURE_COLUMNS)}")
