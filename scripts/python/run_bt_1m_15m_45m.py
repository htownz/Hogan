"""Run backtests on 1m, 15m, 45m for BTC/USD and ETH/USD and print results."""
from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hogan_bot.backtest import run_backtest_on_candles
from hogan_bot.config import load_config
from hogan_bot.ml import load_model, make_backtest_labels
from hogan_bot.storage import get_connection, load_candles

conn = get_connection("data/hogan.db")
cfg = load_config()

ml_model = None
# ML per-bar inference is expensive; skip for large bar counts
# and only enable for shorter runs.
USE_ML_THRESHOLD = 1500
try:
    ml_model = load_model(cfg.ml_model_path)
    print(f"ML model loaded from {cfg.ml_model_path}")
except Exception:
    print("No ML model found — running without ML filter")

CONFIGS = [
    ("BTC/USD", "1m",  1500),
    ("BTC/USD", "15m", 3000),
    ("BTC/USD", "45m", 2878),
    ("ETH/USD", "1m",  1500),
    ("ETH/USD", "15m", 3000),
    ("ETH/USD", "45m", 2878),
]

all_results = []

for sym, tf, lim in CONFIGS:
    candles = load_candles(conn, sym, tf, limit=lim)
    if candles.empty:
        print(f"\n{sym}/{tf}: NO DATA — skipping")
        continue

    use_ml = ml_model if lim <= USE_ML_THRESHOLD else None
    ml_tag = "with ML" if use_ml else "no ML"
    print(f"\nRunning {sym}/{tf} ({len(candles)} bars, {ml_tag})...", flush=True)

    t0 = time.perf_counter()
    result = run_backtest_on_candles(
        candles=candles,
        symbol=sym,
        timeframe=tf,
        starting_balance_usd=cfg.starting_balance_usd,
        aggressive_allocation=cfg.aggressive_allocation,
        max_risk_per_trade=cfg.max_risk_per_trade,
        max_drawdown=cfg.max_drawdown,
        short_ma_window=cfg.short_ma_window,
        long_ma_window=cfg.long_ma_window,
        volume_window=cfg.volume_window,
        volume_threshold=cfg.volume_threshold,
        fee_rate=cfg.fee_rate,
        ml_model=use_ml,
        ml_buy_threshold=cfg.ml_buy_threshold,
        ml_sell_threshold=cfg.ml_sell_threshold,
        use_ema_clouds=cfg.use_ema_clouds,
        ema_fast_short=cfg.ema_fast_short,
        ema_fast_long=cfg.ema_fast_long,
        ema_slow_short=cfg.ema_slow_short,
        ema_slow_long=cfg.ema_slow_long,
        use_fvg=cfg.use_fvg,
        fvg_min_gap_pct=cfg.fvg_min_gap_pct,
        signal_mode=cfg.signal_mode,
        min_vote_margin=cfg.signal_min_vote_margin,
        trailing_stop_pct=cfg.trailing_stop_pct,
        take_profit_pct=cfg.take_profit_pct,
    )
    dt = time.perf_counter() - t0
    s = result.summary_dict()
    s["symbol"] = sym
    s["timeframe"] = tf
    s["bars"] = len(candles)
    s["elapsed_s"] = round(dt, 1)
    s["closed_trades"] = len(result.closed_trades)

    wins = sum(1 for t in result.closed_trades if t["pnl_usd"] > 0)
    losses = len(result.closed_trades) - wins
    total_pnl = sum(t["pnl_usd"] for t in result.closed_trades)

    print(f"\n{'='*60}")
    print(f"  {sym} / {tf}  ({len(candles)} bars, {dt:.1f}s)")
    print(f"{'='*60}")
    print(f"  Return:     {s['total_return_pct']:+.2f}%")
    print(f"  Max DD:     {s['max_drawdown_pct']:.2f}%")
    print(f"  Trades:     {s['trades']}  (closed: {len(result.closed_trades)})")
    print(f"  Win rate:   {s['win_rate']:.2%}")
    print(f"  Wins/Losses:{wins}/{losses}")
    print(f"  Total P&L:  ${total_pnl:+.2f}")
    print(f"  Sharpe:     {s['sharpe_ratio']}")
    print(f"  Sortino:    {s['sortino_ratio']}")
    print(f"  Calmar:     {s['calmar_ratio']}")

    bt_labels = make_backtest_labels(result, candles, sym, db_conn=conn)
    if bt_labels[0] is not None:
        label_dist = bt_labels[1].value_counts().to_dict()
        print(f"  BT labels:  {len(bt_labels[0])} rows  (1={label_dist.get(1,0)} / 0={label_dist.get(0,0)})")
    else:
        print("  BT labels:  too few trades for labels")

    all_results.append(s)

conn.close()

print(f"\n{'='*60}")
print("  SUMMARY TABLE")
print(f"{'='*60}")
header = f"{'Symbol':<10} {'TF':<5} {'Bars':>6} {'Return%':>9} {'MaxDD%':>8} {'Trades':>7} {'WinRate':>8} {'Sharpe':>8}"
print(header)
print("-" * len(header))
for r in all_results:
    print(
        f"{r['symbol']:<10} {r['timeframe']:<5} {r['bars']:>6} "
        f"{r['total_return_pct']:>+8.2f}% {r['max_drawdown_pct']:>7.2f}% "
        f"{r['trades']:>7} {r['win_rate']:>7.2%} "
        f"{r['sharpe_ratio'] or 0:>8.4f}"
    )
print()
