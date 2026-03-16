"""Full-dataset signal funnel diagnostic.

Runs a complete backtest on all available 1h candles and saves the signal
funnel breakdown, regime analytics, and trade-level diagnostics to JSON.

Usage:
    python -u diagnostics/run_funnel.py
"""
from __future__ import annotations

import json
import sys
import time

sys.stdout.reconfigure(line_buffering=True)

from hogan_bot.backtest import (
    diagnose_exits,
    diagnose_long_entries,
    diagnose_longs_by_confidence,
    diagnose_shorts_by_confidence,
    evaluate_market_regimes,
    evaluate_trades_by_regime_side,
    run_backtest_on_candles,
)
from hogan_bot.champion import apply_champion_mode, is_champion_mode
from hogan_bot.config import load_config
from hogan_bot.ml import load_model
from hogan_bot.storage import get_connection, load_candles

SYMBOL = "BTC/USD"
DB_PATH = "data/hogan.db"
OUTPUT = "diagnostics/funnel_report.json"


def main() -> None:
    cfg = load_config()
    if is_champion_mode():
        cfg = apply_champion_mode(cfg)

    timeframe = cfg.timeframe
    conn = get_connection(DB_PATH)
    candles = load_candles(conn, SYMBOL, timeframe)
    conn.close()

    if candles.empty:
        print(f"No candles for {SYMBOL}/{timeframe} in {DB_PATH}.")
        return
    print(f"Loaded {len(candles)} candles ({SYMBOL}/{timeframe})")

    ml_model = None
    if cfg.use_ml_filter:
        try:
            ml_model = load_model(cfg.ml_model_path)
            print(f"ML model loaded from {cfg.ml_model_path}")
        except Exception as e:
            print(f"ML model not loaded: {e}")
    else:
        print("ML filter is OFF (use_ml_filter=False)")

    print(f"\nKey thresholds:")
    print(f"  ml_buy_threshold:     {cfg.ml_buy_threshold}")
    print(f"  min_final_confidence: {cfg.min_final_confidence}")
    print(f"  min_tech_confidence:  {cfg.min_tech_confidence}")
    print(f"  min_regime_confidence:{cfg.min_regime_confidence}")
    print(f"  min_edge_multiple:    {cfg.min_edge_multiple}")
    print(f"  max_whipsaws:         {cfg.max_whipsaws}")
    print(f"  trailing_stop_pct:    {cfg.trailing_stop_pct}")
    print(f"  take_profit_pct:      {cfg.take_profit_pct}")
    print(f"  short_max_hold_hours: {cfg.short_max_hold_hours}")
    print(f"  max_hold_hours:       {cfg.max_hold_hours}")
    print()

    t0 = time.time()
    print("Running backtest (this will take a while)...")

    result = run_backtest_on_candles(
        candles=candles,
        symbol=SYMBOL,
        timeframe=timeframe,
        starting_balance_usd=cfg.starting_balance_usd,
        aggressive_allocation=cfg.aggressive_allocation,
        max_risk_per_trade=cfg.max_risk_per_trade,
        max_drawdown=cfg.max_drawdown,
        short_ma_window=cfg.short_ma_window,
        long_ma_window=cfg.long_ma_window,
        volume_window=cfg.volume_window,
        volume_threshold=cfg.volume_threshold,
        fee_rate=cfg.fee_rate,
        ml_model=ml_model,
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
        ml_confidence_sizing=cfg.ml_confidence_sizing,
        atr_stop_multiplier=cfg.atr_stop_multiplier,
        max_hold_hours=cfg.max_hold_hours if cfg.max_hold_hours > 0 else None,
        loss_cooldown_hours=cfg.loss_cooldown_hours if cfg.loss_cooldown_hours > 0 else None,
        min_edge_multiple=cfg.min_edge_multiple,
        min_final_confidence=cfg.min_final_confidence,
        min_tech_confidence=cfg.min_tech_confidence,
        min_regime_confidence=cfg.min_regime_confidence,
        max_whipsaws=cfg.max_whipsaws,
        reversal_confidence_mult=cfg.reversal_confidence_multiplier,
        enable_shorts=True,
        short_max_hold_hours=cfg.short_max_hold_hours,
        enable_pullback_gate=True,
        enable_close_and_reverse=False,
        db_path=DB_PATH,
    )

    elapsed = time.time() - t0
    print(f"Backtest complete in {elapsed:.0f}s")

    summary = result.summary_dict()
    funnel = summary.pop("signal_funnel", {})

    # ── Funnel analysis ───────────────────────────────────────────────
    bars = funnel.get("bars_evaluated", 1)
    pipeline_buy = funnel.get("pipeline_buy", 0)
    pipeline_sell = funnel.get("pipeline_sell", 0)
    post_ml_buy = funnel.get("post_ml_buy", 0)
    post_ml_sell = funnel.get("post_ml_sell", 0)
    post_edge_buy = funnel.get("post_edge_buy", 0)
    post_edge_sell = funnel.get("post_edge_sell", 0)
    post_quality_buy = funnel.get("post_quality_buy", 0)
    post_quality_sell = funnel.get("post_quality_sell", 0)
    post_ranging_buy = funnel.get("post_ranging_buy", 0)
    post_ranging_sell = funnel.get("post_ranging_sell", 0)
    executed_buy = funnel.get("executed_buy", 0)
    executed_short = funnel.get("executed_short_entry", 0)

    def _drop(before: int, after: int) -> str:
        if before == 0:
            return "N/A"
        pct = (1 - after / before) * 100
        return f"-{before - after} ({pct:.1f}%)"

    print(f"\n{'='*70}")
    print(f"SIGNAL FUNNEL ({bars} bars evaluated)")
    print(f"{'='*70}")
    print(f"  Pipeline output:   buy={pipeline_buy:>5d}  sell={pipeline_sell:>5d}")
    print(f"  After ML filter:   buy={post_ml_buy:>5d}  sell={post_ml_sell:>5d}   drop: buy {_drop(pipeline_buy, post_ml_buy)}  sell {_drop(pipeline_sell, post_ml_sell)}")
    print(f"  After edge gate:   buy={post_edge_buy:>5d}  sell={post_edge_sell:>5d}   drop: buy {_drop(post_ml_buy, post_edge_buy)}  sell {_drop(post_ml_sell, post_edge_sell)}")
    print(f"  After quality gate: buy={post_quality_buy:>5d}  sell={post_quality_sell:>5d}   drop: buy {_drop(post_edge_buy, post_quality_buy)}  sell {_drop(post_edge_sell, post_quality_sell)}")
    print(f"  After ranging gate: buy={post_ranging_buy:>5d}  sell={post_ranging_sell:>5d}   drop: buy {_drop(post_quality_buy, post_ranging_buy)}  sell {_drop(post_quality_sell, post_ranging_sell)}")
    print(f"  Executed:          long={executed_buy:>5d}  short={executed_short:>5d}")
    print()

    pullback_blocked = funnel.get("pullback_blocked", 0)
    pullback_resistance = funnel.get("pullback_blocked_resistance", 0)
    pullback_halved = funnel.get("pullback_halved", 0)
    if pullback_blocked or pullback_halved:
        print(f"  Pullback gate:  blocked={pullback_blocked}  (resistance={pullback_resistance}  chase={pullback_blocked - pullback_resistance})  halved={pullback_halved}")

    for key in ["edge_blocked_atr", "edge_blocked_tp", "edge_blocked_forecast", "edge_blocked_spread"]:
        v = funnel.get(key, 0)
        if v:
            print(f"  {key}: {v}")

    for key in ["ranging_blocked_tech", "ranging_blocked_ml", "ranging_blocked_whipsaw"]:
        v = funnel.get(key, 0)
        if v:
            print(f"  {key}: {v}")

    for key in ["blocked_already_long", "blocked_already_short", "blocked_cooldown",
                 "blocked_regime_no_longs", "blocked_regime_no_shorts"]:
        v = funnel.get(key, 0)
        if v:
            print(f"  {key}: {v}")

    regime_dist = funnel.get("regime_distribution", {})
    if regime_dist:
        total_r = sum(regime_dist.values())
        print(f"\n  Regime distribution:")
        for regime, count in sorted(regime_dist.items(), key=lambda x: -x[1]):
            print(f"    {regime:<14s}  {count:>5d} bars ({count/total_r*100:.1f}%)")

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"BACKTEST SUMMARY")
    print(f"{'='*70}")
    print(f"  return:   {summary['total_return_pct']:.4f}%")
    print(f"  maxdd:    {summary['max_drawdown_pct']:.4f}%")
    print(f"  sharpe:   {summary.get('sharpe_ratio', 'N/A')}")
    print(f"  trades:   {summary['trades']}")
    print(f"  win_rate: {summary['win_rate']:.1%}")

    # ── Trade analytics ───────────────────────────────────────────────
    by_regime_side = evaluate_trades_by_regime_side(result)
    long_conf = diagnose_longs_by_confidence(result.closed_trades)
    short_conf = diagnose_shorts_by_confidence(result.closed_trades)

    if by_regime_side:
        print(f"\n  Trades by regime x side:")
        print(f"  {'bucket':<22s}  {'n':>3s}  {'win%':>5s}  {'avg%':>7s}  {'tot%':>7s}")
        for key, m in by_regime_side.items():
            print(f"  {key:<22s}  {m['count']:>3d}  {m['win_rate']:>5.1%}  {m['avg_pnl_pct']:>+7.2f}  {m['total_pnl_pct']:>+7.2f}")

    if long_conf and long_conf.get("by_regime_confidence"):
        print(f"\n  Longs by regime x confidence:")
        print(f"  {'bucket':<26s}  {'n':>3s}  {'win%':>5s}  {'avg%':>7s}  {'tot%':>7s}")
        for key, m in long_conf["by_regime_confidence"].items():
            print(f"  {key:<26s}  {m['count']:>3d}  {m['win_rate']:>5.1%}  {m['avg_pnl_pct']:>+7.2f}  {m['total_pnl_pct']:>+7.2f}")

    if short_conf and short_conf.get("by_regime_confidence"):
        print(f"\n  Shorts by regime x confidence:")
        print(f"  {'bucket':<26s}  {'n':>3s}  {'win%':>5s}  {'avg%':>7s}  {'tot%':>7s}")
        for key, m in short_conf["by_regime_confidence"].items():
            print(f"  {key:<26s}  {m['count']:>3d}  {m['win_rate']:>5.1%}  {m['avg_pnl_pct']:>+7.2f}  {m['total_pnl_pct']:>+7.2f}")

    # ── Save full report ──────────────────────────────────────────────
    report = {
        "candles": len(candles),
        "symbol": SYMBOL,
        "timeframe": timeframe,
        "elapsed_seconds": round(elapsed),
        "summary": summary,
        "signal_funnel": funnel,
        "trades_by_regime_side": {
            k: {kk: vv for kk, vv in v.items() if kk != "exit_reasons"}
            for k, v in (by_regime_side or {}).items()
        },
        "longs_by_confidence": long_conf.get("by_regime_confidence", {}) if long_conf else {},
        "shorts_by_confidence": short_conf.get("by_regime_confidence", {}) if short_conf else {},
        "config_snapshot": {
            "ml_buy_threshold": cfg.ml_buy_threshold,
            "ml_sell_threshold": cfg.ml_sell_threshold,
            "use_ml_filter": cfg.use_ml_filter,
            "min_final_confidence": cfg.min_final_confidence,
            "min_tech_confidence": cfg.min_tech_confidence,
            "min_regime_confidence": cfg.min_regime_confidence,
            "min_edge_multiple": cfg.min_edge_multiple,
            "max_whipsaws": cfg.max_whipsaws,
            "trailing_stop_pct": cfg.trailing_stop_pct,
            "take_profit_pct": cfg.take_profit_pct,
            "short_max_hold_hours": cfg.short_max_hold_hours,
            "max_hold_hours": cfg.max_hold_hours,
            "fee_rate": cfg.fee_rate,
        },
    }

    with open(OUTPUT, "w") as f:
        json.dump(report, f, indent=2, default=str)
    print(f"\nFull report saved to {OUTPUT}")


if __name__ == "__main__":
    main()
