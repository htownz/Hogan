"""Short-hold sweep: compare 8h, 12h, 16h, 20h, 24h with shorts enabled.

Run from repo root:
    python sweep_short_hold.py [--limit N]

Uses last 8000 candles by default (~11 months of 1h) to keep each run
under 10 minutes while still being statistically meaningful.
Results are saved incrementally to sweep_results.json.
"""
from __future__ import annotations

import json
import sys
import time

from hogan_bot.backtest import (
    evaluate_trades_by_regime_side,
    run_backtest_on_candles,
)
from hogan_bot.champion import apply_champion_mode, is_champion_mode
from hogan_bot.config import load_config
from hogan_bot.ml import load_model
from hogan_bot.storage import get_connection, load_candles

HOLD_HOURS = [8, 12, 16, 20, 24]
SYMBOL = "BTC/USD"
DB_PATH = "data/hogan.db"
RESULTS_FILE = "sweep_results.json"
DEFAULT_LIMIT = 8000


def main() -> None:
    limit = DEFAULT_LIMIT
    if "--limit" in sys.argv:
        idx = sys.argv.index("--limit")
        if idx + 1 < len(sys.argv):
            limit = int(sys.argv[idx + 1])

    cfg = load_config()
    if is_champion_mode():
        cfg = apply_champion_mode(cfg)

    timeframe = cfg.timeframe
    conn = get_connection(DB_PATH)
    candles = load_candles(conn, SYMBOL, timeframe, limit=limit)
    conn.close()

    if candles.empty:
        print(f"No candles for {SYMBOL}/{timeframe} in {DB_PATH}.")
        return
    print(f"Loaded {len(candles)} candles ({SYMBOL}/{timeframe})\n")

    ml_model = None
    try:
        ml_model = load_model(cfg.ml_model_path)
    except Exception:
        pass

    rows = []
    for hours in HOLD_HOURS:
        t0 = time.time()
        print(f"--- Running short_max_hold={hours}h ---")
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
            enable_shorts=True,
            short_max_hold_hours=float(hours),
            enable_pullback_gate=True,
            enable_close_and_reverse=False,
            db_path=DB_PATH,
        )

        summary = result.summary_dict()
        funnel = summary.get("signal_funnel", {})

        short_trades = [t for t in result.closed_trades if t.get("side") == "short"]
        long_trades = [t for t in result.closed_trades if t.get("side") == "long"]

        short_wins = sum(1 for t in short_trades if t.get("pnl_pct", 0) > 0)
        short_total_pnl = sum(t.get("pnl_pct", 0) for t in short_trades)
        short_avg_pnl = short_total_pnl / len(short_trades) if short_trades else 0
        short_win_rate = short_wins / len(short_trades) if short_trades else 0

        short_payoff = 0.0
        s_wins_pnl = [t["pnl_pct"] for t in short_trades if t.get("pnl_pct", 0) > 0]
        s_loss_pnl = [abs(t["pnl_pct"]) for t in short_trades if t.get("pnl_pct", 0) <= 0]
        if s_wins_pnl and s_loss_pnl:
            short_payoff = (sum(s_wins_pnl) / len(s_wins_pnl)) / (sum(s_loss_pnl) / len(s_loss_pnl))

        td_shorts = [t for t in short_trades
                     if t.get("entry_regime", "").startswith("trending_down")]
        td_short_pnl = sum(t.get("pnl_pct", 0) for t in td_shorts)
        td_short_avg = td_short_pnl / len(td_shorts) if td_shorts else 0
        td_short_wins = sum(1 for t in td_shorts if t.get("pnl_pct", 0) > 0)
        td_short_wr = td_short_wins / len(td_shorts) if td_shorts else 0

        row = {
            "hold_h": hours,
            "return%": summary["total_return_pct"],
            "maxdd%": summary["max_drawdown_pct"],
            "sharpe": summary.get("sharpe_ratio"),
            "trades": summary["trades"],
            "win%": summary["win_rate"],
            "longs": funnel.get("executed_buy", 0),
            "shorts": funnel.get("executed_short_entry", 0),
            "s_win%": short_win_rate,
            "s_avg%": short_avg_pnl,
            "s_total%": short_total_pnl,
            "s_payoff": short_payoff,
            "td_s_n": len(td_shorts),
            "td_s_wr": td_short_wr,
            "td_s_avg": td_short_avg,
            "td_s_tot": td_short_pnl,
        }
        rows.append(row)
        elapsed = time.time() - t0
        print(f"  return={row['return%']:.2f}%  dd={row['maxdd%']:.2f}%  "
              f"trades={row['trades']}  shorts={row['shorts']}  "
              f"s_avg={row['s_avg%']:+.3f}%  td_s_avg={row['td_s_avg']:+.3f}%  "
              f"({elapsed:.0f}s)\n")
        with open(RESULTS_FILE, "w") as f:
            json.dump(rows, f, indent=2)

    # ── Comparison table ──────────────────────────────────────────────
    print("\n" + "=" * 120)
    print("SHORT-HOLD SWEEP COMPARISON")
    print("=" * 120)

    hdr = (f"  {'hold':>5s}  {'return%':>8s}  {'maxdd%':>7s}  {'sharpe':>7s}  "
           f"{'trades':>6s}  {'win%':>5s}  {'longs':>5s}  {'shorts':>6s}  "
           f"{'s_win%':>6s}  {'s_avg%':>7s}  {'s_tot%':>7s}  {'s_payoff':>8s}  "
           f"{'td_n':>4s}  {'td_wr':>5s}  {'td_avg%':>7s}  {'td_tot%':>7s}")
    print(hdr)
    print("  " + "-" * 116)

    for r in rows:
        sharpe_s = f"{r['sharpe']:.4f}" if r['sharpe'] is not None else "  N/A"
        print(f"  {r['hold_h']:>4d}h  {r['return%']:>8.2f}  {r['maxdd%']:>7.2f}  {sharpe_s:>7s}  "
              f"{r['trades']:>6d}  {r['win%']:>5.1%}  {r['longs']:>5d}  {r['shorts']:>6d}  "
              f"{r['s_win%']:>6.1%}  {r['s_avg%']:>+7.3f}  {r['s_total%']:>+7.2f}  {r['s_payoff']:>8.2f}  "
              f"{r['td_s_n']:>4d}  {r['td_s_wr']:>5.1%}  {r['td_s_avg']:>+7.3f}  {r['td_s_tot']:>+7.2f}")

    print()
    print(json.dumps(rows, indent=2))


if __name__ == "__main__":
    main()
