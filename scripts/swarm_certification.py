"""Swarm Certification Backtest — accelerates swarm evaluation using historical data.

Runs the backtest twice over the same candle window:
  1. Baseline:  swarm in shadow mode (swarm logs decisions but doesn't trade)
  2. Active:    swarm in active mode (swarm controls trade decisions)

Compares P&L, Sharpe, drawdown, and trade quality between the two runs.
Also writes swarm decisions to SQLite (default: the same DB as --db).  Use
``--scratch-db`` to copy the source DB to a tempfile so certification does
not append swarm rows / weight snapshots to your production file.

Usage:
    python scripts/swarm_certification.py
    python scripts/swarm_certification.py --bars 8000
    python scripts/swarm_certification.py --dry-run
    python scripts/swarm_certification.py --scratch-db
    python scripts/swarm_certification.py --scratch-db --keep-scratch-db
"""
from __future__ import annotations

import argparse
import atexit
import logging
import os
import shutil
import sys
import tempfile
import time
from pathlib import Path

_project_root = str(Path(__file__).resolve().parents[1])
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def _run_backtest(candles, ml_model, db_path, swarm_mode, macro_sitout, bot_cfg, symbol):
    """Run a single backtest with the given swarm mode."""
    from hogan_bot.backtest import run_backtest_on_candles

    bt = run_backtest_on_candles(
        candles,
        symbol=symbol,
        starting_balance_usd=10_000,
        aggressive_allocation=0.50,
        max_risk_per_trade=bot_cfg.max_risk_per_trade,
        max_drawdown=0.20,
        short_ma_window=bot_cfg.short_ma_window,
        long_ma_window=bot_cfg.long_ma_window,
        volume_window=bot_cfg.volume_window,
        volume_threshold=bot_cfg.volume_threshold,
        fee_rate=bot_cfg.fee_rate,
        timeframe=bot_cfg.timeframe,
        ml_model=ml_model,
        ml_buy_threshold=bot_cfg.ml_buy_threshold,
        ml_sell_threshold=bot_cfg.ml_sell_threshold,
        trailing_stop_pct=bot_cfg.trailing_stop_pct,
        take_profit_pct=bot_cfg.take_profit_pct,
        trail_activation_pct=bot_cfg.trail_activation_pct,
        ml_confidence_sizing=False,
        max_hold_hours=bot_cfg.max_hold_hours,
        slippage_bps=5.0,
        enable_shorts=True,
        enable_pullback_gate=True,
        short_max_hold_hours=bot_cfg.short_max_hold_hours,
        min_edge_multiple=bot_cfg.min_edge_multiple,
        min_final_confidence=bot_cfg.min_final_confidence,
        min_tech_confidence=0.15,
        min_regime_confidence=0.30,
        max_whipsaws=3,
        macro_sitout=macro_sitout,
        use_ml_as_sizer=True,
        use_policy_core=True,
        swarm_enabled=True,
        swarm_mode=swarm_mode,
        swarm_active_allow_new_signals=getattr(
            bot_cfg, "swarm_active_allow_new_signals", False,
        ),
        db_path=db_path,
    )
    return bt


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Swarm certification backtest")
    parser.add_argument("--db", default="data/hogan.db", help="SQLite DB path (read candles + macro data)")
    parser.add_argument("--bars", type=int, default=5000, help="Number of recent 1h bars to use")
    parser.add_argument("--symbol", default="BTC/USD", help="Trading pair")
    parser.add_argument("--dry-run", action="store_true", help="Print plan only")
    parser.add_argument(
        "--scratch-db",
        action="store_true",
        help="Copy --db to a temp file; backtests write swarm logs only there (production DB unchanged)",
    )
    parser.add_argument(
        "--keep-scratch-db",
        action="store_true",
        help="With --scratch-db, keep the temp file and print its path (default: delete on exit)",
    )
    args = parser.parse_args(argv)

    from hogan_bot.champion import apply_champion_mode
    from hogan_bot.config import load_config
    from hogan_bot.ml import load_model
    from hogan_bot.storage import get_connection, load_candles

    config = load_config()
    config = apply_champion_mode(config)

    conn = get_connection(args.db)
    candles = load_candles(conn, args.symbol, "1h")
    if len(candles) > args.bars:
        candles = candles.iloc[-args.bars:].reset_index(drop=True)
    logger.info("Loaded %d candles for %s 1h", len(candles), args.symbol)

    ml_model = None
    try:
        ml_model = load_model(config.ml_model_path)
        logger.info("Loaded ML model from %s", config.ml_model_path)
    except Exception as e:
        logger.warning("No ML model loaded: %s", e)

    macro_sitout = None
    try:
        from hogan_bot.macro_sitout import MacroSitout

        macro_sitout = MacroSitout.from_db(conn)
        logger.info("Macro sitout filter loaded")
    except Exception:
        pass

    conn.close()

    scratch_path: str | None = None
    if args.scratch_db and not args.dry_run:
        fd, scratch_path = tempfile.mkstemp(suffix=".db", prefix="hogan_swarm_cert_")
        os.close(fd)
        shutil.copy2(args.db, scratch_path)
        logger.info("Scratch DB: copied %s -> %s", args.db, scratch_path)
        if not args.keep_scratch_db:
            _scratch_copy = scratch_path

            def _rm_scratch(p: str = _scratch_copy) -> None:
                try:
                    os.unlink(p)
                except OSError:
                    pass

            atexit.register(_rm_scratch)

    write_db = scratch_path if scratch_path else args.db

    if args.dry_run:
        print(f"\nDRY RUN — would run {len(candles)} bars x 2 modes (shadow + active)")
        print(f"  Symbol: {args.symbol}, read DB: {args.db}")
        if args.scratch_db:
            print("  Writes: isolated temp copy (--scratch-db)")
        else:
            print(f"  Writes: same as read DB ({args.db})")
        print(f"  ML model: {config.ml_model_path}")
        print(f"  Swarm agents: {config.swarm_agents}")
        return 0

    from dataclasses import replace
    bot_cfg = replace(config, timeframe="1h")
    _symbol = args.symbol

    print(f"\n{'=' * 60}")
    print("SWARM CERTIFICATION BACKTEST")
    print(f"  {len(candles)} bars | {args.symbol} | ML: {config.ml_model_path}")
    print(f"{'=' * 60}\n")

    logger.info("Run 1/2: SHADOW mode (baseline)...")
    t0 = time.perf_counter()
    shadow = _run_backtest(candles, ml_model, write_db, "shadow", macro_sitout, bot_cfg, _symbol)
    t_shadow = time.perf_counter() - t0
    logger.info("Shadow done in %.0fs", t_shadow)

    logger.info("Run 2/2: ACTIVE mode (swarm controls)...")
    t0 = time.perf_counter()
    active = _run_backtest(candles, ml_model, write_db, "active", macro_sitout, bot_cfg, _symbol)
    t_active = time.perf_counter() - t0
    logger.info("Active done in %.0fs", t_active)

    s_ret = shadow.total_return_pct
    s_dd = shadow.max_drawdown_pct
    s_sh = shadow.sharpe_ratio or 0.0
    s_tr = shadow.trades
    s_wr = shadow.win_rate

    a_ret = active.total_return_pct
    a_dd = active.max_drawdown_pct
    a_sh = active.sharpe_ratio or 0.0
    a_tr = active.trades
    a_wr = active.win_rate

    print(f"\n{'=' * 70}")
    print("CERTIFICATION RESULTS")
    print(f"{'=' * 70}")
    print(f"{'Metric':<22} {'Shadow (baseline)':>18} {'Active (swarm)':>18} {'Delta':>12}")
    print(f"{'-' * 70}")
    print(f"{'Trades':<22} {s_tr:>18} {a_tr:>18} {a_tr - s_tr:>+12}")
    print(f"{'Win Rate':<22} {s_wr:>17.1%} {a_wr:>17.1%} {a_wr - s_wr:>+11.1%}")
    print(f"{'Return %':<22} {s_ret:>+17.2f}% {a_ret:>+17.2f}% {a_ret - s_ret:>+11.2f}%")
    print(f"{'Max Drawdown':<22} {s_dd:>17.2f}% {a_dd:>17.2f}% {a_dd - s_dd:>+11.2f}%")
    print(f"{'Sharpe':<22} {s_sh:>18.2f} {a_sh:>18.2f} {a_sh - s_sh:>+12.2f}")
    print(f"{'=' * 70}")

    if a_tr == 0:
        print("\nWARNING: Swarm ACTIVE mode produced 0 trades.")
        print("  Common causes:")
        print("  - All swarm agents in advisory_only / quarantine (check swarm_agent_modes in DB).")
        print("  - Persistent vetoes (data_guardian gaps, risk_steward drawdown).")
        print("  - Gated policy never emits buy/sell while swarm only confirms gated trades")
        print("    (default: swarm_active_allow_new_signals=false).")

    swarm_better = a_sh > s_sh and a_ret > s_ret
    if swarm_better:
        print("\nVERDICT: SWARM IMPROVES PERFORMANCE")
        print("  Consider promoting to active mode (HOGAN_SWARM_MODE=active)")
    elif a_sh > s_sh:
        print("\nVERDICT: SWARM IMPROVES RISK-ADJUSTED RETURNS (higher Sharpe)")
        print("  But absolute returns are lower. Review trade-by-trade before promoting.")
    elif a_dd < s_dd:
        print("\nVERDICT: SWARM REDUCES DRAWDOWN")
        print("  But returns are lower. Swarm is more conservative than baseline.")
    else:
        print("\nVERDICT: BASELINE OUTPERFORMS SWARM")
        print("  Keep swarm in shadow mode. Investigate veto patterns for improvements.")

    if scratch_path:
        if args.keep_scratch_db:
            print(f"\nSwarm certification writes kept at: {scratch_path}")
            print("  (dashboard will not see these unless you point it at this file)")
        else:
            print(f"\nSwarm writes went to scratch DB (deleted on exit): {scratch_path}")
            print(f"  Production DB unchanged: {args.db}")
    else:
        print(f"\nSwarm decisions written to {args.db} — dashboard will update on next refresh.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
