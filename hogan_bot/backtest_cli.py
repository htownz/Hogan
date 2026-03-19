from __future__ import annotations

import argparse
import json

from hogan_bot.backtest import (
    diagnose_exits,
    diagnose_long_entries,
    diagnose_longs_by_confidence,
    diagnose_shorts_by_confidence,
    evaluate_market_regimes,
    evaluate_regimes_by_market,
    evaluate_trades_by_regime_side,
    run_backtest_on_candles,
)
from hogan_bot.config import load_config
from hogan_bot.exchange import ExchangeClient
from hogan_bot.ml import load_model
from hogan_bot.profiles import PROFILES, apply_profile, get_profile

# RL policy is loaded lazily so that missing stable-baselines3 doesn't break
# non-RL backtests.
_RL_POLICY_CACHE: dict[str, object] = {}

# ---------------------------------------------------------------------------
# Pre-defined strategy configurations for --compare mode.
# Each entry is (label, kwargs_override) where kwargs_override is merged over
# the base config values.
# ---------------------------------------------------------------------------
_COMPARE_CONFIGS: list[tuple[str, dict]] = [
    ("ma_only", {"use_ema_clouds": False, "use_fvg": False, "signal_mode": "ma_only"}),
    ("clouds_any", {"use_ema_clouds": True, "use_fvg": False, "signal_mode": "any"}),
    ("fvg_any", {"use_ema_clouds": False, "use_fvg": True, "signal_mode": "any"}),
    ("clouds_fvg_any", {"use_ema_clouds": True, "use_fvg": True, "signal_mode": "any"}),
    ("clouds_fvg_all", {"use_ema_clouds": True, "use_fvg": True, "signal_mode": "all"}),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run Hogan backtest on Kraken OHLCV or local DB")
    parser.add_argument("--profile", choices=list(PROFILES.keys()), default=None,
                        help="Named strategy profile (overrides config defaults)")
    parser.add_argument("--symbol", default="BTC/USD")
    parser.add_argument("--timeframe", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--from-db", action="store_true",
                        help="Load candles from local SQLite DB (data/hogan.db) instead of fetching live")
    parser.add_argument("--db", default="data/hogan.db",
                        help="SQLite DB path (used with --from-db)")
    parser.add_argument("--use-ml", action="store_true")
    parser.add_argument("--use-ict", action="store_true", help="Enable ICT signal pillars")
    parser.add_argument("--use-rl", action="store_true", help="Enable trained RL agent vote (requires models/hogan_rl_policy.zip)")
    parser.add_argument(
        "--regime-report",
        action="store_true",
        help="Print per-regime (bull/bear/sideways) performance breakdown after the backtest",
    )
    parser.add_argument(
        "--compare",
        action="store_true",
        help="Sweep 5 signal configurations and print a comparison table instead of a single result",
    )
    parser.add_argument(
        "--enable-shorts",
        action="store_true",
        help="Enable short selling in backtest (sell signals open shorts when flat)",
    )
    parser.add_argument(
        "--compare-shorts",
        action="store_true",
        help="Run three configs side-by-side: long-only, shorts+regime-gated, shorts-ungated",
    )
    parser.add_argument(
        "--mtf-exec",
        action="store_true",
        help="Use 1h thesis / 15m execution: defer entries to 15m pullback timing",
    )
    parser.add_argument(
        "--mtf-thesis-age",
        type=int,
        default=4,
        help="Max 1h bars a thesis stays active before expiring (default 4)",
    )
    parser.add_argument(
        "--no-pullback-gate",
        action="store_true",
        help="Disable the anti-chase pullback gate (for A/B comparison)",
    )
    parser.add_argument(
        "--enable-close-and-reverse",
        action="store_true",
        help="Enable close-and-reverse (sell signal can open short on same bar as long close; off by default)",
    )
    parser.add_argument(
        "--short-max-hold-hours",
        type=float,
        default=None,
        help="Explicit short max hold in hours (default: use config value or same as long max hold)",
    )
    parser.add_argument(
        "--no-policy-core",
        action="store_true",
        help="Force legacy decision path (bypass policy_core.decide and swarm)",
    )
    return parser.parse_args()


def _load_rl_policy(model_path: str):
    if model_path not in _RL_POLICY_CACHE:
        from hogan_bot.rl_agent import load_rl_policy
        _RL_POLICY_CACHE[model_path] = load_rl_policy(model_path)
    return _RL_POLICY_CACHE[model_path]


def _run_single(cfg, candles, symbol, ml_model, timeframe: str | None = None, overrides: dict | None = None, use_ict: bool = False, use_rl_agent: bool = False, rl_policy=None, db_path: str | None = None, enable_shorts: bool = False, candles_15m=None, mtf_thesis_max_age: int = 4, enable_pullback_gate: bool = True, enable_close_and_reverse: bool = False, short_max_hold_hours: float | None = None, use_policy_core: bool = True):
    """Run one backtest with optional per-key overrides on *cfg*.

    Returns the full :class:`~hogan_bot.backtest.BacktestResult` object so
    callers can access ``equity_curve`` for regime analysis.
    """
    ov = overrides or {}
    return run_backtest_on_candles(
        candles=candles,
        symbol=symbol,
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
        use_ema_clouds=ov.get("use_ema_clouds", cfg.use_ema_clouds),
        ema_fast_short=cfg.ema_fast_short,
        ema_fast_long=cfg.ema_fast_long,
        ema_slow_short=cfg.ema_slow_short,
        ema_slow_long=cfg.ema_slow_long,
        use_fvg=ov.get("use_fvg", cfg.use_fvg),
        fvg_min_gap_pct=cfg.fvg_min_gap_pct,
        signal_mode=ov.get("signal_mode", cfg.signal_mode),
        min_vote_margin=cfg.signal_min_vote_margin,
        trailing_stop_pct=cfg.trailing_stop_pct,
        take_profit_pct=cfg.take_profit_pct,
        ml_confidence_sizing=cfg.ml_confidence_sizing,
        atr_stop_multiplier=cfg.atr_stop_multiplier,
        use_ict=ov.get("use_ict", use_ict or cfg.use_ict),
        ict_swing_left=cfg.ict_swing_left,
        ict_swing_right=cfg.ict_swing_right,
        ict_eq_tolerance_pct=cfg.ict_eq_tolerance_pct,
        ict_min_displacement_pct=cfg.ict_min_displacement_pct,
        ict_require_time_window=cfg.ict_require_time_window,
        ict_time_windows=cfg.ict_time_windows,
        ict_require_pd=cfg.ict_require_pd,
        ict_ote_enabled=cfg.ict_ote_enabled,
        ict_ote_low=cfg.ict_ote_low,
        ict_ote_high=cfg.ict_ote_high,
        use_rl_agent=use_rl_agent or cfg.use_rl_agent,
        rl_policy=rl_policy,
        max_hold_hours=cfg.max_hold_hours if cfg.max_hold_hours > 0 else None,
        loss_cooldown_hours=cfg.loss_cooldown_hours if cfg.loss_cooldown_hours > 0 else None,
        min_edge_multiple=cfg.min_edge_multiple,
        min_final_confidence=cfg.min_final_confidence,
        min_tech_confidence=cfg.min_tech_confidence,
        min_regime_confidence=cfg.min_regime_confidence,
        max_whipsaws=cfg.max_whipsaws,
        reversal_confidence_mult=cfg.reversal_confidence_multiplier,
        db_path=db_path,
        enable_shorts=enable_shorts,
        candles_15m=candles_15m,
        mtf_thesis_max_age=mtf_thesis_max_age,
        enable_pullback_gate=enable_pullback_gate,
        enable_close_and_reverse=enable_close_and_reverse,
        short_max_hold_hours=short_max_hold_hours if short_max_hold_hours is not None else cfg.short_max_hold_hours,
        use_policy_core=use_policy_core,
        use_ml_as_sizer=cfg.use_ml_as_sizer,
        exit_drawdown_pct=cfg.exit_drawdown_pct,
        exit_time_decay=cfg.exit_time_decay,
        exit_vol_expansion=cfg.exit_vol_expansion,
        exit_stagnation_bars=cfg.exit_stagnation_bars,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config()
    from hogan_bot.champion import apply_champion_mode, is_champion_mode
    if is_champion_mode():
        cfg = apply_champion_mode(cfg)

    profile_cli: dict[str, object] = {}
    if args.profile:
        profile = get_profile(args.profile)
        cfg, profile_cli = apply_profile(cfg, profile)
        print(f"Using profile: {args.profile}")

    timeframe = args.timeframe or profile_cli.get("timeframe") or cfg.timeframe
    limit = args.limit or cfg.ohlcv_limit

    if args.from_db:
        from hogan_bot.storage import get_connection
        from hogan_bot.storage import load_candles as _load_candles
        conn = get_connection(args.db)
        candles = _load_candles(conn, args.symbol, timeframe, limit=limit)
        conn.close()
        if candles.empty:
            print(f"No candles found in DB for {args.symbol}/{timeframe}. Run refresh_daily.py first.")
            return
        print(f"Loaded {len(candles)} candles from local DB ({args.symbol}/{timeframe})")
    else:
        client = ExchangeClient(cfg.exchange_id, cfg.kraken_api_key, cfg.kraken_api_secret)
        candles = client.fetch_ohlcv_df(args.symbol, timeframe=timeframe, limit=limit)

    _use_ml = args.use_ml or (cfg.use_ml_filter if args.profile else False) or getattr(cfg, "use_ml_as_sizer", False)
    ml_model = load_model(cfg.ml_model_path) if _use_ml else None

    use_rl = args.use_rl or cfg.use_rl_agent
    rl_policy = _load_rl_policy(cfg.rl_model_path) if use_rl else None

    _db = args.db if args.from_db else None

    _enable_shorts = args.enable_shorts or profile_cli.get("enable_shorts", False)
    _enable_pullback = (not args.no_pullback_gate) and profile_cli.get("enable_pullback_gate", True)
    _enable_car = args.enable_close_and_reverse or profile_cli.get("enable_close_and_reverse", False)
    _use_pc = not args.no_policy_core and cfg.use_policy_core

    _candles_15m = None
    if args.mtf_exec and args.from_db:
        from hogan_bot.storage import get_connection
        from hogan_bot.storage import load_candles as _load_candles
        conn = get_connection(args.db)
        _candles_15m = _load_candles(conn, args.symbol, "15m")
        conn.close()
        if _candles_15m.empty:
            print("No 15m candles in DB. Run backfill_15m.py first.")
            return
        print(f"Loaded {len(_candles_15m)} candles for 15m execution timing")

    if args.compare_shorts:
        _compare_short_configs = [
            ("long_only", False),
            ("shorts_regime_gated", True),
        ]
        rows = []
        for label, shorts_on in _compare_short_configs:
            result = _run_single(
                cfg, candles, args.symbol, ml_model,
                timeframe=timeframe,
                use_ict=args.use_ict, use_rl_agent=use_rl, rl_policy=rl_policy,
                db_path=_db, enable_shorts=shorts_on,
                candles_15m=_candles_15m, mtf_thesis_max_age=args.mtf_thesis_age,
                enable_pullback_gate=_enable_pullback,
                enable_close_and_reverse=_enable_car,
                short_max_hold_hours=args.short_max_hold_hours,
                use_policy_core=_use_pc,
            )
            summary = result.summary_dict()
            funnel = summary.pop("signal_funnel", {})
            summary["config"] = label
            summary["long_entries"] = funnel.get("executed_buy", 0)
            summary["short_entries"] = funnel.get("executed_short_entry", 0)
            summary["regime_blocked_shorts"] = funnel.get("blocked_regime_no_shorts", 0)
            rows.append(summary)

        _print_table(rows)
        for row in rows:
            print(f"  {row['config']:<24s}  longs={row.get('long_entries', 0):>3d}  "
                  f"shorts={row.get('short_entries', 0):>3d}  "
                  f"regime_blocked={row.get('regime_blocked_shorts', 0):>3d}")
        print()
    elif args.compare:
        rows = []
        for label, overrides in _COMPARE_CONFIGS:
            result = _run_single(
                cfg, candles, args.symbol, ml_model,
                timeframe=timeframe, overrides=overrides,
                use_ict=args.use_ict, use_rl_agent=use_rl, rl_policy=rl_policy,
                db_path=_db, enable_shorts=_enable_shorts,
                candles_15m=_candles_15m, mtf_thesis_max_age=args.mtf_thesis_age,
                enable_pullback_gate=_enable_pullback,
                enable_close_and_reverse=_enable_car,
                short_max_hold_hours=args.short_max_hold_hours,
                use_policy_core=_use_pc,
            )
            rows.append({"config": label, **result.summary_dict()})

        print(json.dumps(rows, indent=2))
        _print_table(rows)
    else:
        result = _run_single(
            cfg, candles, args.symbol, ml_model,
            timeframe=timeframe,
            use_ict=args.use_ict, use_rl_agent=use_rl, rl_policy=rl_policy,
            db_path=_db, enable_shorts=_enable_shorts,
            candles_15m=_candles_15m, mtf_thesis_max_age=args.mtf_thesis_age,
            enable_pullback_gate=_enable_pullback,
            enable_close_and_reverse=_enable_car,
            short_max_hold_hours=args.short_max_hold_hours,
            use_policy_core=_use_pc,
        )
        print(json.dumps(result.summary_dict(), indent=2))

        # Signal funnel diagnostics
        if result.signal_funnel:
            _print_signal_funnel(result.signal_funnel)

        if args.regime_report:
            # Bar-level equity curve segmented by actual market regime
            by_market = evaluate_regimes_by_market(result)
            if by_market:
                print("\n-- Performance by market regime (detect_regime) --------")
                print(f"  {'regime':<14s}  {'bars':>5s}  {'return%':>8s}  {'maxdd%':>7s}  {'sharpe':>8s}  {'avg_bar%':>9s}")
                print(f"  {'-'*14}  {'-'*5}  {'-'*8}  {'-'*7}  {'-'*8}  {'-'*9}")
                for regime, m in by_market.items():
                    sharpe_s = f"{m['sharpe']:.4f}" if m['sharpe'] is not None else "N/A"
                    print(f"  {regime:<14s}  {m['bars']:>5d}  {m['total_return_pct']:>8.3f}  "
                          f"{m['max_drawdown_pct']:>7.3f}  {sharpe_s:>8s}  {m['avg_bar_return_pct']:>9.5f}")
                print()

            # Trade-level analytics by entry regime
            market_regimes = evaluate_market_regimes(result)
            if market_regimes:
                print("-- Trade analytics by entry regime ---------------------")
                for regime, metrics in sorted(market_regimes.items()):
                    _p = metrics.get("payoff_ratio", 0)
                    _ps = f"{_p:>5.2f}" if isinstance(_p, (int, float)) else f"{_p:>5s}"
                    print(f"  {regime:<14s}  trades={metrics.get('trade_count', 0):>3d}  "
                          f"win_rate={metrics.get('win_rate', 0):>5.1%}  "
                          f"avg_pnl={metrics.get('avg_gross_pnl_pct', 0):>7.2f}%  "
                          f"payoff={_ps}")
                print()

            # Regime x side breakdown
            by_regime_side = evaluate_trades_by_regime_side(result)
            if by_regime_side:
                print("-- Trade analytics by regime x side --------------------")
                print(f"  {'bucket':<22s}  {'n':>3s}  {'win%':>5s}  {'avg_pnl%':>8s}  {'total%':>7s}  {'payoff':>6s}  exit_reasons")
                print(f"  {'-'*22}  {'-'*3}  {'-'*5}  {'-'*8}  {'-'*7}  {'-'*6}  {'-'*20}")
                for key, m in by_regime_side.items():
                    payoff_s = f"{m['payoff_ratio']:>6.2f}" if m["payoff_ratio"] != "inf" else "   inf"
                    exits = "  ".join(f"{k}={v}" for k, v in sorted(m["exit_reasons"].items()))
                    print(f"  {key:<22s}  {m['count']:>3d}  {m['win_rate']:>5.1%}  "
                          f"{m['avg_pnl_pct']:>8.2f}  {m['total_pnl_pct']:>7.2f}  {payoff_s}  {exits}")
                print()

        diag = diagnose_long_entries(result.closed_trades)
        if diag and diag.get("total_longs", 0) > 0:
            print("-- Long entry timing diagnostic -------------------------")
            for label, group_key in [("ALL LONGS", "all"), ("WINNERS", "winners"), ("LOSERS", "losers")]:
                g = diag.get(group_key, {})
                if isinstance(g, dict) and "avg_range_position" in g:
                    cnt = g.get("count", diag.get("total_longs", "?"))
                    print(f"  {label} (n={cnt}):")
                    print(f"    Avg range position:  {g['avg_range_position']:.2f}  (0=low, 1=high)")
                    print(f"    Avg % from high:     {g['avg_pct_from_high']:+.2f}%")
                    print(f"    Avg run-up before:   {g['avg_run_up_before']:+.2f}%")
                    print(f"    Avg MFE:             {g['avg_mfe']:+.2f}%")
                    print(f"    Avg MAE:             {g['avg_mae']:+.2f}%")
                    if "avg_bars_to_peak" in g:
                        print(f"    Avg bars to peak:    {g['avg_bars_to_peak']:.1f}")
                        print(f"    Avg bars to trough:  {g['avg_bars_to_trough']:.1f}")
            print()
            per_trade = diag.get("per_trade", [])
            if per_trade:
                print("  Per-trade detail:")
                print(f"  {'bar':>5s}  {'regime':<15s}  {'pnl%':>6s}  {'range':>5s}  {'runup':>6s}  "
                      f"{'MFE':>6s}  {'MAE':>6s}  {'pk_bar':>6s}  exit")
                print(f"  {'-'*5}  {'-'*15}  {'-'*6}  {'-'*5}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*6}  {'-'*15}")
                for t in per_trade:
                    print(f"  {t['bar']:>5d}  {(t['regime'] or '?'):<15s}  {t['pnl_pct']:>+6.2f}  "
                          f"{t['range_pos']:>5.2f}  {t['run_up']:>+6.2f}  "
                          f"{t['mfe']:>+6.2f}  {t['mae']:>+6.2f}  {t['bars_to_peak']:>6d}  {t['exit'] or '?'}")
            print()

        # Exit diagnostic: trailing stop analysis
        exit_diag = diagnose_exits(result.closed_trades)
        if exit_diag:
            print("-- Exit diagnostic: trailing stop analysis ---------------")
            for side_label, key in [("LONG STOPS", "long_stops"), ("SHORT STOPS", "short_stops")]:
                s = exit_diag.get(key, {})
                if not s:
                    continue
                print(f"  {side_label} (n={s['count']}):")
                print(f"    Premature (price recovered/hit TP):  {s['premature']}")
                print(f"    Correct (price kept falling):        {s['correct']}")
                print(f"    Early but OK (minor bounce >1%):     {s['early_but_ok']}")
                print(f"    Would have hit take-profit:          {s['would_have_hit_tp']}")
                print(f"    Avg trade PnL at stop:   {s['avg_trade_pnl_pct']:>+7.3f}%")
                print(f"    Avg post-exit MFE:       {s['avg_post_exit_mfe_pct']:>+7.3f}%  (money left on table)")
                print(f"    Avg post-exit MAE:       {s['avg_post_exit_mae_pct']:>+7.3f}%  (avoided loss)")
                print(f"    Avg post-exit final:     {s['avg_post_exit_final_pct']:>+7.3f}%  (where price ended)")
                print()

            per_stop = exit_diag.get("per_trade", [])
            if per_stop:
                print("  Per-trade stop detail:")
                print(f"  {'bar':>5s}  {'side':<6s} {'regime':<14s} {'pnl%':>6s}  {'inMFE':>6s}  "
                      f"{'postMFE':>7s}  {'postMAE':>7s}  {'final':>6s}  {'recover':>7s}  verdict")
                print(f"  {'-'*5}  {'-'*6} {'-'*14} {'-'*6}  {'-'*6}  {'-'*7}  {'-'*7}  {'-'*6}  {'-'*7}  {'-'*12}")
                for t in per_stop:
                    print(f"  {t['bar'] or 0:>5d}  {t['side']:<6s} {(t['regime'] or '?'):<14s} "
                          f"{t['pnl_pct']:>+6.2f}  {t.get('in_trade_mfe', 0) or 0:>+6.2f}  "
                          f"{t['post_mfe']:>+7.3f}  {t['post_mae']:>+7.3f}  {t['post_final']:>+6.3f}  "
                          f"{'yes' if t['recovered'] else 'no':>7s}  {t['verdict']}")
            print()

        # Long confidence bucket analysis
        long_conf = diagnose_longs_by_confidence(result.closed_trades)
        if long_conf:
            by_rc = long_conf.get("by_regime_confidence", {})
            if by_rc:
                print("-- Long trades by regime x confidence bucket -----------")
                print(f"  {'bucket':<26s}  {'n':>3s}  {'win%':>5s}  {'avg_pnl%':>8s}  {'total%':>7s}")
                print(f"  {'-'*26}  {'-'*3}  {'-'*5}  {'-'*8}  {'-'*7}")
                for key, m in by_rc.items():
                    print(f"  {key:<26s}  {m['count']:>3d}  {m['win_rate']:>5.1%}  "
                          f"{m['avg_pnl_pct']:>8.2f}  {m['total_pnl_pct']:>7.2f}")
                print()

            per_long = long_conf.get("per_trade", [])
            if per_long:
                print("  Per-trade long detail (with confidence):")
                print(f"  {'bar':>5s}  {'regime':<14s}  {'conf':>5s}  {'bucket':<7s}  {'pnl%':>6s}  {'hold':>4s}  exit")
                print(f"  {'-'*5}  {'-'*14}  {'-'*5}  {'-'*7}  {'-'*6}  {'-'*4}  {'-'*20}")
                for t in per_long:
                    print(f"  {t['bar'] or 0:>5d}  {(t['regime'] or '?'):<14s}  "
                          f"{t['regime_conf']:>5.3f}  {t['conf_bucket']:<7s}  "
                          f"{t['pnl_pct']:>+6.2f}  {t['hold_bars']:>4d}  {t['exit'] or '?'}")
                print()

        # Short confidence bucket analysis
        short_conf = diagnose_shorts_by_confidence(result.closed_trades)
        if short_conf:
            by_rc = short_conf.get("by_regime_confidence", {})
            if by_rc:
                print("-- Short trades by regime x confidence bucket ----------")
                print(f"  {'bucket':<26s}  {'n':>3s}  {'win%':>5s}  {'avg_pnl%':>8s}  {'total%':>7s}")
                print(f"  {'-'*26}  {'-'*3}  {'-'*5}  {'-'*8}  {'-'*7}")
                for key, m in by_rc.items():
                    print(f"  {key:<26s}  {m['count']:>3d}  {m['win_rate']:>5.1%}  "
                          f"{m['avg_pnl_pct']:>8.2f}  {m['total_pnl_pct']:>7.2f}")
                print()

            per_short = short_conf.get("per_trade", [])
            if per_short:
                print("  Per-trade short detail (with confidence):")
                print(f"  {'bar':>5s}  {'regime':<14s}  {'conf':>5s}  {'bucket':<7s}  {'pnl%':>6s}  {'hold':>4s}  exit")
                print(f"  {'-'*5}  {'-'*14}  {'-'*5}  {'-'*7}  {'-'*6}  {'-'*4}  {'-'*20}")
                for t in per_short:
                    print(f"  {t['bar'] or 0:>5d}  {(t['regime'] or '?'):<14s}  "
                          f"{t['regime_conf']:>5.3f}  {t['conf_bucket']:<7s}  "
                          f"{t['pnl_pct']:>+6.2f}  {t['hold_bars']:>4d}  {t['exit'] or '?'}")
                print()


def _print_signal_funnel(funnel: dict) -> None:
    """Print a human-readable signal funnel showing where signals die."""
    bars = funnel.get("bars_evaluated", 0)
    if bars == 0:
        return
    print(f"\n-- Signal funnel ({bars} bars evaluated) -------------------")

    stages = [
        ("Pipeline output", "pipeline_buy", "pipeline_sell"),
        ("After ML filter", "post_ml_buy", "post_ml_sell"),
        ("After edge gate", "post_edge_buy", "post_edge_sell"),
        ("After quality gate", "post_quality_buy", "post_quality_sell"),
        ("After ranging gate", "post_ranging_buy", "post_ranging_sell"),
    ]

    for label, buy_key, sell_key in stages:
        b = funnel.get(buy_key, 0)
        s = funnel.get(sell_key, 0)
        buy_pct = b / bars * 100 if bars else 0
        sell_pct = s / bars * 100 if bars else 0
        print(f"  {label:<22s}  buy={b:>5d} ({buy_pct:>5.1f}%)  sell={s:>5d} ({sell_pct:>5.1f}%)")

    executed = funnel.get("executed_buy", 0)
    exec_short = funnel.get("executed_short_entry", 0)
    blocked_long = funnel.get("blocked_already_long", 0)
    blocked_short = funnel.get("blocked_already_short", 0)
    blocked_cd = funnel.get("blocked_cooldown", 0)
    if exec_short:
        print(f"  {'Executed entries':<22s}  long={executed:>4d}  short={exec_short:>4d}")
    else:
        print(f"  {'Executed entries':<22s}  buy={executed:>5d}")
    regime_no_long = funnel.get("blocked_regime_no_longs", 0)
    regime_no_short = funnel.get("blocked_regime_no_shorts", 0)
    if any([blocked_long, blocked_short, blocked_cd, regime_no_long, regime_no_short]):
        if blocked_long:
            print(f"  {'Blocked (already long)':<24s}     {blocked_long:>5d}")
        if blocked_short:
            print(f"  {'Blocked (already short)':<24s}    {blocked_short:>5d}")
        if blocked_cd:
            print(f"  {'Blocked (cooldown)':<24s}     {blocked_cd:>5d}")
        if regime_no_long:
            print(f"  {'Blocked (regime no long)':<24s}   {regime_no_long:>5d}")
        if regime_no_short:
            print(f"  {'Blocked (regime no short)':<24s}  {regime_no_short:>5d}")
    close_reverse = funnel.get("close_and_reverse", 0)
    if close_reverse:
        print(f"  {'Close-and-reverse':<24s}       {close_reverse:>5d}")
    pullback_blocked = funnel.get("pullback_blocked", 0)
    pullback_blocked_resistance = funnel.get("pullback_blocked_resistance", 0)
    pullback_halved = funnel.get("pullback_halved", 0)
    if pullback_blocked or pullback_halved:
        print("\n  Pullback gate:")
        if pullback_blocked:
            print(f"    Blocked total:       {pullback_blocked:>5d}")
            if pullback_blocked_resistance:
                print(f"      regime resistance: {pullback_blocked_resistance:>5d}")
            _chase_blocked = pullback_blocked - pullback_blocked_resistance
            if _chase_blocked:
                print(f"      chasing run-up:    {_chase_blocked:>5d}")
        if pullback_halved:
            print(f"    Half-sized (near top):{pullback_halved:>5d}")

    mtf_created = funnel.get("mtf_thesis_created", 0)
    mtf_executed = funnel.get("mtf_thesis_executed", 0)
    mtf_expired = funnel.get("mtf_thesis_expired", 0)
    mtf_15m = funnel.get("mtf_15m_entry_used", 0)
    if mtf_created:
        print("\n  MTF thesis/execution:")
        print(f"    Theses created:      {mtf_created:>5d}")
        print(f"    Theses executed:     {mtf_executed:>5d}")
        print(f"    Theses expired:      {mtf_expired:>5d}")
        print(f"    15m entries used:    {mtf_15m:>5d}")

    s_sig = funnel.get("short_covered_signal", 0)
    s_stop = funnel.get("short_covered_stop", 0)
    s_tp = funnel.get("short_covered_tp", 0)
    s_hold = funnel.get("short_covered_max_hold", 0)
    if any([s_sig, s_stop, s_tp, s_hold]):
        print("\n  Short exits breakdown:")
        if s_sig:
            print(f"    Buy signal cover:    {s_sig:>5d}")
        if s_stop:
            print(f"    Trailing stop:       {s_stop:>5d}")
        if s_tp:
            print(f"    Take profit:         {s_tp:>5d}")
        if s_hold:
            print(f"    Max hold time:       {s_hold:>5d}")

    edge_atr = funnel.get("edge_blocked_atr", 0)
    edge_tp = funnel.get("edge_blocked_tp", 0)
    edge_fc = funnel.get("edge_blocked_forecast", 0)
    edge_sp = funnel.get("edge_blocked_spread", 0)
    if any([edge_atr, edge_tp, edge_fc, edge_sp]):
        print("\n  Edge gate breakdown:")
        if edge_atr:
            print(f"    ATR too low:         {edge_atr:>5d}")
        if edge_tp:
            print(f"    TP too low:          {edge_tp:>5d}")
        if edge_fc:
            print(f"    Forecast too low:    {edge_fc:>5d}")
        if edge_sp:
            print(f"    Spread too wide:     {edge_sp:>5d}")

    rg_tech = funnel.get("ranging_blocked_tech", 0)
    rg_ml = funnel.get("ranging_blocked_ml", 0)
    rg_whip = funnel.get("ranging_blocked_whipsaw", 0)
    if any([rg_tech, rg_ml, rg_whip]):
        print("\n  Ranging gate breakdown:")
        if rg_tech:
            print(f"    Tech disagree:       {rg_tech:>5d}")
        if rg_ml:
            print(f"    ML indifference:     {rg_ml:>5d}")
        if rg_whip:
            print(f"    Whipsaw block:       {rg_whip:>5d}")

    ml_stats = funnel.get("ml_prob_stats")
    if ml_stats:
        print("\n  ML probability distribution:")
        print(f"    mean={ml_stats['mean']:.4f}  std={ml_stats['std']:.4f}  "
              f"median={ml_stats['median']:.4f}")
        print(f"    [min={ml_stats['min']:.4f}  p10={ml_stats['p10']:.4f}  "
              f"p25={ml_stats['p25']:.4f}  p75={ml_stats['p75']:.4f}  "
              f"p90={ml_stats['p90']:.4f}  max={ml_stats['max']:.4f}]")
        print(f"    above buy threshold:  {ml_stats['pct_above_buy_thresh']:.1f}%")
        print(f"    below sell threshold: {ml_stats['pct_below_sell_thresh']:.1f}%")

    regime_dist = funnel.get("regime_distribution")
    if regime_dist:
        total_r = sum(regime_dist.values())
        print("\n  Market regime distribution (detect_regime):")
        for regime, count in sorted(regime_dist.items(), key=lambda x: -x[1]):
            pct = count / total_r * 100 if total_r else 0
            print(f"    {regime:<14s}  {count:>5d} bars ({pct:>5.1f}%)")

    print()


def _print_table(rows: list[dict]) -> None:
    """Print a fixed-width comparison table to stdout."""
    columns = [
        ("config", 18),
        ("total_return_pct", 14),
        ("max_drawdown_pct", 14),
        ("sharpe_ratio", 10),
        ("sortino_ratio", 11),
        ("calmar_ratio", 10),
        ("win_rate", 8),
        ("trades", 6),
    ]
    header = "  ".join(col.ljust(width) for col, width in columns)
    separator = "  ".join("-" * width for _, width in columns)
    print("\n" + header)
    print(separator)
    for row in rows:
        parts = []
        for col, width in columns:
            val = row.get(col)
            if val is None:
                parts.append("N/A".ljust(width))
            elif isinstance(val, float):
                parts.append(f"{val:.4f}".ljust(width))
            else:
                parts.append(str(val).ljust(width))
        print("  ".join(parts))
    print()


if __name__ == "__main__":
    main()
