from __future__ import annotations

import argparse
import json

from hogan_bot.backtest import evaluate_market_regimes, evaluate_regimes, run_backtest_on_candles
from hogan_bot.config import load_config
from hogan_bot.exchange import ExchangeClient
from hogan_bot.ml import load_model

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
    return parser.parse_args()


def _load_rl_policy(model_path: str):
    if model_path not in _RL_POLICY_CACHE:
        from hogan_bot.rl_agent import load_rl_policy
        _RL_POLICY_CACHE[model_path] = load_rl_policy(model_path)
    return _RL_POLICY_CACHE[model_path]


def _run_single(cfg, candles, symbol, ml_model, timeframe: str | None = None, overrides: dict | None = None, use_ict: bool = False, use_rl_agent: bool = False, rl_policy=None, db_path: str | None = None):
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
        db_path=db_path,
    )


def main() -> None:
    args = parse_args()
    cfg = load_config()
    from hogan_bot.champion import apply_champion_mode, is_champion_mode
    if is_champion_mode():
        cfg = apply_champion_mode(cfg)

    timeframe = args.timeframe or cfg.timeframe
    limit = args.limit or cfg.ohlcv_limit

    if args.from_db:
        from hogan_bot.storage import get_connection, load_candles as _load_candles
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

    ml_model = load_model(cfg.ml_model_path) if args.use_ml else None

    use_rl = args.use_rl or cfg.use_rl_agent
    rl_policy = _load_rl_policy(cfg.rl_model_path) if use_rl else None

    _db = args.db if args.from_db else None

    if args.compare:
        rows = []
        for label, overrides in _COMPARE_CONFIGS:
            result = _run_single(
                cfg, candles, args.symbol, ml_model,
                timeframe=timeframe, overrides=overrides,
                use_ict=args.use_ict, use_rl_agent=use_rl, rl_policy=rl_policy,
                db_path=_db,
            )
            rows.append({"config": label, **result.summary_dict()})

        print(json.dumps(rows, indent=2))
        _print_table(rows)
    else:
        result = _run_single(
            cfg, candles, args.symbol, ml_model,
            timeframe=timeframe,
            use_ict=args.use_ict, use_rl_agent=use_rl, rl_policy=rl_policy,
            db_path=_db,
        )
        print(json.dumps(result.summary_dict(), indent=2))

        # Signal funnel diagnostics
        if result.signal_funnel:
            _print_signal_funnel(result.signal_funnel)

        if args.regime_report:
            # Market-condition regime breakdown (from detect_regime)
            market_regimes = evaluate_market_regimes(result)
            if market_regimes:
                print("\n-- Market regime trade analytics -----------------------")
                for regime, metrics in sorted(market_regimes.items()):
                    print(f"  {regime:<14s}  trades={metrics.get('trade_count', 0):>3d}  "
                          f"win_rate={metrics.get('win_rate', 0):>5.1%}  "
                          f"avg_pnl={metrics.get('avg_gross_pnl_pct', 0):>7.2f}%  "
                          f"payoff={metrics.get('payoff_ratio', 0):>5.2f}")
                print()

            # Equity-curve regime breakdown (legacy)
            regimes = evaluate_regimes(result)
            print("-- Equity-curve regime breakdown (legacy) --------------")
            for regime, metrics in regimes.items():
                print(f"  {regime:<10s}  bars={metrics['bars']:>5d}  "
                      f"sharpe={str(metrics['sharpe']):<8s}  "
                      f"return={metrics['total_return_pct']:>7.2f}%  "
                      f"maxdd={metrics['max_drawdown_pct']:>6.2f}%")
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

    prev_buy, prev_sell = bars, bars
    for label, buy_key, sell_key in stages:
        b = funnel.get(buy_key, 0)
        s = funnel.get(sell_key, 0)
        buy_pct = b / bars * 100 if bars else 0
        sell_pct = s / bars * 100 if bars else 0
        print(f"  {label:<22s}  buy={b:>5d} ({buy_pct:>5.1f}%)  sell={s:>5d} ({sell_pct:>5.1f}%)")
        prev_buy, prev_sell = b, s

    executed = funnel.get("executed_buy", 0)
    blocked_long = funnel.get("blocked_already_long", 0)
    blocked_cd = funnel.get("blocked_cooldown", 0)
    print(f"  {'Executed entries':<22s}  buy={executed:>5d}")
    if blocked_long or blocked_cd:
        print(f"  {'Blocked (in position)':<22s}       {blocked_long:>5d}")
        print(f"  {'Blocked (cooldown)':<22s}       {blocked_cd:>5d}")

    edge_atr = funnel.get("edge_blocked_atr", 0)
    edge_tp = funnel.get("edge_blocked_tp", 0)
    edge_fc = funnel.get("edge_blocked_forecast", 0)
    edge_sp = funnel.get("edge_blocked_spread", 0)
    if any([edge_atr, edge_tp, edge_fc, edge_sp]):
        print(f"\n  Edge gate breakdown:")
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
        print(f"\n  Ranging gate breakdown:")
        if rg_tech:
            print(f"    Tech disagree:       {rg_tech:>5d}")
        if rg_ml:
            print(f"    ML indifference:     {rg_ml:>5d}")
        if rg_whip:
            print(f"    Whipsaw block:       {rg_whip:>5d}")

    ml_stats = funnel.get("ml_prob_stats")
    if ml_stats:
        print(f"\n  ML probability distribution:")
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
        print(f"\n  Market regime distribution (detect_regime):")
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
