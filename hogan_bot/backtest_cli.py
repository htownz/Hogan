from __future__ import annotations

import argparse
import json

from hogan_bot.backtest import evaluate_regimes, run_backtest_on_candles
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


def _run_single(cfg, candles, symbol, ml_model, timeframe: str | None = None, overrides: dict | None = None, use_ict: bool = False, use_rl_agent: bool = False, rl_policy=None):
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
    )


def main() -> None:
    args = parse_args()
    cfg = load_config()

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

    if args.compare:
        rows = []
        for label, overrides in _COMPARE_CONFIGS:
            result = _run_single(
                cfg, candles, args.symbol, ml_model,
                timeframe=timeframe, overrides=overrides,
                use_ict=args.use_ict, use_rl_agent=use_rl, rl_policy=rl_policy,
            )
            rows.append({"config": label, **result.summary_dict()})

        print(json.dumps(rows, indent=2))
        _print_table(rows)
    else:
        result = _run_single(
            cfg, candles, args.symbol, ml_model,
            timeframe=timeframe,
            use_ict=args.use_ict, use_rl_agent=use_rl, rl_policy=rl_policy,
        )
        print(json.dumps(result.summary_dict(), indent=2))

        if args.regime_report:
            regimes = evaluate_regimes(result)
            print("\n-- Regime breakdown ------------------------------------")
            for regime, metrics in regimes.items():
                print(f"  {regime:<10s}  bars={metrics['bars']:>5d}  "
                      f"sharpe={str(metrics['sharpe']):<8s}  "
                      f"return={metrics['total_return_pct']:>7.2f}%  "
                      f"maxdd={metrics['max_drawdown_pct']:>6.2f}%")
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
