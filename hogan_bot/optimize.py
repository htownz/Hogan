"""Bayesian + random-search parameter optimizer for the Hogan trading bot.

Uses Optuna (TPE sampler) for efficient Bayesian hyperparameter search when
available, with fallback to pure random search.

Usage
-----
::

    python -m hogan_bot.optimize \\
        --symbol  BTC/USD \\
        --timeframe 5m \\
        --limit  50000 \\
        --trials 200 \\
        --metric sharpe \\
        --max-drawdown 0.20

    # Force random search (skip Optuna)
    python -m hogan_bot.optimize --engine random --trials 200

Output is written to ``models/opt_{SYMBOL}_{TF}.json`` and echoed to stdout.

Metrics
-------
``sharpe``        Annualised Sharpe ratio.
``sortino``       Annualised Sortino ratio.
``calmar``        Calmar ratio (return / max drawdown).
``return``        Total return % (raw, no risk adjustment).
``profit_factor`` Gross-profit / gross-loss computed from bar-by-bar equity.
``expectancy``    Mean bar-level return in basis points.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import math
import os
import random
import subprocess
import sys
from pathlib import Path
from typing import Any

import pandas as pd

from hogan_bot.backtest import run_backtest_on_candles
from hogan_bot.config import load_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Parameter space definition
# ---------------------------------------------------------------------------
#
# Each entry: (type, low, high) for numeric types; (type, choices) for "choice".
# "bool" uses (type, None, None).
# ---------------------------------------------------------------------------

_PARAM_SPACE: dict[str, tuple] = {
    "short_ma_window":          ("int",    5,    25),
    "long_ma_window":           ("int",   30,   120),
    "volume_threshold":         ("float", 0.8,   2.5),
    "atr_stop_multiplier":      ("float", 0.5,   3.5),
    "use_ema_clouds":           ("bool",  None, None),
    "signal_mode":              ("choice", ["ma_only", "any", "all"], None),
    "trailing_stop_pct":        ("float", 0.0,   0.05),
    "take_profit_pct":          ("float", 0.0,   0.10),
    # ICT
    "use_ict":                  ("bool",  None, None),
    "ict_swing_left":           ("int",   1,    5),
    "ict_swing_right":          ("int",   1,    5),
    "ict_eq_tolerance_pct":     ("float", 0.0003, 0.002),
    "ict_min_displacement_pct": ("float", 0.001,  0.01),
    "ict_require_time_window":  ("bool",  None, None),
    "ict_require_pd":           ("bool",  None, None),
    "ict_ote_enabled":          ("bool",  None, None),
}

# ICT-specific params only matter when use_ict=True; seed them with their
# defaults when ICT is disabled so they don't inflate the search noise.
_ICT_PARAMS = {
    "ict_swing_left",
    "ict_swing_right",
    "ict_eq_tolerance_pct",
    "ict_min_displacement_pct",
    "ict_require_time_window",
    "ict_require_pd",
    "ict_ote_enabled",
}


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------


def _sample_param(name: str, rng: random.Random) -> Any:
    spec = _PARAM_SPACE[name]
    ptype = spec[0]
    if ptype == "int":
        return rng.randint(spec[1], spec[2])
    if ptype == "float":
        return rng.uniform(spec[1], spec[2])
    if ptype == "bool":
        return rng.choice([True, False])
    if ptype == "choice":
        return rng.choice(spec[1])
    raise ValueError(f"Unknown param type: {ptype}")


def sample_config(rng: random.Random) -> dict[str, Any]:
    """Draw one random parameter configuration, respecting constraints."""
    cfg: dict[str, Any] = {}

    for name in _PARAM_SPACE:
        cfg[name] = _sample_param(name, rng)

    # Constraint: short_ma < long_ma (resample long until satisfied)
    for _ in range(50):
        if cfg["long_ma_window"] > cfg["short_ma_window"] + 5:
            break
        cfg["long_ma_window"] = _sample_param("long_ma_window", rng)

    # When ICT is disabled freeze its params at sensible defaults
    if not cfg["use_ict"]:
        cfg["ict_swing_left"] = 2
        cfg["ict_swing_right"] = 2
        cfg["ict_eq_tolerance_pct"] = 0.0008
        cfg["ict_min_displacement_pct"] = 0.003
        cfg["ict_require_time_window"] = True
        cfg["ict_require_pd"] = True
        cfg["ict_ote_enabled"] = False

    return cfg


# ---------------------------------------------------------------------------
# Metric extraction
# ---------------------------------------------------------------------------


def _compute_extra_metrics(equity_curve: list[float]) -> dict[str, float]:
    """Compute profit_factor and expectancy from the bar-level equity curve."""
    if len(equity_curve) < 2:
        return {"profit_factor": 0.0, "expectancy": 0.0}

    returns = [
        (equity_curve[i] - equity_curve[i - 1]) / max(equity_curve[i - 1], 1e-9)
        for i in range(1, len(equity_curve))
    ]
    gross_profit = sum(r for r in returns if r > 0)
    gross_loss = abs(sum(r for r in returns if r < 0))
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else (
        float("inf") if gross_profit > 0 else 0.0
    )
    expectancy = (sum(returns) / len(returns)) * 10_000  # in basis points per bar
    return {"profit_factor": profit_factor, "expectancy": expectancy}


def _score(result_dict: dict, metric: str) -> float:
    """Extract the optimisation metric from a result dict (higher = better)."""
    val = result_dict.get(metric)
    if val is None or (isinstance(val, float) and math.isnan(val)):
        return float("-inf")
    return float(val)


# ---------------------------------------------------------------------------
# Single trial
# ---------------------------------------------------------------------------


def _run_trial(
    candles: pd.DataFrame,
    symbol: str,
    cfg_dict: dict[str, Any],
    base_cfg,
    timeframe: str = "5m",
) -> dict[str, Any]:
    """Run one backtest and return the result enriched with computed metrics."""
    try:
        result = run_backtest_on_candles(
            candles,
            symbol=symbol,
            timeframe=timeframe,
            starting_balance_usd=base_cfg.starting_balance_usd,
            aggressive_allocation=base_cfg.aggressive_allocation,
            max_risk_per_trade=base_cfg.max_risk_per_trade,
            max_drawdown=base_cfg.max_drawdown,
            short_ma_window=cfg_dict["short_ma_window"],
            long_ma_window=cfg_dict["long_ma_window"],
            volume_window=base_cfg.volume_window,
            volume_threshold=cfg_dict["volume_threshold"],
            fee_rate=base_cfg.fee_rate,
            use_ema_clouds=cfg_dict["use_ema_clouds"],
            ema_fast_short=base_cfg.ema_fast_short,
            ema_fast_long=base_cfg.ema_fast_long,
            ema_slow_short=base_cfg.ema_slow_short,
            ema_slow_long=base_cfg.ema_slow_long,
            use_fvg=base_cfg.use_fvg and not cfg_dict["use_ict"],
            fvg_min_gap_pct=base_cfg.fvg_min_gap_pct,
            signal_mode=cfg_dict["signal_mode"],
            trailing_stop_pct=cfg_dict["trailing_stop_pct"],
            take_profit_pct=cfg_dict["take_profit_pct"],
            atr_stop_multiplier=cfg_dict["atr_stop_multiplier"],
            use_ict=cfg_dict["use_ict"],
            ict_swing_left=cfg_dict["ict_swing_left"],
            ict_swing_right=cfg_dict["ict_swing_right"],
            ict_eq_tolerance_pct=cfg_dict["ict_eq_tolerance_pct"],
            ict_min_displacement_pct=cfg_dict["ict_min_displacement_pct"],
            ict_require_time_window=cfg_dict["ict_require_time_window"],
            ict_time_windows=base_cfg.ict_time_windows,
            ict_require_pd=cfg_dict["ict_require_pd"],
            ict_ote_enabled=cfg_dict["ict_ote_enabled"],
            ict_ote_low=base_cfg.ict_ote_low,
            ict_ote_high=base_cfg.ict_ote_high,
        )
    except Exception as exc:
        logger.debug("Trial raised %s: %s", type(exc).__name__, exc)
        return {"error": str(exc), "config": cfg_dict}

    summary = result.summary_dict()
    extra = _compute_extra_metrics(result.equity_curve)
    return {**summary, **extra, "config": cfg_dict}


# ---------------------------------------------------------------------------
# Constraint checking
# ---------------------------------------------------------------------------


def _satisfies_constraints(row: dict, constraints: dict[str, float]) -> bool:
    for key, limit in constraints.items():
        # Convention: keys starting with "max_" are upper-bound checks
        if key.startswith("max_"):
            metric = key[4:]  # strip "max_"
            val = row.get(metric)
            if val is None:
                continue
            if float(val) > limit:
                return False
        # Keys starting with "min_" are lower-bound checks
        elif key.startswith("min_"):
            metric = key[4:]
            val = row.get(metric)
            if val is None:
                continue
            if float(val) < limit:
                return False
    return True


# ---------------------------------------------------------------------------
# Optuna Bayesian optimizer
# ---------------------------------------------------------------------------

def _optuna_objective(
    trial,
    candles: pd.DataFrame,
    symbol: str,
    base_cfg,
    metric: str,
    constraints: dict[str, float],
    timeframe: str = "5m",
) -> float:
    """Optuna objective: suggest params, run backtest, return score."""
    cfg_dict: dict[str, Any] = {}

    cfg_dict["short_ma_window"] = trial.suggest_int("short_ma_window", 5, 25)
    cfg_dict["long_ma_window"] = trial.suggest_int("long_ma_window", 30, 120)
    cfg_dict["volume_threshold"] = trial.suggest_float("volume_threshold", 0.8, 2.5)
    cfg_dict["atr_stop_multiplier"] = trial.suggest_float("atr_stop_multiplier", 0.5, 3.5)
    cfg_dict["use_ema_clouds"] = trial.suggest_categorical("use_ema_clouds", [True, False])
    cfg_dict["signal_mode"] = trial.suggest_categorical("signal_mode", ["ma_only", "any", "all"])
    cfg_dict["trailing_stop_pct"] = trial.suggest_float("trailing_stop_pct", 0.0, 0.05)
    cfg_dict["take_profit_pct"] = trial.suggest_float("take_profit_pct", 0.0, 0.10)
    cfg_dict["use_ict"] = trial.suggest_categorical("use_ict", [True, False])

    if cfg_dict["use_ict"]:
        cfg_dict["ict_swing_left"] = trial.suggest_int("ict_swing_left", 1, 5)
        cfg_dict["ict_swing_right"] = trial.suggest_int("ict_swing_right", 1, 5)
        cfg_dict["ict_eq_tolerance_pct"] = trial.suggest_float("ict_eq_tolerance_pct", 0.0003, 0.002)
        cfg_dict["ict_min_displacement_pct"] = trial.suggest_float("ict_min_displacement_pct", 0.001, 0.01)
        cfg_dict["ict_require_time_window"] = trial.suggest_categorical("ict_require_time_window", [True, False])
        cfg_dict["ict_require_pd"] = trial.suggest_categorical("ict_require_pd", [True, False])
        cfg_dict["ict_ote_enabled"] = trial.suggest_categorical("ict_ote_enabled", [True, False])
    else:
        cfg_dict["ict_swing_left"] = 2
        cfg_dict["ict_swing_right"] = 2
        cfg_dict["ict_eq_tolerance_pct"] = 0.0008
        cfg_dict["ict_min_displacement_pct"] = 0.003
        cfg_dict["ict_require_time_window"] = True
        cfg_dict["ict_require_pd"] = True
        cfg_dict["ict_ote_enabled"] = False

    if cfg_dict["long_ma_window"] <= cfg_dict["short_ma_window"] + 5:
        return float("-inf")

    row = _run_trial(candles, symbol, cfg_dict, base_cfg, timeframe=timeframe)

    if "error" in row:
        return float("-inf")
    if not _satisfies_constraints(row, constraints):
        return float("-inf")

    score = _score(row, metric)
    trial.set_user_attr("result", row)
    trial.set_user_attr("config", cfg_dict)
    return score


def optimize_bayesian(
    candles: pd.DataFrame,
    symbol: str,
    base_cfg,
    *,
    timeframe: str = "5m",
    trials: int = 200,
    metric: str = "sharpe",
    constraints: dict[str, float] | None = None,
    top_k: int = 10,
    seed: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Optuna TPE-based Bayesian hyperparameter optimization."""
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    if constraints is None:
        constraints = {}

    actual_seed = seed if seed is not None else random.randint(0, 2**32 - 1)
    sampler = optuna.samplers.TPESampler(seed=actual_seed)

    study = optuna.create_study(
        direction="maximize",
        sampler=sampler,
        study_name=f"hogan_opt_{symbol.replace('/', '_')}",
    )

    study.optimize(
        lambda trial: _optuna_objective(
            trial, candles, symbol, base_cfg, metric, constraints, timeframe=timeframe
        ),
        n_trials=trials,
        show_progress_bar=verbose,
    )

    # Extract results
    all_results = []
    for t in study.trials:
        if t.state == optuna.trial.TrialState.COMPLETE:
            row = t.user_attrs.get("result", {})
            row["trial"] = t.number
            row["seed"] = actual_seed
            row["satisfies_constraints"] = True
            all_results.append(row)

    feasible = [r for r in all_results if "error" not in r]
    feasible.sort(key=lambda r: _score(r, metric), reverse=True)

    leaderboard = []
    for rank, row in enumerate(feasible[:top_k], start=1):
        entry = {"rank": rank, "config": row.get("config", {}), metric: _score(row, metric)}
        for extra_key in ("total_return_pct", "max_drawdown_pct", "win_rate", "trades",
                          "sharpe_ratio", "sortino_ratio", "calmar_ratio", "profit_factor"):
            if extra_key in row:
                entry[extra_key] = row[extra_key]
        leaderboard.append(entry)

    best_config = study.best_trial.user_attrs.get("config", {}) if study.best_trial else {}
    best_score = study.best_value if study.best_trial else float("-inf")

    commit_hash = None
    try:
        commit_hash = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        pass

    return {
        "symbol": symbol,
        "metric": metric,
        "engine": "optuna_tpe",
        "trials": trials,
        "seed": actual_seed,
        "git_commit": commit_hash,
        "best_score": best_score,
        "best_config": best_config,
        "leaderboard": leaderboard,
    }


# ---------------------------------------------------------------------------
# Main optimizer (random search fallback)
# ---------------------------------------------------------------------------


def optimize(
    candles: pd.DataFrame,
    symbol: str,
    base_cfg,
    *,
    timeframe: str = "5m",
    trials: int = 200,
    metric: str = "sharpe",
    constraints: dict[str, float] | None = None,
    top_k: int = 10,
    seed: int | None = None,
    verbose: bool = True,
) -> dict[str, Any]:
    """Run random-search optimisation over the parameter space.

    Parameters
    ----------
    candles:
        Pre-loaded OHLCV DataFrame (fetched once, reused across all trials).
    symbol:
        Pair string, e.g. ``"BTC/USD"``.
    base_cfg:
        A :class:`~hogan_bot.config.BotConfig` for non-optimised parameters.
    trials:
        Number of random configurations to evaluate.
    metric:
        Optimisation target: ``"sharpe"``, ``"sortino"``, ``"calmar"``,
        ``"return"``, ``"profit_factor"``, or ``"expectancy"``.
    constraints:
        Dict of hard constraints.  Keys: ``"max_drawdown_pct"`` (e.g. 20.0),
        ``"min_trades"`` (e.g. 5), etc.
    top_k:
        Number of top results to include in the leaderboard.
    seed:
        RNG seed for reproducibility.

    Returns
    -------
    dict
        ``best_config``, ``best_score``, ``metric``, ``leaderboard`` (top_k
        results), ``all_results`` (every trial).
    """
    if constraints is None:
        constraints = {}

    rng = random.Random(seed)
    actual_seed = seed if seed is not None else rng.randint(0, 2**32 - 1)

    all_results: list[dict] = []

    # Try to capture the current git commit hash for reproducibility
    commit_hash = None
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"],
                                    stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        pass

    logger.info(
        "Starting optimisation: symbol=%s trials=%d metric=%s seed=%d",
        symbol, trials, metric, actual_seed,
    )

    for t in range(trials):
        cfg_dict = sample_config(rng)
        row = _run_trial(candles, symbol, cfg_dict, base_cfg, timeframe=timeframe)

        # Attach trial metadata
        row["trial"] = t
        row["seed"] = actual_seed
        row["satisfies_constraints"] = _satisfies_constraints(row, constraints)

        all_results.append(row)

        if verbose and (t + 1) % max(1, trials // 10) == 0:
            feasible = [r for r in all_results if r.get("satisfies_constraints")]
            best_so_far = max((_score(r, metric) for r in feasible), default=float("-inf"))
            print(
                f"  [{t + 1:>4d}/{trials}]  best_{metric}={best_so_far:.4f}"
                f"  feasible={len(feasible)}",
                file=sys.stderr,
            )

    # Rank feasible results
    feasible = [r for r in all_results if r.get("satisfies_constraints") and "error" not in r]
    feasible.sort(key=lambda r: _score(r, metric), reverse=True)

    leaderboard = []
    for rank, row in enumerate(feasible[:top_k], start=1):
        entry = {
            "rank": rank,
            "config": row["config"],
            metric: _score(row, metric),
        }
        for extra_key in ("total_return_pct", "max_drawdown_pct", "win_rate", "trades",
                          "sharpe_ratio", "sortino_ratio", "calmar_ratio", "profit_factor"):
            if extra_key in row:
                entry[extra_key] = row[extra_key]
        leaderboard.append(entry)

    best_row = feasible[0] if feasible else {}
    best_config = best_row.get("config", {})
    best_score = _score(best_row, metric) if best_row else float("-inf")

    return {
        "symbol": symbol,
        "metric": metric,
        "trials": trials,
        "seed": actual_seed,
        "git_commit": commit_hash,
        "best_score": best_score,
        "best_config": best_config,
        "leaderboard": leaderboard,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Hogan parameter optimiser — random search over strategy config space",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="BTC/USD", help="Trading pair, e.g. BTC/USD")
    p.add_argument("--timeframe", default="5m", help="Bar interval, e.g. 5m 15m 1h")
    p.add_argument("--limit", type=int, default=10_000, help="Number of OHLCV bars to fetch")
    p.add_argument("--trials", type=int, default=200, help="Number of random trials")
    p.add_argument(
        "--metric",
        default="sharpe",
        choices=["sharpe", "sortino", "calmar", "return", "profit_factor", "expectancy"],
        help="Metric to maximise",
    )
    p.add_argument(
        "--max-drawdown",
        type=float,
        default=None,
        metavar="PCT",
        help="Constraint: reject configs whose max drawdown %% exceeds this (e.g. 20 for 20%%)",
    )
    p.add_argument(
        "--min-trades",
        type=int,
        default=None,
        metavar="N",
        help="Constraint: reject configs with fewer than N trades",
    )
    p.add_argument("--top-k", type=int, default=10, help="Leaderboard size")
    p.add_argument("--seed", type=int, default=None, help="RNG seed for reproducibility")
    p.add_argument(
        "--out-dir",
        default="models",
        help="Directory to write output JSON (created if missing)",
    )
    p.add_argument(
        "--from-db",
        action="store_true",
        help="Load candles from local SQLite DB instead of fetching live",
    )
    p.add_argument(
        "--db",
        default=os.getenv("HOGAN_DB_PATH", os.path.join("data", "hogan.db")),
        help="Path to local SQLite DB (used with --from-db)",
    )
    p.add_argument("--quiet", action="store_true", help="Suppress trial progress output")
    p.add_argument(
        "--engine",
        default="auto",
        choices=["auto", "optuna", "random"],
        help="Search engine: 'optuna' (Bayesian TPE), 'random', or 'auto' (Optuna if installed)",
    )
    return p


def _output_path(out_dir: str, symbol: str, timeframe: str) -> Path:
    safe_sym = symbol.replace("/", "-").replace(":", "-")
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    return Path(out_dir) / f"opt_{safe_sym}_{timeframe}.json"


def main(argv: list[str] | None = None) -> None:
    """CLI entry point for ``python -m hogan_bot.optimize``."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%H:%M:%S",
    )
    args = _build_parser().parse_args(argv)

    cfg = load_config()

    # Build constraint dict
    constraints: dict[str, float] = {}
    if args.max_drawdown is not None:
        constraints["max_drawdown_pct"] = float(args.max_drawdown)
    if args.min_trades is not None:
        constraints["min_trades"] = float(args.min_trades)

    # Remap metric name to match BacktestResult fields
    metric_map = {
        "sharpe": "sharpe_ratio",
        "sortino": "sortino_ratio",
        "calmar": "calmar_ratio",
        "return": "total_return_pct",
        "profit_factor": "profit_factor",
        "expectancy": "expectancy",
    }
    metric_key = metric_map[args.metric]
    # Constraints use BacktestResult field names too
    if "max_drawdown_pct" in constraints:
        constraints["max_drawdown_pct"] = constraints["max_drawdown_pct"]

    # Fetch candles once
    if args.from_db:
        from hogan_bot.storage import get_connection, load_candles
        conn = get_connection(args.db)
        candles = load_candles(conn, args.symbol, args.timeframe, limit=args.limit)
        if candles is None or candles.empty:
            logger.error("No candles found in DB for %s", args.symbol)
            sys.exit(1)
    else:
        from hogan_bot.exchange import ExchangeClient
        client = ExchangeClient(cfg.exchange_id, cfg.kraken_api_key, cfg.kraken_api_secret)
        candles = client.fetch_ohlcv_df(args.symbol, timeframe=args.timeframe, limit=args.limit)

    if candles.empty:
        logger.error("No candles returned — aborting.")
        sys.exit(1)

    # Select optimization engine
    use_optuna = False
    if args.engine == "optuna":
        use_optuna = True
    elif args.engine == "auto":
        try:
            import optuna  # noqa: F401
            use_optuna = True
        except ImportError:
            pass

    engine_name = "Optuna TPE" if use_optuna else "Random Search"
    logger.info(
        "Loaded %d bars for %s/%s — running %d trials optimising %s [engine=%s]",
        len(candles), args.symbol, args.timeframe, args.trials, args.metric, engine_name,
    )

    if use_optuna:
        result = optimize_bayesian(
            candles,
            args.symbol,
            cfg,
            timeframe=args.timeframe,
            trials=args.trials,
            metric=metric_key,
            constraints=constraints,
            top_k=args.top_k,
            seed=args.seed,
            verbose=not args.quiet,
        )
    else:
        result = optimize(
            candles,
            args.symbol,
            cfg,
            timeframe=args.timeframe,
            trials=args.trials,
            metric=metric_key,
            constraints=constraints,
            top_k=args.top_k,
            seed=args.seed,
            verbose=not args.quiet,
        )

    # Add result metadata for versioning and reproducibility
    from hogan_bot.timeframe_utils import bars_per_year
    result["metric_version"] = 2
    result["timeframe"] = args.timeframe
    result["bars_per_year"] = int(bars_per_year(args.timeframe))
    result["execution_mode"] = "same_bar"
    ts_col = "ts_ms" if "ts_ms" in candles.columns else "timestamp"
    if ts_col in candles.columns:
        if ts_col == "ts_ms":
            result["data_start_ts"] = int(candles["ts_ms"].iloc[0])
            result["data_end_ts"] = int(candles["ts_ms"].iloc[-1])
        else:
            first_ts = candles["timestamp"].iloc[0]
            last_ts = candles["timestamp"].iloc[-1]
            result["data_start_ts"] = int(pd.Timestamp(first_ts).timestamp() * 1000)
            result["data_end_ts"] = int(pd.Timestamp(last_ts).timestamp() * 1000)
    else:
        result["data_start_ts"] = None
        result["data_end_ts"] = None

    # Write candidate to a separate path so we can compare against incumbent
    # (incumbent lives at opt_{symbol}_{tf}.json). Previously we wrote over
    # incumbent before evaluation, neutering the "beat incumbent" logic.
    from datetime import datetime, timezone
    candidates_dir = Path(args.out_dir) / "candidates"
    candidates_dir.mkdir(parents=True, exist_ok=True)
    ts_str = datetime.now(tz=timezone.utc).strftime("%Y%m%d_%H%M%S")
    safe_sym = args.symbol.replace("/", "-").replace(":", "-")
    candidate_path = candidates_dir / f"opt_{safe_sym}_{args.timeframe}_{ts_str}.json"
    with open(candidate_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    logger.info("Results written to %s", candidate_path)

    # Auto-promotion: compare candidate against incumbent, promote only if better
    try:
        from hogan_bot.auto_promote import evaluate_and_promote
        promo = evaluate_and_promote(
            args.symbol, args.timeframe,
            candidate_path=str(candidate_path),
            models_dir=args.out_dir,
        )
        if promo.promoted:
            print(f"\n  >>> AUTO-PROMOTED: {promo.reason}")
            if promo.backup_file:
                print(f"  >>> Archived incumbent to {promo.backup_file}")
            print(f"  >>> Live bot will use new config on next iteration.\n")
        else:
            print(f"\n  >>> NOT PROMOTED: {promo.reason}\n")
    except Exception as exc:
        logger.warning("Auto-promotion check failed: %s", exc)

    # Also pretty-print the leaderboard to stdout
    print(f"\n{'='*70}")
    print(f"  Hogan Optimizer — {args.symbol} / {args.timeframe}")
    print(f"  Metric: {args.metric}   Trials: {args.trials}   Seed: {result['seed']}")
    if result.get("git_commit"):
        print(f"  Commit: {result['git_commit']}")
    print(f"{'='*70}")
    print(f"\n  Best score ({args.metric}): {result['best_score']:.4f}")
    print(f"  Best config: {json.dumps(result['best_config'], indent=4)}\n")

    if result["leaderboard"]:
        # Display column uses the friendly name; lookup uses the internal key
        cols = ["rank", args.metric, "total_return_pct", "max_drawdown_pct",
                "win_rate", "trades"]
        widths = [6, 12, 16, 16, 10, 6]
        header = "  ".join(str(c).ljust(w) for c, w in zip(cols, widths))
        sep = "  ".join("-" * w for w in widths)
        print(header)
        print(sep)
        for row in result["leaderboard"]:
            vals = [
                row.get("rank", ""),
                row.get(metric_key, ""),   # internal key e.g. "sharpe_ratio"
                row.get("total_return_pct", ""),
                row.get("max_drawdown_pct", ""),
                row.get("win_rate", ""),
                row.get("trades", ""),
            ]
            line = "  ".join(
                (f"{v:.4f}" if isinstance(v, float) else str(v)).ljust(w)
                for v, w in zip(vals, widths)
            )
            print(line)
    print()


if __name__ == "__main__":
    main()
