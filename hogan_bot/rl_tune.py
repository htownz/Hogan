"""Optuna-based hyperparameter search for the Hogan RL (PPO) agent.

Tunes the following hyperparameters:
    learning_rate, clip_range, ent_coef, n_steps, batch_size, gamma

Each Optuna trial trains the agent for ``--trial-steps`` steps on the first
80 % of candles (training window), then evaluates annualised Sharpe on the
remaining 20 % (validation window).  The objective is to maximise Sharpe.

The best hyperparameter set is saved to ``models/best_hparams.json`` and is
automatically picked up by :mod:`hogan_bot.rl_train` on subsequent runs.

Usage
-----
    python -m hogan_bot.rl_tune \\
        --symbol BTC/USD --timeframe 5m --from-db --trials 30

    # Custom trial length and output dir
    python -m hogan_bot.rl_tune --from-db --trial-steps 100000 \\
        --trials 50 --model-dir models
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Dependency checks
# ---------------------------------------------------------------------------

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    _OPTUNA_AVAILABLE = True
except ImportError:
    _OPTUNA_AVAILABLE = False

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.vec_env import DummyVecEnv
    _SB3_AVAILABLE = True
except ImportError:
    _SB3_AVAILABLE = False

try:
    import gymnasium  # noqa: F401
    _GYM_AVAILABLE = True
except ImportError:
    _GYM_AVAILABLE = False


def _check_deps() -> None:
    missing = []
    if not _OPTUNA_AVAILABLE:
        missing.append("optuna")
    if not _SB3_AVAILABLE:
        missing.append("stable-baselines3")
    if not _GYM_AVAILABLE:
        missing.append("gymnasium")
    if missing:
        sys.exit(
            f"Missing required packages: {', '.join(missing)}\n"
            "Run: pip install " + " ".join(missing)
        )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_from_db(symbol: str, timeframe: str) -> pd.DataFrame:
    from hogan_bot.storage import get_connection, load_candles
    conn = get_connection()
    candles = load_candles(conn, symbol=symbol, timeframe=timeframe)
    conn.close()
    if candles is None or len(candles) == 0:
        sys.exit(f"No candles found for {symbol} {timeframe}.  Run backfill first.")
    return candles


# ---------------------------------------------------------------------------
# Sharpe evaluation helper
# ---------------------------------------------------------------------------

def _sharpe_on_env(env, policy, bars_per_year: float) -> float:
    """Run policy deterministically on env and return annualised Sharpe."""
    obs, _ = env.reset()
    equity_curve: list[float] = [env.starting_balance]
    done = False
    while not done:
        action, _ = policy.predict(obs, deterministic=True)
        obs, _, terminated, truncated, info = env.step(int(action))
        equity_curve.append(info["equity"])
        done = terminated or truncated

    arr = np.array(equity_curve)
    rets = np.diff(arr) / np.maximum(arr[:-1], 1e-9)
    if len(rets) < 2 or np.std(rets) < 1e-12:
        return -10.0  # degenerate episode
    return float(np.mean(rets) / np.std(rets) * np.sqrt(bars_per_year))


# ---------------------------------------------------------------------------
# Objective function
# ---------------------------------------------------------------------------

def _make_objective(
    train_candles: pd.DataFrame,
    val_candles: pd.DataFrame,
    reward_type: str,
    starting_balance: float,
    fee_rate: float,
    slippage_pct: float,
    min_trades: int,
    trial_steps: int,
    seed: int,
    device: str,
    timeframe: str = "5m",
):
    """Return an Optuna objective closure."""
    from hogan_bot.rl_env import TradingEnv
    from hogan_bot.timeframe_utils import bars_per_year

    bpy = float(bars_per_year(timeframe))
    env_kwargs = dict(
        starting_balance=starting_balance,
        fee_rate=fee_rate,
        slippage_pct=slippage_pct,
        reward_type=reward_type,
        min_trades=min_trades,
    )

    def objective(trial: "optuna.Trial") -> float:  # type: ignore[name-defined]
        lr = trial.suggest_float("learning_rate", 1e-5, 3e-3, log=True)
        clip_range = trial.suggest_float("clip_range", 0.1, 0.4)
        ent_coef = trial.suggest_float("ent_coef", 1e-4, 0.05, log=True)
        n_steps = trial.suggest_categorical("n_steps", [512, 1024, 2048, 4096])
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])
        gamma = trial.suggest_float("gamma", 0.95, 0.9999)

        # batch_size must not exceed n_steps
        if batch_size > n_steps:
            batch_size = n_steps

        def _make_env():
            return TradingEnv(train_candles, seed=seed, **env_kwargs)

        env = DummyVecEnv([_make_env])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            model = PPO(
                "MlpPolicy",
                env,
                verbose=0,
                seed=seed,
                device=device,
                n_steps=n_steps,
                batch_size=batch_size,
                n_epochs=10,
                gamma=gamma,
                gae_lambda=0.95,
                clip_range=clip_range,
                ent_coef=ent_coef,
                learning_rate=lr,
            )
            model.learn(total_timesteps=trial_steps)

        val_env = TradingEnv(val_candles, seed=seed, **env_kwargs)
        sharpe = _sharpe_on_env(val_env, model, bpy)
        return sharpe

    return objective


# ---------------------------------------------------------------------------
# Main tuning routine
# ---------------------------------------------------------------------------

def tune(
    candles: pd.DataFrame,
    *,
    timeframe: str = "5m",
    reward_type: str = "risk_adjusted",
    model_dir: str = "models",
    starting_balance: float = 10_000.0,
    fee_rate: float = 0.0026,
    slippage_pct: float = 0.001,
    min_trades: int = 3,
    n_trials: int = 30,
    trial_steps: int = 100_000,
    seed: int = 42,
    device: str = "cpu",
) -> dict:
    """Run Optuna hyperparameter search.

    Parameters
    ----------
    candles:
        Full OHLCV history — split 80/20 internally.
    n_trials:
        Number of Optuna trials (each trains for ``trial_steps``).
    trial_steps:
        Training steps per trial (shorter = faster search, less accurate).
    model_dir:
        Directory where ``best_hparams.json`` is written.

    Returns
    -------
    dict
        Best hyperparameters found.
    """
    split = int(len(candles) * 0.8)
    train_candles = candles.iloc[:split].reset_index(drop=True)
    val_candles = candles.iloc[split:].reset_index(drop=True)

    min_bars = 62
    if len(train_candles) < min_bars or len(val_candles) < min_bars:
        sys.exit(
            f"Not enough candles for tuning "
            f"(train={len(train_candles)}, val={len(val_candles)}).  "
            "Accumulate more data first."
        )

    objective = _make_objective(
        train_candles, val_candles,
        reward_type=reward_type,
        starting_balance=starting_balance,
        fee_rate=fee_rate,
        slippage_pct=slippage_pct,
        min_trades=min_trades,
        trial_steps=trial_steps,
        seed=seed,
        device=device,
        timeframe=timeframe,
    )

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    print(
        f"Starting Optuna search: {n_trials} trials x {trial_steps:,} steps each\n"
        f"  reward={reward_type}  train_bars={len(train_candles)}"
        f"  val_bars={len(val_candles)}"
    )

    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    best = study.best_params
    best_value = study.best_value
    print(f"\nBest Sharpe: {best_value:.4f}")
    print("Best hyperparameters:")
    for k, v in best.items():
        print(f"  {k:<20s} {v}")

    # Ensure batch_size <= n_steps constraint is preserved
    if "batch_size" in best and "n_steps" in best:
        if best["batch_size"] > best["n_steps"]:
            best["batch_size"] = best["n_steps"]

    Path(model_dir).mkdir(parents=True, exist_ok=True)
    out_path = Path(model_dir) / "best_hparams.json"
    with open(out_path, "w") as f:
        json.dump(best, f, indent=2)
    print(f"Saved -> {out_path}")

    return best


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Optuna hyperparameter search for the Hogan RL agent",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--symbol", default="BTC/USD")
    p.add_argument("--timeframe", default="5m")
    p.add_argument(
        "--from-db", action="store_true",
        help="Load candles from data/hogan.db",
    )
    p.add_argument("--trials", type=int, default=30, metavar="N",
                   help="Number of Optuna trials")
    p.add_argument("--trial-steps", type=int, default=100_000, metavar="STEPS",
                   help="Training steps per trial")
    p.add_argument(
        "--reward",
        default=os.getenv("HOGAN_RL_REWARD_TYPE", "risk_adjusted"),
        choices=["delta_equity", "risk_adjusted", "sharpe", "sortino"],
    )
    p.add_argument("--model-dir", default="models",
                   help="Directory where best_hparams.json is saved")
    p.add_argument("--balance", type=float, default=10_000.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--device", default="cpu",
        help="PyTorch device: 'cpu' or 'cuda'",
    )
    p.add_argument("--slippage", type=float, default=0.001)
    p.add_argument("--min-trades", type=int, default=3)
    return p.parse_args()


def main() -> None:
    _check_deps()
    args = parse_args()

    if args.from_db:
        candles = _load_from_db(args.symbol, args.timeframe)
    else:
        sys.exit("--from-db is required (live exchange fetch not supported for tuning).")

    print(f"Loaded {len(candles)} candles for {args.symbol} {args.timeframe}")

    tune(
        candles,
        timeframe=args.timeframe,
        reward_type=args.reward,
        model_dir=args.model_dir,
        starting_balance=args.balance,
        fee_rate=0.0026,
        slippage_pct=args.slippage,
        min_trades=args.min_trades,
        n_trials=args.trials,
        trial_steps=args.trial_steps,
        seed=args.seed,
        device=args.device,
    )


if __name__ == "__main__":
    main()
