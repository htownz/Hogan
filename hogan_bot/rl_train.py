"""Training CLI for the Hogan RL agent.

Usage
-----
    python -m hogan_bot.rl_train \\
        --symbol BTC/USD --timeframe 5m \\
        --from-db --timesteps 500000 --reward risk_adjusted

The trained PPO policy is saved to ``models/hogan_rl_policy.zip`` (or the
path specified by ``--model-path``).  An evaluation summary (Sharpe, total
return, max drawdown) over the held-out 20 % window is printed on completion.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Optional heavy dependencies — checked at runtime to give helpful errors
# ---------------------------------------------------------------------------

try:
    from stable_baselines3 import PPO
    from stable_baselines3.common.env_util import make_vec_env
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
    if not _SB3_AVAILABLE:
        missing.append("stable-baselines3")
    if not _GYM_AVAILABLE:
        missing.append("gymnasium")
    if missing:
        sys.exit(
            f"Missing required packages: {', '.join(missing)}\n"
            "Run: pip install stable-baselines3 gymnasium"
        )


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def _load_from_db(symbol: str, timeframe: str) -> pd.DataFrame:
    """Fetch candles from the local SQLite database."""
    from hogan_bot.storage import load_candles
    candles = load_candles(symbol=symbol, timeframe=timeframe)
    if candles is None or len(candles) == 0:
        sys.exit(
            f"No candles found in the database for {symbol} {timeframe}.\n"
            "Run: python -m hogan_bot.fetch_data --symbol BTC/USD --timeframe 5m"
        )
    return candles


def _load_from_exchange(symbol: str, timeframe: str, limit: int) -> pd.DataFrame:
    """Fetch candles live from the exchange."""
    from hogan_bot.config import load_config
    from hogan_bot.exchange import ExchangeClient
    cfg = load_config()
    client = ExchangeClient(cfg.exchange_id, cfg.kraken_api_key, cfg.kraken_api_secret)
    return client.fetch_ohlcv_df(symbol, timeframe=timeframe, limit=limit)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def _evaluate(env, policy, n_episodes: int = 1) -> dict[str, float]:
    """Run *policy* on *env* for *n_episodes* and return performance metrics."""
    all_returns: list[float] = []
    all_equity_curves: list[list[float]] = []

    for _ in range(n_episodes):
        obs, _ = env.reset()
        equity_curve: list[float] = [env.starting_balance]
        done = False
        while not done:
            action, _ = policy.predict(obs, deterministic=True)
            obs, _reward, terminated, truncated, info = env.step(int(action))
            equity_curve.append(info["equity"])
            done = terminated or truncated

        final_equity = equity_curve[-1]
        total_return = (final_equity - env.starting_balance) / env.starting_balance
        all_returns.append(total_return)
        all_equity_curves.append(equity_curve)

    # Aggregate over episodes
    mean_return = float(np.mean(all_returns))
    equity = all_equity_curves[0]
    equity_arr = np.array(equity)
    step_returns = np.diff(equity_arr) / np.maximum(equity_arr[:-1], 1e-9)
    sharpe = (
        float(np.mean(step_returns) / (np.std(step_returns) + 1e-9)) * np.sqrt(252 * 288)
        if len(step_returns) > 1
        else 0.0
    )
    peak = np.maximum.accumulate(equity_arr)
    drawdowns = (peak - equity_arr) / np.maximum(peak, 1e-9)
    max_drawdown = float(np.max(drawdowns))

    return {
        "total_return_pct": round(mean_return * 100, 2),
        "sharpe_ratio": round(sharpe, 3),
        "max_drawdown_pct": round(max_drawdown * 100, 2),
        "final_equity": round(equity[-1], 2),
    }


# ---------------------------------------------------------------------------
# Training entry point
# ---------------------------------------------------------------------------

def train(
    candles: pd.DataFrame,
    *,
    timesteps: int = 500_000,
    reward_type: str = "risk_adjusted",
    model_path: str = "models/hogan_rl_policy.zip",
    starting_balance: float = 10_000.0,
    fee_rate: float = 0.0026,
    seed: int = 42,
    verbose: int = 1,
) -> dict[str, float]:
    """Train a PPO agent and save the policy.

    Parameters
    ----------
    candles:
        Full OHLCV history (oldest first).
    timesteps:
        Total environment steps for training.
    reward_type:
        ``"delta_equity"`` or ``"risk_adjusted"``.
    model_path:
        Destination for the saved ``.zip`` policy file.
    starting_balance:
        Paper-trading starting equity used in both training and eval envs.
    fee_rate:
        Per-side transaction fee fraction.
    seed:
        Random seed for reproducibility.
    verbose:
        SB3 verbosity level (0 = silent, 1 = progress, 2 = debug).

    Returns
    -------
    dict
        Evaluation metrics from the held-out 20 % window.
    """
    from hogan_bot.rl_env import TradingEnv

    split = int(len(candles) * 0.8)
    train_candles = candles.iloc[:split].reset_index(drop=True)
    eval_candles = candles.iloc[split:].reset_index(drop=True)

    min_bars = 62  # min_history (60) + 2 buffer

    if len(train_candles) < min_bars:
        sys.exit(
            f"Not enough training bars ({len(train_candles)}).  "
            "Accumulate more data or reduce the train/eval split."
        )

    def _make_train_env():
        return TradingEnv(
            train_candles,
            starting_balance=starting_balance,
            fee_rate=fee_rate,
            reward_type=reward_type,
            seed=seed,
        )

    env = DummyVecEnv([_make_train_env])

    model = PPO(
        "MlpPolicy",
        env,
        verbose=verbose,
        seed=seed,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        learning_rate=3e-4,
        tensorboard_log=None,
    )

    print(f"Training PPO for {timesteps:,} steps  "
          f"(reward={reward_type}, train_bars={len(train_candles)}) …")
    model.learn(total_timesteps=timesteps)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path.removesuffix(".zip"))
    print(f"Policy saved → {model_path}")

    # Evaluate on held-out window
    if len(eval_candles) >= min_bars:
        eval_env = TradingEnv(
            eval_candles,
            starting_balance=starting_balance,
            fee_rate=fee_rate,
            reward_type=reward_type,
            seed=seed,
        )
        metrics = _evaluate(eval_env, model, n_episodes=1)
    else:
        print("Eval window too small — skipping evaluation.")
        metrics = {}

    return metrics


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the Hogan RL (PPO) agent")
    parser.add_argument("--symbol", default="BTC/USD")
    parser.add_argument("--timeframe", default="5m")
    parser.add_argument(
        "--from-db",
        action="store_true",
        help="Load candles from data/hogan.db instead of a live exchange call",
    )
    parser.add_argument(
        "--limit", type=int, default=5000,
        help="Number of candles to fetch from the exchange (ignored with --from-db)",
    )
    parser.add_argument(
        "--timesteps", type=int,
        default=int(os.getenv("HOGAN_RL_TIMESTEPS", "500000")),
    )
    parser.add_argument(
        "--reward",
        default=os.getenv("HOGAN_RL_REWARD_TYPE", "risk_adjusted"),
        choices=["delta_equity", "risk_adjusted"],
    )
    parser.add_argument(
        "--model-path",
        default=os.getenv("HOGAN_RL_MODEL_PATH", "models/hogan_rl_policy.zip"),
    )
    parser.add_argument("--balance", type=float, default=10_000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    _check_deps()
    args = parse_args()

    if args.from_db:
        candles = _load_from_db(args.symbol, args.timeframe)
    else:
        candles = _load_from_exchange(args.symbol, args.timeframe, args.limit)

    print(f"Loaded {len(candles)} candles for {args.symbol} {args.timeframe}")

    metrics = train(
        candles,
        timesteps=args.timesteps,
        reward_type=args.reward,
        model_path=args.model_path,
        starting_balance=args.balance,
        seed=args.seed,
        verbose=args.verbose,
    )

    if metrics:
        print("\n── Evaluation (held-out 20 %) ──────────────────────────")
        for k, v in metrics.items():
            print(f"  {k:<25s} {v}")
        print()


if __name__ == "__main__":
    main()
