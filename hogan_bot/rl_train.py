"""Training CLI for the Hogan RL agent.

Usage
-----
    python -m hogan_bot.rl_train \\
        --symbol BTC/USD --timeframe 5m \\
        --from-db --timesteps 500000 --reward risk_adjusted

    # With checkpointing and evaluation callbacks:
    python -m hogan_bot.rl_train --from-db --checkpoint-freq 50000 --eval-freq 25000

    # Auto-load best hyperparameters if rl_tune was run first:
    # models/best_hparams.json is picked up automatically.

The trained PPO policy is saved to ``models/hogan_rl_policy.zip`` (or the
path specified by ``--model-path``).  The best model during training is saved
to ``models/best_model.zip`` when --eval-freq is set.
"""
from __future__ import annotations

import argparse
import json
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
    from stable_baselines3.common.callbacks import (
        CallbackList,
        CheckpointCallback,
        EvalCallback,
    )
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
    from hogan_bot.storage import get_connection, load_candles
    conn = get_connection()
    candles = load_candles(conn, symbol=symbol, timeframe=timeframe)
    conn.close()
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


def _load_hparams(model_dir: str = "models") -> dict:
    """Load best hyperparameters from Optuna tune run, if available."""
    path = Path(model_dir) / "best_hparams.json"
    if path.exists():
        with open(path) as f:
            hparams = json.load(f)
        print(f"Loaded hyperparameters from {path}")
        return hparams
    return {}


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
    slippage_pct: float = 0.001,
    min_trades: int = 3,
    seed: int = 42,
    verbose: int = 1,
    device: str = "cpu",
    checkpoint_freq: int = 0,
    eval_freq: int = 0,
    candles_1h: pd.DataFrame | None = None,
    candles_15m: pd.DataFrame | None = None,
    use_ext_features: bool = False,
    symbol: str = "BTC/USD",
) -> dict[str, float]:
    """Train a PPO agent and save the policy.

    Parameters
    ----------
    candles:
        Full 5m OHLCV history (oldest first).
    timesteps:
        Total environment steps for training.
    reward_type:
        One of ``"delta_equity"``, ``"risk_adjusted"``, ``"sharpe"``,
        ``"sortino"``.
    model_path:
        Destination for the saved ``.zip`` policy file.
    starting_balance:
        Paper-trading starting equity used in both training and eval envs.
    fee_rate:
        Per-side transaction fee fraction.
    slippage_pct:
        One-way slippage fraction applied at execution.
    min_trades:
        Minimum completed round-trips per episode; episodes with fewer are
        penalised.
    seed:
        Random seed for reproducibility.
    verbose:
        SB3 verbosity level (0 = silent, 1 = progress, 2 = debug).
    device:
        PyTorch device string (``"cpu"`` or ``"cuda"``).
    checkpoint_freq:
        Save a model checkpoint every this many steps (0 = disabled).
    eval_freq:
        Evaluate on the held-out window every this many steps and save the
        best model to ``models/best_model.zip`` (0 = disabled).
    candles_1h:
        Optional 1h candles for multi-timeframe observation.
    candles_15m:
        Optional 15m candles for multi-timeframe observation.

    Returns
    -------
    dict
        Evaluation metrics from the held-out 20 % window.
    """
    from hogan_bot.rl_env import TradingEnv

    split = int(len(candles) * 0.8)
    train_candles = candles.iloc[:split].reset_index(drop=True)
    eval_candles = candles.iloc[split:].reset_index(drop=True)

    # Split MTF candles at the same ratio if provided
    train_1h = eval_1h = None
    train_15m = eval_15m = None
    if candles_1h is not None and len(candles_1h) > 10:
        sp1h = int(len(candles_1h) * 0.8)
        train_1h = candles_1h.iloc[:sp1h].reset_index(drop=True)
        eval_1h = candles_1h.iloc[sp1h:].reset_index(drop=True)
    if candles_15m is not None and len(candles_15m) > 10:
        sp15 = int(len(candles_15m) * 0.8)
        train_15m = candles_15m.iloc[:sp15].reset_index(drop=True)
        eval_15m = candles_15m.iloc[sp15:].reset_index(drop=True)

    min_bars = 62

    if len(train_candles) < min_bars:
        sys.exit(
            f"Not enough training bars ({len(train_candles)}).  "
            "Accumulate more data or reduce the train/eval split."
        )

    # Open a DB connection for ext features (derivatives / on-chain / SPY)
    _db_conn = None
    if use_ext_features and (candles_1h is not None or candles_15m is not None):
        try:
            from hogan_bot.storage import get_connection
            _db_conn = get_connection()
        except Exception:
            pass

    env_kwargs = dict(
        starting_balance=starting_balance,
        fee_rate=fee_rate,
        slippage_pct=slippage_pct,
        reward_type=reward_type,
        min_trades=min_trades,
        seed=seed,
        symbol=symbol,
    )

    def _make_train_env():
        return TradingEnv(
            train_candles,
            candles_1h=train_1h,
            candles_15m=train_15m,
            db_conn=_db_conn,
            **env_kwargs,
        )

    env = DummyVecEnv([_make_train_env])

    # Load best hyperparameters from Optuna tune run if available
    hparams = _load_hparams(str(Path(model_path).parent))
    model = PPO(
        "MlpPolicy",
        env,
        verbose=verbose,
        seed=seed,
        device=device,
        n_steps=hparams.get("n_steps", 2048),
        batch_size=hparams.get("batch_size", 64),
        n_epochs=10,
        gamma=hparams.get("gamma", 0.99),
        gae_lambda=0.95,
        clip_range=hparams.get("clip_range", 0.2),
        ent_coef=hparams.get("ent_coef", 0.01),
        learning_rate=hparams.get("learning_rate", 3e-4),
        tensorboard_log=None,
    )

    # ── Callbacks ─────────────────────────────────────────────────────
    callbacks = []

    if checkpoint_freq > 0:
        ckpt_dir = str(Path(model_path).parent / "checkpoints")
        callbacks.append(
            CheckpointCallback(
                save_freq=checkpoint_freq,
                save_path=ckpt_dir,
                name_prefix="rl_checkpoint",
                verbose=0,
            )
        )
        print(f"Checkpoints every {checkpoint_freq:,} steps -> {ckpt_dir}/")

    if eval_freq > 0 and len(eval_candles) >= min_bars:
        eval_env = DummyVecEnv([lambda: TradingEnv(
            eval_candles,
            candles_1h=eval_1h,
            candles_15m=eval_15m,
            db_conn=_db_conn,
            **env_kwargs,
        )])
        best_dir = str(Path(model_path).parent)
        callbacks.append(
            EvalCallback(
                eval_env,
                best_model_save_path=best_dir,
                n_eval_episodes=3,
                eval_freq=eval_freq,
                verbose=0,
                warn=False,
            )
        )
        print(f"Eval every {eval_freq:,} steps; best model -> {best_dir}/best_model.zip")

    callback = CallbackList(callbacks) if callbacks else None

    print(
        f"Training PPO for {timesteps:,} steps  "
        f"(reward={reward_type}, slippage={slippage_pct:.3f}, "
        f"train_bars={len(train_candles)}) ..."
    )
    model.learn(total_timesteps=timesteps, callback=callback)

    Path(model_path).parent.mkdir(parents=True, exist_ok=True)
    model.save(model_path.removesuffix(".zip"))
    print(f"Policy saved -> {model_path}")

    # Evaluate on held-out window
    if len(eval_candles) >= min_bars:
        eval_env_single = TradingEnv(
            eval_candles,
            candles_1h=eval_1h,
            candles_15m=eval_15m,
            db_conn=_db_conn,
            **env_kwargs,
        )
        metrics = _evaluate(eval_env_single, model, n_episodes=1)
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
        choices=["delta_equity", "risk_adjusted", "sharpe", "sortino"],
    )
    parser.add_argument(
        "--model-path",
        default=os.getenv("HOGAN_RL_MODEL_PATH", "models/hogan_rl_policy.zip"),
    )
    parser.add_argument("--balance", type=float, default=10_000.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument(
        "--device", default="cpu",
        help="PyTorch device: 'cpu' (default, faster for small MLP) or 'cuda'",
    )
    parser.add_argument(
        "--slippage", type=float, default=0.001,
        help="One-way slippage fraction (default 0.001 = 0.1%%)",
    )
    parser.add_argument(
        "--min-trades", type=int, default=3,
        help="Min round-trips per episode before penalty is applied",
    )
    parser.add_argument(
        "--checkpoint-freq", type=int, default=0,
        metavar="STEPS",
        help="Save a checkpoint every N steps (0 = disabled)",
    )
    parser.add_argument(
        "--eval-freq", type=int, default=0,
        metavar="STEPS",
        help="Evaluate on held-out data every N steps and save best model (0 = disabled)",
    )
    parser.add_argument(
        "--load-1h", action="store_true",
        help="Load 1h candles from DB for multi-timeframe observation",
    )
    parser.add_argument(
        "--load-15m", action="store_true",
        help="Load 15m candles from DB for multi-timeframe observation",
    )
    parser.add_argument(
        "--ext-features", action="store_true",
        help="Load derivatives / on-chain / SPY ext features from DB (requires prior fetch)",
    )
    parser.add_argument(
        "--symbol-str", default="BTC/USD", dest="symbol_str",
        help="Trading symbol for DB lookups (default BTC/USD)",
    )
    return parser.parse_args()


def main() -> None:
    _check_deps()
    args = parse_args()

    if args.from_db:
        candles = _load_from_db(args.symbol, args.timeframe)
    else:
        candles = _load_from_exchange(args.symbol, args.timeframe, args.limit)

    print(f"Loaded {len(candles)} candles ({args.timeframe}) for {args.symbol}")

    candles_1h = None
    candles_15m = None
    if args.load_1h:
        candles_1h = _load_from_db(args.symbol, "1h")
        print(f"Loaded {len(candles_1h)} candles (1h) for {args.symbol}")
    if args.load_15m:
        candles_15m = _load_from_db(args.symbol, "15m")
        print(f"Loaded {len(candles_15m)} candles (15m) for {args.symbol}")

    metrics = train(
        candles,
        timesteps=args.timesteps,
        reward_type=args.reward,
        model_path=args.model_path,
        starting_balance=args.balance,
        fee_rate=0.0026,
        slippage_pct=args.slippage,
        min_trades=args.min_trades,
        seed=args.seed,
        verbose=args.verbose,
        device=args.device,
        checkpoint_freq=args.checkpoint_freq,
        eval_freq=args.eval_freq,
        candles_1h=candles_1h,
        candles_15m=candles_15m,
        use_ext_features=args.ext_features,
        symbol=args.symbol_str,
    )

    if metrics:
        print("\n-- Evaluation (held-out 20 %) --------------------------")
        for k, v in metrics.items():
            print(f"  {k:<25s} {v}")
        print()


if __name__ == "__main__":
    main()
