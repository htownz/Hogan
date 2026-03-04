"""Inference wrapper for the trained RL policy.

Usage
-----
    from hogan_bot.rl_agent import load_rl_policy, predict_rl_action

    policy = load_rl_policy("models/hogan_rl_policy.zip")
    action = predict_rl_action(candles_df, policy)   # "buy" | "sell" | "hold"
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

# stable-baselines3 is an optional dependency
try:
    from stable_baselines3 import PPO as _PPO
    _SB3_AVAILABLE = True
except ImportError:  # pragma: no cover
    _SB3_AVAILABLE = False
    _PPO = None  # type: ignore[assignment,misc]

from hogan_bot.ml import build_feature_row
from hogan_bot.rl_env import N_ML_FEATURES, N_OBS

_ACTION_MAP: dict[int, str] = {0: "hold", 1: "buy", 2: "sell"}


def load_rl_policy(model_path: str) -> Any:
    """Load a PPO policy saved by :mod:`hogan_bot.rl_train`.

    Parameters
    ----------
    model_path:
        Path to the ``.zip`` file produced by ``PPO.save()``.

    Raises
    ------
    ImportError
        When ``stable-baselines3`` is not installed.
    FileNotFoundError
        When *model_path* does not exist on disk.
    """
    if not _SB3_AVAILABLE:
        raise ImportError(
            "stable-baselines3 is not installed.  "
            "Run: pip install stable-baselines3 gymnasium"
        )
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"RL policy not found: {model_path}")
    return _PPO.load(str(path))


def predict_rl_action(
    candles: pd.DataFrame,
    policy: Any,
    *,
    in_position: bool = False,
    unrealized_pnl_pct: float = 0.0,
    bars_in_trade: int = 0,
    max_bars_in_trade: int = 100,
) -> str:
    """Return the RL agent's action for the current bar.

    Constructs the 27-dimensional observation vector from the last window of
    *candles*, calls :meth:`policy.predict`, and maps the integer action to
    ``"buy"``, ``"sell"``, or ``"hold"``.

    Parameters
    ----------
    candles:
        OHLCV DataFrame ending at the current bar.  At least 60 rows
        are needed for the feature-engineering to produce a non-zero vector.
    policy:
        A loaded SB3 PPO model (output of :func:`load_rl_policy`).
    in_position:
        Whether the caller currently holds a long position.
    unrealized_pnl_pct:
        Current unrealised P&L as a fraction (e.g. 0.02 = +2 %).
    bars_in_trade:
        Number of bars the current position has been held.
    max_bars_in_trade:
        Normalisation denominator for *bars_in_trade* (must match the value
        used during training — default 100).

    Returns
    -------
    str
        One of ``"buy"``, ``"sell"``, ``"hold"``.
    """
    features = build_feature_row(candles)
    if features is None:
        ml_part = np.zeros(N_ML_FEATURES, dtype=np.float32)
    else:
        ml_part = np.clip(np.array(features, dtype=np.float32), -10.0, 10.0)

    bars_norm = min(bars_in_trade / max(max_bars_in_trade, 1), 1.0)
    pos_part = np.array(
        [float(in_position), float(unrealized_pnl_pct), float(bars_norm)],
        dtype=np.float32,
    )

    obs = np.clip(
        np.concatenate([ml_part, pos_part]), -10.0, 10.0
    ).astype(np.float32)

    assert obs.shape == (N_OBS,), f"Expected obs shape ({N_OBS},), got {obs.shape}"

    action_int, _ = policy.predict(obs, deterministic=True)
    return _ACTION_MAP.get(int(action_int), "hold")
