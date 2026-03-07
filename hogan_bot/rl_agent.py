"""Inference wrapper for the trained RL policy.

Usage
-----
    from hogan_bot.rl_agent import load_rl_policy, predict_rl_action

    policy = load_rl_policy("models/hogan_rl_policy.zip")

    # Base 27-dim obs (fast, no DB):
    action = predict_rl_action(candles_df, policy)

    # Extended 73-dim obs (use when policy was trained with --ext-features):
    action = predict_rl_action(
        candles_df, policy,
        use_ext_features=True,
        candles_1h=candles_1h_df,
        candles_15m=candles_15m_df,
        db_conn=conn,
        symbol="BTC/USD",
    )
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
from hogan_bot.rl_env import (
    N_ML_FEATURES,
    N_ML_FEATURES_EXTENDED,
    N_OBS,
    N_OBS_EXTENDED,
)

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
    # Extended obs mode — set when policy was trained with --ext-features
    use_ext_features: bool = False,
    candles_1h: pd.DataFrame | None = None,
    candles_15m: pd.DataFrame | None = None,
    candles_10m: pd.DataFrame | None = None,
    candles_30m: pd.DataFrame | None = None,
    extended_mtf: bool = False,
    db_conn: Any | None = None,
    symbol: str = "BTC/USD",
) -> str:
    """Return the RL agent's action for the current bar.

    Supports two observation modes:

    * **Base mode** (default): constructs the 27-dimensional vector using
      :func:`~hogan_bot.ml.build_feature_row`.  Use when the policy was
      trained *without* ``--ext-features``.

    * **Extended mode** (``use_ext_features=True``): constructs the
      73-dimensional vector via
      :func:`~hogan_bot.features_mtf.build_feature_row_extended`, which
      includes MTF candles (1h/15m) and 20 external features from the DB.
      Use when the policy was trained *with* ``--ext-features``.

    Parameters
    ----------
    candles:
        5m OHLCV DataFrame ending at the current bar (oldest-first).
        At least 60 rows are required for non-zero features.
    policy:
        A loaded SB3 PPO model (output of :func:`load_rl_policy`).
    in_position:
        Whether the caller currently holds a long position.
    unrealized_pnl_pct:
        Current unrealised P&L as a fraction (e.g. 0.02 = +2 %).
    bars_in_trade:
        Number of bars the current position has been held.
    max_bars_in_trade:
        Normalisation denominator for *bars_in_trade* (must match
        the value used during training — default 100).
    use_ext_features:
        When ``True``, use the 73-dim extended obs path.
    candles_1h:
        1h OHLCV DataFrame for MTF features (``None`` = zeros).
    candles_15m:
        15m OHLCV DataFrame for MTF features (``None`` = zeros).
    db_conn:
        Open SQLite connection for ext feature lookup (``None`` = zeros).
    symbol:
        Trading symbol used for DB lookups (e.g. ``"BTC/USD"``).

    Returns
    -------
    str
        One of ``"buy"``, ``"sell"``, ``"hold"``.
    """
    if use_ext_features:
        from hogan_bot.features_mtf import build_feature_row_extended
        features = build_feature_row_extended(
            candles,
            candles_1h=candles_1h,
            candles_15m=candles_15m,
            candles_10m=candles_10m,
            candles_30m=candles_30m,
            conn=db_conn,
            symbol=symbol,
            extended_mtf=extended_mtf,
        )
        n_ml = N_ML_FEATURES_EXTENDED
        n_obs = N_OBS_EXTENDED
    else:
        features = build_feature_row(candles)
        n_ml = N_ML_FEATURES
        n_obs = N_OBS

    if features is None:
        ml_part = np.zeros(n_ml, dtype=np.float32)
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

    if obs.shape != (n_obs,):
        raise ValueError(
            f"Obs shape mismatch: expected ({n_obs},), got {obs.shape}. "
            f"Ensure use_ext_features matches the policy's training mode."
        )

    action_int, _ = policy.predict(obs, deterministic=True)
    return _ACTION_MAP.get(int(action_int), "hold")
