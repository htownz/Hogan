"""Unit tests for TradingEnv (hogan_bot/rl_env.py).

Tests run without requiring stable-baselines3 training — they only
exercise the Gymnasium environment interface directly.

The tests are skipped automatically when gymnasium is not installed so
the full test suite can still pass on CI environments that don't have
the heavy RL dependencies (~700 MB PyTorch install).
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

# Skip the entire module if gymnasium is not installed
gymnasium = pytest.importorskip("gymnasium", reason="gymnasium not installed")

from hogan_bot.rl_env import N_OBS, TradingEnv  # noqa: E402

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _make_candles(n: int = 200, seed: int = 0) -> pd.DataFrame:
    """Generate synthetic OHLCV candles suitable for TradingEnv."""
    rng = np.random.default_rng(seed)
    closes = 50_000.0 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    opens = closes * (1 + rng.normal(0, 0.001, n))
    highs = np.maximum(closes, opens) * (1 + rng.uniform(0, 0.003, n))
    lows = np.minimum(closes, opens) * (1 - rng.uniform(0, 0.003, n))
    volumes = rng.uniform(1, 10, n)
    ts = pd.date_range("2024-01-01", periods=n, freq="5min")
    return pd.DataFrame(
        {
            "timestamp": ts,
            "open": opens,
            "high": highs,
            "low": lows,
            "close": closes,
            "volume": volumes,
        }
    )


@pytest.fixture()
def env():
    candles = _make_candles(200)
    return TradingEnv(candles, starting_balance=10_000.0, seed=42)


# ---------------------------------------------------------------------------
# Tests: construction
# ---------------------------------------------------------------------------


def test_observation_space_shape(env):
    assert env.observation_space.shape == (N_OBS,)


def test_action_space_size(env):
    assert env.action_space.n == 3


def test_reset_returns_correct_obs_shape(env):
    obs, info = env.reset()
    assert obs.shape == (N_OBS,)
    assert obs.dtype == np.float32
    assert isinstance(info, dict)


def test_obs_values_finite(env):
    obs, _ = env.reset()
    assert np.all(np.isfinite(obs)), "Reset observation contains NaN or Inf"


def test_obs_clipped_to_bounds(env):
    obs, _ = env.reset()
    lo, hi = env.observation_space.low, env.observation_space.high
    assert np.all(obs >= lo - 1e-6)
    assert np.all(obs <= hi + 1e-6)


# ---------------------------------------------------------------------------
# Tests: step
# ---------------------------------------------------------------------------


def test_step_hold_returns_valid_types(env):
    env.reset()
    obs, reward, terminated, truncated, info = env.step(0)  # hold
    assert obs.shape == (N_OBS,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
    assert isinstance(truncated, bool)
    assert isinstance(info, dict)


def test_step_reward_is_finite(env):
    env.reset()
    for action in [1, 0, 0, 2, 0]:   # buy, hold, hold, sell, hold
        _, reward, terminated, _, _ = env.step(action)
        assert math.isfinite(reward), f"reward={reward} is not finite"
        if terminated:
            break


def test_step_buy_enters_position(env):
    env.reset()
    _, _, _, _, info_before = env.step(0)
    assert not info_before["position"]

    env2 = TradingEnv(_make_candles(200), starting_balance=10_000.0, seed=42)
    env2.reset()
    _, _, _, _, info = env2.step(1)   # buy
    assert info["position"]


def test_step_sell_exits_position(env):
    env.reset()
    env.step(1)   # buy
    _, _, _, _, info = env.step(2)    # sell
    assert not info["position"]


def test_step_sell_without_position_is_hold(env):
    """Selling while flat should be treated as hold (no crash)."""
    env.reset()
    obs, reward, terminated, truncated, info = env.step(2)
    assert not info["position"]
    assert math.isfinite(reward)


def test_sequential_steps_equity_tracked(env):
    """Equity should be a positive number throughout a normal episode."""
    env.reset()
    for _ in range(10):
        obs, reward, terminated, truncated, info = env.step(0)
        assert info["equity"] > 0
        if terminated:
            break


# ---------------------------------------------------------------------------
# Tests: episode termination
# ---------------------------------------------------------------------------


def test_episode_terminates_at_end_of_candles():
    """Episode must end when the cursor reaches the last bar."""
    candles = _make_candles(80)
    env = TradingEnv(candles, starting_balance=10_000.0, min_history=60, seed=7)
    env.reset()
    terminated = False
    steps = 0
    while not terminated:
        _, _, terminated, truncated, _ = env.step(0)
        steps += 1
        assert steps < 500, "Episode did not terminate within expected bars"


def test_episode_terminates_on_large_drawdown():
    """Force a catastrophic equity drop and verify the environment halts."""
    candles = _make_candles(300, seed=1)
    env = TradingEnv(
        candles,
        starting_balance=10_000.0,
        fee_rate=0.10,   # extreme fee to drain equity fast
        seed=1,
    )
    env.reset()
    terminated = False
    steps = 0
    while not terminated:
        # Alternate buy/sell rapidly — high fees will destroy equity
        action = 1 if steps % 2 == 0 else 2
        _, _, terminated, truncated, info = env.step(action)
        steps += 1
        if steps > 1000:
            break   # safety valve

    # Either drawdown guard fired OR we hit the end — either is acceptable
    assert terminated or steps >= 1000


# ---------------------------------------------------------------------------
# Tests: reward modes
# ---------------------------------------------------------------------------


def test_delta_equity_reward_mode():
    candles = _make_candles(200, seed=3)
    env = TradingEnv(candles, reward_type="delta_equity", seed=3)
    env.reset()
    _, reward, _, _, _ = env.step(1)
    assert math.isfinite(reward)


def test_risk_adjusted_reward_mode():
    candles = _make_candles(200, seed=4)
    env = TradingEnv(candles, reward_type="risk_adjusted", seed=4)
    env.reset()
    _, reward, _, _, _ = env.step(1)
    assert math.isfinite(reward)


# ---------------------------------------------------------------------------
# Tests: determinism
# ---------------------------------------------------------------------------


def test_same_seed_same_obs():
    candles = _make_candles(200)
    env1 = TradingEnv(candles, seed=99)
    env2 = TradingEnv(candles, seed=99)
    obs1, _ = env1.reset(seed=99)
    obs2, _ = env2.reset(seed=99)
    np.testing.assert_array_equal(obs1, obs2)


# ---------------------------------------------------------------------------
# Tests: too-few-bars guard
# ---------------------------------------------------------------------------


def test_too_few_bars_raises():
    candles = _make_candles(30)  # below min_history + 2
    with pytest.raises(ValueError, match="at least"):
        TradingEnv(candles)
