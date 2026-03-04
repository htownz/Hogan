"""Custom Gymnasium trading environment for the Hogan RL agent.

``TradingEnv`` wraps a candle DataFrame and exposes the standard Gymnasium
interface so any Stable-Baselines3 algorithm can train directly on historical
price data.

Observation (27-dim float32 vector)
-------------------------------------
The 24 features produced by :func:`~hogan_bot.ml.feature_row` plus three
position-state scalars appended at runtime:

    in_position      0.0 or 1.0
    unrealized_pnl   signed percentage relative to entry price
    bars_in_trade    number of bars held, normalised to [0, 1] over max_bars

Action space
------------
``Discrete(3)``  —  0 = hold, 1 = buy, 2 = sell

Reward modes (``reward_type`` constructor arg)
----------------------------------------------
``"delta_equity"``
    Step reward = fractional change in total equity.  Fast to train,
    aligns the agent to raw profitability.

``"risk_adjusted"``
    Step reward = fractional equity change
        - overtrading penalty if the agent flip-flops more than once per N bars
        - drawdown penalty proportional to depth below the equity high-water mark
    Produces more conservative, realistic trading behaviour.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

# Gymnasium is an optional dependency — import defensively.
try:
    import gymnasium as gym
    from gymnasium import spaces
    _GYM_AVAILABLE = True
except ImportError:  # pragma: no cover
    _GYM_AVAILABLE = False
    gym = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]

from hogan_bot.ml import build_feature_row

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

N_ML_FEATURES: int = 24       # must match _FEATURE_COLUMNS in ml.py
N_OBS: int = N_ML_FEATURES + 3  # + in_position, unrealized_pnl, bars_in_trade

# Reward shaping weights
_OVERTRADING_PENALTY: float = 0.002   # fraction of equity penalised per flip
_DRAWDOWN_PENALTY_SCALE: float = 0.5  # multiplies drawdown depth
_OVERTRADING_WINDOW: int = 6          # bars to check for flip-flops

# Drawdown guard: terminate an episode if drawdown exceeds this fraction
_MAX_EPISODE_DRAWDOWN: float = 0.25


class TradingEnv:
    """Gymnasium-compatible environment for bar-by-bar crypto trading.

    Parameters
    ----------
    candles:
        OHLCV DataFrame (``timestamp``, ``open``, ``high``, ``low``,
        ``close``, ``volume``) sorted oldest-first.
    starting_balance:
        Initial cash in USD.
    fee_rate:
        Round-trip transaction fee as a fraction (e.g. 0.0026 for 0.26 %).
    reward_type:
        ``"delta_equity"`` or ``"risk_adjusted"`` (see module docstring).
    max_bars_in_trade:
        Normalisation denominator for the ``bars_in_trade`` observation feature.
    min_history:
        Minimum number of bars needed before the first observation can be
        computed (must be ≥ the longest window used in feature engineering,
        typically 50).
    seed:
        Optional RNG seed for reproducibility.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        candles: pd.DataFrame,
        starting_balance: float = 10_000.0,
        fee_rate: float = 0.0026,
        reward_type: str = "risk_adjusted",
        max_bars_in_trade: int = 100,
        min_history: int = 60,
        seed: int | None = None,
    ) -> None:
        if not _GYM_AVAILABLE:
            raise ImportError(
                "gymnasium is not installed.  Run: pip install gymnasium stable-baselines3"
            )

        super().__init__()

        self.candles = candles.reset_index(drop=True)
        self.starting_balance = starting_balance
        self.fee_rate = fee_rate
        self.reward_type = reward_type
        self.max_bars_in_trade = max_bars_in_trade
        self.min_history = max(min_history, 60)
        self._rng = np.random.default_rng(seed)

        self.n_bars = len(self.candles)
        if self.n_bars < self.min_history + 2:
            raise ValueError(
                f"TradingEnv needs at least {self.min_history + 2} candles; "
                f"got {self.n_bars}.  Accumulate more data first."
            )

        # Gymnasium spaces
        low = np.full(N_OBS, -10.0, dtype=np.float32)
        high = np.full(N_OBS, 10.0, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        # State (reset on each episode)
        self._cursor: int = self.min_history
        self._cash: float = starting_balance
        self._position_qty: float = 0.0
        self._entry_price: float = 0.0
        self._bars_in_trade: int = 0
        self._equity_peak: float = starting_balance
        self._action_history: list[int] = []

    # ------------------------------------------------------------------
    # Gymnasium interface
    # ------------------------------------------------------------------

    @property
    def observation_space(self) -> "spaces.Box":
        return self._observation_space

    @observation_space.setter
    def observation_space(self, v: "spaces.Box") -> None:
        self._observation_space = v

    @property
    def action_space(self) -> "spaces.Discrete":
        return self._action_space

    @action_space.setter
    def action_space(self, v: "spaces.Discrete") -> None:
        self._action_space = v

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)

        self._cursor = self.min_history
        self._cash = self.starting_balance
        self._position_qty = 0.0
        self._entry_price = 0.0
        self._bars_in_trade = 0
        self._equity_peak = self.starting_balance
        self._action_history = []

        obs = self._get_obs()
        return obs, {}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance the environment by one bar.

        Returns
        -------
        obs, reward, terminated, truncated, info
        """
        assert 0 <= action <= 2, f"Invalid action {action}"

        price = float(self.candles["close"].iloc[self._cursor])
        prev_equity = self._total_equity(price)

        self._action_history.append(action)

        # ── Execute trade ───────────────────────────────────────────────
        if action == 1 and self._position_qty == 0.0:          # buy
            spend = self._cash * 0.95   # use 95% of available cash
            qty = spend / price * (1.0 - self.fee_rate)
            self._cash -= spend
            self._position_qty = qty
            self._entry_price = price
            self._bars_in_trade = 0

        elif action == 2 and self._position_qty > 0.0:         # sell
            proceeds = self._position_qty * price * (1.0 - self.fee_rate)
            self._cash += proceeds
            self._position_qty = 0.0
            self._entry_price = 0.0
            self._bars_in_trade = 0

        # ── Advance cursor ──────────────────────────────────────────────
        if self._position_qty > 0.0:
            self._bars_in_trade += 1

        self._cursor += 1
        next_price = float(self.candles["close"].iloc[self._cursor])
        new_equity = self._total_equity(next_price)

        # ── Update high-water mark ──────────────────────────────────────
        if new_equity > self._equity_peak:
            self._equity_peak = new_equity

        # ── Reward ─────────────────────────────────────────────────────
        reward = self._compute_reward(prev_equity, new_equity, action)

        # ── Termination ────────────────────────────────────────────────
        drawdown = (self._equity_peak - new_equity) / self._equity_peak
        terminated = bool(
            drawdown > _MAX_EPISODE_DRAWDOWN
            or self._cursor >= self.n_bars - 1
        )
        truncated = False

        obs = self._get_obs() if not terminated else self._zero_obs()

        info = {
            "equity": new_equity,
            "drawdown": drawdown,
            "position": self._position_qty > 0,
            "cursor": self._cursor,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        price = float(self.candles["close"].iloc[self._cursor])
        equity = self._total_equity(price)
        dd = (self._equity_peak - equity) / self._equity_peak
        print(
            f"  bar={self._cursor:>5d}  price={price:>10.2f}  "
            f"equity={equity:>10.2f}  dd={dd:.2%}  "
            f"pos={self._position_qty > 0}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _total_equity(self, price: float) -> float:
        return self._cash + self._position_qty * price

    def _get_obs(self) -> np.ndarray:
        window = self.candles.iloc[: self._cursor + 1]
        features = build_feature_row(window)   # 24-dim or None
        if features is None:
            ml_part = np.zeros(N_ML_FEATURES, dtype=np.float32)
        else:
            ml_part = np.array(features, dtype=np.float32)
            ml_part = np.clip(ml_part, -10.0, 10.0)

        price = float(self.candles["close"].iloc[self._cursor])
        in_pos = 1.0 if self._position_qty > 0.0 else 0.0
        upnl = (
            (price - self._entry_price) / max(self._entry_price, 1e-9)
            if self._position_qty > 0.0
            else 0.0
        )
        bars_norm = min(self._bars_in_trade / self.max_bars_in_trade, 1.0)

        pos_part = np.array([in_pos, float(upnl), float(bars_norm)], dtype=np.float32)
        obs = np.concatenate([ml_part, pos_part])
        return np.clip(obs, -10.0, 10.0).astype(np.float32)

    def _zero_obs(self) -> np.ndarray:
        return np.zeros(N_OBS, dtype=np.float32)

    def _compute_reward(
        self, prev_equity: float, new_equity: float, action: int
    ) -> float:
        delta = (new_equity - prev_equity) / max(prev_equity, 1e-9)

        if self.reward_type == "delta_equity":
            return float(delta)

        # "risk_adjusted": penalise overtrading and drawdown
        reward = float(delta)

        # Overtrading penalty: penalise if agent flip-flopped recently
        if len(self._action_history) >= _OVERTRADING_WINDOW:
            recent = self._action_history[-_OVERTRADING_WINDOW:]
            flips = sum(
                1 for i in range(1, len(recent))
                if recent[i] != recent[i - 1] and recent[i] != 0
            )
            if flips > 1:
                reward -= _OVERTRADING_PENALTY * (flips - 1)

        # Drawdown penalty
        if new_equity < self._equity_peak:
            drawdown = (self._equity_peak - new_equity) / self._equity_peak
            reward -= _DRAWDOWN_PENALTY_SCALE * drawdown * abs(delta) if delta < 0 else 0.0

        return float(reward)
