"""Custom Gymnasium trading environment for the Hogan RL agent.

``TradingEnv`` wraps a candle DataFrame and exposes the standard Gymnasium
interface so any Stable-Baselines3 algorithm can train directly on historical
price data.

Observation vector
------------------
Base mode (27-dim):
    24 ML features from :func:`~hogan_bot.ml.build_feature_row`
    + 3 position-state scalars: in_position, unrealized_pnl, bars_in_trade

Extended mode (47-dim, requires MTF candles):
    38 features from :func:`~hogan_bot.features_mtf.build_feature_row_extended`
    + 6 external features (derivatives / on-chain, when available)
    + 3 position-state scalars

Action space
------------
``Discrete(3)``  —  0 = hold, 1 = buy, 2 = sell

Reward modes (``reward_type`` constructor arg)
----------------------------------------------
``"delta_equity"``
    Step reward = fractional change in total equity.

``"risk_adjusted"``
    Step reward = equity change minus overtrading and drawdown penalties.

``"sharpe"``
    Step reward = equity change normalised by rolling volatility of recent
    returns.  Encourages the agent to seek stable positive returns and avoid
    high-variance strategies.

``"sortino"``
    Like ``"sharpe"`` but normalises by downside deviation only, penalising
    losses more than upside variance.
"""

from __future__ import annotations

from collections import deque
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
# Base class — gym.Env when gymnasium is available, plain object otherwise
# so the module stays importable without the dependency.
# ---------------------------------------------------------------------------

_GymBase = gym.Env if _GYM_AVAILABLE else object  # type: ignore[misc]

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# Base feature count — must match _FEATURE_COLUMNS in ml.py (59 as of hardening roadmap)
N_ML_FEATURES: int = 59
# Extended: 59 base + 3 ICT experimental + 14 MTF + 20 ext
N_ML_FEATURES_EXTENDED: int = 96
N_OBS: int = N_ML_FEATURES + 3          # 62 — base mode
N_OBS_EXTENDED: int = N_ML_FEATURES_EXTENDED + 3  # 99 — extended mode

# Reward shaping weights (risk_adjusted mode)
_OVERTRADING_PENALTY: float = 0.002
_DRAWDOWN_PENALTY_SCALE: float = 0.5
_OVERTRADING_WINDOW: int = 6

# Rolling-window size for Sharpe/Sortino reward normalisation
_RETURN_BUFFER_SIZE: int = 50
_MIN_VOL: float = 1e-6  # floor to prevent division by zero

# Drawdown guard: terminate an episode if drawdown exceeds this fraction
_MAX_EPISODE_DRAWDOWN: float = 0.25

# Min-trade penalty applied when episode ends with too few round-trips
_MIN_TRADE_PENALTY: float = 0.05


class TradingEnv(_GymBase):
    """Gymnasium-compatible environment for bar-by-bar crypto trading.

    Parameters
    ----------
    candles:
        5m OHLCV DataFrame sorted oldest-first.
    starting_balance:
        Initial cash in USD.
    fee_rate:
        Per-side transaction fee as a fraction (e.g. 0.0026 for 0.26 %).
    slippage_pct:
        One-way slippage fraction applied at execution (e.g. 0.001 = 0.1 %).
        Buys fill at ``close * (1 + slippage_pct)``, sells at
        ``close * (1 - slippage_pct)``.
    reward_type:
        ``"delta_equity"``, ``"risk_adjusted"``, ``"sharpe"``, or
        ``"sortino"`` (see module docstring).
    min_trades:
        Minimum number of completed round-trips per episode.  If the agent
        finishes an episode with fewer trades a penalty of
        ``_MIN_TRADE_PENALTY`` is deducted from the terminal reward to
        prevent gaming risk metrics by staying flat.
    max_bars_in_trade:
        Normalisation denominator for the ``bars_in_trade`` observation feature.
    min_history:
        Minimum bars needed before the first observation can be computed.
    candles_1h:
        Optional 1-hour OHLCV DataFrame for multi-timeframe features.
    candles_15m:
        Optional 15-minute OHLCV DataFrame for multi-timeframe features.
    db_conn:
        Optional open SQLite connection for ext feature lookup (derivatives /
        on-chain / SPY).  Pass ``get_connection()`` from storage to activate
        the 6 external features.  The env does NOT close this connection.
    symbol:
        Trading symbol string, used for DB lookups (default ``"BTC/USD"``).
    seed:
        Optional RNG seed for reproducibility.
    """

    metadata: dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        candles: pd.DataFrame,
        starting_balance: float = 10_000.0,
        fee_rate: float = 0.0026,
        slippage_pct: float = 0.001,
        reward_type: str = "risk_adjusted",
        min_trades: int = 3,
        max_bars_in_trade: int = 100,
        min_history: int = 60,
        candles_1h: pd.DataFrame | None = None,
        candles_15m: pd.DataFrame | None = None,
        db_conn=None,
        symbol: str = "BTC/USD",
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
        self.slippage_pct = slippage_pct
        self.reward_type = reward_type
        self.min_trades = min_trades
        self.max_bars_in_trade = max_bars_in_trade
        self.min_history = max(min_history, 60)
        self.candles_1h = candles_1h
        self.candles_15m = candles_15m
        self.db_conn = db_conn
        self.symbol = symbol
        self._rng = np.random.default_rng(seed)

        # Determine observation dimensionality
        self._use_extended = (candles_1h is not None) or (candles_15m is not None)
        self._n_obs = N_OBS_EXTENDED if self._use_extended else N_OBS

        self.n_bars = len(self.candles)
        if self.n_bars < self.min_history + 2:
            raise ValueError(
                f"TradingEnv needs at least {self.min_history + 2} candles; "
                f"got {self.n_bars}.  Accumulate more data first."
            )

        # Gymnasium spaces
        low = np.full(self._n_obs, -10.0, dtype=np.float32)
        high = np.full(self._n_obs, 10.0, dtype=np.float32)
        self.observation_space = spaces.Box(low=low, high=high, dtype=np.float32)
        self.action_space = spaces.Discrete(3)

        # Episode state — reset in reset()
        self._cursor: int = self.min_history
        self._cash: float = starting_balance
        self._position_qty: float = 0.0
        self._entry_price: float = 0.0
        self._bars_in_trade: int = 0
        self._equity_peak: float = starting_balance
        self._action_history: list[int] = []
        self._trade_count: int = 0
        self._return_buffer: deque[float] = deque(maxlen=_RETURN_BUFFER_SIZE)

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
        self._trade_count = 0
        self._return_buffer = deque(maxlen=_RETURN_BUFFER_SIZE)

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
        if not (0 <= action <= 2):
            raise ValueError(f"Invalid action {action}; expected 0, 1, or 2")

        price = float(self.candles["close"].iloc[self._cursor])
        prev_equity = self._total_equity(price)

        self._action_history.append(action)

        # ── Execute trade with slippage ─────────────────────────────────
        if action == 1 and self._position_qty == 0.0:          # buy
            fill_price = price * (1.0 + self.slippage_pct)
            spend = self._cash * 0.95
            qty = spend / fill_price * (1.0 - self.fee_rate)
            self._cash -= spend
            self._position_qty = qty
            self._entry_price = fill_price
            self._bars_in_trade = 0

        elif action == 2 and self._position_qty > 0.0:         # sell
            fill_price = price * (1.0 - self.slippage_pct)
            proceeds = self._position_qty * fill_price * (1.0 - self.fee_rate)
            self._cash += proceeds
            self._position_qty = 0.0
            self._entry_price = 0.0
            self._bars_in_trade = 0
            self._trade_count += 1  # count completed round-trips

        # ── Advance cursor ──────────────────────────────────────────────
        if self._position_qty > 0.0:
            self._bars_in_trade += 1

        self._cursor += 1
        next_price = float(self.candles["close"].iloc[self._cursor])
        new_equity = self._total_equity(next_price)

        # ── Update high-water mark ──────────────────────────────────────
        if new_equity > self._equity_peak:
            self._equity_peak = new_equity

        # ── Track step return for Sharpe/Sortino ───────────────────────
        step_ret = (new_equity - prev_equity) / max(prev_equity, 1e-9)
        self._return_buffer.append(step_ret)

        # ── Termination ────────────────────────────────────────────────
        drawdown = (self._equity_peak - new_equity) / max(self._equity_peak, 1e-9)
        terminated = bool(
            drawdown > _MAX_EPISODE_DRAWDOWN
            or self._cursor >= self.n_bars - 1
        )

        # ── Reward ─────────────────────────────────────────────────────
        reward = self._compute_reward(prev_equity, new_equity, action, terminated)

        truncated = False
        obs = self._get_obs() if not terminated else self._zero_obs()

        info = {
            "equity": new_equity,
            "drawdown": drawdown,
            "position": self._position_qty > 0,
            "cursor": self._cursor,
            "trade_count": self._trade_count,
        }
        return obs, float(reward), terminated, truncated, info

    def render(self) -> None:
        price = float(self.candles["close"].iloc[self._cursor])
        equity = self._total_equity(price)
        dd = (self._equity_peak - equity) / max(self._equity_peak, 1e-9)
        print(
            f"  bar={self._cursor:>5d}  price={price:>10.2f}  "
            f"equity={equity:>10.2f}  dd={dd:.2%}  "
            f"pos={self._position_qty > 0}  trades={self._trade_count}"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _total_equity(self, price: float) -> float:
        return self._cash + self._position_qty * price

    def _get_obs(self) -> np.ndarray:
        window_5m = self.candles.iloc[: self._cursor + 1]

        if self._use_extended:
            try:
                from hogan_bot.features_mtf import build_feature_row_extended
                # Slice MTF candles up to current bar timestamp
                ts = self.candles["timestamp"].iloc[self._cursor] if "timestamp" in self.candles.columns else None
                w1h = _slice_up_to(self.candles_1h, ts) if self.candles_1h is not None else None
                w15m = _slice_up_to(self.candles_15m, ts) if self.candles_15m is not None else None
                features = build_feature_row_extended(
                    window_5m, w1h, w15m,
                    conn=self.db_conn,
                    symbol=self.symbol,
                )
            except Exception:
                features = None

            # Pad / truncate to N_ML_FEATURES_EXTENDED
            if features is None:
                ml_part = np.zeros(N_ML_FEATURES_EXTENDED, dtype=np.float32)
            else:
                arr = np.array(features[:N_ML_FEATURES_EXTENDED], dtype=np.float32)
                if len(arr) < N_ML_FEATURES_EXTENDED:
                    arr = np.pad(arr, (0, N_ML_FEATURES_EXTENDED - len(arr)))
                ml_part = np.clip(arr, -10.0, 10.0)
        else:
            features = build_feature_row(window_5m, use_champion_features=False)
            if features is None:
                ml_part = np.zeros(N_ML_FEATURES, dtype=np.float32)
            else:
                arr = np.array(features[:N_ML_FEATURES], dtype=np.float32)
                if len(arr) < N_ML_FEATURES:
                    arr = np.pad(arr, (0, N_ML_FEATURES - len(arr)))
                ml_part = np.clip(arr, -10.0, 10.0)

        price = float(self.candles["close"].iloc[self._cursor])
        in_pos = 1.0 if self._position_qty > 0.0 else 0.0
        upnl = (
            (price - self._entry_price) / max(self._entry_price, 1e-9)
            if self._position_qty > 0.0
            else 0.0
        )
        bars_norm = min(self._bars_in_trade / max(self.max_bars_in_trade, 1), 1.0)

        pos_part = np.array([in_pos, float(upnl), float(bars_norm)], dtype=np.float32)
        obs = np.concatenate([ml_part, pos_part])
        return np.clip(obs, -10.0, 10.0).astype(np.float32)

    def _zero_obs(self) -> np.ndarray:
        return np.zeros(self._n_obs, dtype=np.float32)

    def _compute_reward(
        self,
        prev_equity: float,
        new_equity: float,
        action: int,
        terminated: bool,
    ) -> float:
        delta = (new_equity - prev_equity) / max(prev_equity, 1e-9)

        if self.reward_type == "delta_equity":
            reward = float(delta)

        elif self.reward_type == "sharpe":
            reward = self._sharpe_reward(delta, downside_only=False)

        elif self.reward_type == "sortino":
            reward = self._sharpe_reward(delta, downside_only=True)

        else:  # "risk_adjusted"
            reward = float(delta)

            # Overtrading penalty
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
                drawdown = (self._equity_peak - new_equity) / max(self._equity_peak, 1e-9)
                reward -= _DRAWDOWN_PENALTY_SCALE * drawdown * abs(delta) if delta < 0 else 0.0

        # ── Min-trade penalty at episode end ───────────────────────────
        if terminated and self._trade_count < self.min_trades:
            reward -= _MIN_TRADE_PENALTY

        return float(reward)

    def _sharpe_reward(self, delta: float, downside_only: bool) -> float:
        """Normalise the step return by rolling vol or downside-vol."""
        buf = list(self._return_buffer)
        if len(buf) < 2:
            return float(delta)

        buf_arr = np.array(buf, dtype=np.float64)
        if downside_only:
            neg = buf_arr[buf_arr < 0]
            vol = float(np.sqrt(np.mean(neg ** 2))) if len(neg) > 0 else _MIN_VOL
        else:
            vol = float(np.std(buf_arr))

        vol = max(vol, _MIN_VOL)
        # Scale to keep rewards in a similar range as delta_equity
        return float(delta / vol * 0.01)


# ---------------------------------------------------------------------------
# Helper: slice candles up to a timestamp
# ---------------------------------------------------------------------------

def _slice_up_to(df: pd.DataFrame | None, ts) -> pd.DataFrame | None:
    """Return rows of *df* with timestamp <= *ts*, or the full df if ts is None."""
    if df is None or ts is None:
        return df
    try:
        mask = df["timestamp"] <= ts
        sliced = df.loc[mask]
        return sliced if len(sliced) > 0 else df
    except Exception:
        return df
