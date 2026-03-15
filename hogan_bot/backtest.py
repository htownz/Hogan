from __future__ import annotations

import math
import types
from dataclasses import dataclass, field

import pandas as pd

from hogan_bot.agent_pipeline import AgentPipeline
from hogan_bot.champion import apply_champion_mode, is_champion_mode
from hogan_bot.decision import (
    apply_ml_filter, edge_gate, entry_quality_gate, ml_confidence,
    estimate_spread_from_candles, ranging_gate,
)
from hogan_bot.exit_model import ExitEvaluator
from hogan_bot.expectancy import ExpectancyTracker
from hogan_bot.indicators import compute_atr
from hogan_bot.ml import TrainedModel, predict_up_probability
from hogan_bot.paper import PaperPortfolio
from hogan_bot.regime import detect_regime, reset_regime_history
from hogan_bot.risk import DrawdownGuard, calculate_position_size
from hogan_bot.timeframe_utils import bars_per_year as tf_bars_per_year
from hogan_bot.timeframe_utils import infer_timeframe_from_candles

# Annualisation constant for 1-hour 24/7 crypto bars (fallback when timeframe unknown).
# 365 days × 24 hours = 8 760 bars/year.
_BARS_PER_YEAR_DEFAULT: float = 8_760.0


def _bar_ts_ms(candles: pd.DataFrame, bar_idx: int) -> int:
    """Return ts_ms for the bar at bar_idx (candles may have 'timestamp' or 'ts_ms')."""
    row = candles.iloc[bar_idx]
    if "ts_ms" in row.index:
        return int(row["ts_ms"])
    ts = row.get("timestamp", row)
    if hasattr(ts, "timestamp"):
        return int(ts.timestamp() * 1000)
    return int(ts)


@dataclass
class BacktestResult:
    start_equity: float
    end_equity: float
    total_return_pct: float
    max_drawdown_pct: float
    trades: int
    win_rate: float
    # Risk-adjusted performance ratios (None when not enough data to compute)
    sharpe_ratio: float | None = None
    sortino_ratio: float | None = None
    calmar_ratio: float | None = None
    # bars_per_year used for annualization (for result versioning / reproducibility)
    bars_per_year: float = _BARS_PER_YEAR_DEFAULT
    equity_curve: list[float] = field(default_factory=list)
    trade_log: list[dict] = field(default_factory=list)
    # Full closed-trade records for ML learning: entry_bar_idx, entry_ts_ms, exit_bar_idx,
    # exit_ts_ms, side, entry_price, exit_price, qty, pnl_usd, pnl_pct
    closed_trades: list[dict] = field(default_factory=list)
    expectancy_report: dict = field(default_factory=dict)
    regime_log: list[str | None] = field(default_factory=list)
    signal_funnel: dict = field(default_factory=dict)

    def summary_dict(self) -> dict:
        """Return all scalar fields as a plain dict (omits large lists)."""
        d = {
            "start_equity": self.start_equity,
            "end_equity": self.end_equity,
            "total_return_pct": round(self.total_return_pct, 4),
            "max_drawdown_pct": round(self.max_drawdown_pct, 4),
            "trades": self.trades,
            "win_rate": round(self.win_rate, 4),
            "sharpe_ratio": round(self.sharpe_ratio, 4) if self.sharpe_ratio is not None else None,
            "sortino_ratio": round(self.sortino_ratio, 4) if self.sortino_ratio is not None else None,
            "calmar_ratio": round(self.calmar_ratio, 4) if self.calmar_ratio is not None else None,
        }
        if self.signal_funnel:
            d["signal_funnel"] = self.signal_funnel
        return d


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------


def _compute_max_drawdown(equity_curve: list[float]) -> float:
    peak = equity_curve[0]
    max_dd = 0.0
    for equity in equity_curve:
        peak = max(peak, equity)
        if peak > 0:
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)
    return max_dd


def _equity_returns(equity_curve: list[float]) -> list[float]:
    return [
        equity_curve[i] / equity_curve[i - 1] - 1.0
        for i in range(1, len(equity_curve))
        if equity_curve[i - 1] > 0
    ]


def compute_sharpe(equity_curve: list[float], bars_per_year: float = _BARS_PER_YEAR_DEFAULT) -> float | None:
    """Annualised Sharpe ratio (zero risk-free rate)."""
    rets = _equity_returns(equity_curve)
    if len(rets) < 2:
        return None
    n = len(rets)
    mean = sum(rets) / n
    variance = sum((r - mean) ** 2 for r in rets) / (n - 1)
    std = math.sqrt(variance)
    if std < 1e-12:
        return None
    return mean / std * math.sqrt(bars_per_year)


def compute_sortino(equity_curve: list[float], bars_per_year: float = _BARS_PER_YEAR_DEFAULT) -> float | None:
    """Annualised Sortino ratio (zero target return, only downside deviation)."""
    rets = _equity_returns(equity_curve)
    if len(rets) < 2:
        return None
    mean = sum(rets) / len(rets)
    downside_sq = [r ** 2 for r in rets if r < 0]
    if not downside_sq:
        return None
    downside_dev = math.sqrt(sum(downside_sq) / len(downside_sq))
    if downside_dev < 1e-12:
        return None
    return mean / downside_dev * math.sqrt(bars_per_year)


def compute_calmar(total_return_pct: float, max_drawdown_pct: float) -> float | None:
    """Calmar ratio = total return / max drawdown (both as percentages)."""
    if max_drawdown_pct <= 0:
        return None
    return total_return_pct / max_drawdown_pct


# ---------------------------------------------------------------------------
# Regime-based evaluation
# ---------------------------------------------------------------------------

def _classify_regimes(
    equity_curve: list[float],
    ma_window: int = 50,
    slope_threshold: float = 0.0002,
) -> list[str]:
    """Label each bar of *equity_curve* as ``"bull"``, ``"bear"``, or ``"sideways"``.

    Regime is determined by the slope of a rolling ``ma_window``-bar simple
    moving average of equity, normalised by the current level:
        slope = (ma[i] - ma[i-1]) / ma[i-1]
    If slope > +threshold  -> bull
    If slope < -threshold  -> bear
    Otherwise              -> sideways
    """
    n = len(equity_curve)
    if n < ma_window + 1:
        return ["sideways"] * n

    arr = equity_curve  # list of floats
    labels: list[str] = ["sideways"] * n

    # Compute rolling MA
    ma: list[float | None] = [None] * n
    for i in range(ma_window - 1, n):
        ma[i] = sum(arr[i - ma_window + 1 : i + 1]) / ma_window

    for i in range(ma_window, n):
        prev = ma[i - 1]
        curr = ma[i]
        if prev is None or curr is None or prev <= 0:
            continue
        slope = (curr - prev) / prev
        if slope > slope_threshold:
            labels[i] = "bull"
        elif slope < -slope_threshold:
            labels[i] = "bear"
        else:
            labels[i] = "sideways"

    return labels


def _regime_metrics(
    equity_slice: list[float],
    bars_per_year: float = _BARS_PER_YEAR_DEFAULT,
) -> dict:
    if len(equity_slice) < 2:
        return {"bars": len(equity_slice), "sharpe": None, "total_return_pct": 0.0, "max_drawdown_pct": 0.0}
    sharpe = compute_sharpe(equity_slice, bars_per_year)
    total_ret = (equity_slice[-1] / equity_slice[0] - 1.0) * 100 if equity_slice[0] > 0 else 0.0
    max_dd = _compute_max_drawdown(equity_slice)
    return {
        "bars": len(equity_slice),
        "sharpe": round(sharpe, 4) if sharpe is not None else None,
        "total_return_pct": round(total_ret, 4),
        "max_drawdown_pct": round(max_dd * 100, 4),
    }


def evaluate_regimes(
    result: "BacktestResult",
    ma_window: int = 50,
    slope_threshold: float = 0.0002,
) -> dict[str, dict]:
    """Split *result.equity_curve* into bull/bear/sideways segments and
    compute per-regime performance metrics.

    Parameters
    ----------
    result:
        A :class:`BacktestResult` with a populated ``equity_curve``.
    ma_window:
        Rolling window for the MA slope classifier (default 50 bars).
    slope_threshold:
        Normalised slope above/below which a bar is classified bull/bear.

    Returns
    -------
    dict
        Keys ``"bull"``, ``"bear"``, ``"sideways"`` each mapping to a sub-dict
        with ``bars``, ``sharpe``, ``total_return_pct``, ``max_drawdown_pct``.
    """
    equity = result.equity_curve
    if not equity:
        return {r: {"bars": 0, "sharpe": None, "total_return_pct": 0.0, "max_drawdown_pct": 0.0}
                for r in ("bull", "bear", "sideways")}

    labels = _classify_regimes(equity, ma_window=ma_window, slope_threshold=slope_threshold)

    regime_bars: dict[str, list[float]] = {"bull": [], "bear": [], "sideways": []}
    for eq, lbl in zip(equity, labels):
        regime_bars[lbl].append(eq)

    bpy = getattr(result, "bars_per_year", _BARS_PER_YEAR_DEFAULT)
    return {regime: _regime_metrics(bars, bars_per_year=bpy) for regime, bars in regime_bars.items()}


def evaluate_regimes_by_market(result: "BacktestResult") -> dict[str, dict]:
    """Per-bar equity curve analytics segmented by actual market regime.

    Uses ``result.regime_log`` (from ``detect_regime()`` on price data) to
    group the equity curve by market condition and compute per-regime
    performance metrics.  This answers: "does the strategy make money in
    trends and lose in ranges, or vice versa?"

    Returns
    -------
    dict
        Keys are regime names (``trending_up``, ``trending_down``,
        ``ranging``, ``volatile``) each mapping to a sub-dict with
        ``bars``, ``sharpe``, ``total_return_pct``, ``max_drawdown_pct``,
        ``avg_bar_return_pct``.
    """
    equity = result.equity_curve
    regime_log = result.regime_log
    if not equity or not regime_log:
        return {}

    n = min(len(equity), len(regime_log))
    bpy = getattr(result, "bars_per_year", _BARS_PER_YEAR_DEFAULT)

    regime_equity: dict[str, list[float]] = {}
    regime_returns: dict[str, list[float]] = {}

    for i in range(n):
        regime = regime_log[i] or "unknown"
        regime_equity.setdefault(regime, []).append(equity[i])
        if i > 0 and equity[i - 1] > 0:
            bar_ret = (equity[i] / equity[i - 1]) - 1.0
            regime_returns.setdefault(regime, []).append(bar_ret)

    report: dict[str, dict] = {}
    for regime in sorted(regime_equity.keys()):
        eq_slice = regime_equity[regime]
        rets = regime_returns.get(regime, [])
        bars = len(eq_slice)

        if bars < 2:
            report[regime] = {
                "bars": bars, "sharpe": None,
                "total_return_pct": 0.0, "max_drawdown_pct": 0.0,
                "avg_bar_return_pct": 0.0,
            }
            continue

        total_ret_pct = sum(rets) * 100 if rets else 0.0
        max_dd = _compute_max_drawdown(eq_slice)
        avg_ret = sum(rets) / len(rets) * 100 if rets else 0.0

        sharpe = None
        if len(rets) > 1:
            import statistics
            mean_r = statistics.mean(rets)
            std_r = statistics.stdev(rets)
            if std_r > 1e-12:
                sharpe = mean_r / std_r * math.sqrt(bpy)

        report[regime] = {
            "bars": bars,
            "sharpe": round(sharpe, 4) if sharpe is not None else None,
            "total_return_pct": round(total_ret_pct, 4),
            "max_drawdown_pct": round(max_dd * 100, 4),
            "avg_bar_return_pct": round(avg_ret, 6),
        }

    return report


def evaluate_market_regimes(result: "BacktestResult") -> dict[str, dict]:
    """Per-market-regime trade analytics using the detector's labels at entry.

    Segments ``closed_trades`` by their ``entry_regime`` field and computes:
    ``trade_count``, ``win_rate``, ``avg_gross_pnl_pct``, ``avg_net_pnl_pct``,
    ``payoff_ratio``, and ``sharpe`` per regime.

    This tells us which strategy family (via the regime it ran in) is
    earning money and which is bleeding.
    """
    closed = getattr(result, "closed_trades", None)
    if not closed:
        return {}

    import numpy as np

    regime_trades: dict[str, list[dict]] = {}
    for trade in closed:
        regime = trade.get("entry_regime") or "unknown"
        regime_trades.setdefault(regime, []).append(trade)

    fee_rate = 0.0026  # fallback

    report: dict[str, dict] = {}
    for regime, trades in regime_trades.items():
        n = len(trades)
        wins = sum(1 for t in trades if t.get("pnl_usd", 0) > 0)
        gross = [t.get("pnl_pct", 0.0) for t in trades]
        net = [g - 2 * fee_rate * 100 for g in gross]

        avg_gross = float(np.mean(gross)) if gross else 0.0
        avg_net = float(np.mean(net)) if net else 0.0
        win_rate = wins / n if n else 0.0

        wins_pnl = [g for g in gross if g > 0]
        losses_pnl = [-g for g in gross if g <= 0]
        avg_win = float(np.mean(wins_pnl)) if wins_pnl else 0.0
        avg_loss = float(np.mean(losses_pnl)) if losses_pnl else 0.0
        payoff = avg_win / avg_loss if avg_loss > 0 else float("inf") if avg_win > 0 else 0.0

        ret = np.diff(np.array([0.0] + [t.get("pnl_pct", 0.0) for t in trades]))
        sharpe = None
        if len(ret) > 1 and np.std(ret) > 0:
            bpy = getattr(result, "bars_per_year", _BARS_PER_YEAR_DEFAULT)
            sharpe = float(np.mean(ret) / np.std(ret) * np.sqrt(bpy))

        report[regime] = {
            "trade_count": n,
            "win_rate": round(win_rate, 4),
            "avg_gross_pnl_pct": round(avg_gross, 4),
            "avg_net_pnl_pct": round(avg_net, 4),
            "payoff_ratio": round(payoff, 3) if payoff != float("inf") else "inf",
            "sharpe": round(sharpe, 4) if sharpe is not None else None,
        }

    return report


def evaluate_trades_by_regime_side(result: "BacktestResult") -> dict[str, dict]:
    """Break down closed trades by (regime, side) and exit reason.

    Returns a nested dict:
    ``{ "trending_down|short": { "count": 5, "win_rate": 0.6, ... }, ... }``

    Each bucket includes: ``count``, ``win_rate``, ``avg_pnl_pct``,
    ``total_pnl_pct``, ``payoff_ratio``, ``exit_reasons`` (counter dict).
    """
    closed = getattr(result, "closed_trades", None)
    if not closed:
        return {}

    buckets: dict[str, list[dict]] = {}
    for trade in closed:
        regime = trade.get("entry_regime") or "unknown"
        side = trade.get("side", "long")
        key = f"{regime}|{side}"
        buckets.setdefault(key, []).append(trade)

    report: dict[str, dict] = {}
    for key, trades in sorted(buckets.items()):
        n = len(trades)
        wins = sum(1 for t in trades if t.get("pnl_usd", 0) > 0)
        pnls = [t.get("pnl_pct", 0.0) for t in trades]
        avg_pnl = sum(pnls) / n if n else 0.0
        total_pnl = sum(pnls)
        win_rate = wins / n if n else 0.0

        wins_pnl = [p for p in pnls if p > 0]
        losses_pnl = [-p for p in pnls if p <= 0]
        avg_win = sum(wins_pnl) / len(wins_pnl) if wins_pnl else 0.0
        avg_loss = sum(losses_pnl) / len(losses_pnl) if losses_pnl else 0.0
        payoff = avg_win / avg_loss if avg_loss > 0 else float("inf") if avg_win > 0 else 0.0

        exit_reasons: dict[str, int] = {}
        for t in trades:
            r = t.get("close_reason", "unknown")
            exit_reasons[r] = exit_reasons.get(r, 0) + 1

        report[key] = {
            "count": n,
            "wins": wins,
            "win_rate": round(win_rate, 4),
            "avg_pnl_pct": round(avg_pnl, 4),
            "total_pnl_pct": round(total_pnl, 4),
            "payoff_ratio": round(payoff, 3) if payoff != float("inf") else "inf",
            "exit_reasons": exit_reasons,
        }

    return report


def run_backtest_on_candles(  # noqa: PLR0912,PLR0913
    candles,
    symbol: str,
    starting_balance_usd: float,
    aggressive_allocation: float,
    max_risk_per_trade: float,
    max_drawdown: float,
    short_ma_window: int,
    long_ma_window: int,
    volume_window: int,
    volume_threshold: float,
    fee_rate: float,
    timeframe: str | None = None,
    ml_model: TrainedModel | None = None,
    ml_buy_threshold: float = 0.55,
    ml_sell_threshold: float = 0.45,
    use_ema_clouds: bool = False,
    ema_fast_short: int = 8,
    ema_fast_long: int = 9,
    ema_slow_short: int = 34,
    ema_slow_long: int = 50,
    use_fvg: bool = False,
    fvg_min_gap_pct: float = 0.001,
    signal_mode: str = "any",
    min_vote_margin: int = 1,
    trailing_stop_pct: float = 0.0,
    take_profit_pct: float = 0.0,
    ml_confidence_sizing: bool = False,
    atr_stop_multiplier: float = 1.5,
    use_ict: bool = False,
    ict_swing_left: int = 2,
    ict_swing_right: int = 2,
    ict_eq_tolerance_pct: float = 0.0008,
    ict_min_displacement_pct: float = 0.003,
    ict_require_time_window: bool = True,
    ict_time_windows: str = "03:00-04:00,10:00-11:00,14:00-15:00",
    ict_require_pd: bool = True,
    ict_ote_enabled: bool = False,
    ict_ote_low: float = 0.62,
    ict_ote_high: float = 0.79,
    # RL agent
    use_rl_agent: bool = False,
    rl_policy=None,
    # Max hold time (overridden by max_hold_hours when set)
    max_hold_bars: int = 144,
    # Loss cooldown (overridden by loss_cooldown_hours when set)
    loss_cooldown_bars: int = 12,
    max_hold_hours: float | None = None,
    loss_cooldown_hours: float | None = None,
    # Slippage model: basis points added to buys, subtracted from sells
    slippage_bps: float = 5.0,
    # Execution: "same_bar" = fill at signal bar close; "next_open" = fill at next bar open
    execution_mode: str = "same_bar",
    # Short support: when True, sell signals open shorts when flat
    enable_shorts: bool = False,
    # DB path for sentiment/macro agents (as-of timestamp semantics)
    db_path: str | None = None,
    # Entry quality / edge gate (from config)
    min_edge_multiple: float = 1.5,
    min_final_confidence: float = 0.25,
    min_tech_confidence: float = 0.15,
    min_regime_confidence: float = 0.30,
    max_whipsaws: int = 3,
) -> BacktestResult:
    """Run bar-by-bar paper backtest for a single symbol dataframe."""

    reset_regime_history()

    if candles.empty:
        return BacktestResult(starting_balance_usd, starting_balance_usd, 0.0, 0.0, 0, 0.0)

    use_next_open = execution_mode == "next_open"

    # Resolve bars_per_year for correct Sharpe/Sortino annualization
    if timeframe:
        _bars_per_year = float(tf_bars_per_year(timeframe))
        _tf = timeframe
    else:
        inferred = infer_timeframe_from_candles(candles)
        _bars_per_year = float(tf_bars_per_year(inferred)) if inferred else _BARS_PER_YEAR_DEFAULT
        _tf = inferred or "1h"

    # Hour-based hold/cooldown override (parity with live/paper)
    if max_hold_hours is not None and max_hold_hours > 0:
        from hogan_bot.timeframe_utils import hours_to_bars
        max_hold_bars = hours_to_bars(max_hold_hours, _tf)
    if loss_cooldown_hours is not None and loss_cooldown_hours > 0:
        from hogan_bot.timeframe_utils import hours_to_bars
        loss_cooldown_bars = hours_to_bars(loss_cooldown_hours, _tf)

    slip_mult = slippage_bps / 10000.0
    portfolio = PaperPortfolio(cash_usd=starting_balance_usd, fee_rate=fee_rate)
    guard = DrawdownGuard(starting_balance_usd, max_drawdown)

    _bt_config = types.SimpleNamespace(
        short_ma_window=short_ma_window, long_ma_window=long_ma_window,
        volume_window=volume_window, volume_threshold=volume_threshold,
        use_ema_clouds=use_ema_clouds, ema_fast_short=ema_fast_short,
        ema_fast_long=ema_fast_long, ema_slow_short=ema_slow_short,
        ema_slow_long=ema_slow_long, use_fvg=use_fvg,
        fvg_min_gap_pct=fvg_min_gap_pct, signal_mode=signal_mode,
        signal_min_vote_margin=min_vote_margin,
        atr_stop_multiplier=atr_stop_multiplier,
        use_ict=use_ict, ict_swing_left=ict_swing_left,
        ict_swing_right=ict_swing_right, ict_eq_tolerance_pct=ict_eq_tolerance_pct,
        ict_min_displacement_pct=ict_min_displacement_pct,
        ict_require_time_window=ict_require_time_window,
        ict_time_windows=ict_time_windows, ict_require_pd=ict_require_pd,
        ict_ote_enabled=ict_ote_enabled, ict_ote_low=ict_ote_low,
        ict_ote_high=ict_ote_high, use_rl_agent=use_rl_agent,
        rl_policy=rl_policy, symbols=[symbol],
        # Fields required by effective_thresholds()
        ml_buy_threshold=ml_buy_threshold,
        ml_sell_threshold=ml_sell_threshold,
        trailing_stop_pct=trailing_stop_pct,
        take_profit_pct=take_profit_pct,
        use_regime_detection=True,
    )
    _bt_conn = None
    if db_path:
        import sqlite3
        _bt_conn = sqlite3.connect(db_path, check_same_thread=False)
        _bt_conn.execute("PRAGMA journal_mode=WAL")
        _bt_conn.execute("PRAGMA query_only=ON")
    _pipeline = AgentPipeline(_bt_config, conn=_bt_conn)

    wins = 0
    closed = 0
    trades = 0
    equity_curve: list[float] = []
    trade_log: list[dict] = []
    closed_trades: list[dict] = []
    _expectancy = ExpectancyTracker()

    # Track entry bar index for each symbol (for closed_trades / ML learning)
    _entry_bar: dict[str, int] = {}

    # Cooldown: bars remaining before next entry is allowed
    _cooldown_remaining: int = 0

    _rl_bars_in_trade: int = 0

    # Conviction persistence (parity with event_loop)
    _consecutive_exit_signals: int = 0
    _min_hold_bars: int = 3
    _exit_confirm_bars: int = 2

    # Short position tracking
    _short_entry_bar: dict[str, int] = {}
    _consecutive_short_exit_signals: int = 0
    _pending_shorts: dict[str, float] = {}
    _pending_covers: dict[str, tuple[float, float, int | None, str]] = {}
    short_wins: int = 0
    short_closed: int = 0

    # Signal funnel counters — where do signals die?
    _funnel = {
        "bars_evaluated": 0,
        "pipeline_buy": 0, "pipeline_sell": 0,
        "post_ml_buy": 0, "post_ml_sell": 0,
        "post_edge_buy": 0, "post_edge_sell": 0,
        "post_quality_buy": 0, "post_quality_sell": 0,
        "post_ranging_buy": 0, "post_ranging_sell": 0,
        "executed_buy": 0, "blocked_already_long": 0,
        "blocked_cooldown": 0, "blocked_regime_no_longs": 0, "blocked_regime_no_shorts": 0,
        "executed_short_entry": 0, "blocked_already_short": 0,
        "short_covered_signal": 0, "short_covered_stop": 0,
        "short_covered_tp": 0, "short_covered_max_hold": 0,
        "edge_blocked_atr": 0, "edge_blocked_tp": 0,
        "edge_blocked_forecast": 0, "edge_blocked_spread": 0,
        "ranging_blocked_tech": 0, "ranging_blocked_ml": 0,
        "ranging_blocked_whipsaw": 0,
    }
    # ML probability histogram (to understand model output distribution)
    _ml_probs: list[float] = []

    # ExitEvaluator (parity with event_loop — configurable thresholds)
    _exit_eval = ExitEvaluator(
        drawdown_panic_pct=0.03,
        time_decay_threshold=0.75,
        volatility_expansion_threshold=2.0,
        max_consolidation_bars=12,
    )

    # Regime tracking (parity with event_loop)
    _current_regime: str | None = None
    _regime_conf: float | None = None
    _whipsaw_count: int = 0
    _last_signal: str = "hold"
    _regime_per_bar: list[str | None] = []

    # Next-open execution: pending buys/sells to fill at next bar's open
    _pending_buys: dict[str, float] = {}
    _pending_sells: dict[str, tuple[float, float, int | None, str]] = {}

    min_rows = max(long_ma_window, volume_window) + 2
    _lookback = max(200, long_ma_window * 3, volume_window * 3)
    for i in range(min_rows, len(candles) + 1):
        window = candles.iloc[max(0, i - _lookback):i]
        px = float(window["close"].iloc[-1])
        bar_ts = str(window["timestamp"].iloc[-1]) if "timestamp" in window.columns else str(i)
        # Open of current bar (for next_open fills from previous bar's signal)
        open_px = float(window["open"].iloc[-1]) if "open" in window.columns else px

        if _cooldown_remaining > 0:
            _cooldown_remaining -= 1

        # Process pending next_open fills at this bar's open (we have "next bar" data)
        if use_next_open:
            for sym, size in list(_pending_buys.items()):
                _pending_buys.pop(sym, None)
                buy_px = open_px * (1.0 + slip_mult)
                if portfolio.execute_buy(sym, buy_px, size, trailing_stop_pct=trailing_stop_pct, take_profit_pct=take_profit_pct):
                    trades += 1
                    _entry_bar[sym] = i - 1
                    trade_log.append({"bar": bar_ts, "action": "buy", "reason": "signal", "price": buy_px, "qty": size})
            for sym, (qty, avg_entry, entry_bar_idx, reason) in list(_pending_sells.items()):
                _pending_sells.pop(sym, None)
                _entry_bar.pop(sym, None)
                sell_px = open_px * (1.0 - slip_mult)
                if portfolio.execute_sell(sym, sell_px, qty):
                    trades += 1
                    closed += 1
                    is_win = sell_px > avg_entry
                    if is_win:
                        wins += 1
                    elif loss_cooldown_bars > 0:
                        _cooldown_remaining = loss_cooldown_bars
                    trade_log.append({"bar": bar_ts, "action": "sell", "reason": reason, "price": sell_px, "qty": qty})
                    gross_pct = (sell_px - avg_entry) / avg_entry if avg_entry else 0
                    _expectancy.record_trade(
                        symbol=sym, regime=_current_regime or "backtest",
                        gross_pnl_pct=gross_pct, net_pnl_pct=gross_pct - 2 * fee_rate,
                        hold_bars=(i - 1 - entry_bar_idx) if entry_bar_idx is not None else 0,
                        close_reason=reason,
                    )
                    if entry_bar_idx is not None:
                        exit_bar_idx = min(i - 1, len(candles) - 1)
                        fee = qty * (avg_entry + sell_px) * fee_rate
                        pnl_usd = (sell_px - avg_entry) * qty - fee
                        pnl_pct = (sell_px - avg_entry) / avg_entry * 100 if avg_entry else 0
                        closed_trades.append({
                            "entry_bar_idx": entry_bar_idx,
                            "exit_bar_idx": exit_bar_idx,
                            "entry_ts_ms": _bar_ts_ms(candles, entry_bar_idx),
                            "exit_ts_ms": _bar_ts_ms(candles, exit_bar_idx),
                            "side": "long",
                            "entry_price": avg_entry,
                            "exit_price": sell_px,
                            "qty": qty,
                            "pnl_usd": pnl_usd,
                            "pnl_pct": pnl_pct,
                            "entry_regime": _current_regime,
                            "close_reason": reason,
                        })
            for sym, size in list(_pending_shorts.items()):
                _pending_shorts.pop(sym, None)
                short_px = open_px * (1.0 - slip_mult)
                if portfolio.execute_short(sym, short_px, size, trailing_stop_pct=trailing_stop_pct, take_profit_pct=take_profit_pct):
                    trades += 1
                    _short_entry_bar[sym] = i - 1
                    trade_log.append({"bar": bar_ts, "action": "short", "reason": "signal", "price": short_px, "qty": size})
            for sym, (qty, avg_entry, entry_bar_idx, reason) in list(_pending_covers.items()):
                _pending_covers.pop(sym, None)
                _short_entry_bar.pop(sym, None)
                cover_px = open_px * (1.0 + slip_mult)
                if portfolio.execute_cover(sym, cover_px, qty):
                    trades += 1
                    short_closed += 1
                    is_win = cover_px < avg_entry
                    if is_win:
                        short_wins += 1
                    elif loss_cooldown_bars > 0:
                        _cooldown_remaining = loss_cooldown_bars
                    trade_log.append({"bar": bar_ts, "action": "cover", "reason": reason, "price": cover_px, "qty": qty})
                    gross_pct = (avg_entry - cover_px) / avg_entry if avg_entry else 0
                    _expectancy.record_trade(
                        symbol=sym, regime=_current_regime or "backtest",
                        gross_pnl_pct=gross_pct, net_pnl_pct=gross_pct - 2 * fee_rate,
                        hold_bars=(i - 1 - entry_bar_idx) if entry_bar_idx is not None else 0,
                        close_reason=reason,
                    )
                    if entry_bar_idx is not None:
                        exit_bar_idx = min(i - 1, len(candles) - 1)
                        fee = qty * (avg_entry + cover_px) * fee_rate
                        pnl_usd = (avg_entry - cover_px) * qty - fee
                        pnl_pct = (avg_entry - cover_px) / avg_entry * 100 if avg_entry else 0
                        closed_trades.append({
                            "entry_bar_idx": entry_bar_idx,
                            "exit_bar_idx": exit_bar_idx,
                            "entry_ts_ms": _bar_ts_ms(candles, entry_bar_idx),
                            "exit_ts_ms": _bar_ts_ms(candles, exit_bar_idx),
                            "side": "short",
                            "entry_price": avg_entry,
                            "exit_price": cover_px,
                            "qty": qty,
                            "pnl_usd": pnl_usd,
                            "pnl_pct": pnl_pct,
                            "entry_regime": _current_regime,
                            "close_reason": reason,
                        })

        mark = {symbol: px}
        exits = portfolio.check_exits(mark, max_hold_bars=max_hold_bars)
        for exit_symbol, reason in exits:
            _is_short_exit = reason.startswith("short_")

            if _is_short_exit and enable_shorts:
                spos = portfolio.short_positions.get(exit_symbol)
                if spos is None:
                    continue
                qty = spos.qty
                avg_entry = spos.avg_entry
                _mae = getattr(spos, "max_adverse_pct", 0.0)
                _mfe = getattr(spos, "max_favorable_pct", 0.0)
                entry_bar = _short_entry_bar.get(exit_symbol)
                if use_next_open and i < len(candles):
                    _pending_covers[exit_symbol] = (qty, avg_entry, entry_bar, reason)
                    _short_entry_bar.pop(exit_symbol, None)
                else:
                    cover_px = px * (1.0 + slip_mult)
                    _short_entry_bar.pop(exit_symbol, None)
                    if portfolio.execute_cover(exit_symbol, cover_px, qty):
                        trades += 1
                        short_closed += 1
                        is_win = cover_px < avg_entry
                        if is_win:
                            short_wins += 1
                        elif loss_cooldown_bars > 0:
                            _cooldown_remaining = loss_cooldown_bars
                        trade_log.append(
                            {"bar": bar_ts, "action": "cover", "reason": reason, "price": cover_px, "qty": qty}
                        )
                        if "stop" in reason:
                            _funnel["short_covered_stop"] += 1
                        elif "profit" in reason:
                            _funnel["short_covered_tp"] += 1
                        elif "hold" in reason:
                            _funnel["short_covered_max_hold"] += 1
                        gross_pct = (avg_entry - cover_px) / avg_entry if avg_entry else 0
                        _expectancy.record_trade(
                            symbol=exit_symbol, regime=_current_regime or "backtest",
                            gross_pnl_pct=gross_pct, net_pnl_pct=gross_pct - 2 * fee_rate,
                            hold_bars=(i - 1 - entry_bar) if entry_bar is not None else 0,
                            close_reason=reason,
                            mae_pct=_mae, mfe_pct=_mfe,
                        )
                        if entry_bar is not None:
                            exit_bar_idx = min(i - 1, len(candles) - 1)
                            fee = qty * (avg_entry + cover_px) * fee_rate
                            pnl_usd = (avg_entry - cover_px) * qty - fee
                            pnl_pct = (avg_entry - cover_px) / avg_entry * 100 if avg_entry else 0
                            closed_trades.append({
                                "entry_bar_idx": entry_bar,
                                "exit_bar_idx": exit_bar_idx,
                                "entry_ts_ms": _bar_ts_ms(candles, entry_bar),
                                "exit_ts_ms": _bar_ts_ms(candles, exit_bar_idx),
                                "side": "short",
                                "entry_price": avg_entry,
                                "exit_price": cover_px,
                                "qty": qty,
                                "pnl_usd": pnl_usd,
                                "pnl_pct": pnl_pct,
                                "entry_regime": _current_regime,
                                "close_reason": reason,
                            })
                continue

            pos = portfolio.positions.get(exit_symbol)
            if pos is None:
                continue
            qty = pos.qty
            avg_entry = pos.avg_entry
            _mae = getattr(pos, "max_adverse_pct", 0.0)
            _mfe = getattr(pos, "max_favorable_pct", 0.0)
            entry_bar = _entry_bar.get(exit_symbol)
            if use_next_open and i < len(candles):
                _pending_sells[exit_symbol] = (qty, avg_entry, entry_bar, reason)
                _entry_bar.pop(exit_symbol, None)
            else:
                sell_px = px * (1.0 - slip_mult)
                _entry_bar.pop(exit_symbol, None)
                if portfolio.execute_sell(exit_symbol, sell_px, qty):
                    trades += 1
                    closed += 1
                    is_win = sell_px > avg_entry
                    if is_win:
                        wins += 1
                    elif loss_cooldown_bars > 0:
                        _cooldown_remaining = loss_cooldown_bars
                    trade_log.append(
                        {"bar": bar_ts, "action": "sell", "reason": reason, "price": sell_px, "qty": qty}
                    )
                    gross_pct = (sell_px - avg_entry) / avg_entry if avg_entry else 0
                    _expectancy.record_trade(
                        symbol=exit_symbol, regime=_current_regime or "backtest",
                        gross_pnl_pct=gross_pct, net_pnl_pct=gross_pct - 2 * fee_rate,
                        hold_bars=(i - 1 - entry_bar) if entry_bar is not None else 0,
                        close_reason=reason,
                        mae_pct=_mae, mfe_pct=_mfe,
                    )
                    if entry_bar is not None:
                        exit_bar_idx = min(i - 1, len(candles) - 1)
                        fee = qty * (avg_entry + sell_px) * fee_rate
                        pnl_usd = (sell_px - avg_entry) * qty - fee
                        pnl_pct = (sell_px - avg_entry) / avg_entry * 100 if avg_entry else 0
                        closed_trades.append({
                            "entry_bar_idx": entry_bar,
                            "exit_bar_idx": exit_bar_idx,
                            "entry_ts_ms": _bar_ts_ms(candles, entry_bar),
                            "exit_ts_ms": _bar_ts_ms(candles, exit_bar_idx),
                            "side": "long",
                            "entry_price": avg_entry,
                            "exit_price": sell_px,
                            "qty": qty,
                            "pnl_usd": pnl_usd,
                            "pnl_pct": pnl_pct,
                            "entry_regime": _current_regime,
                            "close_reason": reason,
                        })

        # Build RL position state for this bar
        _rl_pos = portfolio.positions.get(symbol)
        _rl_in_pos = _rl_pos is not None
        if _rl_in_pos:
            _rl_upnl = (px - _rl_pos.avg_entry) / max(_rl_pos.avg_entry, 1e-9)
            _rl_bars_in_trade += 1
        else:
            _rl_upnl = 0.0
            _rl_bars_in_trade = 0

        # Regime detection (parity with event_loop)
        _rstate = None
        if len(window) >= 80:
            try:
                _rstate = detect_regime(window)
                _current_regime = _rstate.regime
                _regime_conf = _rstate.confidence
            except Exception:
                pass
        _regime_per_bar.append(_current_regime)

        # Regime-adjusted thresholds (parity with event_loop)
        _eff: dict[str, float] = {}
        if _rstate is not None:
            try:
                from hogan_bot.regime import effective_thresholds
                _eff = effective_thresholds(_rstate, _bt_config)
            except Exception:
                pass
        _eff_ml_buy = _eff.get("ml_buy_threshold", ml_buy_threshold)
        _eff_ml_sell = _eff.get("ml_sell_threshold", ml_sell_threshold)
        _eff_tp = _eff.get("take_profit_pct", take_profit_pct)
        _eff_ts = _eff.get("trailing_stop_pct", trailing_stop_pct)
        _eff_position_scale = _eff.get("position_scale", 1.0)
        _eff_allow_longs = _eff.get("allow_longs", True)
        _eff_allow_shorts = _eff.get("allow_shorts", True)
        _eff_long_size_scale = _eff.get("long_size_scale", 1.0)
        _eff_short_size_scale = _eff.get("short_size_scale", 1.0)

        _as_of = _bar_ts_ms(candles, i - 1) if _bt_conn is not None else None
        signal = _pipeline.run(
            window,
            symbol=symbol,
            as_of_ms=_as_of,
            rl_in_position=_rl_in_pos,
            rl_unrealized_pnl=_rl_upnl,
            rl_bars_in_trade=_rl_bars_in_trade,
            regime=_current_regime,
            regime_state=_rstate,
        )

        _funnel["bars_evaluated"] += 1
        action = signal.action
        conf_scale = signal.confidence or 1.0

        # Track pipeline output before any filtering
        if action == "buy":
            _funnel["pipeline_buy"] += 1
        elif action == "sell":
            _funnel["pipeline_sell"] += 1

        up_prob = None
        if ml_model is not None:
            up_prob = predict_up_probability(window, ml_model)
            _ml_probs.append(up_prob)
            _ml_gd = apply_ml_filter(action, up_prob, _eff_ml_buy, _eff_ml_sell)
            action = _ml_gd.action
            if ml_confidence_sizing:
                conf_scale *= ml_confidence(up_prob)

        if action == "buy":
            _funnel["post_ml_buy"] += 1
        elif action == "sell":
            _funnel["post_ml_sell"] += 1

        _atr_series = compute_atr(window, window=14)
        _atr_pct = float(_atr_series.iloc[-1]) / max(px, 1e-9)
        _spread_est = estimate_spread_from_candles(window)
        _forecast_ret = None
        if signal.forecast is not None and getattr(signal.forecast, 'confidence', 0) > 0.2:
            _er = signal.forecast.expected_return
            if isinstance(_er, dict) and _er:
                _forecast_ret = max(abs(v) for v in _er.values())
            elif isinstance(_er, (int, float)):
                _forecast_ret = abs(float(_er))
        _edge_gd = edge_gate(
            action,
            atr_pct=_atr_pct,
            take_profit_pct=_eff_tp,
            fee_rate=fee_rate,
            min_edge_multiple=min_edge_multiple,
            forecast_expected_return=_forecast_ret,
            estimated_spread=_spread_est,
        )
        action = _edge_gd.action

        if _edge_gd.blocked_by:
            blk = _edge_gd.blocked_by
            if "atr" in blk:
                _funnel["edge_blocked_atr"] += 1
            elif "tp" in blk:
                _funnel["edge_blocked_tp"] += 1
            elif "forecast" in blk:
                _funnel["edge_blocked_forecast"] += 1
            elif "spread" in blk:
                _funnel["edge_blocked_spread"] += 1

        if action == "buy":
            _funnel["post_edge_buy"] += 1
        elif action == "sell":
            _funnel["post_edge_sell"] += 1

        _tech_conf = signal.tech.confidence if signal.tech else None
        _quality_gd = entry_quality_gate(
            action,
            final_confidence=signal.confidence,
            tech_confidence=_tech_conf,
            regime=_current_regime,
            regime_confidence=_regime_conf,
            recent_whipsaw_count=_whipsaw_count,
            min_final_confidence=min_final_confidence,
            min_tech_confidence=min_tech_confidence,
            min_regime_confidence=min_regime_confidence,
            max_whipsaws=max_whipsaws,
        )
        action = _quality_gd.action
        _quality_scale = _quality_gd.size_scale

        if action == "buy":
            _funnel["post_quality_buy"] += 1
        elif action == "sell":
            _funnel["post_quality_sell"] += 1

        _tech_action = signal.tech.action if signal.tech else None
        _ranging_gd = ranging_gate(
            action,
            regime=_current_regime,
            tech_action=_tech_action,
            up_prob=up_prob if ml_model is not None else None,
            recent_whipsaw_count=_whipsaw_count,
        )
        action = _ranging_gd.action
        _ranging_scale = _ranging_gd.size_scale

        if _ranging_gd.blocked_by:
            blk = _ranging_gd.blocked_by
            if "tech" in blk:
                _funnel["ranging_blocked_tech"] += 1
            elif "ml" in blk:
                _funnel["ranging_blocked_ml"] += 1
            elif "whipsaw" in blk:
                _funnel["ranging_blocked_whipsaw"] += 1

        if action == "buy":
            _funnel["post_ranging_buy"] += 1
        elif action == "sell":
            _funnel["post_ranging_sell"] += 1

        equity = portfolio.total_equity(mark)
        equity_curve.append(equity)

        if not guard.update_and_check(equity):
            break

        size = calculate_position_size(
            equity_usd=equity,
            price=px,
            stop_distance_pct=signal.stop_distance_pct,
            max_risk_per_trade=max_risk_per_trade,
            max_allocation_pct=aggressive_allocation,
            confidence_scale=conf_scale * _quality_scale * _ranging_scale * _eff_position_scale,
            fee_rate=fee_rate,
        )

        # Track whipsaws (signal flipped from last bar)
        if action != "hold" and _last_signal != "hold" and action != _last_signal:
            _whipsaw_count = min(_whipsaw_count + 1, 10)
        elif action == _last_signal:
            _whipsaw_count = max(0, _whipsaw_count - 1)
        _last_signal = action

        if action == "buy":
            _consecutive_exit_signals = 0

            # Cover any existing short first
            if enable_shorts and symbol in portfolio.short_positions:
                _consecutive_short_exit_signals += 1
                spos = portfolio.short_positions[symbol]
                if spos.bars_held >= _min_hold_bars:
                    _consecutive_short_exit_signals = 0
                    cover_qty = spos.qty
                    s_avg_entry = spos.avg_entry
                    s_entry_bar = _short_entry_bar.get(symbol)
                    if use_next_open and i < len(candles):
                        _pending_covers[symbol] = (cover_qty, s_avg_entry, s_entry_bar, "buy_signal")
                        _short_entry_bar.pop(symbol, None)
                    else:
                        cover_px = px * (1.0 + slip_mult)
                        _short_entry_bar.pop(symbol, None)
                        if portfolio.execute_cover(symbol, cover_px, cover_qty):
                            trades += 1
                            short_closed += 1
                            _funnel["short_covered_signal"] += 1
                            is_win = cover_px < s_avg_entry
                            if is_win:
                                short_wins += 1
                            elif loss_cooldown_bars > 0:
                                _cooldown_remaining = loss_cooldown_bars
                            trade_log.append(
                                {"bar": bar_ts, "action": "cover", "reason": "buy_signal", "price": cover_px, "qty": cover_qty}
                            )
                            gross_pct = (s_avg_entry - cover_px) / s_avg_entry if s_avg_entry else 0
                            _expectancy.record_trade(
                                symbol=symbol, regime=_current_regime or "backtest",
                                gross_pnl_pct=gross_pct, net_pnl_pct=gross_pct - 2 * fee_rate,
                                hold_bars=(i - 1 - s_entry_bar) if s_entry_bar is not None else 0,
                                close_reason="buy_signal",
                            )
                            if s_entry_bar is not None:
                                exit_bar_idx = min(i - 1, len(candles) - 1)
                                fee = cover_qty * (s_avg_entry + cover_px) * fee_rate
                                pnl_usd = (s_avg_entry - cover_px) * cover_qty - fee
                                pnl_pct = (s_avg_entry - cover_px) / s_avg_entry * 100 if s_avg_entry else 0
                                closed_trades.append({
                                    "entry_bar_idx": s_entry_bar,
                                    "exit_bar_idx": exit_bar_idx,
                                    "entry_ts_ms": _bar_ts_ms(candles, s_entry_bar),
                                    "exit_ts_ms": _bar_ts_ms(candles, exit_bar_idx),
                                    "side": "short",
                                    "entry_price": s_avg_entry,
                                    "exit_price": cover_px,
                                    "qty": cover_qty,
                                    "pnl_usd": pnl_usd,
                                    "pnl_pct": pnl_pct,
                                    "entry_regime": _current_regime,
                                    "close_reason": "buy_signal",
                                })

            _long_size = size * _eff_long_size_scale
            if not _eff_allow_longs:
                _funnel["blocked_regime_no_longs"] += 1
            elif symbol in portfolio.positions:
                _funnel["blocked_already_long"] += 1
            elif _cooldown_remaining > 0:
                _funnel["blocked_cooldown"] += 1
            elif symbol in portfolio.short_positions:
                pass
            elif _long_size <= 0:
                _funnel["blocked_regime_no_longs"] += 1
            elif use_next_open and i < len(candles):
                _pending_buys[symbol] = _long_size
            else:
                buy_px = px * (1.0 + slip_mult)
                if portfolio.execute_buy(
                    symbol, buy_px, _long_size,
                    trailing_stop_pct=trailing_stop_pct,
                    take_profit_pct=take_profit_pct,
                ):
                    _pos = portfolio.positions.get(symbol)
                    if _pos is not None:
                        _pos.entry_atr_pct = _atr_pct
                    trades += 1
                    _funnel["executed_buy"] += 1
                    _entry_bar[symbol] = i - 1
                    trade_log.append(
                        {"bar": bar_ts, "action": "buy", "reason": "signal", "price": buy_px, "qty": _long_size}
                    )
        elif action == "sell":
            # Close existing long (with ExitEvaluator confirmation)
            if symbol in portfolio.positions:
                pos = portfolio.positions[symbol]
                qty = pos.qty
                if qty <= 0:
                    _consecutive_exit_signals = 0
                else:
                    _exit_dec = _exit_eval.should_exit(
                        candles=window,
                        entry_price=pos.avg_entry,
                        current_price=px,
                        bars_held=pos.bars_held,
                        side="long",
                        max_hold_bars=max_hold_bars,
                        entry_atr=getattr(pos, "entry_atr_pct", None) or None,
                        vol_ratio=signal.volume_ratio,
                    )
                    _consecutive_exit_signals += 1
                    if pos.bars_held < _min_hold_bars:
                        pass
                    elif not _exit_dec.should_exit and _consecutive_exit_signals < _exit_confirm_bars:
                        pass
                    else:
                        _consecutive_exit_signals = 0
                        sell_qty = min(qty, size)
                        avg_entry = pos.avg_entry
                        _mae = getattr(pos, "max_adverse_pct", 0.0)
                        _mfe = getattr(pos, "max_favorable_pct", 0.0)
                        entry_bar_idx = _entry_bar.get(symbol)
                        if use_next_open and i < len(candles):
                            _pending_sells[symbol] = (sell_qty, avg_entry, entry_bar_idx, "signal")
                            _entry_bar.pop(symbol, None)
                        else:
                            sell_px_sig = px * (1.0 - slip_mult)
                            _entry_bar.pop(symbol, None)
                            if portfolio.execute_sell(symbol, sell_px_sig, sell_qty):
                                trades += 1
                                closed += 1
                                is_win = sell_px_sig > avg_entry
                                if is_win:
                                    wins += 1
                                elif loss_cooldown_bars > 0:
                                    _cooldown_remaining = loss_cooldown_bars
                                trade_log.append(
                                    {"bar": bar_ts, "action": "sell", "reason": "signal", "price": sell_px_sig, "qty": sell_qty}
                                )
                                gross_pct = (sell_px_sig - avg_entry) / avg_entry if avg_entry else 0
                                _expectancy.record_trade(
                                    symbol=symbol, regime=_current_regime or "backtest",
                                    gross_pnl_pct=gross_pct, net_pnl_pct=gross_pct - 2 * fee_rate,
                                    hold_bars=(i - 1 - entry_bar_idx) if entry_bar_idx is not None else 0,
                                    close_reason="signal",
                                    mae_pct=_mae, mfe_pct=_mfe,
                                )
                                if entry_bar_idx is not None:
                                    exit_bar_idx = min(i - 1, len(candles) - 1)
                                    fee = sell_qty * (avg_entry + sell_px_sig) * fee_rate
                                    pnl_usd = (sell_px_sig - avg_entry) * sell_qty - fee
                                    pnl_pct = (sell_px_sig - avg_entry) / avg_entry * 100 if avg_entry else 0
                                    closed_trades.append({
                                        "entry_bar_idx": entry_bar_idx,
                                        "exit_bar_idx": exit_bar_idx,
                                        "entry_ts_ms": _bar_ts_ms(candles, entry_bar_idx),
                                        "exit_ts_ms": _bar_ts_ms(candles, exit_bar_idx),
                                        "side": "long",
                                        "entry_price": avg_entry,
                                        "exit_price": sell_px_sig,
                                        "qty": sell_qty,
                                        "pnl_usd": pnl_usd,
                                        "pnl_pct": pnl_pct,
                                        "entry_regime": _current_regime,
                                        "close_reason": "signal",
                                    })
            # Open short when flat and shorts enabled
            elif enable_shorts and symbol not in portfolio.short_positions:
                _consecutive_exit_signals = 0
                _short_size = size * _eff_short_size_scale
                if not _eff_allow_shorts:
                    _funnel["blocked_regime_no_shorts"] += 1
                elif _cooldown_remaining > 0:
                    _funnel["blocked_cooldown"] += 1
                elif _short_size <= 0:
                    _funnel["blocked_regime_no_shorts"] += 1
                elif use_next_open and i < len(candles):
                    _pending_shorts[symbol] = _short_size
                else:
                    short_px = px * (1.0 - slip_mult)
                    if portfolio.execute_short(
                        symbol, short_px, _short_size,
                        trailing_stop_pct=trailing_stop_pct,
                        take_profit_pct=take_profit_pct,
                    ):
                        spos = portfolio.short_positions.get(symbol)
                        if spos is not None:
                            spos.entry_atr_pct = _atr_pct
                        trades += 1
                        _funnel["executed_short_entry"] += 1
                        _short_entry_bar[symbol] = i - 1
                        trade_log.append(
                            {"bar": bar_ts, "action": "short", "reason": "signal", "price": short_px, "qty": _short_size}
                        )
            elif enable_shorts and symbol in portfolio.short_positions:
                _funnel["blocked_already_short"] += 1
            else:
                _consecutive_exit_signals = 0

    if not equity_curve:
        equity_curve = [starting_balance_usd]
    end_equity = equity_curve[-1]
    max_dd = _compute_max_drawdown(equity_curve)
    total_return_pct = (
        ((end_equity / starting_balance_usd) - 1.0) * 100 if starting_balance_usd > 0 else 0.0
    )
    total_closed = closed + short_closed
    total_wins = wins + short_wins
    win_rate = (total_wins / total_closed) if total_closed > 0 else 0.0
    max_dd_pct = max_dd * 100

    if _bt_conn is not None:
        _bt_conn.close()

    # Build ML probability distribution summary
    if _ml_probs:
        import numpy as np
        _probs_arr = np.array(_ml_probs)
        _funnel["ml_prob_stats"] = {
            "mean": round(float(np.mean(_probs_arr)), 4),
            "std": round(float(np.std(_probs_arr)), 4),
            "min": round(float(np.min(_probs_arr)), 4),
            "p10": round(float(np.percentile(_probs_arr, 10)), 4),
            "p25": round(float(np.percentile(_probs_arr, 25)), 4),
            "median": round(float(np.median(_probs_arr)), 4),
            "p75": round(float(np.percentile(_probs_arr, 75)), 4),
            "p90": round(float(np.percentile(_probs_arr, 90)), 4),
            "max": round(float(np.max(_probs_arr)), 4),
            "pct_above_buy_thresh": round(float(np.mean(_probs_arr >= ml_buy_threshold) * 100), 2),
            "pct_below_sell_thresh": round(float(np.mean(_probs_arr <= ml_sell_threshold) * 100), 2),
        }

    # Regime distribution from actual market detection (not equity curve)
    _regime_counts: dict[str, int] = {}
    for _r in _regime_per_bar:
        if _r is not None:
            _regime_counts[_r] = _regime_counts.get(_r, 0) + 1
    _funnel["regime_distribution"] = _regime_counts

    return BacktestResult(
        start_equity=starting_balance_usd,
        end_equity=end_equity,
        total_return_pct=total_return_pct,
        max_drawdown_pct=max_dd_pct,
        trades=trades,
        win_rate=win_rate,
        sharpe_ratio=compute_sharpe(equity_curve, bars_per_year=_bars_per_year),
        sortino_ratio=compute_sortino(equity_curve, bars_per_year=_bars_per_year),
        calmar_ratio=compute_calmar(total_return_pct, max_dd_pct),
        bars_per_year=_bars_per_year,
        equity_curve=equity_curve,
        trade_log=trade_log,
        closed_trades=closed_trades,
        expectancy_report=_expectancy.summary(),
        regime_log=_regime_per_bar,
        signal_funnel=_funnel,
    )
