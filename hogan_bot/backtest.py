from __future__ import annotations

import math
import types
from dataclasses import dataclass, field

import pandas as pd

from hogan_bot.agent_pipeline import AgentPipeline
from hogan_bot.champion import apply_champion_mode, is_champion_mode
from hogan_bot.decision import (
    GateDecision, apply_ml_filter, edge_gate, entry_quality_gate,
    loss_streak_scale, ml_blind_blocks_shorts, ml_blind_scale,
    ml_confidence, ml_probability_sizer,
    estimate_spread_from_candles, pullback_gate, ranging_gate,
)
from hogan_bot.exit_model import ExitEvaluator
from hogan_bot.expectancy import ExpectancyTracker
from hogan_bot.indicators import compute_atr
from hogan_bot.ml import TrainedModel, predict_up_probability
from hogan_bot.paper import PaperPortfolio
from hogan_bot.regime import detect_regime, reset_regime_history, RegimeTransitionTracker
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


def enrich_trades_with_entry_context(
    candles: pd.DataFrame,
    closed_trades: list[dict],
    lookback: int = 12,
    forward: int = 24,
) -> list[dict]:
    """Add entry-timing diagnostics to each closed trade.

    For each trade, measures where the entry price sits relative to
    recent price action and what the price does after entry.

    Added fields per trade:
    - ``pct_from_local_high``:  (entry - N-bar high) / N-bar high
    - ``pct_from_local_low``:   (entry - N-bar low) / N-bar low
    - ``local_range_position``: 0 = entered at low, 1 = entered at high
    - ``run_up_before_entry``:  how much price rose in lookback bars
    - ``max_favorable_excursion``: best unrealized PnL during trade (%)
    - ``max_adverse_excursion``:   worst unrealized PnL during trade (%)
    - ``bars_to_peak``:  bars from entry to MFE
    - ``bars_to_trough``: bars from entry to MAE
    """
    import numpy as np

    highs = candles["high"].values if "high" in candles.columns else candles["close"].values
    lows = candles["low"].values if "low" in candles.columns else candles["close"].values
    closes = candles["close"].values

    for trade in closed_trades:
        entry_bar = trade.get("entry_bar_idx")
        exit_bar = trade.get("exit_bar_idx")
        entry_px = trade.get("entry_price", 0.0)
        side = trade.get("side", "long")

        if entry_bar is None or entry_px <= 0:
            continue

        lb_start = max(0, entry_bar - lookback)
        local_high = float(np.max(highs[lb_start:entry_bar + 1]))
        local_low = float(np.min(lows[lb_start:entry_bar + 1]))
        local_range = local_high - local_low if local_high > local_low else 1e-9

        trade["pct_from_local_high"] = round((entry_px - local_high) / local_high * 100, 3)
        trade["pct_from_local_low"] = round((entry_px - local_low) / local_low * 100, 3)
        trade["local_range_position"] = round((entry_px - local_low) / local_range, 3)

        lb_close_start = float(closes[lb_start])
        trade["run_up_before_entry"] = round((entry_px - lb_close_start) / lb_close_start * 100, 3)

        fw_end = min(len(candles), (exit_bar or entry_bar) + 1)
        fw_highs = highs[entry_bar:fw_end]
        fw_lows = lows[entry_bar:fw_end]

        if side == "long":
            if len(fw_highs) > 0:
                mfe_val = float(np.max(fw_highs))
                mae_val = float(np.min(fw_lows))
                trade["max_favorable_excursion"] = round((mfe_val - entry_px) / entry_px * 100, 3)
                trade["max_adverse_excursion"] = round((mae_val - entry_px) / entry_px * 100, 3)
                trade["bars_to_peak"] = int(np.argmax(fw_highs))
                trade["bars_to_trough"] = int(np.argmin(fw_lows))
            else:
                trade["max_favorable_excursion"] = 0.0
                trade["max_adverse_excursion"] = 0.0
                trade["bars_to_peak"] = 0
                trade["bars_to_trough"] = 0
        else:
            if len(fw_lows) > 0:
                mfe_val = float(np.min(fw_lows))
                mae_val = float(np.max(fw_highs))
                trade["max_favorable_excursion"] = round((entry_px - mfe_val) / entry_px * 100, 3)
                trade["max_adverse_excursion"] = round((entry_px - mae_val) / entry_px * 100, 3)
                trade["bars_to_peak"] = int(np.argmin(fw_lows))
                trade["bars_to_trough"] = int(np.argmax(fw_highs))
            else:
                trade["max_favorable_excursion"] = 0.0
                trade["max_adverse_excursion"] = 0.0
                trade["bars_to_peak"] = 0
                trade["bars_to_trough"] = 0

    return closed_trades


def enrich_trades_with_post_exit(
    closed_trades: list[dict],
    candles: pd.DataFrame,
    post_exit_bars: int = 24,
) -> list[dict]:
    """Add post-exit price analysis to each closed trade.

    For each trade, looks at the ``post_exit_bars`` bars AFTER the exit
    and records:

    - ``post_exit_max_favorable``: best outcome if the stop had NOT fired
      (max price rise for longs, max price drop for shorts, as % of exit px)
    - ``post_exit_max_adverse``: worst case (continued move against, as %)
    - ``post_exit_recovery``: True if price recovered above the trade's
      peak (the high-water-mark the trailing stop tracked)
    - ``post_exit_would_tp``: True if price hit the take-profit level
    - ``post_exit_final_pct``: price change from exit to N bars later (%)
    - ``stop_verdict``: "premature" if recovery, "correct" otherwise
    """
    import numpy as np

    if candles.empty:
        return closed_trades

    closes = candles["close"].values
    highs = candles["high"].values
    lows = candles["low"].values

    for trade in closed_trades:
        exit_idx = trade.get("exit_bar_idx")
        if exit_idx is None:
            continue

        entry_px = trade.get("entry_price", 0)
        exit_px = trade.get("exit_price", 0)
        side = trade.get("side", "long")
        tp_pct = trade.get("take_profit_pct", 0.054)

        start = exit_idx + 1
        end = min(exit_idx + 1 + post_exit_bars, len(candles))
        if start >= len(candles) or start >= end:
            trade["post_exit_max_favorable"] = 0.0
            trade["post_exit_max_adverse"] = 0.0
            trade["post_exit_recovery"] = False
            trade["post_exit_would_tp"] = False
            trade["post_exit_final_pct"] = 0.0
            trade["stop_verdict"] = "no_data"
            continue

        post_highs = highs[start:end]
        post_lows = lows[start:end]
        post_closes = closes[start:end]

        if side == "long":
            post_max_high = float(np.max(post_highs))
            post_min_low = float(np.min(post_lows))
            post_mfe = (post_max_high - exit_px) / exit_px * 100 if exit_px > 0 else 0
            post_mae = (post_min_low - exit_px) / exit_px * 100 if exit_px > 0 else 0
            post_final = (float(post_closes[-1]) - exit_px) / exit_px * 100 if exit_px > 0 else 0

            peak_px = entry_px * (1 + trade.get("max_favorable_excursion", 0) / 100) if entry_px else exit_px
            recovered = post_max_high > peak_px
            tp_level = entry_px * (1 + tp_pct) if entry_px > 0 else 0
            would_tp = post_max_high >= tp_level if tp_level > 0 else False
        else:
            post_min_low = float(np.min(post_lows))
            post_max_high = float(np.max(post_highs))
            post_mfe = (exit_px - post_min_low) / exit_px * 100 if exit_px > 0 else 0
            post_mae = (exit_px - post_max_high) / exit_px * 100 if exit_px > 0 else 0
            post_final = (exit_px - float(post_closes[-1])) / exit_px * 100 if exit_px > 0 else 0

            trough_px = entry_px * (1 - trade.get("max_favorable_excursion", 0) / 100) if entry_px else exit_px
            recovered = post_min_low < trough_px
            tp_level = entry_px * (1 - tp_pct) if entry_px > 0 else 0
            would_tp = post_min_low <= tp_level if tp_level > 0 else False

        trade["post_exit_max_favorable"] = round(post_mfe, 3)
        trade["post_exit_max_adverse"] = round(post_mae, 3)
        trade["post_exit_recovery"] = recovered
        trade["post_exit_would_tp"] = would_tp
        trade["post_exit_final_pct"] = round(post_final, 3)

        if trade.get("close_reason") in ("trailing_stop", "short_trailing_stop"):
            if recovered or would_tp:
                trade["stop_verdict"] = "premature"
            elif post_mfe > 1.0:
                trade["stop_verdict"] = "early_but_ok"
            else:
                trade["stop_verdict"] = "correct"
        else:
            trade["stop_verdict"] = "n/a"

    return closed_trades


def diagnose_exits(closed_trades: list[dict]) -> dict:
    """Aggregate post-exit diagnostics for all trailing-stop exits.

    Requires trades enriched by :func:`enrich_trades_with_post_exit`.
    Returns summary showing how many stops were premature vs correct.
    """
    stop_trades = [
        t for t in closed_trades
        if t.get("close_reason") in ("trailing_stop", "short_trailing_stop")
        and "post_exit_max_favorable" in t
    ]
    if not stop_trades:
        return {}

    long_stops = [t for t in stop_trades if t.get("side") == "long"]
    short_stops = [t for t in stop_trades if t.get("side") == "short"]

    def _summarize(trades: list[dict]) -> dict:
        if not trades:
            return {}
        n = len(trades)
        premature = sum(1 for t in trades if t.get("stop_verdict") == "premature")
        correct = sum(1 for t in trades if t.get("stop_verdict") == "correct")
        early_ok = sum(1 for t in trades if t.get("stop_verdict") == "early_but_ok")
        avg_post_mfe = sum(t.get("post_exit_max_favorable", 0) for t in trades) / n
        avg_post_mae = sum(t.get("post_exit_max_adverse", 0) for t in trades) / n
        avg_post_final = sum(t.get("post_exit_final_pct", 0) for t in trades) / n
        would_tp = sum(1 for t in trades if t.get("post_exit_would_tp"))
        avg_pnl = sum(t.get("pnl_pct", 0) for t in trades) / n
        return {
            "count": n,
            "premature": premature,
            "correct": correct,
            "early_but_ok": early_ok,
            "would_have_hit_tp": would_tp,
            "avg_trade_pnl_pct": round(avg_pnl, 3),
            "avg_post_exit_mfe_pct": round(avg_post_mfe, 3),
            "avg_post_exit_mae_pct": round(avg_post_mae, 3),
            "avg_post_exit_final_pct": round(avg_post_final, 3),
        }

    per_trade = []
    for t in stop_trades:
        per_trade.append({
            "bar": t.get("entry_bar_idx"),
            "side": t.get("side"),
            "regime": t.get("entry_regime"),
            "pnl_pct": round(t.get("pnl_pct", 0), 2),
            "in_trade_mfe": t.get("max_favorable_excursion"),
            "exit_reason": t.get("close_reason"),
            "post_mfe": t.get("post_exit_max_favorable"),
            "post_mae": t.get("post_exit_max_adverse"),
            "post_final": t.get("post_exit_final_pct"),
            "recovered": t.get("post_exit_recovery"),
            "would_tp": t.get("post_exit_would_tp"),
            "verdict": t.get("stop_verdict"),
        })

    return {
        "long_stops": _summarize(long_stops),
        "short_stops": _summarize(short_stops),
        "per_trade": per_trade,
    }


def diagnose_shorts_by_confidence(closed_trades: list[dict]) -> dict:
    """Break down short trades by regime and regime confidence bucket.

    Requires ``regime_confidence`` field on each trade (set by the backtest).
    Returns per-bucket stats that separate genuine regime-classified shorts
    from ambiguity-window trades (low confidence).

    Confidence buckets:
    - high:   >= 0.60
    - medium: 0.40 – 0.59
    - low:    < 0.40  (regime classification is uncertain)
    """
    shorts = [t for t in closed_trades if t.get("side") == "short"]
    if not shorts:
        return {}

    def _bucket(conf: float | None) -> str:
        if conf is None:
            return "unknown"
        if conf >= 0.60:
            return "high"
        if conf >= 0.40:
            return "medium"
        return "low"

    def _summarize(trades: list[dict]) -> dict:
        if not trades:
            return {}
        n = len(trades)
        wins = sum(1 for t in trades if t.get("pnl_usd", 0) > 0)
        pnls = [t.get("pnl_pct", 0.0) for t in trades]
        return {
            "count": n,
            "wins": wins,
            "win_rate": round(wins / n, 3) if n else 0,
            "avg_pnl_pct": round(sum(pnls) / n, 3) if n else 0,
            "total_pnl_pct": round(sum(pnls), 3),
        }

    by_regime_conf: dict[str, list[dict]] = {}
    for t in shorts:
        regime = t.get("entry_regime") or "unknown"
        bucket = _bucket(t.get("regime_confidence"))
        key = f"{regime}|{bucket}"
        by_regime_conf.setdefault(key, []).append(t)

    report: dict[str, dict] = {}
    for key in sorted(by_regime_conf.keys()):
        report[key] = _summarize(by_regime_conf[key])

    per_trade = []
    for t in shorts:
        per_trade.append({
            "bar": t.get("entry_bar_idx"),
            "regime": t.get("entry_regime"),
            "regime_conf": round(t.get("regime_confidence", 0) or 0, 3),
            "conf_bucket": _bucket(t.get("regime_confidence")),
            "pnl_pct": round(t.get("pnl_pct", 0), 2),
            "exit": t.get("close_reason"),
            "hold_bars": (t.get("exit_bar_idx", 0) or 0) - (t.get("entry_bar_idx", 0) or 0),
        })

    return {"by_regime_confidence": report, "per_trade": per_trade}


def diagnose_longs_by_confidence(closed_trades: list[dict]) -> dict:
    """Break down long trades by regime and regime confidence bucket.

    Mirrors :func:`diagnose_shorts_by_confidence` for the long side.

    Confidence buckets:
    - high:   >= 0.60
    - medium: 0.40 – 0.59
    - low:    < 0.40  (regime classification is uncertain)
    """
    longs = [t for t in closed_trades if t.get("side") == "long"]
    if not longs:
        return {}

    def _bucket(conf: float | None) -> str:
        if conf is None:
            return "unknown"
        if conf >= 0.60:
            return "high"
        if conf >= 0.40:
            return "medium"
        return "low"

    def _summarize(trades: list[dict]) -> dict:
        if not trades:
            return {}
        n = len(trades)
        wins = sum(1 for t in trades if t.get("pnl_usd", 0) > 0)
        pnls = [t.get("pnl_pct", 0.0) for t in trades]
        return {
            "count": n,
            "wins": wins,
            "win_rate": round(wins / n, 3) if n else 0,
            "avg_pnl_pct": round(sum(pnls) / n, 3) if n else 0,
            "total_pnl_pct": round(sum(pnls), 3),
        }

    by_regime_conf: dict[str, list[dict]] = {}
    for t in longs:
        regime = t.get("entry_regime") or "unknown"
        bucket = _bucket(t.get("regime_confidence"))
        key = f"{regime}|{bucket}"
        by_regime_conf.setdefault(key, []).append(t)

    report: dict[str, dict] = {}
    for key in sorted(by_regime_conf.keys()):
        report[key] = _summarize(by_regime_conf[key])

    per_trade = []
    for t in longs:
        per_trade.append({
            "bar": t.get("entry_bar_idx"),
            "regime": t.get("entry_regime"),
            "regime_conf": round(t.get("regime_confidence", 0) or 0, 3),
            "conf_bucket": _bucket(t.get("regime_confidence")),
            "pnl_pct": round(t.get("pnl_pct", 0), 2),
            "exit": t.get("close_reason"),
            "hold_bars": (t.get("exit_bar_idx", 0) or 0) - (t.get("entry_bar_idx", 0) or 0),
        })

    return {"by_regime_confidence": report, "per_trade": per_trade}


def diagnose_long_entries(closed_trades: list[dict]) -> dict:
    """Aggregate entry-timing diagnostics for long trades only.

    Requires trades enriched by :func:`enrich_trades_with_entry_context`.
    Returns summary statistics that reveal whether longs are entering
    at local highs (top-chasing) or local lows (dip-buying).
    """
    longs = [t for t in closed_trades if t.get("side") == "long" and "local_range_position" in t]
    if not longs:
        return {}

    winners = [t for t in longs if t.get("pnl_usd", 0) > 0]
    losers = [t for t in longs if t.get("pnl_usd", 0) <= 0]

    def _avg(trades: list[dict], key: str) -> float:
        vals = [t[key] for t in trades if key in t]
        return round(sum(vals) / len(vals), 3) if vals else 0.0

    return {
        "total_longs": len(longs),
        "winners": len(winners),
        "losers": len(losers),
        "all": {
            "avg_range_position": _avg(longs, "local_range_position"),
            "avg_pct_from_high": _avg(longs, "pct_from_local_high"),
            "avg_run_up_before": _avg(longs, "run_up_before_entry"),
            "avg_mfe": _avg(longs, "max_favorable_excursion"),
            "avg_mae": _avg(longs, "max_adverse_excursion"),
            "avg_bars_to_peak": _avg(longs, "bars_to_peak"),
            "avg_bars_to_trough": _avg(longs, "bars_to_trough"),
        },
        "winners": {
            "count": len(winners),
            "avg_range_position": _avg(winners, "local_range_position"),
            "avg_pct_from_high": _avg(winners, "pct_from_local_high"),
            "avg_run_up_before": _avg(winners, "run_up_before_entry"),
            "avg_mfe": _avg(winners, "max_favorable_excursion"),
            "avg_mae": _avg(winners, "max_adverse_excursion"),
        },
        "losers": {
            "count": len(losers),
            "avg_range_position": _avg(losers, "local_range_position"),
            "avg_pct_from_high": _avg(losers, "pct_from_local_high"),
            "avg_run_up_before": _avg(losers, "run_up_before_entry"),
            "avg_mfe": _avg(losers, "max_favorable_excursion"),
            "avg_mae": _avg(losers, "max_adverse_excursion"),
        },
        "per_trade": [
            {
                "bar": t.get("entry_bar_idx"),
                "regime": t.get("entry_regime"),
                "pnl_pct": round(t.get("pnl_pct", 0), 2),
                "range_pos": t.get("local_range_position"),
                "run_up": t.get("run_up_before_entry"),
                "mfe": t.get("max_favorable_excursion"),
                "mae": t.get("max_adverse_excursion"),
                "bars_to_peak": t.get("bars_to_peak"),
                "exit": t.get("close_reason"),
            }
            for t in longs
        ],
    }


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
    trail_activation_pct: float = 0.0,
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
    # MTF execution: use 15m candles for entry timing within 1h thesis
    candles_15m: pd.DataFrame | None = None,
    mtf_thesis_max_age: int = 4,
    # Pullback gate: when False, skip the anti-chase pullback filter
    enable_pullback_gate: bool = True,
    # Close-and-reverse: when True, a sell signal can close a long AND open
    # a short on the same bar.  When False, short entry waits until next signal.
    # Attribution testing showed close-and-reverse adds nothing on BTC/USD;
    # keep off by default, enable via CLI when testing other assets.
    enable_close_and_reverse: bool = False,
    # Short max hold (hours): explicit side-specific hold.  0 = use long max hold.
    short_max_hold_hours: float = 0.0,
    # DB path for sentiment/macro agents (as-of timestamp semantics)
    db_path: str | None = None,
    # Entry quality / edge gate (from config)
    min_edge_multiple: float = 1.5,
    min_final_confidence: float = 0.25,
    min_tech_confidence: float = 0.15,
    min_regime_confidence: float = 0.30,
    max_whipsaws: int = 3,
    reversal_confidence_mult: float = 1.3,
    macro_sitout=None,
    use_ml_as_sizer: bool = False,
    funding_overlay=None,
    use_policy_core: bool = False,
    swarm_enabled: bool = False,
    swarm_mode: str = "shadow",
    swarm_agents: str = "pipeline_v1,risk_steward_v1,data_guardian_v1,execution_cost_v1",
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
        # Fields needed by policy_core.decide()
        fee_rate=fee_rate,
        max_risk_per_trade=max_risk_per_trade,
        aggressive_allocation=aggressive_allocation,
        use_ml_filter=ml_model is not None,
        use_ml_as_sizer=use_ml_as_sizer,
        ml_confidence_sizing=ml_confidence_sizing,
        min_edge_multiple=min_edge_multiple,
        min_final_confidence=min_final_confidence,
        min_tech_confidence=min_tech_confidence,
        min_regime_confidence=min_regime_confidence,
        max_whipsaws=max_whipsaws,
        swarm_enabled=swarm_enabled,
        swarm_mode=swarm_mode,
        swarm_agents=swarm_agents,
        swarm_min_agreement=0.60,
        swarm_min_vote_margin=0.10,
        swarm_max_entropy=0.95,
        swarm_log_full_votes=True,
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

    # Track entry bar index and regime for each symbol (for closed_trades)
    _entry_bar: dict[str, int] = {}
    _entry_regime: dict[str, str | None] = {}
    _entry_regime_conf: dict[str, float | None] = {}

    # Cooldown: bars remaining before next entry is allowed
    _cooldown_remaining: int = 0

    _rl_bars_in_trade: int = 0

    # Conviction persistence (parity with event_loop)
    _consecutive_exit_signals: int = 0
    _min_hold_bars: int = 3
    _exit_confirm_bars: int = 2

    # Short position tracking
    _short_entry_bar: dict[str, int] = {}
    _short_entry_regime: dict[str, str | None] = {}
    _short_entry_regime_conf: dict[str, float | None] = {}
    _consecutive_short_exit_signals: int = 0
    _pending_shorts: dict[str, float] = {}
    _pending_covers: dict[str, tuple[float, float, int | None, str]] = {}
    short_wins: int = 0
    short_closed: int = 0

    # ── MTF thesis/execution state ──────────────────────────────────────
    _use_mtf = candles_15m is not None and not candles_15m.empty
    _active_thesis: Thesis | None = None
    _mtf_map: dict[int, list[int]] = {}
    if _use_mtf:
        from hogan_bot.thesis_executor import (
            Thesis, align_15m_to_1h, find_15m_entry_in_window,
        )
        _mtf_map = align_15m_to_1h(candles, candles_15m)

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
        "mtf_thesis_created": 0, "mtf_thesis_executed": 0,
        "mtf_thesis_expired": 0, "mtf_15m_entry_used": 0,
        "blocked_short_ml_blind": 0, "loss_streak_scaled": 0,
    }
    # ML probability histogram (to understand model output distribution)
    _ml_probs: list[float] = []
    # Trade outcome history for loss-streak dampener (True=win, False=loss)
    _trade_outcomes: list[bool] = []

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
    _regime_transition = RegimeTransitionTracker(cooldown_bars=3, min_scale=0.40)

    # Next-open execution: pending buys/sells to fill at next bar's open
    _pending_buys: dict[str, float] = {}
    _pending_sells: dict[str, tuple[float, float, int | None, str]] = {}

    _pc_state = None
    if use_policy_core:
        from hogan_bot.policy_core import PolicyState, decide as _pc_decide
        _pc_state = PolicyState()

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
                if portfolio.execute_buy(sym, buy_px, size, trailing_stop_pct=trailing_stop_pct, take_profit_pct=take_profit_pct, trail_activation_pct=trail_activation_pct):
                    trades += 1
                    _entry_bar[sym] = i - 1
                    _entry_regime[sym] = _current_regime
                    _entry_regime_conf[sym] = _regime_conf
                    trade_log.append({"bar": bar_ts, "action": "buy", "reason": "signal", "price": buy_px, "qty": size})
            for sym, (qty, avg_entry, entry_bar_idx, reason) in list(_pending_sells.items()):
                _pending_sells.pop(sym, None)
                _entry_bar.pop(sym, None)
                sell_px = open_px * (1.0 - slip_mult)
                if portfolio.execute_sell(sym, sell_px, qty):
                    trades += 1
                    closed += 1
                    is_win = sell_px > avg_entry
                    _trade_outcomes.append(is_win)
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
                            "entry_regime": _entry_regime.pop(sym, _current_regime),
                            "regime_confidence": _entry_regime_conf.pop(sym, _regime_conf),
                            "close_reason": reason,
                        })
            for sym, size in list(_pending_shorts.items()):
                _pending_shorts.pop(sym, None)
                short_px = open_px * (1.0 - slip_mult)
                if portfolio.execute_short(sym, short_px, size, trailing_stop_pct=trailing_stop_pct, take_profit_pct=take_profit_pct, trail_activation_pct=trail_activation_pct):
                    trades += 1
                    _short_entry_bar[sym] = i - 1
                    _short_entry_regime[sym] = _current_regime
                    _short_entry_regime_conf[sym] = _regime_conf
                    trade_log.append({"bar": bar_ts, "action": "short", "reason": "signal", "price": short_px, "qty": size})
            for sym, (qty, avg_entry, entry_bar_idx, reason) in list(_pending_covers.items()):
                _pending_covers.pop(sym, None)
                _short_entry_bar.pop(sym, None)
                cover_px = open_px * (1.0 + slip_mult)
                if portfolio.execute_cover(sym, cover_px, qty):
                    trades += 1
                    short_closed += 1
                    is_win = cover_px < avg_entry
                    _trade_outcomes.append(is_win)
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
                            "entry_regime": _short_entry_regime.pop(sym, _current_regime),
                            "regime_confidence": _short_entry_regime_conf.pop(sym, _regime_conf),
                            "close_reason": reason,
                        })

        mark = {symbol: px}
        if enable_shorts and short_max_hold_hours > 0:
            from hogan_bot.timeframe_utils import hours_to_bars
            _short_max = hours_to_bars(short_max_hold_hours, _tf)
        elif enable_shorts:
            _short_max = max_hold_bars
        else:
            _short_max = 0

        exits = portfolio.check_exits(
            mark, max_hold_bars=max_hold_bars,
            short_max_hold_bars=_short_max,
        )
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
                        _trade_outcomes.append(is_win)
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
                                "entry_regime": _short_entry_regime.pop(exit_symbol, _current_regime),
                                "regime_confidence": _short_entry_regime_conf.pop(exit_symbol, _regime_conf),
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
                    _trade_outcomes.append(is_win)
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
                            "entry_regime": _entry_regime.pop(exit_symbol, _current_regime),
                            "regime_confidence": _entry_regime_conf.pop(exit_symbol, _regime_conf),
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

        # Regime transition dampener: reduce sizing at regime boundaries
        _transition_scale = _regime_transition.update(_current_regime) if _current_regime else 1.0
        if _transition_scale < 1.0:
            _funnel["regime_transition_bars"] = _funnel.get("regime_transition_bars", 0) + 1

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
        _eff_position_scale = _eff.get("position_scale", 1.0) * _transition_scale
        _eff_allow_longs = _eff.get("allow_longs", True)
        _eff_allow_shorts = _eff.get("allow_shorts", True)
        _eff_long_size_scale = _eff.get("long_size_scale", 1.0)
        _eff_short_size_scale = _eff.get("short_size_scale", 1.0)

        _as_of = _bar_ts_ms(candles, i - 1) if _bt_conn is not None else None

        # ── Policy-core delegation ──────────────────────────────────
        if use_policy_core:
            _pc_state.ml_probs = _ml_probs
            _pc_state.trade_outcomes = _trade_outcomes

            _intent = _pc_decide(
                symbol=symbol,
                candles=window,
                equity_usd=equity_curve[-1] if equity_curve else starting_balance_usd,
                config=_bt_config,
                pipeline=_pipeline,
                ml_model=ml_model,
                state=_pc_state,
                conn=_bt_conn,
                as_of_ms=_as_of,
                mode="backtest",
                recent_whipsaw_count=_whipsaw_count,
                macro_sitout=macro_sitout,
                funding_overlay=None,
                enable_pullback_gate=enable_pullback_gate,
                enable_freshness_check=False,
                peak_equity_usd=guard.peak_equity,
            )

            action = _intent.action
            up_prob = _intent.up_prob
            size = _intent.size_usd / px if px > 0 else 0.0
            conf_scale = _intent.conf_scale
            _quality_scale = _intent.quality_scale
            _ranging_scale = _intent.ranging_scale
            _pullback_scale = _intent.pullback_scale
            _momentum_scale = _intent.momentum_scale
            _atr_pct = _intent.atr_pct
            _eff_ts = _intent.eff_trailing_stop_pct or _eff_ts
            _eff_tp = _intent.eff_take_profit_pct or _eff_tp
            _eff_allow_longs = _intent.eff_allow_longs
            _eff_allow_shorts = _intent.eff_allow_shorts
            _eff_long_size_scale = _intent.eff_long_size_scale
            _eff_short_size_scale = _intent.eff_short_size_scale
            signal = types.SimpleNamespace(
                action=_intent.action,
                confidence=_intent.confidence,
                stop_distance_pct=_intent.stop_distance_pct,
                volume_ratio=_intent.vol_ratio,
                tech=None,
                forecast=None,
                explanation=_intent.explanation,
            )

            _funnel["bars_evaluated"] += 1
            if _intent.action == "buy":
                _funnel["pipeline_buy"] += 1
                _funnel["post_ml_buy"] += 1
                _funnel["post_edge_buy"] += 1
                _funnel["post_quality_buy"] += 1
                _funnel["post_ranging_buy"] += 1
            elif _intent.action == "sell":
                _funnel["pipeline_sell"] += 1
                _funnel["post_ml_sell"] += 1
                _funnel["post_edge_sell"] += 1
                _funnel["post_quality_sell"] += 1
                _funnel["post_ranging_sell"] += 1
            for _br in _intent.block_reasons:
                _funnel[_br] = _funnel.get(_br, 0) + 1

            equity = portfolio.total_equity(mark)
            equity_curve.append(equity)
            if not guard.update_and_check(equity):
                break

        if not use_policy_core:
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
                if hasattr(ml_model, "set_regime"):
                    ml_model.set_regime(_current_regime)
                up_prob = predict_up_probability(window, ml_model)
                _ml_probs.append(up_prob)
                if use_ml_as_sizer:
                    conf_scale *= ml_probability_sizer(action, up_prob)
                else:
                    _ml_gd = apply_ml_filter(action, up_prob, _eff_ml_buy, _eff_ml_sell)
                    action = _ml_gd.action
                    if ml_confidence_sizing:
                        conf_scale *= ml_confidence(up_prob)
                _blind = ml_blind_scale(_ml_probs)
                if _blind < 1.0:
                    conf_scale *= _blind
                    _funnel["ml_blind_scaled"] = _funnel.get("ml_blind_scaled", 0) + 1

            _ls_scale = loss_streak_scale(_trade_outcomes)
            if _ls_scale < 1.0:
                conf_scale *= _ls_scale
                _funnel["loss_streak_scaled"] = _funnel.get("loss_streak_scaled", 0) + 1

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

            if enable_pullback_gate:
                _pb_range = 0.70 if _use_mtf else 0.55
                _pb_runup = 3.0 if _use_mtf else 2.0
                _pullback_gd = pullback_gate(
                    action, window,
                    max_range_position=_pb_range,
                    max_run_up_pct=_pb_runup,
                    regime=_current_regime,
                )
                action = _pullback_gd.action
                _pullback_scale = _pullback_gd.size_scale
                if _pullback_gd.blocked_by:
                    _funnel["pullback_blocked"] = _funnel.get("pullback_blocked", 0) + 1
                    if "resistance" in (_pullback_gd.blocked_by or ""):
                        _funnel["pullback_blocked_resistance"] = _funnel.get("pullback_blocked_resistance", 0) + 1
                elif _pullback_scale < 1.0 and action == "buy":
                    _funnel["pullback_halved"] = _funnel.get("pullback_halved", 0) + 1
            else:
                _pullback_scale = 1.0

            if action == "buy":
                _funnel["post_ranging_buy"] += 1
            elif action == "sell":
                _funnel["post_ranging_sell"] += 1

            # ── Long momentum confirmation ───────────────────────────────────
            _momentum_scale = 1.0
            if action == "buy" and len(window) >= 8:
                _ema8 = window["close"].ewm(span=8, min_periods=8).mean().iloc[-1]
                if px < _ema8:
                    _momentum_scale = 0.3
                    _funnel["long_momentum_reduced"] = _funnel.get("long_momentum_reduced", 0) + 1
                else:
                    _funnel["long_momentum_confirmed"] = _funnel.get("long_momentum_confirmed", 0) + 1

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
                confidence_scale=conf_scale * _quality_scale * _ranging_scale * _pullback_scale * _eff_position_scale * _momentum_scale,
                fee_rate=fee_rate,
            )

            # ── Macro sitout filter ────────────────────────────────────────
            if macro_sitout is not None and action != "hold":
                _bar_ts = candles.iloc[i - 1].get("timestamp") if i > 0 else None
                _sitout = macro_sitout.check(_bar_ts)
                if _sitout.should_sitout:
                    _funnel["macro_sitout"] = _funnel.get("macro_sitout", 0) + 1
                    action = "hold"
                elif _sitout.size_scale < 1.0:
                    size *= _sitout.size_scale
                    _funnel["macro_scaled"] = _funnel.get("macro_scaled", 0) + 1

        # Track whipsaws (signal flipped from last bar)
        if action != "hold" and _last_signal != "hold" and action != _last_signal:
            _whipsaw_count = min(_whipsaw_count + 1, 10)
        elif action == _last_signal:
            _whipsaw_count = max(0, _whipsaw_count - 1)
        _last_signal = action

        # ── MTF thesis / 15m execution ──────────────────────────────────
        if _use_mtf:
            # Clear executed thesis once the position has closed
            if _active_thesis is not None and _active_thesis.executed:
                if _active_thesis.direction == "long" and symbol not in portfolio.positions:
                    _active_thesis = None
                elif _active_thesis.direction == "short" and symbol not in portfolio.short_positions:
                    _active_thesis = None

            # Expire stale thesis
            if _active_thesis is not None and not _active_thesis.executed:
                age = (i - 1) - _active_thesis.created_bar_1h
                if age >= _active_thesis.max_age_1h_bars:
                    _active_thesis.expired = True
                    _funnel["mtf_thesis_expired"] += 1
                    _active_thesis = None

            # Check active thesis for 15m entry
            if (
                _active_thesis is not None
                and not _active_thesis.executed
                and symbol not in portfolio.positions
                and symbol not in portfolio.short_positions
                and _cooldown_remaining <= 0
            ):
                _thesis_bar = i - 1
                _15m_indices = _mtf_map.get(_thesis_bar, [])
                if _15m_indices:
                    _trigger = find_15m_entry_in_window(
                        candles_15m, _15m_indices, _active_thesis,
                    )
                    if _trigger is not None:
                        _mtf_px = _trigger.entry_price * (
                            1.0 + slip_mult if _active_thesis.direction == "long"
                            else 1.0 - slip_mult
                        )
                        _mtf_size = size * _eff_long_size_scale
                        if _active_thesis.direction == "long" and _eff_allow_longs and _mtf_size > 0:
                            if portfolio.execute_buy(
                                symbol, _mtf_px, _mtf_size,
                                trailing_stop_pct=_eff_ts,
                                take_profit_pct=_eff_tp,
                                trail_activation_pct=trail_activation_pct,
                            ):
                                _pos = portfolio.positions.get(symbol)
                                if _pos is not None:
                                    _pos.entry_atr_pct = _atr_pct
                                trades += 1
                                _funnel["executed_buy"] += 1
                                _funnel["mtf_15m_entry_used"] += 1
                                _funnel["mtf_thesis_executed"] += 1
                                _entry_bar[symbol] = _thesis_bar
                                _entry_regime[symbol] = _current_regime
                                _entry_regime_conf[symbol] = _regime_conf
                                trade_log.append({
                                    "bar": bar_ts, "action": "buy",
                                    "reason": f"mtf_15m_{_trigger.trigger_reason}",
                                    "price": _mtf_px, "qty": _mtf_size,
                                })
                                _active_thesis.executed = True

            # New thesis creation: defer new ENTRIES to 15m timing.
            # Sell signals for closing existing longs pass through untouched.
            # Regime gating is applied here to prevent thesis creation
            # when the regime blocks the direction.
            _has_open_long = symbol in portfolio.positions
            _has_open_short = symbol in portfolio.short_positions
            # Only defer LONG entries to 15m timing. Short entries execute
            # immediately because waiting for a 15m bounce in a downtrend
            # degrades short entry quality.
            _is_new_entry = action == "buy" and not _has_open_long and _eff_allow_longs
            if action == "buy" and not _eff_allow_longs and not _has_open_long:
                _funnel["blocked_regime_no_longs"] += 1
            elif action == "sell" and enable_shorts and not _eff_allow_shorts and not _has_open_long:
                _funnel["blocked_regime_no_shorts"] += 1
            if _is_new_entry and _active_thesis is None:
                _thesis_dir = "long" if action == "buy" else "short"
                _active_thesis = Thesis(
                    direction=_thesis_dir,
                    created_bar_1h=i - 1,
                    confidence=signal.confidence,
                    creation_price=px,
                    regime=_current_regime,
                    max_age_1h_bars=mtf_thesis_max_age,
                )
                _funnel["mtf_thesis_created"] += 1
                action = "hold"
            elif _is_new_entry and _active_thesis is not None:
                if (action == "buy" and _active_thesis.direction == "long") or \
                   (action == "sell" and _active_thesis.direction == "short"):
                    _active_thesis.created_bar_1h = i - 1
                    _active_thesis.confidence = signal.confidence
                    _active_thesis.creation_price = px
                else:
                    _active_thesis = Thesis(
                        direction="long" if action == "buy" else "short",
                        created_bar_1h=i - 1,
                        confidence=signal.confidence,
                        creation_price=px,
                        regime=_current_regime,
                        max_age_1h_bars=mtf_thesis_max_age,
                    )
                    _funnel["mtf_thesis_created"] += 1
                action = "hold"

        if action == "buy":
            _consecutive_exit_signals = 0

            # Cover any existing short first (with ExitEvaluator confirmation)
            if enable_shorts and symbol in portfolio.short_positions:
                _consecutive_short_exit_signals += 1
                spos = portfolio.short_positions[symbol]
                if spos.bars_held < _min_hold_bars:
                    pass
                elif _consecutive_short_exit_signals < _exit_confirm_bars:
                    pass
                else:
                    _short_exit_dec = _exit_eval.should_exit(
                        candles=window,
                        entry_price=spos.avg_entry,
                        current_price=px,
                        bars_held=spos.bars_held,
                        side="short",
                        max_hold_bars=_short_max if _short_max > 0 else max_hold_bars,
                        entry_atr=getattr(spos, "entry_atr_pct", None) or None,
                        vol_ratio=signal.volume_ratio,
                    )
                    if not _short_exit_dec.should_exit:
                        pass
                    else:
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
                                _trade_outcomes.append(is_win)
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
                                        "entry_regime": _short_entry_regime.pop(symbol, _current_regime),
                                        "regime_confidence": _short_entry_regime_conf.pop(symbol, _regime_conf),
                                        "close_reason": "buy_signal",
                                    })

            _long_size = size * _eff_long_size_scale
            if funding_overlay is not None:
                _bar_ts_val = candles.iloc[i - 1].get("timestamp") if i > 0 else None
                _long_size *= funding_overlay.position_scale("buy", _bar_ts_val)
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
                    trailing_stop_pct=_eff_ts,
                    take_profit_pct=_eff_tp,
                    trail_activation_pct=trail_activation_pct,
                ):
                    _pos = portfolio.positions.get(symbol)
                    if _pos is not None:
                        _pos.entry_atr_pct = _atr_pct
                    trades += 1
                    _funnel["executed_buy"] += 1
                    _entry_bar[symbol] = i - 1
                    _entry_regime[symbol] = _current_regime
                    _entry_regime_conf[symbol] = _regime_conf
                    trade_log.append(
                        {"bar": bar_ts, "action": "buy", "reason": "signal", "price": buy_px, "qty": _long_size}
                    )
        elif action == "sell":
            # Close existing long (with ExitEvaluator confirmation)
            _closed_long_this_bar = False
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
                    _rev_thresh = min_final_confidence * reversal_confidence_mult
                    if pos.bars_held < _min_hold_bars:
                        pass
                    elif signal.confidence < _rev_thresh:
                        pass
                    elif _consecutive_exit_signals < _exit_confirm_bars:
                        pass
                    elif not _exit_dec.should_exit:
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
                                _closed_long_this_bar = True
                                trades += 1
                                closed += 1
                                is_win = sell_px_sig > avg_entry
                                _trade_outcomes.append(is_win)
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
                                        "entry_regime": _entry_regime.pop(symbol, _current_regime),
                                        "regime_confidence": _entry_regime_conf.pop(symbol, _regime_conf),
                                        "close_reason": "signal",
                                    })

            # Open short when flat and shorts enabled.
            # When enable_close_and_reverse is True, uses `if` (not elif) so a
            # sell signal can close a long AND open a short in one step.
            _allow_short_entry = (
                enable_shorts
                and symbol not in portfolio.positions
                and symbol not in portfolio.short_positions
            )
            if _closed_long_this_bar and not enable_close_and_reverse:
                _allow_short_entry = False
            if _allow_short_entry:
                if _closed_long_this_bar:
                    _funnel["close_and_reverse"] = _funnel.get("close_and_reverse", 0) + 1
                    _cooldown_remaining = 0
                _consecutive_exit_signals = 0
                _short_size = size * _eff_short_size_scale
                if funding_overlay is not None:
                    _bar_ts_val = candles.iloc[i - 1].get("timestamp") if i > 0 else None
                    _short_size *= funding_overlay.position_scale("sell", _bar_ts_val)
                if not _eff_allow_shorts:
                    _funnel["blocked_regime_no_shorts"] += 1
                elif ml_blind_blocks_shorts(_ml_probs):
                    _funnel["blocked_short_ml_blind"] += 1
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
                        trailing_stop_pct=_eff_ts,
                        take_profit_pct=_eff_tp,
                        trail_activation_pct=trail_activation_pct,
                    ):
                        spos = portfolio.short_positions.get(symbol)
                        if spos is not None:
                            spos.entry_atr_pct = _atr_pct
                        trades += 1
                        _funnel["executed_short_entry"] += 1
                        _short_entry_bar[symbol] = i - 1
                        _short_entry_regime[symbol] = _current_regime
                        _short_entry_regime_conf[symbol] = _regime_conf
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

    enrich_trades_with_entry_context(candles, closed_trades)
    enrich_trades_with_post_exit(closed_trades, candles)

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
