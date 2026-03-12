from __future__ import annotations

import math
import types
from dataclasses import dataclass, field

import pandas as pd

from hogan_bot.agent_pipeline import AgentPipeline
from hogan_bot.decision import apply_ml_filter, ml_confidence
from hogan_bot.ml import TrainedModel, predict_up_probability
from hogan_bot.paper import PaperPortfolio
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

    def summary_dict(self) -> dict:
        """Return all scalar fields as a plain dict (omits large lists)."""
        return {
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
        return {"bars": len(equity_slice), "sharpe": None, "total_return_pct": 0.0}
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
    # DB path for sentiment/macro agents (as-of timestamp semantics)
    db_path: str | None = None,
) -> BacktestResult:
    """Run bar-by-bar paper backtest for a single symbol dataframe."""

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

    # Track entry bar index for each symbol (for closed_trades / ML learning)
    _entry_bar: dict[str, int] = {}

    # Cooldown: bars remaining before next entry is allowed
    _cooldown_remaining: int = 0

    _rl_bars_in_trade: int = 0

    # Next-open execution: pending buys/sells to fill at next bar's open
    _pending_buys: dict[str, float] = {}
    _pending_sells: dict[str, tuple[float, float, int | None, str]] = {}

    min_rows = max(long_ma_window, volume_window) + 2
    for i in range(min_rows, len(candles) + 1):
        window = candles.iloc[:i]
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
                        })

        mark = {symbol: px}
        exits = portfolio.check_exits(mark, max_hold_bars=max_hold_bars)
        for exit_symbol, reason in exits:
            pos = portfolio.positions.get(exit_symbol)
            if pos is None:
                continue
            qty = pos.qty
            avg_entry = pos.avg_entry
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

        _as_of = _bar_ts_ms(candles, i - 1) if _bt_conn is not None else None
        signal = _pipeline.run(
            window,
            symbol=symbol,
            as_of_ms=_as_of,
            rl_in_position=_rl_in_pos,
            rl_unrealized_pnl=_rl_upnl,
            rl_bars_in_trade=_rl_bars_in_trade,
        )

        action = signal.action
        conf_scale = signal.confidence or 1.0
        if ml_model is not None:
            up_prob = predict_up_probability(window, ml_model)
            action = apply_ml_filter(action, up_prob, ml_buy_threshold, ml_sell_threshold)
            if ml_confidence_sizing:
                conf_scale *= ml_confidence(up_prob)

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
            confidence_scale=conf_scale,
        )

        if action == "buy":
            if symbol in portfolio.positions:
                pass
            elif _cooldown_remaining > 0:
                pass
            elif use_next_open and i < len(candles):
                _pending_buys[symbol] = size
            else:
                buy_px = px * (1.0 + slip_mult)
                if portfolio.execute_buy(
                    symbol, buy_px, size,
                    trailing_stop_pct=trailing_stop_pct,
                    take_profit_pct=take_profit_pct,
                ):
                    trades += 1
                    _entry_bar[symbol] = i - 1
                    trade_log.append(
                        {"bar": bar_ts, "action": "buy", "reason": "signal", "price": buy_px, "qty": size}
                    )
        elif action == "sell":
            if symbol not in portfolio.positions:
                continue
            pos = portfolio.positions[symbol]
            qty = pos.qty
            if qty <= 0:
                continue
            sell_qty = min(qty, size)
            avg_entry = pos.avg_entry
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
                        })

    if not equity_curve:
        equity_curve = [starting_balance_usd]
    end_equity = equity_curve[-1]
    max_dd = _compute_max_drawdown(equity_curve)
    total_return_pct = (
        ((end_equity / starting_balance_usd) - 1.0) * 100 if starting_balance_usd > 0 else 0.0
    )
    win_rate = (wins / closed) if closed > 0 else 0.0
    max_dd_pct = max_dd * 100

    if _bt_conn is not None:
        _bt_conn.close()

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
    )
