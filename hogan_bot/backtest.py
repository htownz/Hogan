from __future__ import annotations

import math
from dataclasses import dataclass, field

from hogan_bot.decision import apply_ml_filter, ml_confidence
from hogan_bot.ml import TrainedModel, predict_up_probability
from hogan_bot.paper import PaperPortfolio
from hogan_bot.risk import DrawdownGuard, calculate_position_size
from hogan_bot.strategy import generate_signal

# Annualisation constant for 5-minute 24/7 crypto bars.
# 365 days × 24 hours × 12 five-minute intervals = 105 120 bars/year.
_BARS_PER_YEAR_5M: float = 105_120.0


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
    equity_curve: list[float] = field(default_factory=list)
    trade_log: list[dict] = field(default_factory=list)

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


def compute_sharpe(equity_curve: list[float], bars_per_year: float = _BARS_PER_YEAR_5M) -> float | None:
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


def compute_sortino(equity_curve: list[float], bars_per_year: float = _BARS_PER_YEAR_5M) -> float | None:
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
) -> BacktestResult:
    """Run bar-by-bar paper backtest for a single symbol dataframe."""

    if candles.empty:
        return BacktestResult(starting_balance_usd, starting_balance_usd, 0.0, 0.0, 0, 0.0)

    portfolio = PaperPortfolio(cash_usd=starting_balance_usd, fee_rate=fee_rate)
    guard = DrawdownGuard(starting_balance_usd, max_drawdown)

    wins = 0
    closed = 0
    trades = 0
    equity_curve: list[float] = []
    trade_log: list[dict] = []

    # RL position-state tracking (used in generate_signal RL vote)
    _rl_bars_in_trade: int = 0

    min_rows = max(long_ma_window, volume_window) + 2
    for i in range(min_rows, len(candles) + 1):
        window = candles.iloc[:i]
        px = float(window["close"].iloc[-1])
        bar_ts = str(window["timestamp"].iloc[-1]) if "timestamp" in window.columns else str(i)

        # Check trailing-stop and take-profit exits before the new signal
        mark = {symbol: px}
        exits = portfolio.check_exits(mark)
        for exit_symbol, reason in exits:
            pos = portfolio.positions.get(exit_symbol)
            if pos is None:
                continue
            qty = pos.qty
            avg_entry = pos.avg_entry
            if portfolio.execute_sell(exit_symbol, px, qty):
                trades += 1
                closed += 1
                if px > avg_entry:
                    wins += 1
                trade_log.append(
                    {
                        "bar": bar_ts,
                        "action": "sell",
                        "reason": reason,
                        "price": px,
                        "qty": qty,
                    }
                )

        # Build RL position state for this bar
        _rl_pos = portfolio.positions.get(symbol)
        _rl_in_pos = _rl_pos is not None
        if _rl_in_pos:
            _rl_upnl = (px - _rl_pos.avg_entry) / max(_rl_pos.avg_entry, 1e-9)
            _rl_bars_in_trade += 1
        else:
            _rl_upnl = 0.0
            _rl_bars_in_trade = 0

        signal = generate_signal(
            window,
            short_window=short_ma_window,
            long_window=long_ma_window,
            volume_window=volume_window,
            volume_threshold=volume_threshold,
            use_ema_clouds=use_ema_clouds,
            ema_fast_short=ema_fast_short,
            ema_fast_long=ema_fast_long,
            ema_slow_short=ema_slow_short,
            ema_slow_long=ema_slow_long,
            use_fvg=use_fvg,
            fvg_min_gap_pct=fvg_min_gap_pct,
            signal_mode=signal_mode,
            atr_stop_multiplier=atr_stop_multiplier,
            use_ict=use_ict,
            ict_swing_left=ict_swing_left,
            ict_swing_right=ict_swing_right,
            ict_eq_tolerance_pct=ict_eq_tolerance_pct,
            ict_min_displacement_pct=ict_min_displacement_pct,
            ict_require_time_window=ict_require_time_window,
            ict_time_windows=ict_time_windows,
            ict_require_pd=ict_require_pd,
            ict_ote_enabled=ict_ote_enabled,
            ict_ote_low=ict_ote_low,
            ict_ote_high=ict_ote_high,
            use_rl_agent=use_rl_agent,
            rl_policy=rl_policy,
            rl_in_position=_rl_in_pos,
            rl_unrealized_pnl=_rl_upnl,
            rl_bars_in_trade=_rl_bars_in_trade,
        )

        action = signal.action
        conf_scale = 1.0
        if ml_model is not None:
            up_prob = predict_up_probability(window, ml_model)
            action = apply_ml_filter(action, up_prob, ml_buy_threshold, ml_sell_threshold)
            if ml_confidence_sizing:
                conf_scale = ml_confidence(up_prob)

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
            if portfolio.execute_buy(
                symbol, px, size,
                trailing_stop_pct=trailing_stop_pct,
                take_profit_pct=take_profit_pct,
            ):
                trades += 1
                trade_log.append(
                    {"bar": bar_ts, "action": "buy", "reason": "signal", "price": px, "qty": size}
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
            if portfolio.execute_sell(symbol, px, sell_qty):
                trades += 1
                closed += 1
                if px > avg_entry:
                    wins += 1
                trade_log.append(
                    {
                        "bar": bar_ts,
                        "action": "sell",
                        "reason": "signal",
                        "price": px,
                        "qty": sell_qty,
                    }
                )

    if not equity_curve:
        equity_curve = [starting_balance_usd]
    end_equity = equity_curve[-1]
    max_dd = _compute_max_drawdown(equity_curve)
    total_return_pct = (
        ((end_equity / starting_balance_usd) - 1.0) * 100 if starting_balance_usd > 0 else 0.0
    )
    win_rate = (wins / closed) if closed > 0 else 0.0
    max_dd_pct = max_dd * 100

    return BacktestResult(
        start_equity=starting_balance_usd,
        end_equity=end_equity,
        total_return_pct=total_return_pct,
        max_drawdown_pct=max_dd_pct,
        trades=trades,
        win_rate=win_rate,
        sharpe_ratio=compute_sharpe(equity_curve),
        sortino_ratio=compute_sortino(equity_curve),
        calmar_ratio=compute_calmar(total_return_pct, max_dd_pct),
        equity_curve=equity_curve,
        trade_log=trade_log,
    )
