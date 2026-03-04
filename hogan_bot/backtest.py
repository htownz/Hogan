from __future__ import annotations

from dataclasses import dataclass, field

from hogan_bot.decision import apply_ml_filter
from hogan_bot.ml import TrainedModel, predict_up_probability
from hogan_bot.paper import PaperPortfolio
from hogan_bot.risk import DrawdownGuard, calculate_position_size
from hogan_bot.strategy import generate_signal


@dataclass
class BacktestResult:
    start_equity: float
    end_equity: float
    total_return_pct: float
    max_drawdown_pct: float
    trades: int
    win_rate: float
    equity_curve: list[float] = field(default_factory=list)
    trade_log: list[dict] = field(default_factory=list)

    def summary_dict(self) -> dict:
        """Return all scalar fields as a plain dict (omits large lists)."""
        return {
            "start_equity": self.start_equity,
            "end_equity": self.end_equity,
            "total_return_pct": self.total_return_pct,
            "max_drawdown_pct": self.max_drawdown_pct,
            "trades": self.trades,
            "win_rate": self.win_rate,
        }


def _compute_max_drawdown(equity_curve: list[float]) -> float:
    peak = equity_curve[0]
    max_dd = 0.0
    for equity in equity_curve:
        peak = max(peak, equity)
        if peak > 0:
            dd = (peak - equity) / peak
            max_dd = max(max_dd, dd)
    return max_dd


def run_backtest_on_candles(
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
        )

        action = signal.action
        if ml_model is not None:
            up_prob = predict_up_probability(window, ml_model)
            action = apply_ml_filter(action, up_prob, ml_buy_threshold, ml_sell_threshold)

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

    return BacktestResult(
        start_equity=starting_balance_usd,
        end_equity=end_equity,
        total_return_pct=total_return_pct,
        max_drawdown_pct=max_dd * 100,
        trades=trades,
        win_rate=win_rate,
        equity_curve=equity_curve,
        trade_log=trade_log,
    )
