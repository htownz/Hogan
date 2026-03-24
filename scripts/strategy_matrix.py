#!/usr/bin/env python
"""Strategy Matrix Tournament — Entry x Exit walk-forward harness.

Run 5 entry families x 3 exit packs across assets using 5 non-overlapping
OOS windows.  Self-contained backtest engine with locked execution semantics,
fixed-risk sizing, and gross+net cost decomposition.

Usage:
    python scripts/strategy_matrix.py --db data/hogan.db
    python scripts/strategy_matrix.py --db data/hogan.db --assets BTC/USD ETH/USD
    python scripts/strategy_matrix.py --db data/hogan.db --long-only
    python scripts/strategy_matrix.py --db data/hogan.db --zero-cost
    python scripts/strategy_matrix.py --db data/hogan.db --entries A_donchian B_rsi_reclaim
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import sqlite3
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from hogan_bot.exit_packs import ALL_EXIT_PACKS, ExitPack
from hogan_bot.indicators import compute_atr
from hogan_bot.storage import load_candles
from hogan_bot.strategy_candidates import ENTRY_FAMILIES, get_entry_family

logger = logging.getLogger(__name__)


# ── Cost Profiles ───────────────────────────────────────────────────────────

CRYPTO_FEE = 0.0026
CRYPTO_SLIP_BPS = 5.0
FX_FEE = 0.0003
FX_SLIP_BPS = 1.0


def _fee_and_slippage(
    symbol: str,
    zero_cost: bool = False,
    custom_fee: float | None = None,
    custom_slip: float | None = None,
):
    if zero_cost:
        return 0.0, 0.0
    if custom_fee is not None:
        return custom_fee, custom_slip if custom_slip is not None else 0.0
    if "GBP" in symbol or "EUR" in symbol or "JPY" in symbol:
        return FX_FEE, FX_SLIP_BPS
    return CRYPTO_FEE, CRYPTO_SLIP_BPS


# ── Trade / Position ───────────────────────────────────────────────────────

@dataclass
class Trade:
    entry_bar: int
    entry_price: float
    side: str              # "long" or "short"
    stop_price: float
    tp_price: float        # 0 = no TP
    trail_distance: float  # 0 = no trail
    max_hold_bars: int
    size_usd: float
    atr_at_entry: float

    bars_held: int = 0
    peak_favorable: float = 0.0
    trail_active: bool = False
    exit_price: float = 0.0
    exit_reason: str = ""
    gross_pnl: float = 0.0
    cost: float = 0.0
    net_pnl: float = 0.0


# ── Matrix Backtest Engine ─────────────────────────────────────────────────

def _run_cell_backtest(
    candles: pd.DataFrame,
    entry_family,
    exit_pack: ExitPack,
    fee_rate: float,
    slippage_bps: float,
    starting_balance: float = 10_000.0,
    risk_per_trade: float = 0.01,
    long_only: bool = False,
) -> list[Trade]:
    """Run a single (entry, exit) backtest on candle data.

    Locked execution semantics:
    - Signal at bar close, fill at next bar open
    - One position per asset, no pyramiding
    - Ignore new entries while in position
    - No reversal while in trade
    - Adverse fill assumption on intrabar stop+TP collision
    - Force close at end with forced_window_exit tag
    """
    close = candles["close"].astype(float).values
    high = candles["high"].astype(float).values
    low = candles["low"].astype(float).values
    opn = candles["open"].astype(float).values
    n = len(candles)

    atr_series = compute_atr(candles, window=14).values

    trades: list[Trade] = []
    equity = starting_balance
    position: Trade | None = None
    pending_signal: str | None = None
    pending_atr: float = 0.0

    min_warmup = 60

    for i in range(min_warmup, n):
        # ── Process pending fill from prior bar's signal ────────────
        if pending_signal is not None and position is None and i < n:
            fill_price = float(opn[i])
            atr_val = pending_atr if pending_atr > 0 else float(atr_series[i])
            side = "long" if pending_signal == "buy" else "short"

            stop_dist = atr_val * exit_pack.stop_atr_mult
            if side == "long":
                stop_px = fill_price - stop_dist
                tp_px = fill_price + atr_val * exit_pack.take_profit_atr_mult if exit_pack.has_take_profit else 0.0
            else:
                stop_px = fill_price + stop_dist
                tp_px = fill_price - atr_val * exit_pack.take_profit_atr_mult if exit_pack.has_take_profit else 0.0

            trail_dist = atr_val * exit_pack.trailing_stop_atr_mult if exit_pack.has_trailing_stop else 0.0
            max_hold_bars = int(exit_pack.max_hold_hours)

            risk_usd = equity * risk_per_trade
            risk_per_unit = stop_dist / max(fill_price, 1e-9)
            size_usd = risk_usd / max(risk_per_unit, 1e-9)
            size_usd = min(size_usd, equity * 0.95)

            entry_cost = size_usd * (fee_rate + slippage_bps / 10_000)

            position = Trade(
                entry_bar=i,
                entry_price=fill_price,
                side=side,
                stop_price=stop_px,
                tp_price=tp_px,
                trail_distance=trail_dist,
                max_hold_bars=max_hold_bars,
                size_usd=size_usd,
                atr_at_entry=atr_val,
                cost=entry_cost,
            )
            pending_signal = None

        # ── Manage open position ────────────────────────────────────
        if position is not None:
            position.bars_held += 1
            bar_high = float(high[i])
            bar_low = float(low[i])
            bar_close = float(close[i])

            exited = False

            if position.side == "long":
                # Update peak for trailing
                if bar_high > position.entry_price + position.peak_favorable:
                    position.peak_favorable = bar_high - position.entry_price

                # Check stop hit
                if bar_low <= position.stop_price:
                    position.exit_price = min(float(opn[i]), position.stop_price)
                    position.exit_reason = "stop"
                    exited = True
                # Check TP hit (only if stop not hit; adverse-first rule)
                elif position.tp_price > 0 and bar_high >= position.tp_price:
                    position.exit_price = max(float(opn[i]), position.tp_price)
                    position.exit_reason = "take_profit"
                    exited = True

                # Trailing stop update
                if not exited and position.trail_distance > 0:
                    new_trail = bar_high - position.trail_distance
                    if new_trail > position.stop_price:
                        position.stop_price = new_trail

            else:  # short
                if bar_low < position.entry_price - position.peak_favorable:
                    position.peak_favorable = position.entry_price - bar_low

                if bar_high >= position.stop_price:
                    position.exit_price = max(float(opn[i]), position.stop_price)
                    position.exit_reason = "stop"
                    exited = True
                elif position.tp_price > 0 and bar_low <= position.tp_price:
                    position.exit_price = min(float(opn[i]), position.tp_price)
                    position.exit_reason = "take_profit"
                    exited = True

                if not exited and position.trail_distance > 0:
                    new_trail = bar_low + position.trail_distance
                    if new_trail < position.stop_price:
                        position.stop_price = new_trail

            # Max hold
            if not exited and position.bars_held >= position.max_hold_bars:
                position.exit_price = bar_close
                position.exit_reason = "max_hold"
                exited = True

            if exited:
                _finalize_trade(position, fee_rate, slippage_bps)
                equity += position.net_pnl
                trades.append(position)
                position = None

        # ── Generate signal at bar close (only if flat) ─────────────
        if position is None and pending_signal is None and i < n - 1:
            window = candles.iloc[max(0, i - 300):i + 1]
            sig = entry_family.generate_signal(window)

            if sig.action == "buy":
                pending_signal = "buy"
                pending_atr = float(atr_series[i]) if i < len(atr_series) else 0.0
            elif sig.action == "sell" and not long_only:
                pending_signal = "sell"
                pending_atr = float(atr_series[i]) if i < len(atr_series) else 0.0

    # Force close at end
    if position is not None:
        position.exit_price = float(close[-1])
        position.exit_reason = "forced_window_exit"
        _finalize_trade(position, fee_rate, slippage_bps)
        equity += position.net_pnl
        trades.append(position)

    return trades


def _finalize_trade(t: Trade, fee_rate: float, slippage_bps: float):
    if t.side == "long":
        ret = (t.exit_price - t.entry_price) / max(t.entry_price, 1e-9)
    else:
        ret = (t.entry_price - t.exit_price) / max(t.entry_price, 1e-9)
    t.gross_pnl = t.size_usd * ret
    exit_cost = t.size_usd * (fee_rate + slippage_bps / 10_000)
    t.cost += exit_cost
    t.net_pnl = t.gross_pnl - t.cost


# ── Metrics Computation ────────────────────────────────────────────────────

@dataclass
class CellMetrics:
    entry: str
    exit_pack: str
    asset: str
    window_idx: int

    n_trades: int = 0
    n_wins: int = 0
    win_rate: float = 0.0
    gross_return_pct: float = 0.0
    net_return_pct: float = 0.0
    gross_expectancy_r: float = 0.0
    net_expectancy_r: float = 0.0
    sharpe: float = 0.0
    calmar: float = 0.0
    max_drawdown_pct: float = 0.0
    profit_factor: float = 0.0
    avg_cost_per_trade: float = 0.0
    cost_drag_pct: float = 0.0
    turnover_per_month: float = 0.0
    time_in_market: float = 0.0
    total_bars: int = 0


def _compute_metrics(
    trades: list[Trade],
    starting_balance: float,
    total_bars: int,
    entry_name: str,
    exit_name: str,
    asset: str,
    window_idx: int,
) -> CellMetrics:
    m = CellMetrics(
        entry=entry_name, exit_pack=exit_name, asset=asset,
        window_idx=window_idx, total_bars=total_bars,
    )
    if not trades:
        return m

    m.n_trades = len(trades)
    m.n_wins = sum(1 for t in trades if t.net_pnl > 0)
    m.win_rate = m.n_wins / m.n_trades

    gross_pnls = [t.gross_pnl for t in trades]
    net_pnls = [t.net_pnl for t in trades]
    costs = [t.cost for t in trades]
    bars_held = [t.bars_held for t in trades]

    m.gross_return_pct = 100.0 * sum(gross_pnls) / starting_balance
    m.net_return_pct = 100.0 * sum(net_pnls) / starting_balance

    # Expectancy in R (risk units = 1% equity)
    risk_usd = starting_balance * 0.01
    m.gross_expectancy_r = (sum(gross_pnls) / m.n_trades) / risk_usd if risk_usd > 0 else 0.0
    m.net_expectancy_r = (sum(net_pnls) / m.n_trades) / risk_usd if risk_usd > 0 else 0.0

    # Sharpe (annualized from per-trade returns)
    per_trade_rets = [p / starting_balance for p in net_pnls]
    if len(per_trade_rets) > 1:
        mean_r = np.mean(per_trade_rets)
        std_r = np.std(per_trade_rets, ddof=1)
        trades_per_year = m.n_trades * (8760.0 / max(total_bars, 1))
        m.sharpe = (mean_r / max(std_r, 1e-9)) * math.sqrt(max(trades_per_year, 1))

    # Max drawdown
    equity_curve = [starting_balance]
    for pnl in net_pnls:
        equity_curve.append(equity_curve[-1] + pnl)
    peak = starting_balance
    max_dd = 0.0
    for eq in equity_curve:
        if eq > peak:
            peak = eq
        dd = (peak - eq) / max(peak, 1e-9)
        if dd > max_dd:
            max_dd = dd
    m.max_drawdown_pct = 100.0 * max_dd

    # Calmar
    ann_ret = (m.net_return_pct / 100.0) * (8760.0 / max(total_bars, 1))
    m.calmar = ann_ret / max(max_dd, 1e-9) if max_dd > 0 else 0.0

    # Profit factor
    gross_wins = sum(p for p in net_pnls if p > 0)
    gross_losses = abs(sum(p for p in net_pnls if p < 0))
    m.profit_factor = gross_wins / max(gross_losses, 1e-9)

    # Cost decomposition
    m.avg_cost_per_trade = sum(costs) / m.n_trades
    gross_edge = sum(gross_pnls) / m.n_trades
    m.cost_drag_pct = 100.0 * (sum(costs) / max(abs(sum(gross_pnls)), 1e-9))

    # Turnover (trades per 30-day month ≈ 720 1h bars)
    m.turnover_per_month = m.n_trades * (720.0 / max(total_bars, 1))

    # Time in market
    m.time_in_market = sum(bars_held) / max(total_bars, 1)

    return m


# ── Walk-Forward Splitter ──────────────────────────────────────────────────

def _split_windows(n_bars: int, n_splits: int = 5):
    """Return (train_end, test_start, test_end) tuples for non-overlapping OOS windows."""
    test_size = n_bars // (n_splits + 1)
    windows = []
    for w in range(n_splits):
        test_start = n_bars - (n_splits - w) * test_size
        test_end = test_start + test_size
        train_end = test_start
        windows.append((train_end, test_start, test_end))
    return windows


# ── Aggregation ────────────────────────────────────────────────────────────

@dataclass
class AggregatedCell:
    entry: str
    exit_pack: str
    asset: str
    mean_net_return_pct: float = 0.0
    median_net_return_pct: float = 0.0
    mean_sharpe: float = 0.0
    mean_calmar: float = 0.0
    max_drawdown_pct: float = 0.0
    mean_profit_factor: float = 0.0
    mean_net_expectancy_r: float = 0.0
    mean_gross_expectancy_r: float = 0.0
    mean_gross_return_pct: float = 0.0
    mean_win_rate: float = 0.0
    total_trades: int = 0
    positive_windows: int = 0
    n_windows: int = 0
    mean_cost_drag_pct: float = 0.0
    mean_turnover: float = 0.0
    mean_time_in_market: float = 0.0
    avg_cost_per_trade: float = 0.0
    worst_window_return_pct: float = 0.0
    window_metrics: list = field(default_factory=list)


def _aggregate_windows(window_metrics: list[CellMetrics]) -> AggregatedCell:
    if not window_metrics:
        return AggregatedCell(entry="", exit_pack="", asset="")
    m0 = window_metrics[0]
    agg = AggregatedCell(
        entry=m0.entry, exit_pack=m0.exit_pack, asset=m0.asset,
        n_windows=len(window_metrics),
    )
    rets = [m.net_return_pct for m in window_metrics]
    agg.mean_net_return_pct = float(np.mean(rets))
    agg.median_net_return_pct = float(np.median(rets))
    agg.mean_sharpe = float(np.mean([m.sharpe for m in window_metrics]))
    agg.mean_calmar = float(np.mean([m.calmar for m in window_metrics]))
    agg.max_drawdown_pct = float(max(m.max_drawdown_pct for m in window_metrics))
    agg.mean_profit_factor = float(np.mean([m.profit_factor for m in window_metrics]))
    agg.mean_net_expectancy_r = float(np.mean([m.net_expectancy_r for m in window_metrics]))
    agg.mean_gross_expectancy_r = float(np.mean([m.gross_expectancy_r for m in window_metrics]))
    agg.mean_gross_return_pct = float(np.mean([m.gross_return_pct for m in window_metrics]))
    agg.mean_win_rate = float(np.mean([m.win_rate for m in window_metrics]))
    agg.total_trades = sum(m.n_trades for m in window_metrics)
    agg.positive_windows = sum(1 for r in rets if r > 0)
    agg.mean_cost_drag_pct = float(np.mean([m.cost_drag_pct for m in window_metrics]))
    agg.mean_turnover = float(np.mean([m.turnover_per_month for m in window_metrics]))
    agg.mean_time_in_market = float(np.mean([m.time_in_market for m in window_metrics]))
    agg.avg_cost_per_trade = float(np.mean([m.avg_cost_per_trade for m in window_metrics]))
    agg.worst_window_return_pct = float(min(rets))
    agg.window_metrics = window_metrics
    return agg


# ── Gates ──────────────────────────────────────────────────────────────────

def passes_screen_gate(agg: AggregatedCell) -> bool:
    return (
        agg.mean_net_return_pct > 0
        and agg.mean_net_expectancy_r > 0
        and agg.positive_windows >= 3
        and agg.max_drawdown_pct < 15.0
        and agg.total_trades >= 30
        and agg.mean_profit_factor > 1.05
    )


def passes_promotion_gate(agg: AggregatedCell) -> bool:
    return (
        passes_screen_gate(agg)
        and agg.mean_profit_factor > 1.10
        and agg.positive_windows >= 4
        and agg.max_drawdown_pct < 10.0
        and agg.worst_window_return_pct > -5.0
        and agg.mean_net_expectancy_r > 0.05
    )


# ── Single Cell Runner ─────────────────────────────────────────────────────

def run_cell(
    candles: pd.DataFrame,
    entry_key: str,
    exit_pack: ExitPack,
    asset: str,
    n_splits: int = 5,
    starting_balance: float = 10_000.0,
    risk_per_trade: float = 0.01,
    zero_cost: bool = False,
    long_only: bool = False,
    custom_fee: float | None = None,
    custom_slip: float | None = None,
) -> AggregatedCell:
    fee, slip = _fee_and_slippage(asset, zero_cost=zero_cost,
                                   custom_fee=custom_fee, custom_slip=custom_slip)
    entry_family = get_entry_family(entry_key)
    windows = _split_windows(len(candles), n_splits)
    window_metrics: list[CellMetrics] = []

    for w_idx, (train_end, test_start, test_end) in enumerate(windows):
        test_candles = candles.iloc[max(0, test_start - 300):test_end].copy()
        actual_test_bars = test_end - test_start

        trades = _run_cell_backtest(
            test_candles,
            entry_family=entry_family,
            exit_pack=exit_pack,
            fee_rate=fee,
            slippage_bps=slip,
            starting_balance=starting_balance,
            risk_per_trade=risk_per_trade,
            long_only=long_only,
        )

        m = _compute_metrics(
            trades, starting_balance, actual_test_bars,
            entry_key, exit_pack.name, asset, w_idx,
        )
        window_metrics.append(m)

    return _aggregate_windows(window_metrics)


# ── Full Matrix Runner ─────────────────────────────────────────────────────

def run_matrix(
    db_path: str,
    assets: list[str],
    entry_keys: list[str] | None = None,
    exit_packs: list[ExitPack] | None = None,
    n_splits: int = 5,
    zero_cost: bool = False,
    long_only: bool = False,
    custom_fee: float | None = None,
    custom_slip: float | None = None,
    timeframe: str = "1h",
) -> list[AggregatedCell]:
    if entry_keys is None:
        entry_keys = list(ENTRY_FAMILIES.keys())
    if exit_packs is None:
        exit_packs = ALL_EXIT_PACKS

    conn = sqlite3.connect(db_path)
    results: list[AggregatedCell] = []
    total_cells = len(entry_keys) * len(exit_packs) * len(assets)
    cell_num = 0

    for asset in assets:
        candles = load_candles(conn, asset, timeframe)
        if candles is None or len(candles) < 500:
            print(f"  SKIP {asset}: insufficient data ({len(candles) if candles is not None else 0} bars)")
            continue
        print(f"\n  {asset}: {len(candles)} {timeframe} bars loaded")

        for entry_key in entry_keys:
            for ep in exit_packs:
                cell_num += 1
                label = f"  [{cell_num}/{total_cells}] {entry_key} x {ep.name} x {asset}"
                t0 = time.time()
                agg = run_cell(
                    candles, entry_key, ep, asset,
                    n_splits=n_splits,
                    zero_cost=zero_cost,
                    long_only=long_only,
                    custom_fee=custom_fee,
                    custom_slip=custom_slip,
                )
                elapsed = time.time() - t0
                flag = "PASS" if passes_screen_gate(agg) else "fail"
                print(f"{label}: {agg.total_trades} trades, "
                      f"net={agg.mean_net_return_pct:+.2f}%, "
                      f"calmar={agg.mean_calmar:+.2f}, "
                      f"PF={agg.mean_profit_factor:.2f}, "
                      f"DD={agg.max_drawdown_pct:.1f}%, "
                      f"{flag}  ({elapsed:.1f}s)")
                results.append(agg)

    conn.close()
    return results


# ── Output ─────────────────────────────────────────────────────────────────

def _save_leaderboard(results: list[AggregatedCell], out_dir: str, suffix: str = ""):
    btc_cells = [r for r in results if "BTC" in r.asset]
    btc_cells.sort(key=lambda r: r.mean_calmar, reverse=True)
    other = [r for r in results if "BTC" not in r.asset]
    ranked = btc_cells + other

    fname = f"leaderboard{suffix}.csv"
    path = os.path.join(out_dir, fname)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rank", "entry", "exit_pack", "asset",
            "mean_net_return_pct", "median_net_return_pct",
            "mean_calmar", "mean_sharpe", "max_drawdown_pct",
            "mean_profit_factor", "mean_net_expectancy_r",
            "mean_gross_expectancy_r", "mean_gross_return_pct",
            "total_trades", "mean_win_rate",
            "positive_windows", "n_windows",
            "avg_cost_per_trade", "mean_cost_drag_pct",
            "mean_turnover", "mean_time_in_market",
            "worst_window_return_pct",
            "screen_gate", "promotion_gate",
        ])
        for i, r in enumerate(ranked):
            w.writerow([
                i + 1, r.entry, r.exit_pack, r.asset,
                f"{r.mean_net_return_pct:.4f}", f"{r.median_net_return_pct:.4f}",
                f"{r.mean_calmar:.4f}", f"{r.mean_sharpe:.4f}",
                f"{r.max_drawdown_pct:.4f}", f"{r.mean_profit_factor:.4f}",
                f"{r.mean_net_expectancy_r:.4f}", f"{r.mean_gross_expectancy_r:.4f}",
                f"{r.mean_gross_return_pct:.4f}",
                r.total_trades, f"{r.mean_win_rate:.4f}",
                r.positive_windows, r.n_windows,
                f"{r.avg_cost_per_trade:.4f}", f"{r.mean_cost_drag_pct:.2f}",
                f"{r.mean_turnover:.2f}", f"{r.mean_time_in_market:.4f}",
                f"{r.worst_window_return_pct:.4f}",
                passes_screen_gate(r), passes_promotion_gate(r),
            ])
    print(f"\n  Leaderboard saved: {path}")
    return ranked


def _save_json_results(results: list[AggregatedCell], out_dir: str, suffix: str = ""):
    for r in results:
        fname = f"results_{r.entry}_{r.exit_pack}_{r.asset.replace('/', '_')}{suffix}.json"
        path = os.path.join(out_dir, fname)
        data = {
            "entry": r.entry,
            "exit_pack": r.exit_pack,
            "asset": r.asset,
            "mean_net_return_pct": r.mean_net_return_pct,
            "median_net_return_pct": r.median_net_return_pct,
            "mean_calmar": r.mean_calmar,
            "mean_sharpe": r.mean_sharpe,
            "max_drawdown_pct": r.max_drawdown_pct,
            "mean_profit_factor": r.mean_profit_factor,
            "mean_net_expectancy_r": r.mean_net_expectancy_r,
            "mean_gross_expectancy_r": r.mean_gross_expectancy_r,
            "mean_gross_return_pct": r.mean_gross_return_pct,
            "total_trades": r.total_trades,
            "mean_win_rate": r.mean_win_rate,
            "positive_windows": r.positive_windows,
            "n_windows": r.n_windows,
            "avg_cost_per_trade": r.avg_cost_per_trade,
            "mean_cost_drag_pct": r.mean_cost_drag_pct,
            "mean_turnover": r.mean_turnover,
            "mean_time_in_market": r.mean_time_in_market,
            "worst_window_return_pct": r.worst_window_return_pct,
            "screen_gate": passes_screen_gate(r),
            "promotion_gate": passes_promotion_gate(r),
            "windows": [
                {
                    "window_idx": wm.window_idx,
                    "n_trades": wm.n_trades,
                    "net_return_pct": wm.net_return_pct,
                    "gross_return_pct": wm.gross_return_pct,
                    "sharpe": wm.sharpe,
                    "calmar": wm.calmar,
                    "max_drawdown_pct": wm.max_drawdown_pct,
                    "profit_factor": wm.profit_factor,
                    "net_expectancy_r": wm.net_expectancy_r,
                    "gross_expectancy_r": wm.gross_expectancy_r,
                    "win_rate": wm.win_rate,
                    "cost_drag_pct": wm.cost_drag_pct,
                    "turnover_per_month": wm.turnover_per_month,
                    "time_in_market": wm.time_in_market,
                }
                for wm in r.window_metrics
            ],
        }
        with open(path, "w") as f:
            json.dump(data, f, indent=2)


def _save_summary_md(results: list[AggregatedCell], out_dir: str, suffix: str = ""):
    btc_cells = sorted(
        [r for r in results if "BTC" in r.asset],
        key=lambda r: r.mean_calmar, reverse=True,
    )
    survivors = [r for r in btc_cells if passes_screen_gate(r)]
    promoted = [r for r in btc_cells if passes_promotion_gate(r)]

    lines = [
        "# Strategy Matrix Tournament Results\n",
        f"Total cells evaluated: {len(results)}",
        f"BTC cells: {len(btc_cells)}",
        f"BTC screen gate survivors: {len(survivors)}",
        f"BTC promotion gate candidates: {len(promoted)}\n",
        "## Top 5 BTC Cells (by Calmar)\n",
        "| Rank | Entry | Exit | Calmar | Net Ret% | PF | DD% | Trades | Win% | Screen | Promo |",
        "|------|-------|------|--------|----------|----|-----|--------|------|--------|-------|",
    ]
    for i, r in enumerate(btc_cells[:5]):
        lines.append(
            f"| {i+1} | {r.entry} | {r.exit_pack} | {r.mean_calmar:+.2f} | "
            f"{r.mean_net_return_pct:+.2f} | {r.mean_profit_factor:.2f} | "
            f"{r.max_drawdown_pct:.1f} | {r.total_trades} | "
            f"{r.mean_win_rate:.0%} | {passes_screen_gate(r)} | {passes_promotion_gate(r)} |"
        )

    lines.append("\n## Bottom 5 BTC Cells\n")
    lines.append("| Rank | Entry | Exit | Calmar | Net Ret% | PF | DD% | Trades |")
    lines.append("|------|-------|------|--------|----------|----|-----|--------|")
    for r in btc_cells[-5:]:
        lines.append(
            f"| - | {r.entry} | {r.exit_pack} | {r.mean_calmar:+.2f} | "
            f"{r.mean_net_return_pct:+.2f} | {r.mean_profit_factor:.2f} | "
            f"{r.max_drawdown_pct:.1f} | {r.total_trades} |"
        )

    lines.append("\n## Cost Decomposition (Top 5 BTC)\n")
    lines.append("| Entry | Exit | Gross Ret% | Net Ret% | Avg Cost | Cost Drag% | Turnover/mo |")
    lines.append("|-------|------|-----------|---------|----------|-----------|-------------|")
    for r in btc_cells[:5]:
        lines.append(
            f"| {r.entry} | {r.exit_pack} | {r.mean_gross_return_pct:+.2f} | "
            f"{r.mean_net_return_pct:+.2f} | ${r.avg_cost_per_trade:.2f} | "
            f"{r.mean_cost_drag_pct:.1f}% | {r.mean_turnover:.1f} |"
        )

    # ETH confirmation for BTC survivors
    eth_cells = {(r.entry, r.exit_pack): r for r in results if "ETH" in r.asset}
    if survivors and eth_cells:
        lines.append("\n## ETH Confirmation for BTC Survivors\n")
        lines.append("| Entry | Exit | BTC Calmar | ETH Calmar | ETH Net% | Portable? |")
        lines.append("|-------|------|-----------|-----------|---------|-----------|")
        for r in survivors[:5]:
            eth = eth_cells.get((r.entry, r.exit_pack))
            if eth:
                portable = "YES" if eth.mean_net_return_pct > 0 else "no"
                lines.append(
                    f"| {r.entry} | {r.exit_pack} | {r.mean_calmar:+.2f} | "
                    f"{eth.mean_calmar:+.2f} | {eth.mean_net_return_pct:+.2f} | {portable} |"
                )

    # Four-quadrant reading
    lines.append("\n## Four-Quadrant Reading\n")
    btc_any = any(passes_screen_gate(r) for r in btc_cells)
    eth_any = any(passes_screen_gate(r) for r in results if "ETH" in r.asset)
    gbp_cells = [r for r in results if "GBP" in r.asset]
    gbp_any = any(passes_screen_gate(r) for r in gbp_cells) if gbp_cells else None

    if btc_any and eth_any and gbp_any:
        lines.append("**BTC + ETH + GBP all work**: True price/volatility edge with broader validity.")
    elif btc_any and eth_any:
        lines.append("**BTC + ETH work" + (", GBP weak" if gbp_any is False else "") +
                      "**: Crypto-native edge. Good enough to build Hogan around.")
    elif btc_any:
        lines.append("**Only BTC works**: High risk of overfit. Do not trust it yet.")
    elif eth_any:
        lines.append("**ETH works but BTC does not**: Optimizing for crypto beta noise, not the target market.")
    else:
        lines.append("**Nothing passes BTC**: Stop tuning overlays. The base idea needs replacement.")

    path = os.path.join(out_dir, f"summary{suffix}.md")
    with open(path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Summary saved: {path}")


# ── CLI ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Strategy Matrix Tournament")
    parser.add_argument("--db", default="data/hogan.db", help="SQLite DB path")
    parser.add_argument("--assets", nargs="+", default=["BTC/USD", "ETH/USD"])
    parser.add_argument("--entries", nargs="+", default=None,
                        help="Entry keys to test (default: all)")
    parser.add_argument("--exits", nargs="+", default=None,
                        help="Exit pack names to test (default: all)")
    parser.add_argument("--n-splits", type=int, default=5)
    parser.add_argument("--zero-cost", action="store_true",
                        help="Run in zero-cost mode (fee=0, slippage=0)")
    parser.add_argument("--long-only", action="store_true",
                        help="Only take long entries")
    parser.add_argument("--output-dir", default="reports/tournament")
    parser.add_argument("--suffix", default="",
                        help="Suffix for output filenames")
    parser.add_argument("--fee-rate", type=float, default=None,
                        help="Custom fee rate per side (e.g. 0.001 for 0.10%%)")
    parser.add_argument("--slippage-bps", type=float, default=None,
                        help="Custom slippage in basis points")
    parser.add_argument("--timeframe", default="1h",
                        help="Candle timeframe (default: 1h)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, "window_details"), exist_ok=True)

    exit_packs = ALL_EXIT_PACKS
    if args.exits:
        from hogan_bot.exit_packs import EXIT_PACKS
        exit_packs = [EXIT_PACKS[k] for k in args.exits]

    suffix = args.suffix
    if args.zero_cost:
        suffix = suffix or "_zero_cost"
    if args.long_only:
        suffix = suffix or "_long_only"

    print("=" * 80)
    print("STRATEGY MATRIX TOURNAMENT")
    print("=" * 80)
    print(f"  Assets:     {args.assets}")
    print(f"  Entries:    {args.entries or list(ENTRY_FAMILIES.keys())}")
    print(f"  Exits:      {[ep.name for ep in exit_packs]}")
    print(f"  Timeframe:  {args.timeframe}")
    print(f"  Splits:     {args.n_splits}")
    print(f"  Zero-cost:  {args.zero_cost}")
    print(f"  Long-only:  {args.long_only}")
    print(f"  Fee rate:   {args.fee_rate or 'default'}")
    print(f"  Slip bps:   {args.slippage_bps or 'default'}")
    print(f"  Output:     {args.output_dir}")
    print("=" * 80)

    t0 = time.time()
    results = run_matrix(
        db_path=args.db,
        assets=args.assets,
        entry_keys=args.entries,
        exit_packs=exit_packs,
        n_splits=args.n_splits,
        zero_cost=args.zero_cost,
        long_only=args.long_only,
        custom_fee=args.fee_rate,
        custom_slip=args.slippage_bps,
        timeframe=args.timeframe,
    )
    elapsed = time.time() - t0

    print(f"\n{'=' * 80}")
    print(f"MATRIX COMPLETE — {len(results)} cells in {elapsed:.1f}s")
    print(f"{'=' * 80}")

    ranked = _save_leaderboard(results, args.output_dir, suffix)
    _save_json_results(results, args.output_dir, suffix)
    _save_summary_md(results, args.output_dir, suffix)

    # Print BTC summary
    btc_cells = [r for r in results if "BTC" in r.asset]
    survivors = [r for r in btc_cells if passes_screen_gate(r)]
    promoted = [r for r in btc_cells if passes_promotion_gate(r)]

    print(f"\n  BTC screen gate survivors: {len(survivors)}/{len(btc_cells)}")
    print(f"  BTC promotion candidates:  {len(promoted)}/{len(btc_cells)}")

    if survivors:
        print("\n  TOP BTC SURVIVORS:")
        for r in sorted(survivors, key=lambda x: x.mean_calmar, reverse=True)[:5]:
            promo = " [PROMOTION]" if passes_promotion_gate(r) else ""
            print(f"    {r.entry} x {r.exit_pack}: "
                  f"calmar={r.mean_calmar:+.2f}, net={r.mean_net_return_pct:+.2f}%, "
                  f"PF={r.mean_profit_factor:.2f}, DD={r.max_drawdown_pct:.1f}%{promo}")
    else:
        print("\n  NO BTC SURVIVORS. Base idea needs replacement.")

    print()


if __name__ == "__main__":
    main()
