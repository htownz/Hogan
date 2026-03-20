"""Compute and persist forward outcomes for swarm decisions.

Reads candles from the DB, computes forward markouts (5m/15m/30m/60m),
MAE, MFE, and veto correctness, then writes to ``swarm_outcomes``.

Can be called:
  - As a periodic job after each candle cycle
  - As a standalone backfill script
"""
from __future__ import annotations

import logging
import sqlite3
import time

import pandas as pd

logger = logging.getLogger(__name__)


def backfill_outcomes(
    conn: sqlite3.Connection,
    symbol: str | None = None,
    lookback_hours: int = 72,
    forward_bars: int = 60,
) -> int:
    """Compute forward outcomes for decisions that lack them.

    Returns the number of outcomes written.
    """
    cutoff_ms = int((time.time() - lookback_hours * 3600) * 1000) if lookback_hours else 0

    sym_filter = "AND sd.symbol = ?" if symbol else ""
    params: list = [cutoff_ms]
    if symbol:
        params.append(symbol)

    pending = pd.read_sql_query(
        f"""SELECT sd.id, sd.ts_ms, sd.symbol, sd.timeframe,
                   sd.final_action, sd.vetoed, sd.mode
            FROM swarm_decisions sd
            LEFT JOIN swarm_outcomes so ON sd.id = so.decision_id
            WHERE so.decision_id IS NULL
              AND sd.ts_ms >= ?
              {sym_filter}
            ORDER BY sd.ts_ms""",
        conn, params=params,
    )

    if pending.empty:
        return 0

    written = 0
    for _, row in pending.iterrows():
        try:
            n = _write_one_outcome(conn, row)
            written += n
        except Exception as exc:
            logger.debug("Outcome write failed for decision %s: %s", row["id"], exc)

    return written


def _write_one_outcome(
    conn: sqlite3.Connection,
    decision_row: pd.Series,
) -> int:
    """Compute and insert one outcome row.  Returns 1 on success, 0 on skip."""
    dec_id = int(decision_row["id"])
    ts_ms = int(decision_row["ts_ms"])
    symbol = decision_row["symbol"]
    tf = decision_row["timeframe"]
    action = decision_row["final_action"]
    vetoed = bool(decision_row["vetoed"])

    candles = pd.read_sql_query(
        """SELECT ts_ms, open, high, low, close
           FROM candles
           WHERE symbol = ? AND timeframe = ?
             AND ts_ms >= ?
           ORDER BY ts_ms
           LIMIT 120""",
        conn, params=(symbol, tf, ts_ms),
    )

    if candles.empty:
        return 0

    entry_row = candles[candles["ts_ms"] == ts_ms]
    if entry_row.empty:
        entry_row = candles.iloc[:1]
    entry_price = float(entry_row.iloc[0]["close"])

    if entry_price <= 0:
        return 0

    future = candles[candles["ts_ms"] > ts_ms]
    if future.empty:
        return 0

    # Forward markouts in basis points (relative to entry close)
    def _bps(price: float) -> float:
        return (price / entry_price - 1) * 10_000

    tf_minutes = _tf_to_minutes(tf)
    fwd_5m = _forward_bps(future, tf_minutes, 5, entry_price)
    fwd_15m = _forward_bps(future, tf_minutes, 15, entry_price)
    fwd_30m = _forward_bps(future, tf_minutes, 30, entry_price)
    fwd_60m = _forward_bps(future, tf_minutes, 60, entry_price)

    # MAE / MFE over available future bars (up to 60 bars)
    window = future.head(60)
    direction = 1.0 if action == "buy" else (-1.0 if action == "sell" else 0.0)
    if direction != 0 and not window.empty:
        returns_bps = (window["close"] / entry_price - 1) * 10_000 * direction
        mae_bps = float(-returns_bps.min()) if returns_bps.min() < 0 else 0.0
        mfe_bps = float(returns_bps.max()) if returns_bps.max() > 0 else 0.0
    else:
        mae_bps = None
        mfe_bps = None

    # Was the trade taken? (non-hold, non-vetoed)
    was_trade_taken = 1 if action in ("buy", "sell") and not vetoed else 0

    # Baseline comparison — use pipeline_action (pre-swarm baseline) linked
    # via swarm_decision_id.  Falls back to approximate ts_ms match if the
    # linkage column is unavailable.
    bl_action: str | None = None
    baseline_would_trade = None
    try:
        baseline_row = pd.read_sql_query(
            "SELECT pipeline_action FROM decision_log "
            "WHERE swarm_decision_id = ? AND symbol = ? LIMIT 1",
            conn, params=(dec_id, symbol),
        )
        if not baseline_row.empty:
            bl_action = baseline_row.iloc[0]["pipeline_action"]
    except Exception as exc:
        logger.debug("Baseline lookup by id failed for %s: %s", dec_id, exc)
    if bl_action is None:
        try:
            baseline_row = pd.read_sql_query(
                "SELECT pipeline_action, final_action FROM decision_log "
                "WHERE symbol = ? AND ts_ms BETWEEN ? AND ? LIMIT 1",
                conn, params=(symbol, ts_ms - 5000, ts_ms + 5000),
            )
            if not baseline_row.empty:
                bl_action = (
                    baseline_row.iloc[0]["pipeline_action"]
                    if pd.notna(baseline_row.iloc[0]["pipeline_action"])
                    else baseline_row.iloc[0]["final_action"]
                )
        except Exception as exc:
            logger.debug("Baseline lookup by ts_ms failed for %s/%s: %s", symbol, ts_ms, exc)
    if bl_action is not None:
        baseline_would_trade = 1 if bl_action in ("buy", "sell") else 0

    swarm_would_trade = 1 if action in ("buy", "sell") else 0

    # Veto correctness: veto was correct if forward 60m return was negative
    # (for the direction the baseline wanted to trade)
    was_veto_correct = None
    if vetoed and fwd_60m is not None and baseline_would_trade:
        if bl_action == "buy":
            was_veto_correct = 1 if fwd_60m < 0 else 0
        elif bl_action == "sell":
            was_veto_correct = 1 if fwd_60m > 0 else 0

    # Skip correctness (swarm said hold when baseline said trade)
    was_skip_correct = None
    if not vetoed and action == "hold" and baseline_would_trade and fwd_60m is not None:
        if bl_action == "buy":
            was_skip_correct = 1 if fwd_60m < 0 else 0
        elif bl_action == "sell":
            was_skip_correct = 1 if fwd_60m > 0 else 0

    # Outcome label — for hold (direction=0), label as "no_trade" rather
    # than incorrectly mapping market direction to win/loss.
    if fwd_60m is not None:
        if direction == 0:
            outcome_label = "no_trade"
        elif abs(fwd_60m) < 5:
            outcome_label = "scratch"
        elif fwd_60m > 0:
            outcome_label = "win" if direction > 0 else "loss"
        else:
            outcome_label = "loss" if direction > 0 else "win"
    else:
        outcome_label = "pending"

    now_ms = int(time.time() * 1000)

    conn.execute(
        """INSERT OR REPLACE INTO swarm_outcomes
           (decision_id, forward_5m_bps, forward_15m_bps, forward_30m_bps,
            forward_60m_bps, mae_bps, mfe_bps, realized_slippage_bps,
            was_trade_taken, baseline_would_trade, swarm_would_trade,
            was_veto_correct, was_skip_correct, outcome_label, updated_ms)
           VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (dec_id, fwd_5m, fwd_15m, fwd_30m, fwd_60m,
         mae_bps, mfe_bps, None,
         was_trade_taken, baseline_would_trade, swarm_would_trade,
         was_veto_correct, was_skip_correct, outcome_label, now_ms),
    )
    conn.commit()
    return 1


def _tf_to_minutes(tf: str) -> int:
    """Convert timeframe string to minutes (minimum 1)."""
    tf = tf.lower().strip()
    if tf.endswith("m"):
        val = int(tf[:-1])
    elif tf.endswith("h"):
        val = int(tf[:-1]) * 60
    elif tf.endswith("d"):
        val = int(tf[:-1]) * 1440
    else:
        val = 60
    return max(1, val)


def _forward_bps(
    future: pd.DataFrame,
    tf_minutes: int,
    target_minutes: int,
    entry_price: float,
) -> float | None:
    """Get forward return in bps at approximately target_minutes."""
    bars_needed = max(1, target_minutes // tf_minutes)
    if len(future) < bars_needed:
        return None
    target_close = float(future.iloc[bars_needed - 1]["close"])
    return round((target_close / entry_price - 1) * 10_000, 2)
