"""Trade Outcome Labeler — Phase 4a.

After a trade closes, this module:
1. Looks up the closing fill in the ``fills`` table.
2. Computes the realized P&L % relative to entry price.
3. Labels the feature row at entry time as 1 (profitable) or 0 (not).
4. Appends the labeled row to ``online_training_buffer`` for the online
   learner to consume.

Designed to be called from the event loop or a background thread after
each position close:

    from hogan_bot.labeler import label_closed_trade
    label_closed_trade(conn, fill_id="abc123", entry_ts_ms=...,
                       entry_features=[...], horizon_bars=3)

Or run as a batch job to back-fill labels for completed trades::

    python -m hogan_bot.labeler --db data/hogan.db --backfill
"""
from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import time

logger = logging.getLogger(__name__)

_DEFAULT_PROFIT_THRESHOLD = 0.0   # any positive P&L = label 1


def label_closed_trade(
    conn: sqlite3.Connection,
    fill_id: str,
    entry_ts_ms: int,
    entry_features: list[float],
    symbol: str,
    horizon_bars: int = 3,
    profit_threshold: float = _DEFAULT_PROFIT_THRESHOLD,
    position_side: str | None = None,
) -> dict | None:
    """Score a completed trade and write a labeled row.

    Parameters
    ----------
    conn:
        Open SQLite connection (WAL-mode recommended).
    fill_id:
        The closing fill ID to look up P&L from.
    entry_ts_ms:
        Millisecond timestamp of the entry candle.
    entry_features:
        Feature vector at the time of entry (list[float], 36 or 70 dims).
    symbol:
        Trading pair e.g. ``"BTC/USD"``.
    horizon_bars:
        How many bars the trade was held (informational).
    profit_threshold:
        Minimum P&L % to be labelled as 1 (profitable).  Defaults to 0.
    position_side:
        ``"long"`` or ``"short"`` — the side that opened the position. When
        omitted it is inferred from the closing fill: ``sell`` close closes a
        long, ``buy`` close closes a short.

    Returns
    -------
    dict with label info, or None if the fill was not found.
    """
    cur = conn.execute(
        "SELECT side, amount, price, fee FROM fills WHERE fill_id = ?",
        (fill_id,),
    )
    row = cur.fetchone()
    if row is None:
        logger.warning("labeler: fill_id=%s not found in fills table.", fill_id)
        return None

    side, amount, close_price, fee = row

    # Infer the direction of the position that this fill closed when the
    # caller did not tell us explicitly. A *buy* fill can only close a short
    # (you buy to cover); a *sell* fill can only close a long.
    if position_side is None:
        position_side = "short" if side == "buy" else "long"
    position_side = position_side.lower()
    if position_side not in ("long", "short"):
        logger.warning(
            "labeler: invalid position_side=%r for fill_id=%s; defaulting to long",
            position_side,
            fill_id,
        )
        position_side = "long"

    # The opening fill is on the *opposite* side of the closing fill.
    entry_side = "buy" if position_side == "long" else "sell"
    entry_cur = conn.execute(
        """
        SELECT price FROM fills
        WHERE symbol=? AND side=? AND ts_ms <= ?
        ORDER BY ts_ms DESC LIMIT 1
        """,
        (symbol, entry_side, entry_ts_ms + 60_000),
    )
    entry_row = entry_cur.fetchone()
    entry_price = entry_row[0] if entry_row else close_price

    if entry_price <= 0:
        logger.warning("labeler: entry_price=0 for fill_id=%s, skipping.", fill_id)
        return None

    # Raw long-side return; flip for shorts so that "profit ⇒ positive pnl_pct"
    # holds in both directions.
    pnl_pct = (close_price - entry_price) / entry_price * 100.0
    if position_side == "short":
        pnl_pct = -pnl_pct

    label = 1 if pnl_pct > profit_threshold else 0
    features_json = json.dumps([round(float(f), 8) for f in entry_features])

    conn.execute(
        """
        INSERT OR IGNORE INTO online_training_buffer
            (symbol, ts_ms, features_json, label, fill_ts_ms, pnl_pct, horizon_bars)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (symbol, entry_ts_ms, features_json, label,
         int(time.time() * 1000), round(pnl_pct, 4), horizon_bars),
    )
    conn.commit()

    result = {
        "symbol": symbol,
        "entry_ts_ms": entry_ts_ms,
        "fill_id": fill_id,
        "entry_price": entry_price,
        "close_price": close_price,
        "pnl_pct": round(pnl_pct, 4),
        "label": label,
        "horizon_bars": horizon_bars,
    }
    logger.info(
        "Labeled trade: symbol=%s pnl=%.2f%% label=%d fill_id=%s",
        symbol, pnl_pct, label, fill_id,
    )
    return result


def label_pending_trades(
    conn: sqlite3.Connection,
    min_pnl_threshold: float = _DEFAULT_PROFIT_THRESHOLD,
) -> int:
    """Back-fill labels for rows in ``online_training_buffer`` where label IS NULL.

    Looks up matching sell fills after each NULL-label row's timestamp,
    computes P&L, and writes the label.

    Returns the number of rows labeled.
    """
    cur = conn.execute(
        """
        SELECT row_id, symbol, ts_ms, features_json, horizon_bars
        FROM online_training_buffer
        WHERE label IS NULL
        ORDER BY ts_ms
        LIMIT 500
        """
    )
    pending = cur.fetchall()
    if not pending:
        return 0

    labeled = 0
    for row_id, symbol, ts_ms, _features_json, _horizon_bars in pending:
        # Find the *earliest* close fill after this entry — either side. The
        # closing side tells us whether the position was long (closed by sell)
        # or short (closed by buy). Previously this only looked for sell fills,
        # which silently discarded every short trade.
        close_cur = conn.execute(
            """
            SELECT fill_id, price, ts_ms, side FROM fills
            WHERE symbol=? AND ts_ms > ?
            ORDER BY ts_ms ASC LIMIT 1
            """,
            (symbol, ts_ms),
        )
        close_row = close_cur.fetchone()
        if close_row is None:
            continue

        fill_id, close_price, fill_ts_ms, close_side = close_row
        position_side = "short" if close_side == "buy" else "long"
        entry_side = "buy" if position_side == "long" else "sell"

        entry_cur = conn.execute(
            """
            SELECT price FROM fills
            WHERE symbol=? AND side=? AND ts_ms <= ?
            ORDER BY ts_ms DESC LIMIT 1
            """,
            (symbol, entry_side, ts_ms + 60_000),
        )
        entry_row = entry_cur.fetchone()
        if not entry_row:
            continue
        entry_price = entry_row[0]

        if entry_price <= 0:
            continue

        pnl_pct = (close_price - entry_price) / entry_price * 100.0
        if position_side == "short":
            pnl_pct = -pnl_pct
        label = 1 if pnl_pct > min_pnl_threshold else 0

        conn.execute(
            """
            UPDATE online_training_buffer
            SET label=?, fill_ts_ms=?, pnl_pct=?
            WHERE row_id=?
            """,
            (label, fill_ts_ms, round(pnl_pct, 4), row_id),
        )
        labeled += 1

    conn.commit()
    logger.info("label_pending_trades: labeled %d rows.", labeled)
    return labeled


def get_labeled_dataset(
    conn: sqlite3.Connection,
    symbol: str | None = None,
    min_rows: int = 10,
) -> tuple[list[list[float]], list[int]] | None:
    """Return (X, y) from the labeled online_training_buffer.

    Returns None if fewer than *min_rows* labeled rows exist.
    """
    query = """
        SELECT features_json, label
        FROM online_training_buffer
        WHERE label IS NOT NULL
    """
    params: list = []
    if symbol:
        query += " AND symbol=?"
        params.append(symbol)
    query += " ORDER BY ts_ms"

    cur = conn.execute(query, params)
    rows = cur.fetchall()

    if len(rows) < min_rows:
        return None

    X: list[list[float]] = []
    y: list[int] = []
    for features_json, label in rows:
        try:
            feat = json.loads(features_json)
            X.append([float(f) for f in feat])
            y.append(int(label))
        except (json.JSONDecodeError, ValueError, TypeError):
            continue

    if len(X) < min_rows:
        return None
    return X, y


# ---------------------------------------------------------------------------
# CLI — backfill labels for existing fills in DB
# ---------------------------------------------------------------------------
def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Back-fill trade outcome labels")
    p.add_argument("--db", default="data/hogan.db")
    p.add_argument("--backfill", action="store_true", help="Label all pending rows")
    p.add_argument("--threshold", type=float, default=0.0,
                   help="Minimum P&L%% for label=1")
    args = p.parse_args()

    from hogan_bot.storage import get_connection
    conn = get_connection(args.db)

    if args.backfill:
        n = label_pending_trades(conn, min_pnl_threshold=args.threshold)
        print(f"Labeled {n} pending trades.")
    else:
        print("Use --backfill to label pending trades.")

    conn.close()


if __name__ == "__main__":
    _main()
