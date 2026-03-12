"""SQLite-backed candle store for Hogan.

Provides a simple way to persist OHLCV data locally so that training,
backtesting, and the dashboard can work without a live exchange connection.

Typical workflow::

    from hogan_bot.storage import get_connection, upsert_candles, load_candles

    conn = get_connection("data/hogan.db")

    # Populate from exchange
    from hogan_bot.exchange import KrakenClient
    client = KrakenClient(None, None)
    df = client.fetch_ohlcv_df("BTC/USD", timeframe="1h", limit=5000)
    upsert_candles(conn, "BTC/USD", "1h", df)

    # Load back for training / backtesting
    candles = load_candles(conn, "BTC/USD", "1h")
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pandas as pd


def get_connection(db_path: str = "data/hogan.db") -> sqlite3.Connection:
    """Open (or create) the SQLite database and ensure the schema exists.

    WAL mode is enabled so the dashboard, MCP server, and online learner
    can read concurrently while the bot loop writes candles/fills.
    """
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA synchronous=NORMAL")
    conn.execute("PRAGMA cache_size=-32000")  # 32 MB page cache
    _create_schema(conn)
    return conn


def _create_schema(conn: sqlite3.Connection) -> None:
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS candles (
            symbol    TEXT    NOT NULL,
            timeframe TEXT    NOT NULL,
            ts_ms     INTEGER NOT NULL,
            open      REAL,
            high      REAL,
            low       REAL,
            close     REAL,
            volume    REAL,
            PRIMARY KEY (symbol, timeframe, ts_ms)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_candles_symbol_tf ON candles (symbol, timeframe)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS derivatives_metrics (
            symbol TEXT    NOT NULL,
            ts_ms  INTEGER NOT NULL,
            metric TEXT    NOT NULL,
            value  REAL    NOT NULL,
            PRIMARY KEY (symbol, ts_ms, metric)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_deriv_symbol ON derivatives_metrics (symbol, metric)"
    )
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS onchain_metrics (
            symbol TEXT NOT NULL,
            date   TEXT NOT NULL,
            metric TEXT NOT NULL,
            value  REAL NOT NULL,
            PRIMARY KEY (symbol, date, metric)
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_onchain_symbol ON onchain_metrics (symbol, metric)"
    )
    
    # -------------------------------------------------------------------
    # Trading / execution journaling (paper + live)
    # -------------------------------------------------------------------
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS orders (
            order_id   TEXT PRIMARY KEY,
            exchange   TEXT NOT NULL,
            symbol     TEXT NOT NULL,
            side       TEXT NOT NULL,     -- buy/sell
            type       TEXT NOT NULL,     -- market/limit
            status     TEXT NOT NULL,     -- open/closed/canceled/rejected
            amount     REAL NOT NULL,
            price      REAL,             -- limit price (nullable)
            filled     REAL NOT NULL DEFAULT 0,
            avg_price  REAL,
            fee        REAL,
            fee_ccy    TEXT,
            ts_ms      INTEGER NOT NULL,  -- created timestamp (ms)
            raw_json   TEXT              -- exchange payload
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_orders_symbol ON orders (symbol, ts_ms)")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fills (
            fill_id    TEXT PRIMARY KEY,
            order_id   TEXT NOT NULL,
            exchange   TEXT NOT NULL,
            symbol     TEXT NOT NULL,
            side       TEXT NOT NULL,
            amount     REAL NOT NULL,
            price      REAL NOT NULL,
            fee        REAL,
            fee_ccy    TEXT,
            ts_ms      INTEGER NOT NULL,
            raw_json   TEXT,
            FOREIGN KEY(order_id) REFERENCES orders(order_id)
        )
        """
    )
    conn.execute("CREATE INDEX IF NOT EXISTS idx_fills_symbol ON fills (symbol, ts_ms)")

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS equity_snapshots (
            ts_ms      INTEGER PRIMARY KEY,
            cash_usd   REAL NOT NULL,
            equity_usd REAL NOT NULL,
            drawdown   REAL NOT NULL
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS position_state (
            symbol      TEXT PRIMARY KEY,
            entry_price REAL NOT NULL DEFAULT 0,
            peak_price  REAL NOT NULL DEFAULT 0,
            updated_ms  INTEGER NOT NULL
        )
        """
    )

    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS positions (
            symbol     TEXT PRIMARY KEY,
            qty        REAL NOT NULL,
            avg_entry  REAL NOT NULL,
            updated_ms INTEGER NOT NULL
        )
        """
    )

    # -------------------------------------------------------------------
    # Paper trade journal — persists paper buys/sells/shorts/covers
    # Survives restarts; feeds ML retraining as additional labeled data
    # -------------------------------------------------------------------
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS paper_trades (
            trade_id       TEXT PRIMARY KEY,    -- uuid
            symbol         TEXT NOT NULL,
            side           TEXT NOT NULL,       -- long / short
            entry_price    REAL NOT NULL,
            exit_price     REAL,               -- NULL while open
            qty            REAL NOT NULL,
            entry_fee      REAL NOT NULL DEFAULT 0,
            exit_fee       REAL NOT NULL DEFAULT 0,
            realized_pnl   REAL,               -- NULL while open
            pnl_pct        REAL,               -- NULL while open
            open_ts_ms     INTEGER NOT NULL,
            close_ts_ms    INTEGER,            -- NULL while open
            close_reason   TEXT,               -- signal / trailing_stop / take_profit / auto_exit
            ml_up_prob     REAL,               -- ML up-probability at entry
            strategy_conf  REAL,               -- strategy confidence at entry
            vol_ratio      REAL                -- volume ratio at entry
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_pt_symbol ON paper_trades (symbol, open_ts_ms)"
    )

    # -------------------------------------------------------------------
    # Online / continuous learning buffer
    # -------------------------------------------------------------------
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS online_training_buffer (
            row_id        INTEGER PRIMARY KEY AUTOINCREMENT,
            symbol        TEXT    NOT NULL,
            ts_ms         INTEGER NOT NULL,
            features_json TEXT    NOT NULL,  -- JSON array of 36/70-dim feature vector
            label         INTEGER,           -- 1=profitable, 0=not, NULL=pending
            fill_ts_ms    INTEGER,           -- timestamp of the closing fill
            pnl_pct       REAL,              -- realized P&L % of entry price
            horizon_bars  INTEGER NOT NULL DEFAULT 3
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_otb_symbol_ts ON online_training_buffer (symbol, ts_ms)"
    )

    # -------------------------------------------------------------------
    # LLM explanations for trades (Phase 8c)
    # -------------------------------------------------------------------
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS trade_explanations (
            fill_id     TEXT PRIMARY KEY,
            symbol      TEXT NOT NULL,
            ts_ms       INTEGER NOT NULL,
            explanation TEXT NOT NULL,
            model_used  TEXT NOT NULL DEFAULT 'unknown'
        )
        """
    )

    # -------------------------------------------------------------------
    # Structured decision telemetry (Phase 2 — Gap 0)
    # -------------------------------------------------------------------
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS decision_log (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            ts_ms             INTEGER NOT NULL,
            symbol            TEXT    NOT NULL,
            regime            TEXT,
            tech_action       TEXT,
            tech_confidence   REAL,
            sent_bias         REAL,
            sent_strength     REAL,
            macro_regime      TEXT,
            macro_risk_on     INTEGER,
            meta_weights_json TEXT,
            forecast_4h       REAL,
            forecast_12h      REAL,
            forecast_24h      REAL,
            forecast_conf     REAL,
            risk_vol_pct      REAL,
            risk_mae_pct      REAL,
            risk_stop_hit     REAL,
            risk_regime       TEXT,
            risk_pos_scale    REAL,
            freshness_json    TEXT,
            final_action      TEXT    NOT NULL,
            final_confidence  REAL,
            position_size     REAL    DEFAULT 0.0,
            ml_up_prob        REAL,
            conf_scale        REAL,
            explanation       TEXT,
            realized_pnl      REAL,
            outcome_ts_ms     INTEGER
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_decision_log_ts "
        "ON decision_log (ts_ms)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_decision_log_symbol_ts "
        "ON decision_log (symbol, ts_ms)"
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_decision_log_action "
        "ON decision_log (final_action)"
    )

    conn.commit()


def upsert_candles(
    conn: sqlite3.Connection,
    symbol: str,
    timeframe: str,
    df: pd.DataFrame,
) -> int:
    """Insert or replace candles from *df* into the database.

    *df* must have a ``timestamp`` column (datetime or int ms) plus
    ``open``, ``high``, ``low``, ``close``, ``volume`` columns.

    Returns the number of rows written.
    """
    rows = []
    for _, row in df.iterrows():
        ts = row["timestamp"]
        if hasattr(ts, "timestamp"):
            ts_ms = int(ts.timestamp() * 1000)
        else:
            ts_ms = int(ts)
        rows.append(
            (
                symbol,
                timeframe,
                ts_ms,
                float(row["open"]),
                float(row["high"]),
                float(row["low"]),
                float(row["close"]),
                float(row["volume"]),
            )
        )
    conn.executemany(
        """
        INSERT OR REPLACE INTO candles
            (symbol, timeframe, ts_ms, open, high, low, close, volume)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def load_candles(
    conn: sqlite3.Connection,
    symbol: str,
    timeframe: str,
    limit: int | None = None,
) -> pd.DataFrame:
    """Return stored candles as a DataFrame sorted oldest → newest.

    If *limit* is given, the most recent *limit* rows are returned.
    The ``timestamp`` column is returned as UTC-aware datetime objects.
    """
    if limit:
        query = """
            SELECT ts_ms, open, high, low, close, volume
            FROM candles
            WHERE symbol = ? AND timeframe = ?
            ORDER BY ts_ms DESC
            LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(symbol, timeframe, limit))
        df = df.sort_values("ts_ms").reset_index(drop=True)
    else:
        query = """
            SELECT ts_ms, open, high, low, close, volume
            FROM candles
            WHERE symbol = ? AND timeframe = ?
            ORDER BY ts_ms
        """
        df = pd.read_sql_query(query, conn, params=(symbol, timeframe))

    df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    df = df.drop(columns=["ts_ms"])
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def candle_count(conn: sqlite3.Connection, symbol: str, timeframe: str) -> int:
    """Return the number of stored candles for *symbol* / *timeframe*."""
    row = conn.execute(
        "SELECT COUNT(*) FROM candles WHERE symbol = ? AND timeframe = ?",
        (symbol, timeframe),
    ).fetchone()
    return int(row[0]) if row else 0


def oldest_ts_ms(
    conn: sqlite3.Connection, symbol: str, timeframe: str
) -> int | None:
    """Return the oldest candle timestamp (ms) for *symbol* / *timeframe*, or None if empty."""
    row = conn.execute(
        "SELECT MIN(ts_ms) FROM candles WHERE symbol = ? AND timeframe = ?",
        (symbol, timeframe),
    ).fetchone()
    val = row[0] if row else None
    return int(val) if val is not None else None


def upsert_derivatives(
    conn: sqlite3.Connection,
    symbol: str,
    records: list[tuple[int, str, float]],
) -> int:
    """Insert or replace derivatives metrics.

    Parameters
    ----------
    symbol:
        Trading symbol, e.g. ``"BTC/USD"``.
    records:
        List of ``(ts_ms, metric_name, value)`` tuples.

    Returns
    -------
    int
        Number of rows written.
    """
    rows = [(symbol, ts_ms, metric, value) for ts_ms, metric, value in records]
    conn.executemany(
        """
        INSERT OR REPLACE INTO derivatives_metrics (symbol, ts_ms, metric, value)
        VALUES (?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def load_derivatives(
    conn: sqlite3.Connection,
    symbol: str,
    metric: str,
    limit: int | None = None,
) -> pd.DataFrame:
    """Load derivatives metrics as a DataFrame with columns ``ts_ms``, ``value``."""
    if limit:
        query = """
            SELECT ts_ms, value FROM derivatives_metrics
            WHERE symbol = ? AND metric = ?
            ORDER BY ts_ms DESC LIMIT ?
        """
        df = pd.read_sql_query(query, conn, params=(symbol, metric, limit))
    else:
        query = """
            SELECT ts_ms, value FROM derivatives_metrics
            WHERE symbol = ? AND metric = ?
            ORDER BY ts_ms
        """
        df = pd.read_sql_query(query, conn, params=(symbol, metric))
    df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
    return df


def upsert_onchain(
    conn: sqlite3.Connection,
    symbol: str,
    records: list[tuple[str, str, float]],
) -> int:
    """Insert or replace on-chain metrics.

    Parameters
    ----------
    symbol:
        Trading symbol.
    records:
        List of ``(date_str, metric_name, value)`` tuples where ``date_str``
        is ``"YYYY-MM-DD"``.

    Returns
    -------
    int
        Number of rows written.
    """
    rows = [(symbol, date, metric, value) for date, metric, value in records]
    conn.executemany(
        """
        INSERT OR REPLACE INTO onchain_metrics (symbol, date, metric, value)
        VALUES (?, ?, ?, ?)
        """,
        rows,
    )
    conn.commit()
    return len(rows)


def load_onchain(
    conn: sqlite3.Connection,
    symbol: str,
    metric: str,
) -> pd.DataFrame:
    """Load on-chain metrics as a DataFrame with columns ``date``, ``value``."""
    query = """
        SELECT date, value FROM onchain_metrics
        WHERE symbol = ? AND metric = ?
        ORDER BY date
    """
    return pd.read_sql_query(query, conn, params=(symbol, metric))


def available_symbols(conn: sqlite3.Connection) -> list[tuple[str, str, int]]:
    """Return a list of (symbol, timeframe, count) for all stored series."""
    rows = conn.execute(
        """
        SELECT symbol, timeframe, COUNT(*) AS cnt
        FROM candles
        GROUP BY symbol, timeframe
        ORDER BY symbol, timeframe
        """
    ).fetchall()
    return [(r[0], r[1], r[2]) for r in rows]


# ---------------------------------------------------------------------------
# Execution journaling helpers
# ---------------------------------------------------------------------------

def record_order(conn: sqlite3.Connection, order: dict) -> None:
    """Upsert an order payload (CCXT-style dict)."""
    conn.execute(
        """
        INSERT OR REPLACE INTO orders
        (order_id, exchange, symbol, side, type, status, amount, price, filled, avg_price, fee, fee_ccy, ts_ms, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(order.get("id")),
            str(order.get("exchange", "")),
            str(order.get("symbol")),
            str(order.get("side")),
            str(order.get("type")),
            str(order.get("status", "open")),
            float(order.get("amount") or 0),
            None if order.get("price") is None else float(order.get("price")),
            float(order.get("filled") or 0),
            None if order.get("average") is None else float(order.get("average")),
            None if (order.get("fee") or {}).get("cost") is None else float(order["fee"]["cost"]),
            None if (order.get("fee") or {}).get("currency") is None else str(order["fee"]["currency"]),
            int(order.get("timestamp") or 0),
            json.dumps(order, default=str),
        ),
    )
    conn.commit()


def record_fill(conn: sqlite3.Connection, fill: dict) -> None:
    """Insert a trade/fill payload (CCXT-style dict)."""
    conn.execute(
        """
        INSERT OR REPLACE INTO fills
        (fill_id, order_id, exchange, symbol, side, amount, price, fee, fee_ccy, ts_ms, raw_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            str(fill.get("id") or f"{fill.get('order')}_{fill.get('timestamp')}"),
            str(fill.get("order") or ""),
            str(fill.get("exchange", "")),
            str(fill.get("symbol")),
            str(fill.get("side")),
            float(fill.get("amount") or 0),
            float(fill.get("price") or 0),
            None if (fill.get("fee") or {}).get("cost") is None else float(fill["fee"]["cost"]),
            None if (fill.get("fee") or {}).get("currency") is None else str(fill["fee"]["currency"]),
            int(fill.get("timestamp") or 0),
            json.dumps(fill, default=str),
        ),
    )
    conn.commit()


def record_equity(conn: sqlite3.Connection, ts_ms: int, cash_usd: float, equity_usd: float, drawdown: float) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO equity_snapshots (ts_ms, cash_usd, equity_usd, drawdown)
        VALUES (?, ?, ?, ?)
        """,
        (int(ts_ms), float(cash_usd), float(equity_usd), float(drawdown)),
    )
    conn.commit()


def upsert_position(conn: sqlite3.Connection, symbol: str, qty: float, avg_entry: float, updated_ms: int) -> None:
    conn.execute(
        """
        INSERT OR REPLACE INTO positions (symbol, qty, avg_entry, updated_ms)
        VALUES (?, ?, ?, ?)
        """,
        (symbol, float(qty), float(avg_entry), int(updated_ms)),
    )
    conn.commit()


def load_positions(conn: sqlite3.Connection) -> pd.DataFrame:
    return pd.read_sql_query("SELECT * FROM positions ORDER BY symbol", conn)

def load_equity(conn: sqlite3.Connection, limit: int = 2000) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM equity_snapshots ORDER BY ts_ms DESC LIMIT ?",
        conn,
        params=(int(limit),),
    ).sort_values("ts_ms")

def load_fills(conn: sqlite3.Connection, limit: int = 2000) -> pd.DataFrame:
    return pd.read_sql_query(
        "SELECT * FROM fills ORDER BY ts_ms DESC LIMIT ?",
        conn,
        params=(int(limit),),
    ).sort_values("ts_ms")


def load_latest_fill_ts(conn: sqlite3.Connection, exchange: str, symbol: str | None = None) -> int:
    """Return latest fill timestamp (ms) for an exchange (optionally per-symbol)."""
    if symbol:
        row = conn.execute(
            "SELECT COALESCE(MAX(ts_ms), 0) FROM fills WHERE exchange=? AND symbol=?",
            (exchange, symbol),
        ).fetchone()
    else:
        row = conn.execute(
            "SELECT COALESCE(MAX(ts_ms), 0) FROM fills WHERE exchange=?",
            (exchange,),
        ).fetchone()
    return int(row[0] or 0)


def upsert_position_state(conn: sqlite3.Connection, symbol: str, entry_price: float, peak_price: float, updated_ms: int) -> None:
    conn.execute(
        """INSERT OR REPLACE INTO position_state (symbol, entry_price, peak_price, updated_ms)
           VALUES (?, ?, ?, ?)""",
        (symbol, float(entry_price), float(peak_price), int(updated_ms)),
    )
    conn.commit()


def load_position_state(conn: sqlite3.Connection, symbol: str) -> tuple[float, float] | None:
    row = conn.execute(
        "SELECT entry_price, peak_price FROM position_state WHERE symbol=?",
        (symbol,),
    ).fetchone()
    if not row:
        return None
    return float(row[0]), float(row[1])


# ---------------------------------------------------------------------------
# Paper trade journal helpers
# ---------------------------------------------------------------------------

import uuid as _uuid


def open_paper_trade(
    conn: sqlite3.Connection,
    symbol: str,
    side: str,
    entry_price: float,
    qty: float,
    entry_fee: float,
    open_ts_ms: int,
    ml_up_prob: float | None = None,
    strategy_conf: float | None = None,
    vol_ratio: float | None = None,
) -> str:
    """Insert an open paper trade. Returns the trade_id for later closing."""
    trade_id = str(_uuid.uuid4())
    conn.execute(
        """
        INSERT INTO paper_trades
        (trade_id, symbol, side, entry_price, qty, entry_fee,
         open_ts_ms, ml_up_prob, strategy_conf, vol_ratio)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            trade_id, symbol, side,
            float(entry_price), float(qty), float(entry_fee),
            int(open_ts_ms),
            None if ml_up_prob is None else float(ml_up_prob),
            None if strategy_conf is None else float(strategy_conf),
            None if vol_ratio is None else float(vol_ratio),
        ),
    )
    conn.commit()
    return trade_id


def close_paper_trade(
    conn: sqlite3.Connection,
    symbol: str,
    side: str,
    exit_price: float,
    exit_fee: float,
    close_ts_ms: int,
    close_reason: str = "signal",
) -> None:
    """Close the most-recent open paper trade for this symbol/side, recording P&L."""
    row = conn.execute(
        """
        SELECT trade_id, entry_price, qty, entry_fee
        FROM paper_trades
        WHERE symbol=? AND side=? AND exit_price IS NULL
        ORDER BY open_ts_ms DESC
        LIMIT 1
        """,
        (symbol, side),
    ).fetchone()
    if not row:
        return

    trade_id, entry_price, qty, entry_fee = row
    entry_price = float(entry_price)
    qty = float(qty)

    if side == "long":
        raw_pnl = (float(exit_price) - entry_price) * qty
    else:  # short
        raw_pnl = (entry_price - float(exit_price)) * qty

    realized_pnl = raw_pnl - float(entry_fee) - float(exit_fee)
    pnl_pct = realized_pnl / (entry_price * qty) if entry_price * qty > 0 else 0.0

    conn.execute(
        """
        UPDATE paper_trades
        SET exit_price=?, exit_fee=?, realized_pnl=?, pnl_pct=?,
            close_ts_ms=?, close_reason=?
        WHERE trade_id=?
        """,
        (
            float(exit_price), float(exit_fee),
            float(realized_pnl), float(pnl_pct),
            int(close_ts_ms), close_reason, trade_id,
        ),
    )
    conn.commit()


def load_paper_trades(
    conn: sqlite3.Connection,
    limit: int = 100,
    symbol: str | None = None,
    closed_only: bool = False,
) -> pd.DataFrame:
    """Load paper trades for analysis and ML labeling."""
    where_parts = []
    params: list = []
    if symbol:
        where_parts.append("symbol=?")
        params.append(symbol)
    if closed_only:
        where_parts.append("exit_price IS NOT NULL")

    where = ("WHERE " + " AND ".join(where_parts)) if where_parts else ""
    params.append(int(limit))
    return pd.read_sql_query(
        f"SELECT * FROM paper_trades {where} ORDER BY open_ts_ms DESC LIMIT ?",
        conn,
        params=params,
    )


# -------------------------------------------------------------------
# Decision telemetry (Phase 2 — Gap 0)
# -------------------------------------------------------------------

def log_decision(
    conn: sqlite3.Connection,
    *,
    ts_ms: int,
    symbol: str,
    regime: str | None = None,
    tech_action: str | None = None,
    tech_confidence: float | None = None,
    sent_bias: float | None = None,
    sent_strength: float | None = None,
    macro_regime: str | None = None,
    macro_risk_on: bool | None = None,
    meta_weights: dict | None = None,
    forecast_4h: float | None = None,
    forecast_12h: float | None = None,
    forecast_24h: float | None = None,
    forecast_conf: float | None = None,
    risk_vol_pct: float | None = None,
    risk_mae_pct: float | None = None,
    risk_stop_hit: float | None = None,
    risk_regime: str | None = None,
    risk_pos_scale: float | None = None,
    freshness: dict | None = None,
    final_action: str = "hold",
    final_confidence: float | None = None,
    position_size: float = 0.0,
    ml_up_prob: float | None = None,
    conf_scale: float | None = None,
    explanation: str | None = None,
) -> int:
    """Insert a structured decision packet.  Returns the row id."""
    cur = conn.execute(
        """
        INSERT INTO decision_log (
            ts_ms, symbol, regime,
            tech_action, tech_confidence,
            sent_bias, sent_strength,
            macro_regime, macro_risk_on,
            meta_weights_json,
            forecast_4h, forecast_12h, forecast_24h, forecast_conf,
            risk_vol_pct, risk_mae_pct, risk_stop_hit, risk_regime, risk_pos_scale,
            freshness_json,
            final_action, final_confidence, position_size,
            ml_up_prob, conf_scale, explanation
        ) VALUES (?,?,?, ?,?, ?,?, ?,?, ?, ?,?,?,?, ?,?,?,?,?, ?, ?,?,?, ?,?,?)
        """,
        (
            int(ts_ms), symbol, regime,
            tech_action, tech_confidence,
            sent_bias, sent_strength,
            macro_regime,
            (1 if macro_risk_on else 0) if macro_risk_on is not None else None,
            json.dumps(meta_weights) if meta_weights else None,
            forecast_4h, forecast_12h, forecast_24h, forecast_conf,
            risk_vol_pct, risk_mae_pct, risk_stop_hit, risk_regime, risk_pos_scale,
            json.dumps(freshness) if freshness else None,
            final_action, final_confidence, position_size,
            ml_up_prob, conf_scale, explanation,
        ),
    )
    conn.commit()
    return cur.lastrowid  # type: ignore[return-value]


def update_decision_outcome(
    conn: sqlite3.Connection,
    symbol: str,
    entry_ts_ms: int,
    realized_pnl: float,
    outcome_ts_ms: int,
) -> None:
    """Back-fill realized outcome on the decision that opened this trade."""
    conn.execute(
        """
        UPDATE decision_log
        SET realized_pnl = ?, outcome_ts_ms = ?
        WHERE id = (
            SELECT id FROM decision_log
            WHERE symbol = ? AND ts_ms <= ? AND final_action IN ('buy', 'sell')
            ORDER BY ts_ms DESC LIMIT 1
        )
        """,
        (float(realized_pnl), int(outcome_ts_ms), symbol, int(entry_ts_ms)),
    )
    conn.commit()


def load_decision_log(
    conn: sqlite3.Connection,
    symbol: str | None = None,
    limit: int = 500,
    action_filter: str | None = None,
) -> pd.DataFrame:
    """Load recent decision packets for analysis."""
    clauses: list[str] = []
    params: list = []
    if symbol:
        clauses.append("symbol = ?")
        params.append(symbol)
    if action_filter:
        clauses.append("final_action = ?")
        params.append(action_filter)
    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    params.append(int(limit))
    return pd.read_sql_query(
        f"SELECT * FROM decision_log {where} ORDER BY ts_ms DESC LIMIT ?",
        conn,
        params=params,
    )
