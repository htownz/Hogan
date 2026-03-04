"""SQLite-backed candle store for Hogan.

Provides a simple way to persist OHLCV data locally so that training,
backtesting, and the dashboard can work without a live exchange connection.

Typical workflow::

    from hogan_bot.storage import get_connection, upsert_candles, load_candles

    conn = get_connection("data/hogan.db")

    # Populate from exchange
    from hogan_bot.exchange import KrakenClient
    client = KrakenClient(None, None)
    df = client.fetch_ohlcv_df("BTC/USD", timeframe="5m", limit=5000)
    upsert_candles(conn, "BTC/USD", "5m", df)

    # Load back for training / backtesting
    candles = load_candles(conn, "BTC/USD", "5m")
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import pandas as pd


def get_connection(db_path: str = "data/hogan.db") -> sqlite3.Connection:
    """Open (or create) the SQLite database and ensure the schema exists."""
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
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
