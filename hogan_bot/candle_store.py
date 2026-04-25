"""Candle storage abstraction for SQLite and TimescaleDB/Postgres.

SQLite remains Hogan's default runtime store. TimescaleDB is introduced as an
opt-in backend for large candle/sub-minute history without forcing execution,
paper-trade, or swarm tables off SQLite yet.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


class CandleStore(Protocol):
    """Repository API used by ingestion, training, and backtest code."""

    def upsert_candles(self, symbol: str, timeframe: str, df: pd.DataFrame) -> int:
        ...

    def load_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int | None = None,
    ) -> pd.DataFrame:
        ...

    def candle_count(self, symbol: str, timeframe: str) -> int:
        ...

    def oldest_ts_ms(self, symbol: str, timeframe: str) -> int | None:
        ...

    def available_symbols(self) -> list[tuple[str, str, int]]:
        ...

    def close(self) -> None:
        ...


@dataclass
class SQLiteCandleStore:
    """Adapter over the existing SQLite candle helpers in ``storage.py``."""

    conn: object

    def upsert_candles(self, symbol: str, timeframe: str, df: pd.DataFrame) -> int:
        from hogan_bot.storage import upsert_candles
        return upsert_candles(self.conn, symbol, timeframe, df)

    def load_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int | None = None,
    ) -> pd.DataFrame:
        from hogan_bot.storage import load_candles
        return load_candles(self.conn, symbol, timeframe, limit=limit)

    def candle_count(self, symbol: str, timeframe: str) -> int:
        from hogan_bot.storage import candle_count
        return candle_count(self.conn, symbol, timeframe)

    def oldest_ts_ms(self, symbol: str, timeframe: str) -> int | None:
        from hogan_bot.storage import oldest_ts_ms
        return oldest_ts_ms(self.conn, symbol, timeframe)

    def available_symbols(self) -> list[tuple[str, str, int]]:
        from hogan_bot.storage import available_symbols
        return available_symbols(self.conn)

    def close(self) -> None:
        close = getattr(self.conn, "close", None)
        if close is not None:
            close()


class TimescaleCandleStore:
    """TimescaleDB/Postgres candle store.

    The schema matches SQLite's candle table shape and uses
    ``ON CONFLICT(symbol, timeframe, ts_ms)`` for idempotent bar replacement.
    """

    def __init__(self, database_url: str):
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError(
                "Timescale backend requires psycopg[binary]. "
                "Install requirements.txt or set HOGAN_STORAGE_BACKEND=sqlite."
            ) from exc
        self._psycopg = psycopg
        self.conn = psycopg.connect(database_url)
        self.ensure_schema()

    def ensure_schema(self) -> None:
        with self.conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS timescaledb")
            cur.execute(
                """
                CREATE TABLE IF NOT EXISTS candles (
                    symbol    TEXT    NOT NULL,
                    timeframe TEXT    NOT NULL,
                    ts_ms     BIGINT  NOT NULL,
                    open      DOUBLE PRECISION,
                    high      DOUBLE PRECISION,
                    low       DOUBLE PRECISION,
                    close     DOUBLE PRECISION,
                    volume    DOUBLE PRECISION,
                    PRIMARY KEY (symbol, timeframe, ts_ms)
                )
                """
            )
            cur.execute(
                """
                SELECT create_hypertable(
                    'candles',
                    'ts_ms',
                    chunk_time_interval => 86400000,
                    if_not_exists => TRUE
                )
                """
            )
            cur.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_candles_symbol_tf_ts
                ON candles (symbol, timeframe, ts_ms DESC)
                """
            )
        self.conn.commit()

    @staticmethod
    def _rows(symbol: str, timeframe: str, df: pd.DataFrame) -> list[tuple]:
        rows: list[tuple] = []
        for _, row in df.iterrows():
            ts = row["timestamp"] if "timestamp" in row else row["ts_ms"]
            if hasattr(ts, "timestamp"):
                ts_ms = int(ts.timestamp() * 1000)
            else:
                ts_ms = int(ts)
                ts = pd.to_datetime(ts_ms, unit="ms", utc=True)
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
        return rows

    def upsert_candles(self, symbol: str, timeframe: str, df: pd.DataFrame) -> int:
        rows = self._rows(symbol, timeframe, df)
        if not rows:
            return 0
        with self.conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO candles
                    (symbol, timeframe, ts_ms, open, high, low, close, volume)
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                ON CONFLICT (symbol, timeframe, ts_ms) DO UPDATE SET
                    open = EXCLUDED.open,
                    high = EXCLUDED.high,
                    low = EXCLUDED.low,
                    close = EXCLUDED.close,
                    volume = EXCLUDED.volume
                """,
                rows,
            )
        self.conn.commit()
        return len(rows)

    def load_candles(
        self,
        symbol: str,
        timeframe: str,
        limit: int | None = None,
    ) -> pd.DataFrame:
        params: tuple
        if limit:
            query = """
                SELECT ts_ms, open, high, low, close, volume
                FROM (
                    SELECT ts_ms, open, high, low, close, volume
                    FROM candles
                    WHERE symbol = %s AND timeframe = %s
                    ORDER BY ts_ms DESC
                    LIMIT %s
                ) recent
                ORDER BY ts_ms
            """
            params = (symbol, timeframe, int(limit))
        else:
            query = """
                SELECT ts_ms, open, high, low, close, volume
                FROM candles
                WHERE symbol = %s AND timeframe = %s
                ORDER BY ts_ms
            """
            params = (symbol, timeframe)
        with self.conn.cursor() as cur:
            cur.execute(query, params)
            rows = cur.fetchall()
        df = pd.DataFrame(
            rows,
            columns=["ts_ms", "open", "high", "low", "close", "volume"],
        )
        if df.empty:
            df["timestamp"] = pd.to_datetime([], utc=True)
            return df[["ts_ms", "timestamp", "open", "high", "low", "close", "volume"]]
        df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
        return df[["ts_ms", "timestamp", "open", "high", "low", "close", "volume"]]

    def candle_count(self, symbol: str, timeframe: str) -> int:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT COUNT(*) FROM candles WHERE symbol = %s AND timeframe = %s",
                (symbol, timeframe),
            )
            row = cur.fetchone()
        return int(row[0]) if row else 0

    def oldest_ts_ms(self, symbol: str, timeframe: str) -> int | None:
        with self.conn.cursor() as cur:
            cur.execute(
                "SELECT MIN(ts_ms) FROM candles WHERE symbol = %s AND timeframe = %s",
                (symbol, timeframe),
            )
            row = cur.fetchone()
        return int(row[0]) if row and row[0] is not None else None

    def available_symbols(self) -> list[tuple[str, str, int]]:
        with self.conn.cursor() as cur:
            cur.execute(
                """
                SELECT symbol, timeframe, COUNT(*) AS cnt
                FROM candles
                GROUP BY symbol, timeframe
                ORDER BY symbol, timeframe
                """
            )
            rows = cur.fetchall()
        return [(str(r[0]), str(r[1]), int(r[2])) for r in rows]

    def close(self) -> None:
        self.conn.close()


def open_candle_store(config_or_backend, conn=None) -> CandleStore:
    """Open the configured candle store.

    ``config_or_backend`` may be a ``BotConfig``-like object or a backend
    string. Passing ``conn`` with ``sqlite`` wraps an existing SQLite
    connection, preserving current tests and call sites.
    """
    backend = config_or_backend
    database_url = ""
    if not isinstance(config_or_backend, str):
        backend = getattr(config_or_backend, "storage_backend", "sqlite")
        database_url = getattr(config_or_backend, "database_url", "")
    backend = str(backend).lower()
    if backend == "sqlite":
        if conn is None:
            from hogan_bot.storage import get_connection
            db_path = getattr(config_or_backend, "db_path", "data/hogan.db")
            conn = get_connection(db_path)
        return SQLiteCandleStore(conn)
    if backend in ("timescale", "postgres", "postgresql"):
        if not database_url:
            raise ValueError("Timescale backend requires HOGAN_DATABASE_URL")
        return TimescaleCandleStore(database_url)
    raise ValueError(f"Unsupported candle storage backend: {backend!r}")
