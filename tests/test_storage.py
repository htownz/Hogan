"""Tests for hogan_bot.storage SQLite candle store."""
import tempfile
import os
import unittest

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover
    pd = None

from hogan_bot.storage import (
    available_symbols,
    candle_count,
    get_connection,
    load_candles,
    upsert_candles,
)


def _sample_df(n: int = 10) -> "pd.DataFrame":
    import numpy as np
    timestamps = pd.date_range("2024-01-01", periods=n, freq="5min", tz="UTC")
    close = 30_000.0 + np.arange(n, dtype=float)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "open": close - 5,
            "high": close + 10,
            "low": close - 10,
            "close": close,
            "volume": [100.0] * n,
        }
    )


@unittest.skipUnless(pd is not None, "pandas not installed")
class StorageTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mktemp(suffix=".db")

    def tearDown(self):
        if os.path.exists(self.tmp):
            os.remove(self.tmp)

    def test_get_connection_creates_file(self):
        conn = get_connection(self.tmp)
        conn.close()
        self.assertTrue(os.path.exists(self.tmp))

    def test_upsert_returns_row_count(self):
        conn = get_connection(self.tmp)
        df = _sample_df(20)
        n = upsert_candles(conn, "BTC/USD", "5m", df)
        conn.close()
        self.assertEqual(n, 20)

    def test_candle_count(self):
        conn = get_connection(self.tmp)
        upsert_candles(conn, "BTC/USD", "5m", _sample_df(15))
        count = candle_count(conn, "BTC/USD", "5m")
        conn.close()
        self.assertEqual(count, 15)

    def test_load_candles_sorted_oldest_first(self):
        conn = get_connection(self.tmp)
        upsert_candles(conn, "BTC/USD", "5m", _sample_df(10))
        df = load_candles(conn, "BTC/USD", "5m")
        conn.close()
        self.assertEqual(len(df), 10)
        self.assertTrue((df["close"].diff().dropna() >= 0).all())

    def test_load_candles_limit(self):
        conn = get_connection(self.tmp)
        upsert_candles(conn, "BTC/USD", "5m", _sample_df(50))
        df = load_candles(conn, "BTC/USD", "5m", limit=10)
        conn.close()
        self.assertEqual(len(df), 10)

    def test_upsert_is_idempotent(self):
        conn = get_connection(self.tmp)
        df = _sample_df(10)
        upsert_candles(conn, "BTC/USD", "5m", df)
        upsert_candles(conn, "BTC/USD", "5m", df)  # same data again
        count = candle_count(conn, "BTC/USD", "5m")
        conn.close()
        self.assertEqual(count, 10)  # no duplicates

    def test_available_symbols(self):
        conn = get_connection(self.tmp)
        upsert_candles(conn, "BTC/USD", "5m", _sample_df(5))
        upsert_candles(conn, "ETH/USD", "5m", _sample_df(8))
        series = available_symbols(conn)
        conn.close()
        symbols = {s[0] for s in series}
        self.assertIn("BTC/USD", symbols)
        self.assertIn("ETH/USD", symbols)

    def test_load_candles_has_correct_columns(self):
        conn = get_connection(self.tmp)
        upsert_candles(conn, "BTC/USD", "5m", _sample_df(5))
        df = load_candles(conn, "BTC/USD", "5m")
        conn.close()
        for col in ("timestamp", "open", "high", "low", "close", "volume"):
            self.assertIn(col, df.columns)

    def test_timestamp_is_datetime(self):
        conn = get_connection(self.tmp)
        upsert_candles(conn, "BTC/USD", "5m", _sample_df(5))
        df = load_candles(conn, "BTC/USD", "5m")
        conn.close()
        self.assertTrue(pd.api.types.is_datetime64_any_dtype(df["timestamp"]))


if __name__ == "__main__":
    unittest.main()
