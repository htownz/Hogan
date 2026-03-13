"""Dedicated tests for hogan_bot.storage — trade journaling, side normalization, candle CRUD."""
from __future__ import annotations

import sqlite3

import pandas as pd
import pytest

from hogan_bot.storage import (
    _create_schema,
    candle_count,
    close_paper_trade,
    get_connection,
    load_candles,
    load_paper_trades,
    log_decision,
    normalize_side,
    open_paper_trade,
    record_equity,
    record_fill,
    record_order,
    upsert_candles,
    upsert_position,
    load_positions,
    load_equity,
    load_fills,
    load_latest_fill_ts,
)


def _mem_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    _create_schema(conn)
    return conn


# ---------------------------------------------------------------------------
# Side normalization
# ---------------------------------------------------------------------------

class TestNormalizeSide:
    def test_buy_to_long(self):
        assert normalize_side("buy") == "long"

    def test_sell_to_short(self):
        assert normalize_side("sell") == "short"

    def test_long_stays(self):
        assert normalize_side("long") == "long"

    def test_short_stays(self):
        assert normalize_side("short") == "short"

    def test_case_insensitive(self):
        assert normalize_side("BUY") == "long"
        assert normalize_side("SELL") == "short"

    def test_whitespace_stripped(self):
        assert normalize_side("  buy  ") == "long"


# ---------------------------------------------------------------------------
# Candle CRUD
# ---------------------------------------------------------------------------

class TestCandleCrud:
    def test_upsert_and_load(self):
        conn = _mem_db()
        df = pd.DataFrame({
            "timestamp": [1000, 2000, 3000],
            "open": [100, 101, 102],
            "high": [105, 106, 107],
            "low": [95, 96, 97],
            "close": [103, 104, 105],
            "volume": [1000, 2000, 3000],
        })
        n = upsert_candles(conn, "BTC/USD", "1h", df)
        assert n == 3
        assert candle_count(conn, "BTC/USD", "1h") == 3

        loaded = load_candles(conn, "BTC/USD", "1h")
        assert len(loaded) == 3
        assert "timestamp" in loaded.columns
        assert float(loaded["close"].iloc[0]) == 103.0

    def test_upsert_idempotent(self):
        conn = _mem_db()
        df = pd.DataFrame({
            "timestamp": [1000, 2000],
            "open": [100, 101], "high": [105, 106],
            "low": [95, 96], "close": [103, 104],
            "volume": [1000, 2000],
        })
        upsert_candles(conn, "BTC/USD", "1h", df)
        upsert_candles(conn, "BTC/USD", "1h", df)
        assert candle_count(conn, "BTC/USD", "1h") == 2

    def test_load_with_limit(self):
        conn = _mem_db()
        df = pd.DataFrame({
            "timestamp": list(range(100)),
            "open": [100] * 100, "high": [105] * 100,
            "low": [95] * 100, "close": [103] * 100,
            "volume": [1000] * 100,
        })
        upsert_candles(conn, "BTC/USD", "1h", df)
        loaded = load_candles(conn, "BTC/USD", "1h", limit=10)
        assert len(loaded) == 10


# ---------------------------------------------------------------------------
# Paper trade journal
# ---------------------------------------------------------------------------

class TestPaperTradeJournal:
    def test_open_and_close_trade(self):
        conn = _mem_db()
        trade_id = open_paper_trade(conn, "BTC/USD", "buy", 100.0, 10.0, 0.1, 1000)
        assert trade_id is not None

        trades = load_paper_trades(conn)
        assert len(trades) == 1
        assert trades.iloc[0]["side"] == "long"  # normalized from "buy"

        close_paper_trade(conn, "BTC/USD", "long", 110.0, 0.1, 2000, "take_profit")
        trades = load_paper_trades(conn, closed_only=True)
        assert len(trades) == 1
        assert float(trades.iloc[0]["exit_price"]) == 110.0
        assert float(trades.iloc[0]["realized_pnl"]) > 0

    def test_close_nonexistent_returns_none(self):
        conn = _mem_db()
        result = close_paper_trade(conn, "BTC/USD", "long", 100.0, 0.0, 1000)
        assert result is None

    def test_short_pnl_correct(self):
        conn = _mem_db()
        open_paper_trade(conn, "BTC/USD", "sell", 100.0, 10.0, 0.0, 1000)
        close_paper_trade(conn, "BTC/USD", "short", 90.0, 0.0, 2000, "take_profit")
        trades = load_paper_trades(conn, closed_only=True)
        pnl = float(trades.iloc[0]["realized_pnl"])
        assert pnl == pytest.approx(100.0)  # (100-90)*10


# ---------------------------------------------------------------------------
# Order / fill journaling
# ---------------------------------------------------------------------------

class TestOrderFillJournal:
    def test_record_order(self):
        conn = _mem_db()
        order = {
            "id": "order-1", "exchange": "paper", "symbol": "BTC/USD",
            "side": "buy", "type": "market", "status": "filled",
            "amount": 10.0, "price": 100.0, "filled": 10.0,
            "timestamp": 1000,
        }
        record_order(conn, order)
        row = conn.execute("SELECT * FROM orders WHERE order_id='order-1'").fetchone()
        assert row is not None

    def test_record_fill(self):
        conn = _mem_db()
        fill = {
            "id": "fill-1", "order": "order-1", "exchange": "paper",
            "symbol": "BTC/USD", "side": "buy", "amount": 10.0,
            "price": 100.0, "timestamp": 1000,
        }
        record_fill(conn, fill)
        row = conn.execute("SELECT * FROM fills WHERE fill_id='fill-1'").fetchone()
        assert row is not None


# ---------------------------------------------------------------------------
# Equity / position state
# ---------------------------------------------------------------------------

class TestEquityPosition:
    def test_record_and_load_equity(self):
        conn = _mem_db()
        record_equity(conn, 1000, 5000.0, 10000.0, 0.05)
        df = load_equity(conn)
        assert len(df) == 1
        assert float(df.iloc[0]["equity_usd"]) == 10000.0

    def test_upsert_and_load_position(self):
        conn = _mem_db()
        upsert_position(conn, "BTC/USD", 10.0, 100.0, 1000)
        df = load_positions(conn)
        assert len(df) == 1
        assert float(df.iloc[0]["qty"]) == 10.0

    def test_latest_fill_ts(self):
        conn = _mem_db()
        assert load_latest_fill_ts(conn, "paper") == 0
        fill = {
            "id": "f1", "order": "o1", "exchange": "paper",
            "symbol": "BTC/USD", "side": "buy",
            "amount": 1, "price": 100, "timestamp": 5000,
        }
        record_fill(conn, fill)
        assert load_latest_fill_ts(conn, "paper") == 5000


# ---------------------------------------------------------------------------
# Decision log
# ---------------------------------------------------------------------------

class TestDecisionLog:
    def test_log_and_query(self):
        conn = _mem_db()
        dec_id = log_decision(
            conn, ts_ms=1000, symbol="BTC/USD",
            final_action="buy", final_confidence=0.8,
        )
        assert dec_id > 0

        from hogan_bot.storage import load_decision_log
        df = load_decision_log(conn)
        assert len(df) == 1
        assert df.iloc[0]["final_action"] == "buy"
