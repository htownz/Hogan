"""Dedicated tests for hogan_bot.execution — PaperExecution and RealisticPaperExecution."""
from __future__ import annotations

import sqlite3

import pytest

from hogan_bot.execution import (
    ExecResult,
    FillSimConfig,
    PaperExecution,
    RealisticPaperExecution,
)
from hogan_bot.paper import PaperPortfolio
from hogan_bot.storage import _create_schema


def _mem_db() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    _create_schema(conn)
    return conn


# ---------------------------------------------------------------------------
# ExecResult
# ---------------------------------------------------------------------------

class TestExecResult:
    def test_defaults(self):
        r = ExecResult(ok=True)
        assert r.ok is True
        assert r.order_id is None
        assert r.error is None


# ---------------------------------------------------------------------------
# PaperExecution
# ---------------------------------------------------------------------------

class TestPaperExecution:
    def test_buy_success(self):
        p = PaperPortfolio(cash_usd=10_000, fee_rate=0.001)
        conn = _mem_db()
        ex = PaperExecution(p, conn=conn)
        result = ex.buy("BTC/USD", 100.0, 10.0)
        assert result.ok is True
        assert "BTC/USD" in p.positions

    def test_buy_fail_no_cash(self):
        p = PaperPortfolio(cash_usd=10, fee_rate=0.0)
        ex = PaperExecution(p)
        result = ex.buy("BTC/USD", 100.0, 10.0)
        assert result.ok is False

    def test_sell_success(self):
        p = PaperPortfolio(cash_usd=10_000, fee_rate=0.0)
        ex = PaperExecution(p)
        ex.buy("BTC/USD", 100.0, 10.0)
        result = ex.sell("BTC/USD", 110.0, 10.0)
        assert result.ok is True
        assert "BTC/USD" not in p.positions

    def test_sell_no_position(self):
        p = PaperPortfolio(cash_usd=10_000, fee_rate=0.0)
        ex = PaperExecution(p)
        result = ex.sell("BTC/USD", 100.0, 10.0)
        assert result.ok is False

    def test_buy_journals_to_db(self):
        p = PaperPortfolio(cash_usd=10_000, fee_rate=0.0)
        conn = _mem_db()
        ex = PaperExecution(p, conn=conn)
        ex.buy("BTC/USD", 100.0, 10.0)
        row = conn.execute("SELECT * FROM positions WHERE symbol='BTC/USD'").fetchone()
        assert row is not None
        assert float(row[1]) == pytest.approx(10.0)  # qty

    def test_sell_journals_zero_to_db(self):
        p = PaperPortfolio(cash_usd=10_000, fee_rate=0.0)
        conn = _mem_db()
        ex = PaperExecution(p, conn=conn)
        ex.buy("BTC/USD", 100.0, 10.0)
        ex.sell("BTC/USD", 110.0, 10.0)
        row = conn.execute("SELECT * FROM positions WHERE symbol='BTC/USD'").fetchone()
        assert float(row[1]) == pytest.approx(0.0)  # qty = 0


# ---------------------------------------------------------------------------
# RealisticPaperExecution
# ---------------------------------------------------------------------------

class TestRealisticPaperExecution:
    def test_buy_applies_slippage(self):
        p = PaperPortfolio(cash_usd=100_000, fee_rate=0.0)
        cfg = FillSimConfig(slippage_bps=10.0, spread_half_bps=5.0)
        ex = RealisticPaperExecution(p, config=cfg)
        ex.buy("BTC/USD", 100.0, 10.0)
        pos = p.positions["BTC/USD"]
        # Should fill worse than 100: price * (1 + 15bps)
        assert pos.avg_entry > 100.0
        expected = 100.0 * (1 + 15 / 10_000)
        assert pos.avg_entry == pytest.approx(expected)

    def test_sell_applies_slippage(self):
        p = PaperPortfolio(cash_usd=100_000, fee_rate=0.0)
        cfg = FillSimConfig(slippage_bps=10.0, spread_half_bps=5.0)
        ex = RealisticPaperExecution(p, config=cfg)
        p.execute_buy("BTC/USD", 100.0, 10.0)
        cash_before = p.cash_usd
        result = ex.sell("BTC/USD", 100.0, 10.0)
        assert result.ok is True
        # Sell fill should be worse than 100
        expected_fill = 100.0 * (1 - 15 / 10_000)
        proceeds = 10 * expected_fill
        assert p.cash_usd == pytest.approx(cash_before + proceeds)

    def test_default_config_works(self):
        p = PaperPortfolio(cash_usd=10_000, fee_rate=0.001)
        ex = RealisticPaperExecution(p)
        result = ex.buy("BTC/USD", 100.0, 5.0)
        assert result.ok is True

    def test_partial_fill_when_configured(self):
        p = PaperPortfolio(cash_usd=100_000, fee_rate=0.0)
        cfg = FillSimConfig(slippage_bps=0, spread_half_bps=0,
                            partial_fill_probability=1.0, min_fill_ratio=0.5)
        ex = RealisticPaperExecution(p, config=cfg)
        ex.buy("BTC/USD", 100.0, 10.0)
        pos = p.positions["BTC/USD"]
        assert pos.qty == pytest.approx(5.0)  # 50% fill

    def test_journals_to_db(self):
        p = PaperPortfolio(cash_usd=10_000, fee_rate=0.0)
        conn = _mem_db()
        ex = RealisticPaperExecution(p, conn=conn)
        ex.buy("BTC/USD", 100.0, 5.0)
        row = conn.execute("SELECT * FROM positions WHERE symbol='BTC/USD'").fetchone()
        assert row is not None
