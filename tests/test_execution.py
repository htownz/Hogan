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
        assert r.fill_price is None
        assert r.fill_qty is None


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
# PaperExecution — Intent-based API (open_long/close_long/open_short/close_short)
# ---------------------------------------------------------------------------

class TestPaperExecutionIntentAPI:
    def test_open_long_single_mutation(self):
        p = PaperPortfolio(cash_usd=10_000, fee_rate=0.0)
        ex = PaperExecution(p)
        result = ex.open_long("BTC/USD", 100.0, 5.0)
        assert result.ok is True
        assert result.fill_price == pytest.approx(100.0)
        assert result.fill_qty == pytest.approx(5.0)
        assert "BTC/USD" in p.positions
        assert p.positions["BTC/USD"].qty == pytest.approx(5.0)
        assert p.cash_usd == pytest.approx(9_500.0)

    def test_close_long_removes_position(self):
        p = PaperPortfolio(cash_usd=10_000, fee_rate=0.0)
        ex = PaperExecution(p)
        ex.open_long("BTC/USD", 100.0, 5.0)
        result = ex.close_long("BTC/USD", 110.0, 5.0, reason="take_profit")
        assert result.ok is True
        assert result.fill_price == pytest.approx(110.0)
        assert result.fill_qty == pytest.approx(5.0)
        assert "BTC/USD" not in p.positions
        assert p.cash_usd == pytest.approx(10_050.0)

    def test_open_short_single_mutation(self):
        p = PaperPortfolio(cash_usd=10_000, fee_rate=0.0)
        ex = PaperExecution(p)
        result = ex.open_short("BTC/USD", 100.0, 3.0)
        assert result.ok is True
        assert "BTC/USD" in p.short_positions
        assert p.short_positions["BTC/USD"].qty == pytest.approx(3.0)

    def test_close_short_removes_position(self):
        p = PaperPortfolio(cash_usd=10_000, fee_rate=0.0)
        ex = PaperExecution(p)
        ex.open_short("BTC/USD", 100.0, 3.0)
        result = ex.close_short("BTC/USD", 90.0, 3.0, reason="take_profit")
        assert result.ok is True
        assert "BTC/USD" not in p.short_positions

    def test_close_long_no_position_fails(self):
        p = PaperPortfolio(cash_usd=10_000, fee_rate=0.0)
        ex = PaperExecution(p)
        result = ex.close_long("BTC/USD", 100.0, 5.0)
        assert result.ok is False

    def test_close_short_no_position_fails(self):
        p = PaperPortfolio(cash_usd=10_000, fee_rate=0.0)
        ex = PaperExecution(p)
        result = ex.close_short("BTC/USD", 100.0, 5.0)
        assert result.ok is False

    def test_open_long_with_stops_sets_params(self):
        p = PaperPortfolio(cash_usd=50_000, fee_rate=0.0)
        ex = PaperExecution(p)
        ex.open_long("BTC/USD", 100.0, 5.0, trailing_stop_pct=0.02, take_profit_pct=0.05)
        pos = p.positions["BTC/USD"]
        assert pos.qty == pytest.approx(5.0)

    def test_open_short_journals_to_db(self):
        p = PaperPortfolio(cash_usd=10_000, fee_rate=0.0)
        conn = _mem_db()
        ex = PaperExecution(p, conn=conn)
        ex.open_short("BTC/USD", 100.0, 3.0)
        row = conn.execute("SELECT qty FROM positions WHERE symbol='BTC/USD'").fetchone()
        assert row is not None
        assert float(row[0]) == pytest.approx(-3.0)

    def test_emergency_flatten_closes_both(self):
        p = PaperPortfolio(cash_usd=50_000, fee_rate=0.0)
        ex = PaperExecution(p)
        ex.open_long("BTC/USD", 100.0, 2.0)
        ex.open_short("ETH/USD", 50.0, 5.0)
        ex.emergency_flatten("BTC/USD", 100.0)
        assert "BTC/USD" not in p.positions
        ex.emergency_flatten("ETH/USD", 50.0)
        assert "ETH/USD" not in p.short_positions


# ---------------------------------------------------------------------------
# RealisticPaperExecution
# ---------------------------------------------------------------------------

class TestRealisticPaperExecution:
    def test_buy_applies_slippage(self):
        p = PaperPortfolio(cash_usd=100_000, fee_rate=0.0)
        cfg = FillSimConfig(slippage_bps=10.0, spread_half_bps=5.0)
        ex = RealisticPaperExecution(p, config=cfg)
        r = ex.buy("BTC/USD", 100.0, 10.0)
        pos = p.positions["BTC/USD"]
        # Should fill worse than 100: price * (1 + 15bps)
        assert pos.avg_entry > 100.0
        expected = 100.0 * (1 + 15 / 10_000)
        assert pos.avg_entry == pytest.approx(expected)
        assert r.fill_price == pytest.approx(expected)
        assert r.fill_qty == pytest.approx(10.0)

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
        assert result.fill_price == pytest.approx(expected_fill)
        assert result.fill_qty == pytest.approx(10.0)
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
