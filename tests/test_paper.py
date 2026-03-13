"""Dedicated tests for hogan_bot.paper — PaperPortfolio, Position, exits, MAE/MFE."""
from __future__ import annotations

import pytest

from hogan_bot.paper import PaperPortfolio, Position, ShortPosition


# ---------------------------------------------------------------------------
# Position dataclass
# ---------------------------------------------------------------------------

class TestPosition:
    def test_defaults(self):
        p = Position()
        assert p.qty == 0.0
        assert p.avg_entry == 0.0
        assert p.bars_held == 0
        assert p.max_adverse_pct == 0.0
        assert p.max_favorable_pct == 0.0
        assert p.entry_atr_pct == 0.0


# ---------------------------------------------------------------------------
# PaperPortfolio — buy/sell basics
# ---------------------------------------------------------------------------

class TestPortfolioBuySell:
    def test_buy_deducts_cash(self):
        p = PaperPortfolio(cash_usd=10_000, fee_rate=0.001)
        ok = p.execute_buy("BTC/USD", 100.0, 10.0)
        assert ok is True
        expected_cost = 10 * 100 + 10 * 100 * 0.001
        assert p.cash_usd == pytest.approx(10_000 - expected_cost)

    def test_buy_creates_position(self):
        p = PaperPortfolio(cash_usd=10_000, fee_rate=0.0)
        p.execute_buy("BTC/USD", 100.0, 5.0)
        assert "BTC/USD" in p.positions
        assert p.positions["BTC/USD"].qty == 5.0
        assert p.positions["BTC/USD"].avg_entry == pytest.approx(100.0)

    def test_buy_rejected_insufficient_cash(self):
        p = PaperPortfolio(cash_usd=100, fee_rate=0.0)
        ok = p.execute_buy("BTC/USD", 100.0, 10.0)
        assert ok is False
        assert "BTC/USD" not in p.positions

    def test_buy_rejected_zero_qty(self):
        p = PaperPortfolio(cash_usd=10_000, fee_rate=0.0)
        assert p.execute_buy("BTC/USD", 100.0, 0.0) is False

    def test_buy_rejected_zero_price(self):
        p = PaperPortfolio(cash_usd=10_000, fee_rate=0.0)
        assert p.execute_buy("BTC/USD", 0.0, 10.0) is False

    def test_sell_adds_cash(self):
        p = PaperPortfolio(cash_usd=10_000, fee_rate=0.001)
        p.execute_buy("BTC/USD", 100.0, 10.0)
        cash_after_buy = p.cash_usd
        p.execute_sell("BTC/USD", 110.0, 10.0)
        proceeds = 10 * 110 - 10 * 110 * 0.001
        assert p.cash_usd == pytest.approx(cash_after_buy + proceeds)

    def test_sell_removes_position(self):
        p = PaperPortfolio(cash_usd=10_000, fee_rate=0.0)
        p.execute_buy("BTC/USD", 100.0, 10.0)
        p.execute_sell("BTC/USD", 110.0, 10.0)
        assert "BTC/USD" not in p.positions

    def test_partial_sell_reduces_qty(self):
        p = PaperPortfolio(cash_usd=10_000, fee_rate=0.0)
        p.execute_buy("BTC/USD", 100.0, 10.0)
        p.execute_sell("BTC/USD", 110.0, 5.0)
        assert p.positions["BTC/USD"].qty == pytest.approx(5.0)

    def test_sell_rejected_no_position(self):
        p = PaperPortfolio(cash_usd=10_000, fee_rate=0.0)
        assert p.execute_sell("BTC/USD", 100.0, 10.0) is False

    def test_sell_rejected_oversell(self):
        p = PaperPortfolio(cash_usd=10_000, fee_rate=0.0)
        p.execute_buy("BTC/USD", 100.0, 5.0)
        assert p.execute_sell("BTC/USD", 100.0, 10.0) is False

    def test_add_to_position_averages_entry(self):
        p = PaperPortfolio(cash_usd=100_000, fee_rate=0.0)
        p.execute_buy("BTC/USD", 100.0, 10.0)
        p.execute_buy("BTC/USD", 200.0, 10.0)
        pos = p.positions["BTC/USD"]
        assert pos.qty == 20.0
        assert pos.avg_entry == pytest.approx(150.0)


# ---------------------------------------------------------------------------
# PaperPortfolio — equity
# ---------------------------------------------------------------------------

class TestPortfolioEquity:
    def test_equity_with_position(self):
        p = PaperPortfolio(cash_usd=10_000, fee_rate=0.0)
        p.execute_buy("BTC/USD", 100.0, 10.0)
        equity = p.total_equity({"BTC/USD": 110.0})
        assert equity == pytest.approx(10_000 - 1000 + 1100)

    def test_equity_no_positions(self):
        p = PaperPortfolio(cash_usd=10_000, fee_rate=0.0)
        assert p.total_equity({}) == 10_000


# ---------------------------------------------------------------------------
# PaperPortfolio — shorts
# ---------------------------------------------------------------------------

class TestPortfolioShorts:
    def test_short_entry(self):
        p = PaperPortfolio(cash_usd=10_000, fee_rate=0.001)
        ok = p.execute_short("BTC/USD", 100.0, 10.0)
        assert ok is True
        assert "BTC/USD" in p.short_positions
        assert p.short_positions["BTC/USD"].qty == 10.0

    def test_short_rejected_insufficient_margin(self):
        p = PaperPortfolio(cash_usd=100, fee_rate=0.0)
        ok = p.execute_short("BTC/USD", 100.0, 10.0)
        assert ok is False

    def test_cover_short_profit(self):
        p = PaperPortfolio(cash_usd=10_000, fee_rate=0.0)
        p.execute_short("BTC/USD", 100.0, 10.0)
        cash_before = p.cash_usd
        p.execute_cover("BTC/USD", 90.0, 10.0)
        pnl = (100.0 - 90.0) * 10.0
        assert p.cash_usd == pytest.approx(cash_before + pnl)
        assert "BTC/USD" not in p.short_positions

    def test_cover_short_loss(self):
        p = PaperPortfolio(cash_usd=10_000, fee_rate=0.0)
        p.execute_short("BTC/USD", 100.0, 10.0)
        cash_before = p.cash_usd
        p.execute_cover("BTC/USD", 110.0, 10.0)
        pnl = (100.0 - 110.0) * 10.0
        assert p.cash_usd == pytest.approx(cash_before + pnl)

    def test_short_equity_calculation(self):
        p = PaperPortfolio(cash_usd=10_000, fee_rate=0.0)
        p.execute_short("BTC/USD", 100.0, 10.0)
        # Price dropped to 90 → short is +100 pnl
        equity = p.total_equity({"BTC/USD": 90.0})
        assert equity == pytest.approx(10_000 + 100.0)


# ---------------------------------------------------------------------------
# PaperPortfolio — check_exits
# ---------------------------------------------------------------------------

class TestCheckExits:
    def test_trailing_stop_fires(self):
        p = PaperPortfolio(cash_usd=10_000, fee_rate=0.0)
        p.execute_buy("BTC/USD", 100.0, 10.0, trailing_stop_pct=0.05)
        # Price goes up to 110, then drops to 104
        p.check_exits({"BTC/USD": 110.0})
        exits = p.check_exits({"BTC/USD": 104.0})
        assert len(exits) == 1
        assert exits[0] == ("BTC/USD", "trailing_stop")

    def test_take_profit_fires(self):
        p = PaperPortfolio(cash_usd=10_000, fee_rate=0.0)
        p.execute_buy("BTC/USD", 100.0, 10.0, take_profit_pct=0.10)
        exits = p.check_exits({"BTC/USD": 111.0})
        assert len(exits) == 1
        assert exits[0] == ("BTC/USD", "take_profit")

    def test_max_hold_time_fires(self):
        p = PaperPortfolio(cash_usd=10_000, fee_rate=0.0)
        p.execute_buy("BTC/USD", 100.0, 10.0)
        for _ in range(24):
            exits = p.check_exits({"BTC/USD": 100.0}, max_hold_bars=24)
        assert len(exits) == 1
        assert exits[0] == ("BTC/USD", "max_hold_time")

    def test_no_exit_when_healthy(self):
        p = PaperPortfolio(cash_usd=10_000, fee_rate=0.0)
        p.execute_buy("BTC/USD", 100.0, 10.0, trailing_stop_pct=0.10, take_profit_pct=0.20)
        exits = p.check_exits({"BTC/USD": 105.0})
        assert len(exits) == 0

    def test_bars_held_increments(self):
        p = PaperPortfolio(cash_usd=10_000, fee_rate=0.0)
        p.execute_buy("BTC/USD", 100.0, 10.0)
        p.check_exits({"BTC/USD": 100.0})
        p.check_exits({"BTC/USD": 100.0})
        assert p.positions["BTC/USD"].bars_held == 2

    def test_mae_mfe_tracked(self):
        p = PaperPortfolio(cash_usd=10_000, fee_rate=0.0)
        p.execute_buy("BTC/USD", 100.0, 10.0)
        p.check_exits({"BTC/USD": 95.0})   # 5% adverse
        p.check_exits({"BTC/USD": 108.0})  # 8% favorable
        pos = p.positions["BTC/USD"]
        assert pos.max_adverse_pct == pytest.approx(0.05)
        assert pos.max_favorable_pct == pytest.approx(0.08)

    def test_short_trailing_stop_fires(self):
        p = PaperPortfolio(cash_usd=10_000, fee_rate=0.0)
        p.execute_short("BTC/USD", 100.0, 10.0, trailing_stop_pct=0.05)
        p.check_exits({"BTC/USD": 90.0})   # trough = 90
        exits = p.check_exits({"BTC/USD": 95.0})  # 5.5% above trough
        assert any(r == "short_trailing_stop" for _, r in exits)

    def test_short_take_profit_fires(self):
        p = PaperPortfolio(cash_usd=10_000, fee_rate=0.0)
        p.execute_short("BTC/USD", 100.0, 10.0, take_profit_pct=0.10)
        exits = p.check_exits({"BTC/USD": 89.0})
        assert any(r == "short_take_profit" for _, r in exits)
