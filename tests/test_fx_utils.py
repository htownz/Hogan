"""Tests for FX-specific utilities — session filter, pip math, position sizing."""
from __future__ import annotations

import datetime

import pytest

from hogan_bot.fx_utils import (
    SessionFilter,
    current_session,
    fx_position_size,
    is_weekend,
    pip_size,
    pip_stop_loss,
    pip_take_profit,
    pips_to_price,
)


class TestPipMath:
    def test_pip_size_major_pairs(self):
        assert pip_size("EUR/USD") == 0.0001
        assert pip_size("GBP/USD") == 0.0001

    def test_pip_size_jpy(self):
        assert pip_size("USD/JPY") == 0.01

    def test_pip_size_unknown_defaults(self):
        assert pip_size("UNKNOWN/PAIR") == 0.0001

    def test_pips_to_price(self):
        assert pips_to_price("EUR/USD", 50) == pytest.approx(0.005)
        assert pips_to_price("USD/JPY", 50) == pytest.approx(0.5)

    def test_pip_stop_loss_long(self):
        sl = pip_stop_loss("EUR/USD", 1.1000, "long", 50)
        assert sl == pytest.approx(1.0950)

    def test_pip_stop_loss_short(self):
        sl = pip_stop_loss("EUR/USD", 1.1000, "short", 50)
        assert sl == pytest.approx(1.1050)

    def test_pip_take_profit_long(self):
        tp = pip_take_profit("EUR/USD", 1.1000, "long", 100)
        assert tp == pytest.approx(1.1100)

    def test_pip_take_profit_short(self):
        tp = pip_take_profit("GBP/USD", 1.2500, "short", 100)
        assert tp == pytest.approx(1.2400)


class TestPositionSizing:
    def test_basic_sizing(self):
        units = fx_position_size(account_balance=10_000, risk_pct=0.01, stop_pips=50, symbol="EUR/USD", price=1.1000)
        assert units > 0
        risk_usd = units * 50 * pip_size("EUR/USD")
        assert risk_usd == pytest.approx(100, abs=5)

    def test_jpy_pair_sizing(self):
        units = fx_position_size(account_balance=10_000, risk_pct=0.01, stop_pips=50, symbol="USD/JPY", price=150.0)
        assert units > 0

    def test_zero_stop_returns_zero(self):
        units = fx_position_size(account_balance=10_000, risk_pct=0.01, stop_pips=0, symbol="EUR/USD", price=1.1)
        assert units == 0


class TestSessions:
    def test_is_weekend_friday_late(self):
        friday_2200 = datetime.datetime(2026, 3, 13, 22, 0, tzinfo=datetime.timezone.utc)
        assert is_weekend(friday_2200) is True

    def test_is_weekend_monday(self):
        monday_10 = datetime.datetime(2026, 3, 16, 10, 0, tzinfo=datetime.timezone.utc)
        assert is_weekend(monday_10) is False

    def test_current_session_london(self):
        london_time = datetime.datetime(2026, 3, 16, 9, 0, tzinfo=datetime.timezone.utc)
        session = current_session(london_time)
        assert session == "london"

    def test_current_session_overlap(self):
        overlap_time = datetime.datetime(2026, 3, 16, 14, 0, tzinfo=datetime.timezone.utc)
        session = current_session(overlap_time)
        assert session == "overlap_london_ny"

    def test_current_session_asia(self):
        asia_time = datetime.datetime(2026, 3, 16, 2, 0, tzinfo=datetime.timezone.utc)
        session = current_session(asia_time)
        assert session == "asia"


class TestSessionFilter:
    def test_default_allows_trade(self):
        sf = SessionFilter()
        allowed, scale, reason = sf.should_trade()
        # Depends on current time — just check the contract
        assert isinstance(allowed, bool)
        assert isinstance(scale, float)
        assert isinstance(reason, str)

    def test_weekend_blocking(self):
        sf = SessionFilter(block_weekends=True)
        friday_2200 = datetime.datetime(2026, 3, 13, 22, 0, tzinfo=datetime.timezone.utc)
        allowed, _, reason = sf.should_trade(utc_now=friday_2200)
        assert allowed is False
        assert "weekend" in reason.lower()

    def test_session_restriction(self):
        sf = SessionFilter(allowed_sessions=["london", "overlap_london_ny"])
        asia_time = datetime.datetime(2026, 3, 16, 2, 0, tzinfo=datetime.timezone.utc)
        allowed, _, reason = sf.should_trade(utc_now=asia_time)
        assert allowed is False
