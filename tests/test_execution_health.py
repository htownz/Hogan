"""Tests for hogan_bot.execution_health module."""
from __future__ import annotations

import time

from hogan_bot.execution_health import (
    ExecutionHealthState,
    OrderOutcome,
    record_exec_outcome,
)


def _ok_outcome(symbol: str = "BTC/USD", side: str = "open_long", dp: float = 100.0, fp: float = 100.05) -> OrderOutcome:
    return OrderOutcome(ts=time.time(), symbol=symbol, side=side, decision_price=dp, fill_price=fp, ok=True)


def _fail_outcome(symbol: str = "BTC/USD", side: str = "open_long", dp: float = 100.0) -> OrderOutcome:
    return OrderOutcome(ts=time.time(), symbol=symbol, side=side, decision_price=dp, fill_price=None, ok=False, error="test_error")


class TestOrderOutcome:
    def test_slippage_bps_calculation(self) -> None:
        o = OrderOutcome(ts=0, symbol="X", side="open_long", decision_price=100.0, fill_price=100.50, ok=True)
        assert o.slippage_bps is not None
        assert abs(o.slippage_bps - 50.0) < 0.01

    def test_slippage_none_when_no_fill(self) -> None:
        o = OrderOutcome(ts=0, symbol="X", side="open_long", decision_price=100.0, fill_price=None, ok=False)
        assert o.slippage_bps is None

    def test_slippage_none_when_zero_decision(self) -> None:
        o = OrderOutcome(ts=0, symbol="X", side="open_long", decision_price=0.0, fill_price=50.0, ok=True)
        assert o.slippage_bps is None


class TestExecutionHealthState:
    def test_empty_snapshot(self) -> None:
        h = ExecutionHealthState()
        snap = h.health_snapshot()
        assert snap["orders_total"] == 0
        assert snap["fill_rate"] == 1.0
        assert snap["alerts"] == []

    def test_records_success(self) -> None:
        h = ExecutionHealthState()
        h.record_order(_ok_outcome())
        snap = h.health_snapshot()
        assert snap["orders_total"] == 1
        assert snap["orders_ok"] == 1
        assert snap["orders_failed"] == 0

    def test_records_failure(self) -> None:
        h = ExecutionHealthState()
        h.record_order(_fail_outcome())
        snap = h.health_snapshot()
        assert snap["orders_failed"] == 1
        assert snap["consecutive_failures"] == 1

    def test_consecutive_failures_reset(self) -> None:
        h = ExecutionHealthState()
        for _ in range(3):
            h.record_order(_fail_outcome())
        assert h.consecutive_order_failures == 3
        h.record_order(_ok_outcome())
        assert h.consecutive_order_failures == 0

    def test_order_failure_circuit(self) -> None:
        h = ExecutionHealthState()
        for _ in range(5):
            h.record_order(_fail_outcome())
        assert h.order_failure_circuit_open is True
        snap = h.health_snapshot()
        assert snap["order_failure_circuit_open"] is True
        alerts = [a for a in snap["alerts"] if a["code"] == "order_failure_circuit"]
        assert len(alerts) == 1
        assert alerts[0]["level"] == "critical"

    def test_slippage_stats(self) -> None:
        h = ExecutionHealthState()
        h.record_order(OrderOutcome(ts=time.time(), symbol="X", side="open_long",
                                    decision_price=100.0, fill_price=100.10, ok=True))
        h.record_order(OrderOutcome(ts=time.time(), symbol="X", side="open_long",
                                    decision_price=100.0, fill_price=100.30, ok=True))
        snap = h.health_snapshot()
        assert snap["avg_slippage_bps"] > 0
        assert snap["max_slippage_bps"] >= snap["avg_slippage_bps"]

    def test_dead_man_alert(self) -> None:
        h = ExecutionHealthState()
        h.dead_man_triggered = True
        snap = h.health_snapshot()
        alerts = [a for a in snap["alerts"] if a["code"] == "dead_man"]
        assert len(alerts) == 1

    def test_rest_fallback_alert(self) -> None:
        h = ExecutionHealthState()
        h.rest_fallback_active = True
        snap = h.health_snapshot()
        alerts = [a for a in snap["alerts"] if a["code"] == "rest_fallback"]
        assert len(alerts) == 1
        assert alerts[0]["level"] == "info"

    def test_drawdown_circuit_alert(self) -> None:
        h = ExecutionHealthState()
        h.drawdown_circuit_open = True
        snap = h.health_snapshot()
        alerts = [a for a in snap["alerts"] if a["code"] == "drawdown_circuit"]
        assert len(alerts) == 1
        assert alerts[0]["level"] == "critical"

    def test_gap_guard_alert(self) -> None:
        h = ExecutionHealthState()
        h.gap_guard_active = True
        snap = h.health_snapshot()
        alerts = [a for a in snap["alerts"] if a["code"] == "gap_guard"]
        assert len(alerts) == 1

    def test_stale_data_alert(self) -> None:
        h = ExecutionHealthState()
        h.last_candle_ts = time.time() - 600
        snap = h.health_snapshot()
        alerts = [a for a in snap["alerts"] if a["code"] == "stale_data"]
        assert len(alerts) == 1

    def test_high_failure_rate_alert(self) -> None:
        h = ExecutionHealthState()
        h.record_order(_fail_outcome())
        h.record_order(_fail_outcome())
        h.record_order(_ok_outcome())
        snap = h.health_snapshot()
        alerts = [a for a in snap["alerts"] if a["code"] == "high_failure_rate"]
        assert len(alerts) == 1

    def test_ws_reconnect_tracking(self) -> None:
        h = ExecutionHealthState()
        h.record_ws_reconnect()
        h.record_ws_reconnect()
        assert h.ws_reconnect_count == 2
        assert h.last_ws_reconnect_ts > 0

    def test_candle_received_clears_dead_man(self) -> None:
        h = ExecutionHealthState()
        h.dead_man_triggered = True
        h.record_candle_received()
        assert h.dead_man_triggered is False
        assert h.last_candle_ts > 0

    def test_prune_old_orders(self) -> None:
        h = ExecutionHealthState(window_seconds=60.0)
        old = OrderOutcome(ts=time.time() - 120, symbol="X", side="open_long",
                           decision_price=100.0, fill_price=100.0, ok=True)
        h._orders.append(old)
        h.record_order(_ok_outcome())
        assert len(h._orders) == 1


class TestRecordExecOutcome:
    def test_convenience_wrapper_ok(self) -> None:
        h = ExecutionHealthState()

        class FakeResult:
            ok = True
            error = None

        record_exec_outcome(h, symbol="BTC/USD", side="open_long",
                            decision_price=100.0, result=FakeResult())
        assert h.health_snapshot()["orders_total"] == 1
        assert h.health_snapshot()["orders_ok"] == 1

    def test_convenience_wrapper_fail(self) -> None:
        h = ExecutionHealthState()

        class FakeResult:
            ok = False
            error = "exchange_down"

        record_exec_outcome(h, symbol="BTC/USD", side="open_long",
                            decision_price=100.0, result=FakeResult())
        assert h.health_snapshot()["orders_failed"] == 1

    def test_swallows_exceptions(self) -> None:
        h = ExecutionHealthState()
        record_exec_outcome(h, symbol="BTC/USD", side="open_long",
                            decision_price=100.0, result=None)


class TestRunbookExists:
    def test_execution_runbook_doc(self) -> None:
        import pathlib
        p = pathlib.Path("docs/EXECUTION_RUNBOOK.md")
        assert p.exists(), "docs/EXECUTION_RUNBOOK.md should exist"
