"""Execution health monitor — bridges the gap between policy and realized fills.

Tracks order outcomes, slippage, fill rates, data-feed state, and circuit-
breaker conditions.  Pure state object that the event loop updates each bar;
exposes a ``health_snapshot()`` for dashboards and operator alerts.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


@dataclass
class OrderOutcome:
    """Single order attempt result."""

    ts: float
    symbol: str
    side: str  # "open_long", "close_long", "open_short", "close_short", "flatten"
    decision_price: float
    fill_price: float | None
    ok: bool
    error: str | None = None

    @property
    def slippage_bps(self) -> float | None:
        if self.fill_price is None or self.decision_price <= 0:
            return None
        return abs(self.fill_price - self.decision_price) / self.decision_price * 10_000


@dataclass
class ExecutionHealthState:
    """Rolling execution health state.  Thread-safe reads, single-writer updates."""

    window_seconds: float = 3600.0

    _orders: list[OrderOutcome] = field(default_factory=list, repr=False)

    # Data feed state
    rest_fallback_active: bool = False
    last_ws_reconnect_ts: float = 0.0
    ws_reconnect_count: int = 0
    last_candle_ts: float = 0.0
    dead_man_triggered: bool = False

    # Circuit breaker
    drawdown_circuit_open: bool = False
    gap_guard_active: bool = False
    consecutive_order_failures: int = 0
    _ORDER_FAILURE_CIRCUIT_THRESHOLD: int = 5

    def record_order(self, outcome: OrderOutcome) -> None:
        """Record an order outcome and update rolling state."""
        self._orders.append(outcome)
        self._prune()

        if outcome.ok:
            self.consecutive_order_failures = 0
        else:
            self.consecutive_order_failures += 1

        self._update_metrics(outcome)

    def record_ws_reconnect(self) -> None:
        self.last_ws_reconnect_ts = time.time()
        self.ws_reconnect_count += 1

    def record_candle_received(self) -> None:
        self.last_candle_ts = time.time()
        self.dead_man_triggered = False

    def _prune(self) -> None:
        cutoff = time.time() - self.window_seconds
        self._orders = [o for o in self._orders if o.ts >= cutoff]

    def _update_metrics(self, outcome: OrderOutcome) -> None:
        """Push to Prometheus counters/histograms.  Never raises."""
        try:
            from hogan_bot.metrics import FILLS, ORDER_FAILS, ORDERS, SLIPPAGE_BPS

            mode = "live" if not outcome.side.startswith("paper") else "paper"
            exchange = "unknown"

            ORDERS.labels(side=outcome.side, mode=mode, exchange=exchange).inc()

            if not outcome.ok:
                ORDER_FAILS.labels(
                    side=outcome.side, mode=mode, exchange=exchange
                ).inc()
            else:
                FILLS.labels(exchange=exchange).inc()
                slip = outcome.slippage_bps
                if slip is not None:
                    SLIPPAGE_BPS.observe(slip)
        except Exception:
            pass

    @property
    def order_failure_circuit_open(self) -> bool:
        return self.consecutive_order_failures >= self._ORDER_FAILURE_CIRCUIT_THRESHOLD

    def health_snapshot(self) -> dict:
        """Current execution health for dashboards and operator alerts."""
        now = time.time()
        recent = self._orders

        total = len(recent)
        failures = sum(1 for o in recent if not o.ok)
        successes = total - failures
        slippages = [o.slippage_bps for o in recent if o.ok and o.slippage_bps is not None]

        avg_slippage = sum(slippages) / len(slippages) if slippages else 0.0
        max_slippage = max(slippages) if slippages else 0.0
        p95_slippage = _percentile(slippages, 95) if len(slippages) >= 5 else max_slippage

        data_age_seconds = now - self.last_candle_ts if self.last_candle_ts > 0 else None

        alerts = self._compute_alerts(
            failures=failures,
            total=total,
            avg_slippage=avg_slippage,
            data_age_seconds=data_age_seconds,
        )

        return {
            "timestamp": now,
            "window_seconds": self.window_seconds,
            "orders_total": total,
            "orders_ok": successes,
            "orders_failed": failures,
            "fill_rate": round(successes / total, 4) if total else 1.0,
            "avg_slippage_bps": round(avg_slippage, 2),
            "max_slippage_bps": round(max_slippage, 2),
            "p95_slippage_bps": round(p95_slippage, 2),
            "consecutive_failures": self.consecutive_order_failures,
            "order_failure_circuit_open": self.order_failure_circuit_open,
            "rest_fallback_active": self.rest_fallback_active,
            "ws_reconnect_count": self.ws_reconnect_count,
            "dead_man_triggered": self.dead_man_triggered,
            "drawdown_circuit_open": self.drawdown_circuit_open,
            "gap_guard_active": self.gap_guard_active,
            "data_age_seconds": round(data_age_seconds, 1) if data_age_seconds is not None else None,
            "alerts": alerts,
        }

    def _compute_alerts(
        self,
        *,
        failures: int,
        total: int,
        avg_slippage: float,
        data_age_seconds: float | None,
    ) -> list[dict]:
        alerts: list[dict] = []

        if self.order_failure_circuit_open:
            alerts.append({
                "level": "critical",
                "code": "order_failure_circuit",
                "message": f"{self.consecutive_order_failures} consecutive order failures — new orders should be paused",
            })

        if total >= 3 and failures / total > 0.5:
            alerts.append({
                "level": "warning",
                "code": "high_failure_rate",
                "message": f"{failures}/{total} orders failed in window ({failures / total:.0%})",
            })

        if avg_slippage > 25.0:
            alerts.append({
                "level": "warning",
                "code": "high_slippage",
                "message": f"Average slippage {avg_slippage:.1f} bps exceeds 25 bps threshold",
            })

        if self.drawdown_circuit_open:
            alerts.append({
                "level": "critical",
                "code": "drawdown_circuit",
                "message": "Drawdown circuit breaker is open — trading halted",
            })

        if self.dead_man_triggered:
            alerts.append({
                "level": "warning",
                "code": "dead_man",
                "message": "No candle data received for >15 minutes",
            })

        if self.rest_fallback_active:
            alerts.append({
                "level": "info",
                "code": "rest_fallback",
                "message": "WebSocket unavailable — using REST polling fallback",
            })

        if data_age_seconds is not None and data_age_seconds > 300:
            alerts.append({
                "level": "warning",
                "code": "stale_data",
                "message": f"Last candle received {data_age_seconds:.0f}s ago (>{300}s threshold)",
            })

        if self.gap_guard_active:
            alerts.append({
                "level": "warning",
                "code": "gap_guard",
                "message": "GAP_GUARD active — candle >4h old, trading skipped",
            })

        return alerts


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    k = (len(s) - 1) * pct / 100.0
    f = int(k)
    c = f + 1
    if c >= len(s):
        return s[-1]
    return s[f] + (k - f) * (s[c] - s[f])


def record_exec_outcome(
    health: ExecutionHealthState,
    *,
    symbol: str,
    side: str,
    decision_price: float,
    result: object,
    fill_price: float | None = None,
) -> None:
    """Convenience wrapper called from event_loop after each _safe_exec.

    Safe to call unconditionally — swallows all exceptions.
    """
    try:
        ok = bool(getattr(result, "ok", False))
        error = getattr(result, "error", None)
        fp = fill_price if fill_price is not None else (decision_price if ok else None)

        outcome = OrderOutcome(
            ts=time.time(),
            symbol=symbol,
            side=side,
            decision_price=decision_price,
            fill_price=fp,
            ok=ok,
            error=str(error) if error else None,
        )
        health.record_order(outcome)
    except Exception:
        pass
