
from __future__ import annotations

import time
from dataclasses import dataclass

from prometheus_client import Counter, Gauge, Histogram, start_http_server

# Core metrics
LOOP_SECONDS = Histogram("hogan_loop_seconds", "Main loop duration (seconds)")
EQUITY = Gauge("hogan_equity", "Account equity (quote currency)")
CASH = Gauge("hogan_cash", "Account cash (quote currency)")
DRAWDOWN = Gauge("hogan_drawdown", "Drawdown fraction [0,1]")

ORDERS = Counter("hogan_orders_total", "Total orders attempted", ["side", "mode", "exchange"])
ORDER_FAILS = Counter("hogan_order_fail_total", "Total order failures", ["side", "mode", "exchange"])
FILLS = Counter("hogan_fills_total", "Total fills recorded", ["exchange"])
SLIPPAGE_BPS = Histogram("hogan_slippage_bps", "Slippage in bps vs decision price", buckets=(0, 1, 2, 5, 10, 25, 50, 100, 250, 500))
STALE_DATA = Counter("hogan_stale_data_total", "Stale data detections", ["symbol"])
EXCEPTIONS = Counter("hogan_exceptions_total", "Unhandled exceptions in loop")

# Phase 2 — event-driven data engine
WS_RECONNECTS = Counter("hogan_ws_reconnects_total", "WebSocket reconnection attempts", ["symbol"])
CANDLES_RECEIVED = Counter("hogan_candles_received_total", "Candles received via WS/REST", ["symbol", "timeframe"])
DATA_LAG_SECONDS = Gauge("hogan_data_lag_seconds", "Seconds since last candle received", ["symbol"])

# Execution health — data feed
REST_FALLBACK_ACTIVE = Gauge("hogan_rest_fallback_active", "1 when WebSocket is down and REST polling is active")
ORDER_FAILURE_CIRCUIT = Gauge("hogan_order_failure_circuit_open", "1 when consecutive order failures exceed threshold")
GAP_GUARD_ACTIVE = Gauge("hogan_gap_guard_active", "1 when GAP_GUARD is blocking trades due to stale candles")

# Phase 4 — online / continuous learning
FEATURE_DRIFT = Counter("hogan_feature_drift_total", "Feature distribution drift detections", ["symbol"])
ONLINE_UPDATES = Counter("hogan_online_updates_total", "Online model partial_fit calls", ["model_name"])
MODEL_PROMOTED = Counter("hogan_model_promoted_total", "Challenger promotions to champion", ["symbol"])

# Phase 7 — production watchdog
DEAD_MAN_ALERTS = Counter("hogan_dead_man_alerts_total", "Dead-man switch alerts fired")
SIGNAL_QUALITY = Gauge("hogan_signal_quality", "Fraction of recent signals that are non-hold", ["symbol"])

# Decision transparency — all gate block reasons (not just swarm)
BLOCK_REASONS = Counter(
    "hogan_block_reasons_total",
    "Block reasons from policy_core.decide() — counts every gate hit per bar",
    ["reason"],
)
HOLD_NO_REASON = Counter(
    "hogan_hold_no_reason_total",
    "Hold decisions with empty block_reasons — potential logging gap",
)

# Swarm — policy merge layer (after ML + edge/quality/ranging/pullback gates)
SWARM_MERGE_BLOCKS = Counter(
    "hogan_swarm_merge_blocks_total",
    "Extra hold reasons from merge_swarm_with_gated_action (swarm_* tags on block_reasons)",
    ["reason"],
)
SWARM_FINAL_VETO = Counter(
    "hogan_swarm_final_veto_total",
    "Swarm veto forced hold (size scale applied); counted once per bar when vetoed",
    ["swarm_mode"],
)
SWARM_DOMINANT_VETO = Counter(
    "hogan_swarm_dominant_veto_agent_total",
    "Which agent ID was recorded as dominant when a swarm veto fired",
    ["agent", "swarm_mode"],
)


def record_swarm_policy_events(
    *,
    swarm_mode: str,
    swarm_decision: object | None,
    block_reasons: list[str],
) -> None:
    """Increment Prometheus counters for swarm vetoes and gated-merge hold tags.

    Safe to call every bar; failures are swallowed (metrics must never break trading).
    """
    try:
        mode = str(swarm_mode or "unknown")
        if swarm_decision is not None and bool(getattr(swarm_decision, "vetoed", False)):
            SWARM_FINAL_VETO.labels(swarm_mode=mode).inc()
            dom = getattr(swarm_decision, "dominant_veto_agent", None)
            agent = dom if isinstance(dom, str) and dom.strip() else "unknown"
            SWARM_DOMINANT_VETO.labels(agent=agent, swarm_mode=mode).inc()
        for r in block_reasons:
            if isinstance(r, str) and r.startswith("swarm_"):
                SWARM_MERGE_BLOCKS.labels(reason=r).inc()
    except Exception:
        return


@dataclass
class MetricsServer:
    port: int = 8000
    started: bool = False

    def start(self) -> None:
        if self.started:
            return
        start_http_server(self.port)
        self.started = True


class LoopTimer:
    def __enter__(self):
        self.t0 = time.time()
        return self

    def __exit__(self, exc_type, exc, tb):
        LOOP_SECONDS.observe(max(0.0, time.time() - self.t0))
        return False
