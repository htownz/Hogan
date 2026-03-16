
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

# Phase 4 — online / continuous learning
FEATURE_DRIFT = Counter("hogan_feature_drift_total", "Feature distribution drift detections", ["symbol"])
ONLINE_UPDATES = Counter("hogan_online_updates_total", "Online model partial_fit calls", ["model_name"])
MODEL_PROMOTED = Counter("hogan_model_promoted_total", "Challenger promotions to champion", ["symbol"])

# Phase 7 — production watchdog
DEAD_MAN_ALERTS = Counter("hogan_dead_man_alerts_total", "Dead-man switch alerts fired")
SIGNAL_QUALITY = Gauge("hogan_signal_quality", "Fraction of recent signals that are non-hold", ["symbol"])


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
