"""Data Watchdog — Phase 7.

Monitors data freshness for all candle series and external metrics.  When any
feed goes stale (no new data for >2× its expected interval), it:
  1. Emits a Prometheus ``STALE_DATA`` counter.
  2. Sends a Telegram/webhook alert via the MultiNotifier.
  3. Returns a structured staleness report.

Also provides:
  * ``run_health_server()`` — simple HTTP /health endpoint on port 8080.
  * Dead-man's switch — if no candle arrives for ``DEAD_MAN_MINUTES`` minutes,
    sends an alert and pauses signal generation.

Run standalone::

    python -m hogan_bot.data_watchdog --db data/hogan.db --interval 300

Or import::

    from hogan_bot.data_watchdog import DataWatchdog
    watchdog = DataWatchdog("data/hogan.db")
    report = watchdog.check()
"""
from __future__ import annotations

import argparse
import http.server
import json
import logging
import sqlite3
import threading
import time
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

DEAD_MAN_MINUTES = 15
STALE_MULTIPLIER = 2.0   # consider stale when age > STALE_MULTIPLIER × expected_interval

# Expected intervals in seconds per timeframe
_TIMEFRAME_SECONDS = {
    "1m": 60, "3m": 180, "5m": 300, "15m": 900, "30m": 1800,
    "1h": 3600, "2h": 7200, "4h": 14400, "6h": 21600, "12h": 43200,
    "1d": 86400,
}
# External metrics expected to be updated at most once per day
_EXT_METRIC_INTERVAL_SECONDS = 86400 * 1.5  # allow 1.5 days of slack


class DataWatchdog:
    """Monitors data freshness and fires alerts when feeds go stale."""

    def __init__(
        self,
        db_path: str = "data/hogan.db",
        notifier=None,
        stale_multiplier: float = STALE_MULTIPLIER,
    ) -> None:
        self.db_path = db_path
        self.notifier = notifier
        self.stale_multiplier = stale_multiplier
        self._last_check: dict[str, float] = {}

    def check(self) -> dict[str, Any]:
        """Run a full freshness check. Returns a report dict."""
        now = time.time()
        stale: list[dict] = []
        healthy: list[dict] = []

        try:
            conn = sqlite3.connect(self.db_path)
        except Exception as exc:
            return {"ok": False, "error": str(exc), "timestamp": now}

        try:
            # ── Candle series ──────────────────────────────────────────────
            cur = conn.execute(
                "SELECT symbol, timeframe, MAX(ts_ms) FROM candles GROUP BY symbol, timeframe"
            )
            for symbol, timeframe, max_ts_ms in cur.fetchall():
                if max_ts_ms is None:
                    continue
                interval = _TIMEFRAME_SECONDS.get(timeframe, 3600)
                threshold = interval * self.stale_multiplier
                age_s = now - max_ts_ms / 1000.0
                entry = {
                    "type": "candle",
                    "symbol": symbol,
                    "timeframe": timeframe,
                    "age_seconds": round(age_s, 1),
                    "threshold_seconds": threshold,
                    "latest_ts": datetime.fromtimestamp(
                        max_ts_ms / 1000.0, tz=timezone.utc
                    ).isoformat(),
                }
                if age_s > threshold:
                    entry["status"] = "STALE"
                    stale.append(entry)
                    self._fire_stale_alert(symbol, timeframe, age_s)
                else:
                    entry["status"] = "OK"
                    healthy.append(entry)

            # ── External / on-chain metrics ────────────────────────────────
            cur = conn.execute(
                "SELECT symbol, metric, MAX(date) FROM onchain_metrics GROUP BY symbol, metric"
            )
            for symbol, metric, max_date in cur.fetchall():
                if max_date is None:
                    continue
                try:
                    latest_dt = datetime.strptime(max_date, "%Y-%m-%d").replace(
                        tzinfo=timezone.utc
                    )
                    age_s = now - latest_dt.timestamp()
                except Exception:
                    continue
                entry = {
                    "type": "ext_metric",
                    "symbol": symbol,
                    "metric": metric,
                    "age_seconds": round(age_s, 1),
                    "threshold_seconds": _EXT_METRIC_INTERVAL_SECONDS,
                    "latest_date": max_date,
                }
                if age_s > _EXT_METRIC_INTERVAL_SECONDS:
                    entry["status"] = "STALE"
                    stale.append(entry)
                    self._fire_stale_alert(symbol, metric, age_s)
                else:
                    entry["status"] = "OK"
                    healthy.append(entry)

            # ── Derivatives metrics ────────────────────────────────────────
            cur = conn.execute(
                "SELECT symbol, metric, MAX(ts_ms) FROM derivatives_metrics GROUP BY symbol, metric"
            )
            for symbol, metric, max_ts_ms in cur.fetchall():
                if max_ts_ms is None:
                    continue
                age_s = now - max_ts_ms / 1000.0
                entry = {
                    "type": "derivatives",
                    "symbol": symbol,
                    "metric": metric,
                    "age_seconds": round(age_s, 1),
                    "threshold_seconds": _EXT_METRIC_INTERVAL_SECONDS,
                    "latest_ts": datetime.fromtimestamp(
                        max_ts_ms / 1000.0, tz=timezone.utc
                    ).isoformat(),
                }
                if age_s > _EXT_METRIC_INTERVAL_SECONDS:
                    entry["status"] = "STALE"
                    stale.append(entry)
                    self._fire_stale_alert(symbol, metric, age_s)
                else:
                    entry["status"] = "OK"
                    healthy.append(entry)

        finally:
            conn.close()

        report = {
            "ok": len(stale) == 0,
            "timestamp": now,
            "timestamp_iso": datetime.fromtimestamp(now, tz=timezone.utc).isoformat(),
            "stale_count": len(stale),
            "healthy_count": len(healthy),
            "stale": stale,
            "healthy": healthy,
        }

        if stale:
            logger.warning(
                "DataWatchdog: %d stale feeds detected: %s",
                len(stale),
                [f"{s['symbol']}/{s.get('timeframe', s.get('metric', '?'))}" for s in stale[:5]],
            )
            if self.notifier:
                self.notifier.notify("data_stale", {
                    "stale_count": len(stale),
                    "stale_feeds": [
                        f"{s['symbol']}/{s.get('timeframe', s.get('metric','?'))}"
                        for s in stale[:10]
                    ],
                })
        else:
            logger.info("DataWatchdog: all %d feeds healthy.", len(healthy))

        return report

    def _fire_stale_alert(self, symbol: str, identifier: str, age_s: float) -> None:
        try:
            from hogan_bot.metrics import STALE_DATA
            STALE_DATA.labels(symbol=symbol).inc()
        except Exception:
            pass
        logger.warning(
            "STALE_DATA: %s/%s age=%.0fs (threshold=%.0fs)",
            symbol, identifier, age_s, _EXT_METRIC_INTERVAL_SECONDS,
        )

    def check_dead_man(self, symbol: str, timeframe: str = "1h") -> bool:
        """Return True if the dead-man switch is triggered (no candle for 15min)."""
        try:
            conn = sqlite3.connect(self.db_path)
            cur = conn.execute(
                "SELECT MAX(ts_ms) FROM candles WHERE symbol=? AND timeframe=?",
                (symbol, timeframe),
            )
            row = cur.fetchone()
            conn.close()
            if row[0] is None:
                return True
            age_s = time.time() - row[0] / 1000.0
            triggered = age_s > DEAD_MAN_MINUTES * 60
            if triggered:
                logger.warning(
                    "DEAD_MAN_SWITCH: %s/%s last candle %.0f min ago",
                    symbol, timeframe, age_s / 60.0,
                )
                try:
                    from hogan_bot.metrics import DEAD_MAN_ALERTS
                    DEAD_MAN_ALERTS.inc()
                except Exception:
                    pass
                if self.notifier:
                    self.notifier.notify("dead_man_switch", {
                        "symbol": symbol,
                        "timeframe": timeframe,
                        "age_minutes": round(age_s / 60.0, 1),
                    })
            return triggered
        except Exception as exc:
            logger.error("Dead-man check failed: %s", exc)
            return False


# ---------------------------------------------------------------------------
# Health HTTP server — GET /health returns JSON
# ---------------------------------------------------------------------------
class _HealthHandler(http.server.BaseHTTPRequestHandler):
    watchdog: DataWatchdog = None  # type: ignore[assignment]
    _cache: dict = {}
    _cache_ts: float = 0.0
    _cache_ttl: float = 30.0

    def do_GET(self):
        if self.path not in ("/health", "/health/"):
            self.send_response(404)
            self.end_headers()
            return

        now = time.time()
        if now - self.__class__._cache_ts > self.__class__._cache_ttl:
            try:
                report = self.__class__.watchdog.check()
                self.__class__._cache = report
                self.__class__._cache_ts = now
            except Exception as exc:
                self.__class__._cache = {"ok": False, "error": str(exc)}

        body = json.dumps(self.__class__._cache).encode("utf-8")
        status = 200 if self.__class__._cache.get("ok", False) else 503
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *_):
        pass  # suppress access logs


def run_health_server(
    watchdog: DataWatchdog,
    port: int = 8080,
    cache_ttl: float = 30.0,
) -> threading.Thread:
    """Start the /health HTTP server in a daemon thread.

    Returns the thread (already started).
    """
    _HealthHandler.watchdog = watchdog
    _HealthHandler._cache_ttl = cache_ttl

    server = http.server.HTTPServer(("0.0.0.0", port), _HealthHandler)
    t = threading.Thread(target=server.serve_forever, daemon=True, name="health-server")
    t.start()
    logger.info("Health server running at http://0.0.0.0:%d/health", port)
    return t


# ---------------------------------------------------------------------------
# CLI daemon
# ---------------------------------------------------------------------------
def _main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description="Hogan Data Watchdog")
    p.add_argument("--db", default="data/hogan.db")
    p.add_argument("--interval", type=float, default=300.0,
                   help="Check interval in seconds (default 300)")
    p.add_argument("--once", action="store_true", help="Run once and exit")
    p.add_argument("--health-port", type=int, default=8080,
                   help="Health endpoint port (default 8080)")
    args = p.parse_args()

    from hogan_bot.notifier import make_notifier
    notifier = make_notifier()
    watchdog = DataWatchdog(db_path=args.db, notifier=notifier)

    # Start health server
    run_health_server(watchdog, port=args.health_port)

    while True:
        report = watchdog.check()
        logger.info(
            "Watchdog: stale=%d healthy=%d",
            report.get("stale_count", 0),
            report.get("healthy_count", 0),
        )
        if args.once:
            print(json.dumps(report, indent=2, default=str))
            break
        try:
            time.sleep(args.interval)
        except KeyboardInterrupt:
            break


if __name__ == "__main__":
    _main()
