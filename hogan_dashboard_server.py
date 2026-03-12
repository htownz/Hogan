"""Hogan Animated Dashboard — lightweight local server.

Serves the animated HTML dashboard and exposes JSON API endpoints
that read from the live SQLite database.  Supports SSE for real-time push.

Launch:
    python hogan_dashboard_server.py
    # opens http://localhost:8777
"""
from __future__ import annotations

import csv
import io
import json
import sqlite3
import threading
import time
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path
from urllib.parse import urlparse, parse_qs

DB_PATH = "data/hogan.db"
PORT = 8777
DASHBOARD_HTML = Path(__file__).parent / "hogan_dashboard.html"

_last_decision_ts = 0
_last_trade_ts = 0
_lock = threading.Lock()


def _conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False, timeout=5)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA query_only=ON")
    c.row_factory = sqlite3.Row
    return c


def _json_rows(rows):
    return [dict(r) for r in rows]


def _safe_json_loads(s):
    if not s:
        return {}
    try:
        return json.loads(s)
    except Exception:
        return {}


class DashboardHandler(SimpleHTTPRequestHandler):

    def do_GET(self):
        parsed = urlparse(self.path)
        path = parsed.path

        if path == "/" or path == "/index.html":
            self._serve_html()
        elif path.startswith("/api/"):
            self._serve_api(path, parse_qs(parsed.query))
        else:
            self.send_error(404)

    # ── HTML ──────────────────────────────────────────────────────
    def _serve_html(self):
        content = DASHBOARD_HTML.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(content))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content)

    # ── API ROUTER ────────────────────────────────────────────────
    def _serve_api(self, path, qs):
        try:
            conn = _conn()
            data = {}

            if path == "/api/equity":
                hours = int(qs.get("hours", ["0"])[0])
                limit = int(qs.get("limit", ["2000"])[0])
                sql = "SELECT ts_ms, equity_usd, drawdown FROM equity_snapshots "
                params: list = []
                if hours > 0:
                    cutoff = int((time.time() - hours * 3600) * 1000)
                    sql += "WHERE ts_ms >= ? "
                    params.append(cutoff)
                sql += f"ORDER BY ts_ms DESC LIMIT {limit}"
                rows = conn.execute(sql, params).fetchall()
                data = _json_rows(rows)

            elif path == "/api/trades":
                rows = conn.execute(
                    "SELECT trade_id, symbol, side, entry_price, exit_price, qty, "
                    "realized_pnl, pnl_pct, entry_fee, exit_fee, "
                    "open_ts_ms, close_ts_ms, close_reason, "
                    "ml_up_prob, strategy_conf, vol_ratio, entry_decision_id "
                    "FROM paper_trades ORDER BY open_ts_ms DESC LIMIT 100"
                ).fetchall()
                data = _json_rows(rows)

            elif path == "/api/open_positions":
                rows = conn.execute(
                    "SELECT trade_id, symbol, side, entry_price, qty, "
                    "open_ts_ms, ml_up_prob, strategy_conf, entry_decision_id "
                    "FROM paper_trades WHERE exit_price IS NULL "
                    "ORDER BY open_ts_ms DESC"
                ).fetchall()
                data = _json_rows(rows)

            elif path == "/api/decisions":
                limit = int(qs.get("limit", ["100"])[0])
                rows = conn.execute(
                    "SELECT id, ts_ms, symbol, regime, "
                    "tech_action, tech_confidence, "
                    "sent_bias, sent_strength, "
                    "macro_regime, macro_risk_on, "
                    "meta_weights_json, "
                    "forecast_4h, forecast_12h, forecast_24h, forecast_conf, "
                    "risk_vol_pct, risk_mae_pct, risk_stop_hit, risk_regime, risk_pos_scale, "
                    "freshness_json, "
                    "final_action, final_confidence, position_size, "
                    "ml_up_prob, conf_scale, explanation, "
                    "realized_pnl, outcome_ts_ms, linked_trade_id "
                    "FROM decision_log WHERE final_action != 'shadow_weight_update' "
                    f"ORDER BY ts_ms DESC LIMIT {limit}"
                ).fetchall()
                data = _json_rows(rows)

            elif path == "/api/summary":
                eq = conn.execute(
                    "SELECT ts_ms, equity_usd, drawdown FROM equity_snapshots ORDER BY ts_ms DESC LIMIT 1"
                ).fetchone()
                trades = conn.execute(
                    "SELECT COUNT(*) as total, "
                    "SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins, "
                    "SUM(CASE WHEN realized_pnl <= 0 THEN 1 ELSE 0 END) as losses, "
                    "SUM(realized_pnl) as total_pnl, "
                    "AVG(realized_pnl) as avg_pnl, "
                    "MIN(realized_pnl) as worst_trade, "
                    "MAX(realized_pnl) as best_trade "
                    "FROM paper_trades WHERE exit_price IS NOT NULL"
                ).fetchone()
                open_trades = conn.execute(
                    "SELECT COUNT(*) as n FROM paper_trades WHERE exit_price IS NULL"
                ).fetchone()
                last_dec = conn.execute(
                    "SELECT id, ts_ms, regime, final_action, final_confidence, "
                    "tech_action, tech_confidence, sent_bias, sent_strength, "
                    "macro_regime, macro_risk_on, "
                    "forecast_4h, forecast_12h, forecast_24h, "
                    "risk_vol_pct, risk_mae_pct, risk_stop_hit, "
                    "ml_up_prob, meta_weights_json, freshness_json, "
                    "explanation, conf_scale, position_size "
                    "FROM decision_log WHERE final_action != 'shadow_weight_update' "
                    "ORDER BY ts_ms DESC LIMIT 1"
                ).fetchone()
                shadow = conn.execute(
                    "SELECT explanation FROM decision_log "
                    "WHERE final_action = 'shadow_weight_update' "
                    "ORDER BY ts_ms DESC LIMIT 1"
                ).fetchone()
                recent_regimes = conn.execute(
                    "SELECT ts_ms, regime FROM decision_log "
                    "WHERE final_action != 'shadow_weight_update' AND regime IS NOT NULL "
                    "ORDER BY ts_ms DESC LIMIT 50"
                ).fetchall()
                last_closed = conn.execute(
                    "SELECT realized_pnl, pnl_pct, symbol, side, close_ts_ms "
                    "FROM paper_trades WHERE exit_price IS NOT NULL "
                    "ORDER BY close_ts_ms DESC LIMIT 1"
                ).fetchone()
                streak = self._compute_streak(conn)

                open_detail = conn.execute(
                    "SELECT trade_id, symbol, side, entry_price, qty, "
                    "open_ts_ms, ml_up_prob, strategy_conf "
                    "FROM paper_trades WHERE exit_price IS NULL "
                    "ORDER BY open_ts_ms DESC"
                ).fetchall()

                now_ms = int(time.time() * 1000)
                eq_ts_ms = eq["ts_ms"] if eq else 0
                dec_ts_ms = last_dec["ts_ms"] if last_dec else 0
                data = {
                    "equity": dict(eq) if eq else {"ts_ms": 0, "equity_usd": 0, "drawdown": 0},
                    "trades": dict(trades) if trades else {},
                    "open_positions": dict(open_trades)["n"] if open_trades else 0,
                    "open_positions_detail": _json_rows(open_detail),
                    "last_decision": dict(last_dec) if last_dec else {},
                    "shadow_weights": _safe_json_loads(shadow["explanation"]) if shadow else {},
                    "regime_history": _json_rows(recent_regimes),
                    "last_closed_trade": dict(last_closed) if last_closed else None,
                    "streak": streak,
                    "health": {
                        "server_ts": now_ms,
                        "equity_age_s": (now_ms - eq_ts_ms) / 1000,
                        "decision_age_s": (now_ms - dec_ts_ms) / 1000,
                    },
                }

            elif path == "/api/candles":
                sym = qs.get("symbol", ["BTC/USD"])[0]
                limit = int(qs.get("limit", ["200"])[0])
                rows = conn.execute(
                    "SELECT ts_ms, open, high, low, close, volume FROM candles "
                    "WHERE symbol=? ORDER BY ts_ms DESC LIMIT ?",
                    (sym, limit)
                ).fetchall()
                data = _json_rows(rows)

            elif path == "/api/health":
                eq_ts = conn.execute(
                    "SELECT ts_ms FROM equity_snapshots ORDER BY ts_ms DESC LIMIT 1"
                ).fetchone()
                dec_ts = conn.execute(
                    "SELECT ts_ms FROM decision_log ORDER BY ts_ms DESC LIMIT 1"
                ).fetchone()
                now = int(time.time() * 1000)
                data = {
                    "server_ts": now,
                    "last_equity_ts": eq_ts["ts_ms"] if eq_ts else 0,
                    "last_decision_ts": dec_ts["ts_ms"] if dec_ts else 0,
                    "equity_age_s": (now - (eq_ts["ts_ms"] if eq_ts else 0)) / 1000,
                    "decision_age_s": (now - (dec_ts["ts_ms"] if dec_ts else 0)) / 1000,
                }

            elif path == "/api/events":
                self._serve_sse(conn)
                return

            elif path == "/api/export/trades":
                self._serve_csv_export(conn, "trades")
                conn.close()
                return

            elif path == "/api/export/decisions":
                self._serve_csv_export(conn, "decisions")
                conn.close()
                return

            elif path == "/api/export/equity":
                self._serve_csv_export(conn, "equity")
                conn.close()
                return

            conn.close()
            self._send_json(data)

        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

    # ── HELPERS ───────────────────────────────────────────────────

    @staticmethod
    def _compute_streak(conn) -> dict:
        rows = conn.execute(
            "SELECT realized_pnl FROM paper_trades "
            "WHERE exit_price IS NOT NULL ORDER BY close_ts_ms DESC LIMIT 20"
        ).fetchall()
        if not rows:
            return {"type": "none", "count": 0}
        first_sign = rows[0]["realized_pnl"] >= 0
        count = 0
        for r in rows:
            if (r["realized_pnl"] >= 0) == first_sign:
                count += 1
            else:
                break
        return {"type": "win" if first_sign else "loss", "count": count}

    def _serve_sse(self, conn):
        """Server-Sent Events stream for real-time updates."""
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "keep-alive")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()

        global _last_decision_ts, _last_trade_ts
        try:
            while True:
                c2 = _conn()
                dec = c2.execute("SELECT ts_ms FROM decision_log ORDER BY ts_ms DESC LIMIT 1").fetchone()
                tr = c2.execute("SELECT close_ts_ms FROM paper_trades WHERE exit_price IS NOT NULL ORDER BY close_ts_ms DESC LIMIT 1").fetchone()
                op = c2.execute("SELECT open_ts_ms FROM paper_trades WHERE exit_price IS NULL ORDER BY open_ts_ms DESC LIMIT 1").fetchone()
                c2.close()

                events = []
                dec_ts = dec["ts_ms"] if dec else 0
                tr_ts = tr["close_ts_ms"] if tr else 0
                op_ts = op["open_ts_ms"] if op else 0

                with _lock:
                    if dec_ts > _last_decision_ts:
                        _last_decision_ts = dec_ts
                        events.append("decision")
                    trade_ts = max(tr_ts, op_ts)
                    if trade_ts > _last_trade_ts:
                        _last_trade_ts = trade_ts
                        events.append("trade")

                if events:
                    msg = json.dumps({"events": events, "ts": int(time.time() * 1000)})
                    self.wfile.write(f"data: {msg}\n\n".encode())
                    self.wfile.flush()
                else:
                    self.wfile.write(": keepalive\n\n".encode())
                    self.wfile.flush()

                time.sleep(3)
        except (BrokenPipeError, ConnectionError, OSError):
            pass

    def _serve_csv_export(self, conn, table: str):
        buf = io.StringIO()
        writer = csv.writer(buf)
        if table == "trades":
            rows = conn.execute(
                "SELECT trade_id, symbol, side, entry_price, exit_price, qty, "
                "realized_pnl, pnl_pct, open_ts_ms, close_ts_ms, close_reason "
                "FROM paper_trades ORDER BY open_ts_ms DESC"
            ).fetchall()
        elif table == "decisions":
            rows = conn.execute(
                "SELECT id, ts_ms, symbol, regime, tech_action, sent_bias, "
                "macro_regime, forecast_4h, forecast_12h, forecast_24h, "
                "final_action, final_confidence, ml_up_prob, explanation "
                "FROM decision_log ORDER BY ts_ms DESC"
            ).fetchall()
        else:
            rows = conn.execute(
                "SELECT ts_ms, equity_usd, drawdown FROM equity_snapshots ORDER BY ts_ms DESC"
            ).fetchall()

        if rows:
            writer.writerow(rows[0].keys())
            for r in rows:
                writer.writerow(dict(r).values())

        body = buf.getvalue().encode()
        self.send_response(200)
        self.send_header("Content-Type", "text/csv")
        self.send_header("Content-Disposition", f"attachment; filename=hogan_{table}.csv")
        self.send_header("Content-Length", len(body))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, data, status=200):
        body = json.dumps(data, default=str).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", len(body))
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, fmt, *args):
        if "/api/" not in str(args[0]):
            super().log_message(fmt, *args)


class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True


def main():
    if not DASHBOARD_HTML.exists():
        print(f"ERROR: {DASHBOARD_HTML} not found")
        return
    server = ThreadedHTTPServer(("127.0.0.1", PORT), DashboardHandler)
    url = f"http://localhost:{PORT}"
    print(f"\n  HOGAN DASHBOARD running at {url}\n")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down.")
        server.shutdown()


if __name__ == "__main__":
    main()
