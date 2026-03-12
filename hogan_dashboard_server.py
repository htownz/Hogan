"""Hogan Animated Dashboard — lightweight local server.

Serves the animated HTML dashboard and exposes JSON API endpoints
that read from the live SQLite database.

Launch:
    python hogan_dashboard_server.py
    # opens http://localhost:8777
"""
from __future__ import annotations

import json
import sqlite3
import webbrowser
from http.server import HTTPServer, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from pathlib import Path
from urllib.parse import urlparse, parse_qs

DB_PATH = "data/hogan.db"
PORT = 8777
DASHBOARD_HTML = Path(__file__).parent / "hogan_dashboard.html"


def _conn():
    c = sqlite3.connect(DB_PATH, check_same_thread=False)
    c.execute("PRAGMA journal_mode=WAL")
    c.execute("PRAGMA query_only=ON")
    c.row_factory = sqlite3.Row
    return c


def _json_rows(rows):
    return [dict(r) for r in rows]


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

    def _serve_html(self):
        content = DASHBOARD_HTML.read_bytes()
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", len(content))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(content)

    def _serve_api(self, path, qs):
        try:
            conn = _conn()
            data = {}

            if path == "/api/equity":
                rows = conn.execute(
                    "SELECT ts_ms, equity_usd, drawdown FROM equity_snapshots "
                    "ORDER BY ts_ms DESC LIMIT 500"
                ).fetchall()
                data = _json_rows(rows)

            elif path == "/api/trades":
                rows = conn.execute(
                    "SELECT symbol, side, entry_price, exit_price, qty, "
                    "realized_pnl, pnl_pct, open_ts_ms, close_ts_ms, close_reason "
                    "FROM paper_trades ORDER BY open_ts_ms DESC LIMIT 50"
                ).fetchall()
                data = _json_rows(rows)

            elif path == "/api/decisions":
                rows = conn.execute(
                    "SELECT ts_ms, symbol, regime, tech_action, tech_confidence, "
                    "sent_bias, sent_strength, macro_regime, macro_risk_on, "
                    "meta_weights_json, forecast_4h, forecast_12h, forecast_24h, "
                    "risk_vol_pct, final_action, final_confidence, position_size, "
                    "ml_up_prob, explanation "
                    "FROM decision_log WHERE final_action != 'shadow_weight_update' "
                    "ORDER BY ts_ms DESC LIMIT 100"
                ).fetchall()
                data = _json_rows(rows)

            elif path == "/api/summary":
                eq = conn.execute(
                    "SELECT equity_usd, drawdown FROM equity_snapshots ORDER BY ts_ms DESC LIMIT 1"
                ).fetchone()
                trades = conn.execute(
                    "SELECT COUNT(*) as total, "
                    "SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins, "
                    "SUM(CASE WHEN realized_pnl <= 0 THEN 1 ELSE 0 END) as losses, "
                    "SUM(realized_pnl) as total_pnl "
                    "FROM paper_trades WHERE exit_price IS NOT NULL"
                ).fetchone()
                open_trades = conn.execute(
                    "SELECT COUNT(*) as n FROM paper_trades WHERE exit_price IS NULL"
                ).fetchone()
                last_decision = conn.execute(
                    "SELECT ts_ms, regime, final_action, final_confidence, "
                    "tech_action, sent_bias, macro_regime, forecast_4h, forecast_12h, "
                    "forecast_24h, ml_up_prob, meta_weights_json "
                    "FROM decision_log WHERE final_action != 'shadow_weight_update' "
                    "ORDER BY ts_ms DESC LIMIT 1"
                ).fetchone()
                shadow = conn.execute(
                    "SELECT explanation FROM decision_log "
                    "WHERE final_action = 'shadow_weight_update' "
                    "ORDER BY ts_ms DESC LIMIT 1"
                ).fetchone()
                data = {
                    "equity": dict(eq) if eq else {"equity_usd": 0, "drawdown": 0},
                    "trades": dict(trades) if trades else {},
                    "open_positions": dict(open_trades)["n"] if open_trades else 0,
                    "last_decision": dict(last_decision) if last_decision else {},
                    "shadow_weights": json.loads(shadow["explanation"]) if shadow else {},
                }

            elif path == "/api/candles":
                sym = qs.get("symbol", ["BTC/USD"])[0]
                rows = conn.execute(
                    "SELECT ts_ms, open, high, low, close, volume FROM candles "
                    "WHERE symbol=? ORDER BY ts_ms DESC LIMIT 200",
                    (sym,)
                ).fetchall()
                data = _json_rows(rows)

            conn.close()
            self._send_json(data)

        except Exception as exc:
            self._send_json({"error": str(exc)}, status=500)

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
