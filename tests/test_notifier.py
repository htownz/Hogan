"""Tests for hogan_bot.notifier."""
from __future__ import annotations

import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import pytest

from hogan_bot.notifier import NullNotifier, WebhookNotifier, make_notifier

# ---------------------------------------------------------------------------
# NullNotifier
# ---------------------------------------------------------------------------


class TestNullNotifier:
    def test_notify_returns_none(self):
        n = NullNotifier()
        assert n.notify("any_event", {"key": "value"}) is None

    def test_notify_accepts_empty_payload(self):
        n = NullNotifier()
        n.notify("ping", {})  # must not raise


# ---------------------------------------------------------------------------
# make_notifier factory
# ---------------------------------------------------------------------------


class TestMakeNotifier:
    def test_returns_null_for_none(self):
        assert isinstance(make_notifier(None), NullNotifier)

    def test_returns_null_for_empty_string(self):
        assert isinstance(make_notifier(""), NullNotifier)

    def test_returns_webhook_for_url(self):
        assert isinstance(make_notifier(webhook_url="https://hooks.example.com/xyz"), WebhookNotifier)

    def test_webhook_url_stored(self):
        url = "https://hooks.example.com/xyz"
        n = make_notifier(webhook_url=url)
        assert n.url == url  # type: ignore[union-attr]

    def test_returns_discord_for_discord_url(self):
        from hogan_bot.notifier import DiscordNotifier
        url = "https://discord.com/api/webhooks/123/abc"
        assert isinstance(make_notifier(discord_webhook_url=url), DiscordNotifier)


# ---------------------------------------------------------------------------
# WebhookNotifier — integration with a real local HTTP server
# ---------------------------------------------------------------------------

_received: list[dict] = []


class _Handler(BaseHTTPRequestHandler):
    def do_POST(self):  # noqa: N802
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        _received.append(json.loads(body))
        self.send_response(200)
        self.end_headers()

    def log_message(self, fmt, *args):  # silence server logs during tests
        pass


@pytest.fixture()
def local_server():
    """Start a throwaway local HTTP server for one test, then shut it down."""
    _received.clear()
    server = HTTPServer(("127.0.0.1", 0), _Handler)
    port = server.server_address[1]
    thread = Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{port}"
    server.shutdown()


class TestWebhookNotifier:
    def test_posts_json_to_endpoint(self, local_server):
        n = WebhookNotifier(local_server)
        n.notify("buy", {"symbol": "BTC/USD", "price": 60000.0})
        assert len(_received) == 1
        payload = _received[0]
        assert payload["event"] == "buy"
        assert payload["symbol"] == "BTC/USD"
        assert payload["price"] == 60000.0

    def test_includes_timestamp(self, local_server):
        n = WebhookNotifier(local_server)
        n.notify("test", {})
        assert "timestamp" in _received[0]

    def test_bad_url_does_not_raise(self):
        n = WebhookNotifier("http://127.0.0.1:1", timeout_seconds=1)
        # Connection refused — must be silently swallowed
        n.notify("ping", {})  # must not raise

    def test_multiple_events(self, local_server):
        n = WebhookNotifier(local_server)
        n.notify("buy", {"price": 1})
        n.notify("sell", {"price": 2})
        assert len(_received) == 2
        assert _received[0]["event"] == "buy"
        assert _received[1]["event"] == "sell"
