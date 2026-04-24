"""Tests for hogan_bot.notifier."""
from __future__ import annotations

import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from threading import Thread

import pytest

from hogan_bot.notifier import (
    DiscordNotifier,
    NullNotifier,
    WebhookNotifier,
    _redact_webhook_url,
    make_notifier,
)

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


# ---------------------------------------------------------------------------
# Webhook URL redaction — secrets must never hit the log stream
# ---------------------------------------------------------------------------


_SECRET_TOKEN = "S3CR3T_TOKEN_NEVER_LOG_ME_ABCDEF123456"


class TestWebhookRedaction:
    """Direct unit tests for the redaction helper."""

    def test_discord_url_hides_token_keeps_id(self):
        url = f"https://discord.com/api/webhooks/1234567890/{_SECRET_TOKEN}"
        redacted = _redact_webhook_url(url)
        assert _SECRET_TOKEN not in redacted
        assert "1234567890" in redacted
        assert redacted == "https://discord.com/api/webhooks/1234567890/***"

    def test_generic_webhook_hides_path(self):
        url = f"https://hooks.slack.com/services/T000/B000/{_SECRET_TOKEN}"
        redacted = _redact_webhook_url(url)
        assert _SECRET_TOKEN not in redacted
        assert redacted == "https://hooks.slack.com/***"

    def test_empty_url_is_empty(self):
        assert _redact_webhook_url("") == ""

    def test_malformed_url_never_raises(self):
        for bad in ("not a url", "://broken", "://"):
            out = _redact_webhook_url(bad)
            assert _SECRET_TOKEN not in out


class TestWebhookLogRedaction:
    """End-to-end: logger output must not contain the secret token."""

    def test_discord_failure_log_redacts_token(self, caplog):
        # Use a Discord-shaped URL pointing at an unreachable port so the
        # transport raises URLError — this is the branch that previously
        # logged ``self.webhook_url`` verbatim and leaked the token.
        url = f"http://127.0.0.1:1/api/webhooks/1234567890/{_SECRET_TOKEN}"
        n = DiscordNotifier(url, timeout_seconds=1)
        with caplog.at_level(logging.WARNING, logger="hogan_bot.notifier"):
            n.notify("error", {"msg": "probe"})
        combined = "\n".join(rec.getMessage() for rec in caplog.records)
        assert _SECRET_TOKEN not in combined
        assert "1234567890" in combined  # non-secret id still useful for ops

    def test_generic_webhook_failure_log_redacts_token(self, caplog):
        url = f"https://hooks.example.com/services/{_SECRET_TOKEN}"
        n = WebhookNotifier(url, timeout_seconds=1)
        with caplog.at_level(logging.WARNING, logger="hogan_bot.notifier"):
            n.notify("ping", {})
        combined = "\n".join(rec.getMessage() for rec in caplog.records)
        assert _SECRET_TOKEN not in combined
