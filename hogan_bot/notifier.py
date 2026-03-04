"""Lightweight event notification for the Hogan trading bot.

Two concrete implementations are provided:

* ``WebhookNotifier`` — fires a JSON POST to any HTTP endpoint (Slack
  incoming-webhook, Discord, Make/Zapier, custom server, …).
* ``NullNotifier`` — silently discards all events; used when no webhook URL
  is configured so callers never need to check for ``None``.

Usage::

    from hogan_bot.notifier import make_notifier
    notifier = make_notifier(webhook_url="https://hooks.slack.com/…")
    notifier.notify("buy", {"symbol": "BTC/USD", "price": 60000})
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
from datetime import datetime, timezone

logger = logging.getLogger(__name__)


class NullNotifier:
    """A no-op notifier used when notifications are disabled."""

    def notify(self, event_type: str, payload: dict) -> None:  # noqa: D102
        pass


class WebhookNotifier:
    """Fire-and-forget JSON webhook notifier.

    The POST body is::

        {
            "event": "<event_type>",
            "timestamp": "2024-01-01T00:00:00+00:00",
            … <payload fields> …
        }

    Network errors are caught and logged as warnings so the bot loop is never
    interrupted by a notification failure.
    """

    def __init__(self, url: str, timeout_seconds: int = 5) -> None:
        self.url = url
        self.timeout_seconds = timeout_seconds

    def notify(self, event_type: str, payload: dict) -> None:
        body = {
            "event": event_type,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            **payload,
        }
        data = json.dumps(body, default=str).encode("utf-8")
        req = urllib.request.Request(
            self.url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds):
                pass
        except urllib.error.URLError as exc:
            logger.warning("Webhook POST failed (%s): %s", self.url, exc)
        except Exception as exc:  # pragma: no cover
            logger.warning("Unexpected webhook error: %s", exc)


def make_notifier(webhook_url: str | None = None) -> NullNotifier | WebhookNotifier:
    """Return a :class:`WebhookNotifier` when *webhook_url* is truthy, else
    a :class:`NullNotifier`."""
    if webhook_url:
        return WebhookNotifier(webhook_url)
    return NullNotifier()
