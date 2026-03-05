"""Lightweight event notification for the Hogan trading bot.

Concrete implementations:

* ``DiscordNotifier``  — rich embeds to a Discord webhook (preferred).
* ``WebhookNotifier``  — generic JSON POST (Slack, Make/Zapier, etc.).
* ``EmailNotifier``    — SMTP email delivery.
* ``NullNotifier``     — silently discards all events (no channels configured).

Usage::

    from hogan_bot.notifier import make_notifier
    notifier = make_notifier()   # reads env vars automatically
    notifier.notify("buy", {"symbol": "BTC/USD", "price": 60000})
"""

from __future__ import annotations

import json
import logging
import urllib.error
import urllib.request
import urllib.parse
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



_EVENT_COLORS = {
    "buy":      0x00C853,   # green
    "sell":     0xFF1744,   # red
    "drawdown": 0xFF6D00,   # orange
    "error":    0xD50000,   # dark red
    "info":     0x2196F3,   # blue
    "stale":    0xFFC107,   # amber
    "retrain":  0x9C27B0,   # purple
}


class DiscordNotifier:
    """Send rich embeds to a Discord channel via an incoming webhook URL.

    Webhook URL format:
        https://discord.com/api/webhooks/{webhook_id}/{webhook_token}

    Create via: Server Settings → Integrations → Webhooks → New Webhook.
    """

    def __init__(self, webhook_url: str, timeout_seconds: int = 5) -> None:
        self.webhook_url = webhook_url
        self.timeout_seconds = timeout_seconds

    def notify(self, event_type: str, payload: dict) -> None:
        ts = datetime.now(tz=timezone.utc).isoformat()
        color = _EVENT_COLORS.get(event_type.lower(), 0x607D8B)

        # Build embed fields from payload (skip internal/noisy keys)
        _SKIP = {"event", "timestamp"}
        fields = [
            {"name": str(k), "value": f"`{str(v)[:100]}`", "inline": True}
            for k, v in payload.items()
            if k not in _SKIP and v is not None
        ]

        body = {
            "username": "Hogan Bot",
            "embeds": [{
                "title": f"[Hogan] {event_type.upper()}",
                "color": color,
                "timestamp": ts,
                "fields": fields[:25],  # Discord limit
                "footer": {"text": "Hogan Trading Bot"},
            }],
        }
        data = json.dumps(body, default=str).encode("utf-8")
        req = urllib.request.Request(
            self.webhook_url,
            data=data,
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds):
                pass
        except urllib.error.URLError as exc:
            logger.warning("Discord webhook failed (%s): %s", self.webhook_url, exc)
        except Exception as exc:
            logger.warning("Discord unexpected error: %s", exc)


class EmailNotifier:
    """Send events to an email address via SMTP (best-effort)."""

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        username: str,
        password: str,
        from_addr: str,
        to_addr: str,
        use_tls: bool = True,
    ) -> None:
        self.smtp_host = smtp_host
        self.smtp_port = int(smtp_port)
        self.username = username
        self.password = password
        self.from_addr = from_addr
        self.to_addr = to_addr
        self.use_tls = use_tls

    def notify(self, event_type: str, payload: dict) -> None:
        import smtplib
        from email.message import EmailMessage

        body = json.dumps(
            {"event": event_type, "timestamp": datetime.now(tz=timezone.utc).isoformat(), **payload},
            default=str,
            indent=2,
        )
        msg = EmailMessage()
        msg["Subject"] = f"[Hogan] {event_type}"
        msg["From"] = self.from_addr
        msg["To"] = self.to_addr
        msg.set_content(body)

        try:
            with smtplib.SMTP(self.smtp_host, self.smtp_port, timeout=10) as s:
                if self.use_tls:
                    s.starttls()
                if self.username:
                    s.login(self.username, self.password)
                s.send_message(msg)
        except Exception as exc:
            logger.warning("Email notify failed: %s", exc)


class MultiNotifier:
    def __init__(self, notifiers: list) -> None:
        self.notifiers = [n for n in notifiers if n is not None]

    def notify(self, event_type: str, payload: dict) -> None:
        for n in self.notifiers:
            try:
                n.notify(event_type, payload)
            except Exception as exc:
                logger.warning("Notifier failed (%s): %s", type(n).__name__, exc)

def make_notifier(
    discord_webhook_url: str | None = None,
    webhook_url: str | None = None,
    email: dict | None = None,
) -> "NullNotifier | DiscordNotifier | WebhookNotifier | MultiNotifier":
    """Build and return the appropriate notifier from args or environment variables.

    Discord is the primary notification channel. Generic webhook and SMTP email
    are also supported and can run simultaneously via ``MultiNotifier``.

    Explicit args override env vars. Reads these environment variables:

    * ``HOGAN_DISCORD_WEBHOOK_URL``  — Discord incoming webhook (primary)
    * ``HOGAN_WEBHOOK_URL``          — Generic JSON webhook (Slack, Make, etc.)
    * ``HOGAN_EMAIL_SMTP_HOST``      — SMTP host for email
    * ``HOGAN_EMAIL_SMTP_PORT``      — SMTP port (default 587)
    * ``HOGAN_EMAIL_USERNAME``       — SMTP username
    * ``HOGAN_EMAIL_PASSWORD``       — SMTP password
    * ``HOGAN_EMAIL_FROM``           — From address
    * ``HOGAN_EMAIL_TO``             — To address
    """
    import os

    notifiers: list = []

    discord_url = discord_webhook_url or os.getenv("HOGAN_DISCORD_WEBHOOK_URL", "")
    if discord_url:
        notifiers.append(DiscordNotifier(discord_url))

    generic_url = webhook_url or os.getenv("HOGAN_WEBHOOK_URL", "")
    if generic_url:
        notifiers.append(WebhookNotifier(generic_url))

    if email:
        try:
            notifiers.append(EmailNotifier(**email))
        except Exception as exc:
            logger.warning("EmailNotifier setup failed: %s", exc)
    else:
        smtp_host = os.getenv("HOGAN_EMAIL_SMTP_HOST", "")
        if smtp_host:
            notifiers.append(
                EmailNotifier(
                    smtp_host=smtp_host,
                    smtp_port=int(os.getenv("HOGAN_EMAIL_SMTP_PORT", "587")),
                    username=os.getenv("HOGAN_EMAIL_USERNAME", ""),
                    password=os.getenv("HOGAN_EMAIL_PASSWORD", ""),
                    from_addr=os.getenv("HOGAN_EMAIL_FROM", "hogan@localhost"),
                    to_addr=os.getenv("HOGAN_EMAIL_TO", ""),
                    use_tls=os.getenv("HOGAN_EMAIL_TLS", "true").lower() != "false",
                )
            )

    if not notifiers:
        return NullNotifier()
    if len(notifiers) == 1:
        return notifiers[0]
    return MultiNotifier(notifiers)
