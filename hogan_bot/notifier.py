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



class TelegramNotifier:
    """Send events to a Telegram chat via bot token + chat id."""

    def __init__(self, bot_token: str, chat_id: str, timeout_seconds: int = 5) -> None:
        self.bot_token = bot_token
        self.chat_id = chat_id
        self.timeout_seconds = timeout_seconds

    def notify(self, event_type: str, payload: dict) -> None:
        text = {
            "event": event_type,
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            **payload,
        }
        msg = json.dumps(text, default=str)
        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        data = urllib.parse.urlencode({"chat_id": self.chat_id, "text": msg}).encode("utf-8")
        req = urllib.request.Request(url, data=data, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_seconds):
                pass
        except Exception as exc:
            logger.warning("Telegram notify failed: %s", exc)


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
    webhook_url: str | None = None,
    telegram_token: str | None = None,
    telegram_chat_id: str | None = None,
    email: dict | None = None,
) -> "NullNotifier | WebhookNotifier | TelegramNotifier | MultiNotifier":
    """Build and return the appropriate notifier from args or environment variables.

    Priority order: webhook -> Telegram -> Email. Any subset can be active
    simultaneously; all active notifiers are combined into a ``MultiNotifier``.

    Explicit args override env vars. Reads these environment variables (all optional):

    * ``HOGAN_WEBHOOK_URL``     — HTTP webhook endpoint
    * ``HOGAN_TELEGRAM_TOKEN``  — Telegram bot token
    * ``HOGAN_TELEGRAM_CHAT``   — Telegram chat ID
    * ``HOGAN_SMTP_HOST``       — SMTP host for email
    * ``HOGAN_SMTP_PORT``       — SMTP port (default 587)
    * ``HOGAN_SMTP_USER``       — SMTP username
    * ``HOGAN_SMTP_PASS``       — SMTP password
    * ``HOGAN_EMAIL_FROM``      — From address
    * ``HOGAN_EMAIL_TO``        — To address
    """
    import os

    notifiers: list = []

    url = webhook_url or os.getenv("HOGAN_WEBHOOK_URL", "")
    if url:
        notifiers.append(WebhookNotifier(url))

    tg_token = telegram_token or os.getenv("HOGAN_TELEGRAM_TOKEN", "")
    tg_chat = telegram_chat_id or os.getenv("HOGAN_TELEGRAM_CHAT", "")
    if tg_token and tg_chat:
        notifiers.append(TelegramNotifier(tg_token, tg_chat))

    if email:
        try:
            notifiers.append(EmailNotifier(**email))
        except Exception as exc:
            logger.warning("EmailNotifier setup failed: %s", exc)
    else:
        smtp_host = os.getenv("HOGAN_SMTP_HOST", "")
        if smtp_host:
            notifiers.append(
                EmailNotifier(
                    smtp_host=smtp_host,
                    smtp_port=int(os.getenv("HOGAN_SMTP_PORT", "587")),
                    username=os.getenv("HOGAN_SMTP_USER", ""),
                    password=os.getenv("HOGAN_SMTP_PASS", ""),
                    from_addr=os.getenv("HOGAN_EMAIL_FROM", "hogan@localhost"),
                    to_addr=os.getenv("HOGAN_EMAIL_TO", ""),
                    use_tls=os.getenv("HOGAN_SMTP_TLS", "true").lower() != "false",
                )
            )

    if not notifiers:
        return NullNotifier()
    if len(notifiers) == 1:
        return notifiers[0]
    return MultiNotifier(notifiers)
