"""Discord command listener for Hogan.

Polls a Discord channel for messages starting with ``!hogan`` (or just ``!``)
and responds via the existing webhook with live bot data.

Runs as a daemon thread inside the main bot process — zero extra processes
or libraries required (pure ``urllib``).

Setup (one-time)
----------------
1. Go to https://discord.com/developers/applications → New Application
2. Bot tab → Add Bot → copy the **Token** → set DISCORD_BOT_TOKEN in .env
3. Bot tab → enable **Message Content Intent** (required to read message text)
4. OAuth2 → URL Generator → scope: bot → permissions: Read Messages, Send Messages
5. Paste the generated URL in a browser → invite bot to your server
6. Right-click your trading channel → Copy Channel ID → set DISCORD_CHANNEL_ID in .env

Available commands (type in your Discord channel)
-------------------------------------------------
  !balance    Current equity, cash, open positions, unrealized P&L
  !pnl        Realized P&L summary from fill history
  !positions  Open positions with entry price, qty, current P&L %
  !status     Bot mode, model info, market regime signals
  !signals    Latest ML up-probability for each tracked symbol
  !market     Key macro signals: Fear & Greed, BTC dom, DXY, funding rate
  !fills      Last 10 executed trades
  !help       This list
"""
from __future__ import annotations

import json
import logging
import os
import sqlite3
import threading
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from hogan_bot.paper import PaperPortfolio

logger = logging.getLogger(__name__)

_DISCORD_API = "https://discord.com/api/v10"
_PREFIX = ("!hogan ", "!hogan\n", "!h ")  # accepted prefixes


# ---------------------------------------------------------------------------
# Low-level Discord REST helpers
# ---------------------------------------------------------------------------

def _headers(token: str) -> dict[str, str]:
    return {
        "Authorization": f"Bot {token}",
        "Content-Type": "application/json",
        "User-Agent": "HoganBot/1.0",
    }


def _get_messages(token: str, channel_id: str, after: str | None = None) -> list[dict]:
    url = f"{_DISCORD_API}/channels/{channel_id}/messages?limit=10"
    if after:
        url += f"&after={after}"
    req = urllib.request.Request(url, headers=_headers(token))
    try:
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())
    except Exception as exc:
        logger.debug("Discord poll error: %s", exc)
        return []


def _post_webhook(webhook_url: str, embeds: list[dict]) -> None:
    payload = json.dumps({"embeds": embeds}).encode()
    req = urllib.request.Request(
        webhook_url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "User-Agent": "HoganBot/1.0 (trading bot; +https://github.com/htownz/Hogan)",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=10):
            pass
    except Exception as exc:
        logger.warning("Discord webhook error: %s", exc)


# ---------------------------------------------------------------------------
# Command handlers — each returns a list of Discord embed dicts
# ---------------------------------------------------------------------------

def _embed(title: str, fields: list[tuple[str, str, bool]], color: int = 0x5865F2) -> dict:
    """Build a Discord embed with name/value fields."""
    return {
        "title": title,
        "color": color,
        "fields": [{"name": n, "value": v, "inline": i} for n, v, i in fields],
        "footer": {"text": f"Hogan — {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}"},
    }


def cmd_balance(portfolio: "PaperPortfolio | None", mark_prices: dict, db_path: str) -> list[dict]:
    if portfolio is None:
        return [_embed("Balance", [("Status", "Bot not yet initialised", False)], 0xFEE75C)]

    equity = portfolio.total_equity(mark_prices)
    cash   = portfolio.cash_usd
    long_unrealized = sum(
        pos.qty * (mark_prices.get(sym, pos.avg_entry) - pos.avg_entry)
        for sym, pos in portfolio.positions.items()
    )
    short_unrealized = sum(
        (pos.avg_entry - mark_prices.get(sym, pos.avg_entry)) * pos.qty
        for sym, pos in portfolio.short_positions.items()
    )
    unrealized = long_unrealized + short_unrealized

    n_longs = len(portfolio.positions)
    n_shorts = len(portfolio.short_positions)
    pos_str = f"{n_longs}L / {n_shorts}S" if n_longs or n_shorts else "None"

    fields = [
        ("Total Equity", f"**${equity:,.2f}**", True),
        ("Cash", f"${cash:,.2f}", True),
        ("Open Positions", pos_str, True),
        ("Unrealized P&L", f"{'🟢' if unrealized >= 0 else '🔴'} ${unrealized:+,.2f}", True),
    ]

    # Fetch starting balance from equity_snapshots
    try:
        conn = sqlite3.connect(db_path)
        row = conn.execute("SELECT equity FROM equity_snapshots ORDER BY ts_ms ASC LIMIT 1").fetchone()
        conn.close()
        if row:
            start = row[0]
            total_pnl = equity - start
            pct = (total_pnl / start * 100) if start > 0 else 0.0
            fields.append(("Total Return", f"{'🟢' if total_pnl >= 0 else '🔴'} ${total_pnl:+,.2f} ({pct:+.2f}%)", False))
    except Exception:
        pass

    color = 0x57F287 if unrealized >= 0 else 0xED4245
    return [_embed("💰 Account Balance", fields, color)]


def cmd_positions(portfolio: "PaperPortfolio | None", mark_prices: dict) -> list[dict]:
    if portfolio is None:
        return [_embed("Positions", [("Open Positions", "None — no open trades", False)], 0xFEE75C)]

    fields = []

    for sym, pos in portfolio.positions.items():
        px = mark_prices.get(sym, 0.0)
        pnl = pos.qty * (px - pos.avg_entry)
        pnl_pct = ((px / pos.avg_entry) - 1.0) * 100.0 if pos.avg_entry > 0 else 0.0
        icon = "🟢" if pnl >= 0 else "🔴"
        fields.append((
            f"LONG {sym}",
            f"Qty: `{pos.qty:.6f}`\nEntry: `${pos.avg_entry:,.2f}`\nNow: `${px:,.2f}`\nP&L: {icon} `${pnl:+,.2f}` (`{pnl_pct:+.2f}%`)",
            True,
        ))

    for sym, pos in portfolio.short_positions.items():
        px = mark_prices.get(sym, 0.0)
        pnl = (pos.avg_entry - px) * pos.qty
        pnl_pct = ((pos.avg_entry / px) - 1.0) * 100.0 if px > 0 else 0.0
        icon = "🟢" if pnl >= 0 else "🔴"
        fields.append((
            f"SHORT {sym}",
            f"Qty: `{pos.qty:.6f}`\nEntry: `${pos.avg_entry:,.2f}`\nNow: `${px:,.2f}`\nP&L: {icon} `${pnl:+,.2f}` (`{pnl_pct:+.2f}%`)",
            True,
        ))

    if not fields:
        return [_embed("Positions", [("Open Positions", "None — no open trades", False)], 0xFEE75C)]

    total = len(portfolio.positions) + len(portfolio.short_positions)
    return [_embed(f"📊 Open Positions ({total})", fields, 0x5865F2)]


def cmd_pnl(db_path: str) -> list[dict]:
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            """SELECT symbol, side, realized_pnl, pnl_pct, entry_fee, exit_fee, close_reason
               FROM paper_trades
               WHERE exit_price IS NOT NULL
               ORDER BY close_ts_ms DESC LIMIT 50"""
        ).fetchall()
        conn.close()
    except Exception as exc:
        return [_embed("P&L", [("Error", str(exc), False)], 0xED4245)]

    if not rows:
        return [_embed("P&L", [("Realized P&L", "No closed trades yet — positions still open or bot just started", False)], 0xFEE75C)]

    total_pnl = 0.0
    total_fees = 0.0
    wins = 0
    by_symbol: dict[str, float] = {}
    for sym, side, pnl, pnl_pct, entry_fee, exit_fee, reason in rows:
        pnl = pnl or 0.0
        total_pnl += pnl
        total_fees += (entry_fee or 0.0) + (exit_fee or 0.0)
        if pnl > 0:
            wins += 1
        by_symbol[sym] = by_symbol.get(sym, 0.0) + pnl

    win_rate = wins / len(rows) * 100 if rows else 0.0
    color = 0x57F287 if total_pnl >= 0 else 0xED4245
    fields = [
        ("Closed Trades", str(len(rows)), True),
        ("Win Rate", f"{win_rate:.0f}%", True),
        ("Total Fees", f"${total_fees:.4f}", True),
        ("Net Realized P&L", f"{'🟢' if total_pnl >= 0 else '🔴'} ${total_pnl:+,.4f}", False),
    ]
    for sym, pnl in sorted(by_symbol.items()):
        icon = "🟢" if pnl >= 0 else "🔴"
        fields.append((sym, f"{icon} ${pnl:+,.4f}", True))

    return [_embed("📈 Realized P&L", fields, color)]


def cmd_fills(db_path: str) -> list[dict]:
    try:
        conn = sqlite3.connect(db_path)
        rows = conn.execute(
            """SELECT symbol, side, qty, entry_price, exit_price, realized_pnl, pnl_pct,
                      open_ts_ms, close_ts_ms, close_reason, ml_up_prob
               FROM paper_trades
               ORDER BY open_ts_ms DESC LIMIT 10"""
        ).fetchall()
        conn.close()
    except Exception as exc:
        return [_embed("Fills", [("Error", str(exc), False)], 0xED4245)]

    if not rows:
        return [_embed("Recent Trades", [("Status", "No paper trades yet — waiting for first signal", False)], 0xFEE75C)]

    fields = []
    for sym, side, qty, entry_px, exit_px, pnl, pnl_pct, open_ms, close_ms, reason, ml_prob in rows:
        dt_open = datetime.fromtimestamp(open_ms / 1000, tz=timezone.utc).strftime("%m-%d %H:%M")
        if exit_px is not None:
            dt_close = datetime.fromtimestamp(close_ms / 1000, tz=timezone.utc).strftime("%H:%M") if close_ms else "?"
            pnl_str = f"P&L ${pnl:+,.4f} ({pnl_pct*100:+.2f}%)" if pnl is not None else ""
            status = f"CLOSED @ ${exit_px:,.2f} [{reason}] {pnl_str}"
            icon = "🟢" if (pnl or 0) >= 0 else "🔴"
        else:
            status = f"OPEN @ ${entry_px:,.2f} (ml={ml_prob:.2f})" if ml_prob else f"OPEN @ ${entry_px:,.2f}"
            icon = "🔵"
        label = f"{icon} {side.upper()} {sym}"
        fields.append((label, f"`{qty:.6f}` entry ${entry_px:,.2f} — {status} @ {dt_open}", False))

    return [_embed("🧾 Last 10 Trades", fields, 0x5865F2)]


def cmd_status(db_path: str, config_summary: dict) -> list[dict]:
    fields = [
        ("Mode",    config_summary.get("mode", "PAPER"), True),
        ("Symbols", config_summary.get("symbols", "—"),  True),
        ("Model",   config_summary.get("model", "—"),    True),
        ("Timeframe", config_summary.get("timeframe", "5m"), True),
        ("Signal Mode", config_summary.get("signal_mode", "any"), True),
        ("ICT",     "✅ enabled" if config_summary.get("use_ict") else "❌ off", True),
        ("EMA Cloud", "✅ enabled" if config_summary.get("use_ema_clouds") else "❌ off", True),
        ("ML Filter", f"buy ≥ {config_summary.get('ml_buy_threshold', 0.50):.2f}", True),
    ]

    try:
        conn = sqlite3.connect(db_path)
        snap = conn.execute("SELECT equity, ts_ms FROM equity_snapshots ORDER BY ts_ms DESC LIMIT 1").fetchone()
        conn.close()
        if snap:
            dt = datetime.fromtimestamp(snap[1] / 1000, tz=timezone.utc).strftime("%H:%M UTC")
            fields.append(("Last Equity Snapshot", f"${snap[0]:,.2f} @ {dt}", False))
    except Exception:
        pass

    return [_embed("⚙️ Bot Status", fields, 0x5865F2)]


def cmd_market(db_path: str) -> list[dict]:
    fields = []
    metrics = {
        "fear_greed_value":        ("Fear & Greed", lambda v: f"`{v:.0f}/100` {'😱 Extreme Fear' if v < 25 else '😨 Fear' if v < 45 else '😐 Neutral' if v < 55 else '😀 Greed' if v < 75 else '🤑 Extreme Greed'}"),
        "cg_btc_dominance":        ("BTC Dominance", lambda v: f"`{v:.1f}%`"),
        "cmc_eth_dominance":       ("ETH Dominance", lambda v: f"`{v:.1f}%`"),
        "cg_mcap_change_24h":      ("Market Cap 24h Δ", lambda v: f"{'🟢' if v >= 0 else '🔴'} `{v:+.2f}%`"),
        "fred_dgs10":              ("10Y Treasury", lambda v: f"`{v:.2f}%`"),
        "fred_t10y2y":             ("Yield Curve (10Y-2Y)", lambda v: f"{'🔴 Inverted' if v < 0 else '🟢 Normal'} `{v:+.2f}%`"),
        "defi_total_tvl_b":        ("DeFi TVL", lambda v: f"`${v:.1f}B`"),
        "btc_mempool_mb":          ("BTC Mempool", lambda v: f"`{v:.1f} MB`"),
    }
    try:
        conn = sqlite3.connect(db_path)
        for metric, (label, fmt) in metrics.items():
            row = conn.execute(
                "SELECT value FROM onchain_metrics WHERE metric = ? ORDER BY date DESC LIMIT 1",
                (metric,)
            ).fetchone()
            if row:
                fields.append((label, fmt(row[0]), True))

        # Funding rate from derivatives table
        dr = conn.execute(
            "SELECT value FROM derivatives_metrics WHERE metric = 'funding_rate' ORDER BY ts_ms DESC LIMIT 1"
        ).fetchone()
        if dr:
            fr = dr[0]
            fields.append(("Funding Rate", f"{'📉' if fr < 0 else '📈'} `{fr:+.4f}` ({'shorts paying' if fr < 0 else 'longs paying'})", True))

        conn.close()
    except Exception as exc:
        fields.append(("Error", str(exc), False))

    return [_embed("🌍 Market Intelligence", fields or [("Status", "No macro data yet — run refresh_daily.py", False)], 0x5865F2)]


def cmd_signals(signal_cache: dict) -> list[dict]:
    if not signal_cache:
        return [_embed("Signals", [("Status", "No signals yet — bot still warming up", False)], 0xFEE75C)]

    fields = []
    for sym, data in signal_cache.items():
        ml_up  = data.get("ml_up", 0.0)
        action = data.get("action", "hold")
        conf   = data.get("conf", 0.0)
        px     = data.get("price", 0.0)
        icon   = "🟢" if action == "buy" else "🔴" if action == "sell" else "⚪"
        bar = "█" * int(ml_up * 10) + "░" * (10 - int(ml_up * 10))
        fields.append((
            f"{icon} {sym}",
            f"Action: **{action.upper()}**\nML up-prob: `{bar}` `{ml_up:.3f}`\nConf: `{conf:.2f}` | Price: `${px:,.2f}`",
            True,
        ))

    return [_embed("🤖 Latest ML Signals", fields, 0x5865F2)]


def cmd_help() -> list[dict]:
    commands = [
        ("!balance",   "Equity, cash, open positions, total return"),
        ("!positions", "Open trades with entry price and unrealized P&L"),
        ("!pnl",       "Realized P&L summary from fill history"),
        ("!fills",     "Last 10 executed trades"),
        ("!status",    "Bot configuration: mode, model, signals enabled"),
        ("!market",    "Macro intelligence: Fear & Greed, BTC dom, DXY, funding"),
        ("!signals",   "Latest ML up-probability for each symbol"),
        ("!help",      "This list"),
    ]
    fields = [(cmd, desc, False) for cmd, desc in commands]
    return [_embed("📖 Hogan Command Guide", fields, 0x5865F2)]


# ---------------------------------------------------------------------------
# Command dispatcher
# ---------------------------------------------------------------------------

def _parse_command(content: str) -> str | None:
    """Extract the command word from a message, stripping any prefix."""
    content = content.strip().lower()
    for prefix in _PREFIX:
        if content.startswith(prefix):
            return content[len(prefix):].split()[0] if content[len(prefix):].strip() else "help"
    if content.startswith("!"):
        return content[1:].split()[0] if len(content) > 1 else "help"
    return None


def dispatch(
    command: str,
    *,
    portfolio=None,
    mark_prices: dict | None = None,
    db_path: str = "data/hogan.db",
    config_summary: dict | None = None,
    signal_cache: dict | None = None,
) -> list[dict]:
    mark_prices   = mark_prices or {}
    config_summary = config_summary or {}
    signal_cache  = signal_cache or {}

    cmd = command.lower().strip()
    if cmd == "balance":
        return cmd_balance(portfolio, mark_prices, db_path)
    if cmd == "positions":
        return cmd_positions(portfolio, mark_prices)
    if cmd == "pnl":
        return cmd_pnl(db_path)
    if cmd == "fills":
        return cmd_fills(db_path)
    if cmd == "status":
        return cmd_status(db_path, config_summary)
    if cmd == "market":
        return cmd_market(db_path)
    if cmd == "signals":
        return cmd_signals(signal_cache)
    if cmd in ("help", "?", "commands"):
        return cmd_help()

    return [_embed("Unknown Command", [("Tip", f"Unknown: `!{cmd}` — type `!help` for the full list", False)], 0xFEE75C)]


# ---------------------------------------------------------------------------
# Background listener thread
# ---------------------------------------------------------------------------

class DiscordCommandListener:
    """Daemon thread that polls a Discord channel and responds to commands."""

    def __init__(
        self,
        bot_token: str,
        channel_id: str,
        webhook_url: str,
        db_path: str = "data/hogan.db",
        poll_interval: float = 5.0,
    ) -> None:
        self.token         = bot_token
        self.channel_id    = channel_id
        self.webhook_url   = webhook_url
        self.db_path       = db_path
        self.poll_interval = poll_interval
        self._last_msg_id: str | None = None
        self._stop_event   = threading.Event()

        # Shared state — set by main loop each iteration
        self.portfolio    = None
        self.mark_prices: dict = {}
        self.config_summary: dict = {}
        self.signal_cache: dict = {}

    def update_state(self, portfolio, mark_prices: dict, config_summary: dict, signal_cache: dict) -> None:
        """Called by the main loop every iteration to keep state fresh."""
        self.portfolio      = portfolio
        self.mark_prices    = mark_prices
        self.config_summary = config_summary
        self.signal_cache   = signal_cache

    def _seed_last_message(self) -> None:
        """On startup, record the most recent message ID so we only respond to NEW commands."""
        msgs = _get_messages(self.token, self.channel_id)
        if msgs:
            self._last_msg_id = msgs[0]["id"]
            logger.info("Discord listener ready — watching channel %s after message %s", self.channel_id, self._last_msg_id)
        else:
            logger.info("Discord listener ready — no existing messages found")

    def _poll_once(self) -> None:
        msgs = _get_messages(self.token, self.channel_id, after=self._last_msg_id)
        if not msgs:
            return

        # Discord returns newest-first — process oldest-first
        for msg in reversed(msgs):
            self._last_msg_id = msg["id"]
            content = msg.get("content", "")
            author  = msg.get("author", {}).get("username", "unknown")

            # Skip bot's own messages
            if msg.get("author", {}).get("bot"):
                continue

            cmd = _parse_command(content)
            if cmd is None:
                continue

            logger.info("Discord command from %s: !%s", author, cmd)
            try:
                embeds = dispatch(
                    cmd,
                    portfolio=self.portfolio,
                    mark_prices=self.mark_prices,
                    db_path=self.db_path,
                    config_summary=self.config_summary,
                    signal_cache=self.signal_cache,
                )
                _post_webhook(self.webhook_url, embeds)
            except Exception as exc:
                logger.warning("Discord command handler error: %s", exc)
                _post_webhook(self.webhook_url, [_embed("Error", [("Command failed", str(exc), False)], 0xED4245)])

    def run(self) -> None:
        self._seed_last_message()
        while not self._stop_event.is_set():
            try:
                self._poll_once()
            except Exception as exc:
                logger.debug("Discord poll cycle error: %s", exc)
            time.sleep(self.poll_interval)

    def start(self) -> threading.Thread:
        t = threading.Thread(target=self.run, name="discord-cmd", daemon=True)
        t.start()
        logger.info("Discord command listener started (polling every %.0fs)", self.poll_interval)
        return t

    def stop(self) -> None:
        self._stop_event.set()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_command_listener(
    webhook_url: str,
    db_path: str = "data/hogan.db",
) -> DiscordCommandListener | None:
    """Create and return a listener if DISCORD_BOT_TOKEN and DISCORD_CHANNEL_ID are set."""
    token      = os.getenv("DISCORD_BOT_TOKEN", "").strip()
    channel_id = os.getenv("DISCORD_CHANNEL_ID", "").strip()

    if not token or not channel_id:
        logger.info(
            "Discord commands disabled — set DISCORD_BOT_TOKEN and DISCORD_CHANNEL_ID in .env to enable"
        )
        return None

    if not webhook_url:
        logger.warning("Discord commands: webhook URL not set — responses will fail")

    return DiscordCommandListener(
        bot_token=token,
        channel_id=channel_id,
        webhook_url=webhook_url,
        db_path=db_path,
    )
