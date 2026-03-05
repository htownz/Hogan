"""Oanda REST v20 integration for Hogan.

Oanda provides:
  * Crypto CFD pricing  (BTC_USD, ETH_USD, LTC_USD, XRP_USD, BCH_USD)
  * Forex pairs         (EUR_USD, GBP_USD, XAU_USD …)
  * Account information (balance, NAV, open positions, P&L)
  * Practice / paper trading via fxTrade Practice environment

Auth
----
    Set in ``.env``::

        OANDA_ACCESS_TOKEN=<your-token>
        OANDA_ACCOUNT_ID=<your-account-id>       # e.g. 001-001-1234567-001
        OANDA_ENVIRONMENT=practice               # or "live"

    Find your account ID::

        python -m hogan_bot.fetch_oanda --list-accounts

API docs: https://developer.oanda.com/rest-live-v20/introduction/

Data stored in ``onchain_metrics`` (daily snapshots)
-----------------------------------------------------
``oanda_btc_mid``        — BTC_USD mid price (for cross-validation vs exchange)
``oanda_eth_mid``        — ETH_USD mid price
``oanda_xau_mid``        — Gold/USD mid price (macro risk-off signal)
``oanda_eur_mid``        — EUR/USD (DXY proxy for macro context)

Usage
-----
    python -m hogan_bot.fetch_oanda                  # fetch daily price snapshot
    python -m hogan_bot.fetch_oanda --list-accounts  # print your account IDs
    python -m hogan_bot.fetch_oanda --account-summary  # print balance and P&L
"""
from __future__ import annotations

import argparse
import json
import logging
import os
from datetime import datetime, timezone
from urllib.error import URLError
from urllib.request import Request, urlopen

logger = logging.getLogger(__name__)

_TIMEOUT = 15

# Oanda uses instrument codes like BTC_USD (not BTC/USD)
_INSTRUMENTS = {
    "BTC_USD": "oanda_btc_mid",
    "ETH_USD": "oanda_eth_mid",
    "XAU_USD": "oanda_xau_mid",
    "EUR_USD": "oanda_eur_mid",
}

_ENVIRONMENTS = {
    "practice": "https://api-fxpractice.oanda.com",
    "live":     "https://api-fxtrade.oanda.com",
}


# ---------------------------------------------------------------------------
# Config helpers
# ---------------------------------------------------------------------------

def _get_config() -> tuple[str, str, str]:
    """Return (access_token, account_id, base_url).  Raises if token missing."""
    token = os.getenv("OANDA_ACCESS_TOKEN", "").strip()
    if not token:
        raise RuntimeError(
            "OANDA_ACCESS_TOKEN not set.\n"
            "Add to .env:  OANDA_ACCESS_TOKEN=<your-token>\n"
            "Get a token at: https://www.oanda.com/account/#/access/fxtrade/personal-token"
        )
    account_id = os.getenv("OANDA_ACCOUNT_ID", "").strip()
    env = os.getenv("OANDA_ENVIRONMENT", "practice").strip().lower()
    if env not in _ENVIRONMENTS:
        env = "practice"
    base_url = _ENVIRONMENTS[env]
    return token, account_id, base_url


def _get(path: str, token: str, base_url: str) -> dict:
    """GET from Oanda REST v20 with Bearer auth."""
    url = f"{base_url}{path}"
    req = Request(url, headers={"Authorization": f"Bearer {token}", "Accept": "application/json"})
    with urlopen(req, timeout=_TIMEOUT) as resp:
        return json.loads(resp.read().decode())


# ---------------------------------------------------------------------------
# Account helpers
# ---------------------------------------------------------------------------

def list_accounts(token: str, base_url: str) -> list[dict]:
    """Return all accounts accessible with this token."""
    data = _get("/v3/accounts", token, base_url)
    return data.get("accounts", [])


def account_summary(token: str, account_id: str, base_url: str) -> dict:
    """Return account summary (balance, NAV, open trade count, etc.)."""
    data = _get(f"/v3/accounts/{account_id}/summary", token, base_url)
    return data.get("account", {})


# ---------------------------------------------------------------------------
# Price fetchers
# ---------------------------------------------------------------------------

def fetch_prices(
    token: str,
    base_url: str,
    account_id: str = "",
    instruments: list[str] | None = None,
) -> dict[str, float]:
    """Fetch mid prices for a list of instruments.

    Parameters
    ----------
    account_id : str
        Oanda account ID — required for the v20 pricing endpoint.
    instruments : list[str]
        Oanda instrument codes, e.g. ``["BTC_USD", "ETH_USD"]``.
        Defaults to all instruments in ``_INSTRUMENTS``.

    Returns
    -------
    dict
        Mapping of instrument code → mid price.
    """
    instruments = instruments or list(_INSTRUMENTS.keys())
    inst_str = "%2C".join(instruments)   # URL-encoded comma
    # Correct v20 endpoint: /v3/accounts/{id}/pricing?instruments=...
    path = f"/v3/accounts/{account_id}/pricing?instruments={inst_str}"
    try:
        data = _get(path, token, base_url)
    except URLError as exc:
        logger.warning("Oanda prices request failed: %s", exc)
        return {}

    prices: dict[str, float] = {}
    for quote in data.get("prices", []):
        code = quote.get("instrument", "")
        bids = quote.get("bids", [{}])
        asks = quote.get("asks", [{}])
        try:
            bid = float(bids[0].get("price", 0))
            ask = float(asks[0].get("price", 0))
            prices[code] = (bid + ask) / 2
        except (IndexError, TypeError, ValueError):
            pass
    return prices


# ---------------------------------------------------------------------------
# Main fetch → DB
# ---------------------------------------------------------------------------

def fetch_all_oanda(
    symbol: str = "BTC/USD",
    db_path: str = "data/hogan.db",
) -> dict[str, int]:
    """Fetch Oanda price snapshot and upsert into ``onchain_metrics``.

    Parameters
    ----------
    symbol : str
        Informational — used as the symbol label in the DB row.
    db_path : str
        Path to the SQLite database.

    Returns
    -------
    dict
        ``{metric_name: rows_written}``
    """
    from hogan_bot.storage import get_connection, upsert_onchain

    token, account_id, base_url = _get_config()
    if not account_id:
        logger.warning("Oanda: OANDA_ACCOUNT_ID not set — cannot fetch prices")
        return {}
    prices = fetch_prices(token, base_url, account_id=account_id)

    if not prices:
        logger.warning("Oanda: no prices returned")
        return {}

    today = datetime.now(timezone.utc).date().isoformat()
    records: list[tuple[str, str, float]] = []
    for inst_code, metric_name in _INSTRUMENTS.items():
        if inst_code in prices:
            records.append((today, metric_name, prices[inst_code]))
            logger.info("Oanda: %s = %.5f", metric_name, prices[inst_code])

    if not records:
        return {}

    conn = get_connection(db_path)
    written = upsert_onchain(conn, symbol, records)
    conn.close()
    return {r[1]: 1 for r in records}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Oanda REST v20 price & account data fetcher",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--list-accounts", action="store_true",
                   help="Print all accounts accessible with OANDA_ACCESS_TOKEN and exit")
    p.add_argument("--account-summary", action="store_true",
                   help="Print account balance / NAV summary and exit")
    p.add_argument("--symbol", default="BTC/USD", help="Symbol label for DB storage")
    p.add_argument("--db", default="data/hogan.db", help="SQLite database path")
    return p.parse_args()


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = _parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    token, account_id, base_url = _get_config()

    if args.list_accounts:
        accounts = list_accounts(token, base_url)
        if accounts:
            print("\nYour Oanda accounts:\n")
            for acc in accounts:
                print(f"  ID: {acc.get('id')}  tags: {acc.get('tags', [])}")
            print(
                "\nCopy the account ID you want to use and set:\n"
                "  OANDA_ACCOUNT_ID=<id>  in your .env file\n"
            )
        else:
            print("No accounts found — check your OANDA_ACCESS_TOKEN and OANDA_ENVIRONMENT")
        return

    if args.account_summary:
        if not account_id:
            print(
                "OANDA_ACCOUNT_ID is not set.\n"
                "Run with --list-accounts first to find your account ID."
            )
            return
        summary = account_summary(token, account_id, base_url)
        print(json.dumps(summary, indent=2))
        return

    result = fetch_all_oanda(symbol=args.symbol, db_path=args.db)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
