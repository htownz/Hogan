"""Daily data refresh script — run once per day before training or live trading.

Refreshes all external data sources in dependency order:
  1.  Fear & Greed Index        (free, Alternative.me)
  2.  CoinGecko macro data      (free demo key)
  3.  GPR Index                 (free, academic download)
  4.  Kraken Futures / Deriv.   (free public API)
  5.  Blockchain.com + Mempool  (free, no key — BTC hash rate, mempool, fees)
  6.  DeFi Llama                (free, no key — TVL, stablecoin mcap)
  7.  CryptoPanic news          (Developer plan key — 1 req/day)
  8.  FRED macro data           (free key — yields, M2, CPI, DXY)
  9.  CryptoQuant on-chain      (paid — skipped if CRYPTOQUANT_KEY absent)
  10. Glassnode                 (paid — skipped if GLASSNODE_KEY absent)
  11. Santiment                 (paid — skipped if SANTIMENT_KEY absent)
  12. SPY macro backfill        (yfinance, free)
  13. OpenBB macro: DXY/VIX     (yfinance fallback, free)
  14. Messari fundamentals      (free tier — skipped if MESSARI_KEY absent)
  15. CoinMarketCap             (free key — BTC/ETH dominance, total mcap, DeFi%)
  16. Alpaca market data        (free key — SPY close + BTC/ETH bid-ask spread)
  17. Dune Analytics on-chain   (paid — skipped if DUNE_API_KEY absent)
  18. Oanda prices              (OANDA_ACCESS_TOKEN required — BTC/ETH/XAU/EUR)

Usage
-----
    python refresh_daily.py              # all sources
    python refresh_daily.py --dry-run    # print plan only
    python refresh_daily.py --source feargreed coingecko gpr
"""
from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import date
from typing import Callable

_GREEN = "\033[92m"
_RED = "\033[91m"
_YELLOW = "\033[93m"
_RESET = "\033[0m"
_BOLD = "\033[1m"


def _ok(msg: str) -> None:
    print(f"  {_GREEN}OK{_RESET}  {msg}")


def _fail(msg: str) -> None:
    print(f"  {_RED}FAIL{_RESET} {msg}")


def _skip(msg: str) -> None:
    print(f"  {_YELLOW}SKIP{_RESET} {msg}")


def _run_step(name: str, fn: Callable, dry_run: bool) -> bool:
    print(f"\n{_BOLD}[{name}]{_RESET}")
    if dry_run:
        _skip("dry-run mode — skipping execution")
        return True
    t0 = time.perf_counter()
    try:
        fn()
        elapsed = time.perf_counter() - t0
        _ok(f"completed in {elapsed:.1f}s")
        return True
    except Exception as exc:
        elapsed = time.perf_counter() - t0
        _fail(f"{exc!r}  ({elapsed:.1f}s)")
        return False


# ---------------------------------------------------------------------------
# Individual refresh functions
# ---------------------------------------------------------------------------

def _refresh_blockchain() -> None:
    """Fetch BTC hash rate, difficulty, mempool, tx count (blockchain.com + mempool.space)."""
    from hogan_bot.fetch_blockchain import fetch_all_blockchain
    fetch_all_blockchain(symbol=_primary_symbol(), db_path=_db_path(), days=30)


def _refresh_defillama() -> None:
    """Fetch DeFi TVL, ETH chain dominance, stablecoin market cap (DeFi Llama)."""
    from hogan_bot.fetch_defillama import fetch_all_defillama
    fetch_all_defillama(symbol=_primary_symbol(), db_path=_db_path())


def _refresh_fred() -> None:
    """Fetch FRED macro data: 10Y yield, yield curve, M2, CPI, Fed rate."""
    key = os.getenv("FRED_API_KEY", "")
    if not key:
        raise RuntimeError(
            "FRED_API_KEY not set — skipping FRED macro data\n"
            "Free key at: https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    from hogan_bot.fetch_fred import fetch_all_fred
    fetch_all_fred(symbol=_primary_symbol(), db_path=_db_path(), days=60)


def _refresh_feargreed() -> None:
    from hogan_bot.fetch_feargreed import fetch_and_store
    # backfill=False fetches only the most recent reading
    fetch_and_store(backfill=False)


def _refresh_coingecko() -> None:
    key = os.getenv("COINGECKO_KEY", "").strip()
    if not key:
        raise RuntimeError("COINGECKO_KEY not set — skipping CoinGecko")
    from hogan_bot.fetch_coingecko import CoinGeckoClient, fetch_today
    client = CoinGeckoClient(key)
    fetch_today(client)


def _refresh_gpr() -> None:
    from hogan_bot.fetch_gpr import fetch_and_store
    # force=False skips download if the file was already fetched today
    fetch_and_store(force=False)


def _refresh_derivatives() -> None:
    from hogan_bot.fetch_derivatives import fetch_derivatives
    fetch_derivatives(days=7)


def _refresh_news_sentiment() -> None:
    key = os.getenv("CRYPTOPANIC_KEY", "")
    if not key:
        raise RuntimeError("CRYPTOPANIC_KEY not set — skipping news sentiment")
    from hogan_bot.fetch_news_sentiment import fetch_and_store
    # pages=1 → 20 recent posts, 1 API request (Developer plan: 100 req/mo limit)
    fetch_and_store(pages=1)


def _refresh_onchain() -> None:
    key = os.getenv("CRYPTOQUANT_KEY", "")
    if not key:
        raise RuntimeError("CRYPTOQUANT_KEY not set — skipping CryptoQuant on-chain")
    from hogan_bot.fetch_onchain import fetch_onchain
    # API key is read from CRYPTOQUANT_KEY env var inside the module
    fetch_onchain(days=7)


def _refresh_glassnode() -> None:
    key = os.getenv("GLASSNODE_KEY", "")
    if not key:
        raise RuntimeError("GLASSNODE_KEY not set — skipping Glassnode")
    from hogan_bot.fetch_glassnode import fetch_and_store
    # API key is read from GLASSNODE_KEY env var inside the module
    fetch_and_store(days=7)


def _refresh_santiment() -> None:
    key = os.getenv("SANTIMENT_KEY", "")
    if not key:
        raise RuntimeError("SANTIMENT_KEY not set — skipping Santiment")
    from hogan_bot.fetch_santiment import fetch_and_store
    # API key is read from SANTIMENT_KEY env var inside the module
    fetch_and_store(days=7)


def _refresh_spy() -> None:
    import subprocess
    result = subprocess.run(
        [sys.executable, "-m", "hogan_bot.backfill", "--macro"],
        capture_output=True, text=True, timeout=120,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "backfill --macro failed")


def _primary_symbol() -> str:
    """Return the first configured trading symbol (HOGAN_SYMBOLS, comma-separated)."""
    return os.getenv("HOGAN_SYMBOLS", "BTC/USD").split(",")[0].strip()


def _db_path() -> str:
    """Return the canonical database path (HOGAN_DB_PATH)."""
    return os.getenv("HOGAN_DB_PATH", "data/hogan.db")


def _refresh_messari() -> None:
    """Fetch NVT ratio, realized cap, ROI, dev activity from Messari (free tier)."""
    key = os.getenv("MESSARI_KEY", "")
    if not key:
        raise RuntimeError("MESSARI_KEY not set — skipping Messari")
    from hogan_bot.fetch_messari import fetch_all_messari
    fetch_all_messari(symbol=_primary_symbol(), db_path=_db_path())


def _refresh_dune() -> None:
    """Fetch BTC exchange flow and whale count from Dune Analytics."""
    key = os.getenv("DUNE_API_KEY", "")
    if not key:
        raise RuntimeError("DUNE_API_KEY not set — skipping Dune Analytics")
    from hogan_bot.fetch_dune import fetch_all_dune
    fetch_all_dune(symbol=_primary_symbol(), db_path=_db_path())


def _refresh_oanda() -> None:
    """Fetch BTC/ETH/XAU/EUR mid prices from Oanda REST v20."""
    token = os.getenv("OANDA_ACCESS_TOKEN", "")
    if not token:
        raise RuntimeError("OANDA_ACCESS_TOKEN not set — skipping Oanda")
    from hogan_bot.fetch_oanda import fetch_all_oanda
    fetch_all_oanda(symbol=_primary_symbol(), db_path=_db_path())


def _refresh_openbb() -> None:
    """Fetch DXY, VIX, SPY return, FOMC calendar via OpenBB / yfinance."""
    from hogan_bot.fetch_openbb import (
        fetch_dxy, fetch_vix, fetch_spy_return, fetch_fed_calendar,
        fetch_btc_options_skew, store_records,
    )
    from hogan_bot.storage import get_connection
    conn = get_connection(_db_path())
    symbol = _primary_symbol()
    total = 0
    for fn in [fetch_dxy, fetch_vix, fetch_spy_return]:
        records = fn(days=30)
        total += store_records(records, symbol, conn)
    records = fetch_fed_calendar()
    total += store_records(records, symbol, conn)
    records = fetch_btc_options_skew()
    total += store_records(records, symbol, conn)
    conn.close()
    if total == 0:
        raise RuntimeError("OpenBB refresh produced 0 rows — check yfinance install")


def _refresh_cmc() -> None:
    """Fetch BTC/ETH dominance, total market cap, DeFi % from CoinMarketCap (CMC_API_KEY)."""
    k = os.getenv("CMC_API_KEY", "").strip()
    if not k:
        raise RuntimeError(
            "CMC_API_KEY not set — skipping CoinMarketCap\n"
            "Free Basic plan (10k credits/month) at: pro.coinmarketcap.com"
        )
    from hogan_bot.fetch_cmc import fetch_all_cmc
    result = fetch_all_cmc(db_path=_db_path())
    total = sum(result.values())
    print(f"  CMC: global metrics + BTC/ETH quotes — {total} records stored")


def _refresh_alpaca() -> None:
    """Fetch SPY close, crypto bid-ask spread from Alpaca (ALPACA_API_KEY required)."""
    k = os.getenv("ALPACA_API_KEY", "").strip()
    if not k:
        raise RuntimeError(
            "ALPACA_API_KEY not set — skipping Alpaca\n"
            "Free paper account at: https://alpaca.markets"
        )
    from hogan_bot.fetch_alpaca import fetch_all_alpaca
    result = fetch_all_alpaca(db_path=_db_path(), stock_days=10, include_spread=True)
    total = sum(result.values())
    if total == 0:
        raise RuntimeError("Alpaca refresh produced 0 rows — check API keys")


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

_SOURCES: list[tuple[str, str, Callable]] = [
    # ── Completely free (no key) ─────────────────────────────────────────────
    ("feargreed",    "Fear & Greed Index (Alternative.me, no key)",            _refresh_feargreed),
    ("gpr",          "GPR Index (Caldara & Iacoviello, free download)",        _refresh_gpr),
    ("derivatives",  "Kraken Futures — funding rate + open interest",          _refresh_derivatives),
    ("blockchain",   "BTC on-chain: hash rate, mempool, fees (no key)",        _refresh_blockchain),
    ("defillama",    "DeFi TVL + stablecoin mcap (DeFi Llama, no key)",        _refresh_defillama),
    ("spy",          "SPY daily macro candles (yfinance, free)",                _refresh_spy),
    ("openbb",       "OpenBB macro: DXY, VIX, SPY return, FOMC (yfinance)",   _refresh_openbb),
    # ── Free with API key ────────────────────────────────────────────────────
    ("coingecko",    "CoinGecko market intelligence (COINGECKO_KEY)",          _refresh_coingecko),
    ("cmc",          "CoinMarketCap: BTC/ETH dominance, market cap, DeFi% (CMC_API_KEY)", _refresh_cmc),
    ("fred",         "FRED macro: 10Y yield, M2, CPI, Fed rate (FRED_API_KEY)", _refresh_fred),
    ("news",         "CryptoPanic news sentiment 1 req/day (CRYPTOPANIC_KEY)", _refresh_news_sentiment),
    ("messari",      "Messari fundamentals: NVT, realized cap (MESSARI_KEY)",  _refresh_messari),
    ("alpaca",       "Alpaca: SPY close + BTC/ETH bid-ask spread (ALPACA_API_KEY)", _refresh_alpaca),
    ("oanda",        "Oanda prices: BTC/ETH/XAU/EUR mid (OANDA_ACCESS_TOKEN)", _refresh_oanda),
    # ── Paid / key-gated ────────────────────────────────────────────────────
    ("dune",         "Dune Analytics: BTC exchange flow, whales (DUNE_API_KEY)", _refresh_dune),
    ("onchain",      "CryptoQuant on-chain metrics (CRYPTOQUANT_KEY)",          _refresh_onchain),
    ("glassnode",    "Glassnode on-chain analytics (GLASSNODE_KEY)",            _refresh_glassnode),
    ("santiment",    "Santiment social/dev intelligence (SANTIMENT_KEY)",       _refresh_santiment),
]

_SOURCE_KEYS = [s[0] for s in _SOURCES]


def main() -> None:
    parser = argparse.ArgumentParser(description="Refresh all Hogan daily data sources")
    parser.add_argument(
        "--source", nargs="*", metavar="SRC",
        help=f"Run only specific sources. Choices: {', '.join(_SOURCE_KEYS)}",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print what would run without executing",
    )
    args = parser.parse_args()

    # Load .env if python-dotenv is available
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    active_keys = set(args.source) if args.source else set(_SOURCE_KEYS)
    unknown = active_keys - set(_SOURCE_KEYS)
    if unknown:
        sys.exit(f"Unknown sources: {', '.join(sorted(unknown))}. Valid: {', '.join(_SOURCE_KEYS)}")

    print(f"\n{_BOLD}Hogan Daily Data Refresh — {date.today().isoformat()}{_RESET}")
    if args.dry_run:
        print(f"{_YELLOW}DRY-RUN mode — no data will be fetched{_RESET}")

    results: dict[str, bool] = {}
    for key, description, fn in _SOURCES:
        if key not in active_keys:
            continue
        results[key] = _run_step(description, fn, args.dry_run)

    # Summary
    ok_count = sum(1 for v in results.values() if v)
    fail_count = len(results) - ok_count
    print(f"\n{_BOLD}Summary:{_RESET} {ok_count} succeeded, {fail_count} failed")
    if fail_count:
        _KEY_GATED = {"news", "onchain", "glassnode", "santiment", "coingecko", "messari", "dune", "oanda", "fred"}
        failed = [k for k, v in results.items() if not v]
        key_failures = [k for k in failed if k in _KEY_GATED]
        other_failures = [k for k in failed if k not in _KEY_GATED]
        if key_failures:
            print(
                f"  {_YELLOW}Key-gated:{_RESET} add API keys to .env for: "
                + ", ".join(key_failures)
            )
        if other_failures:
            print(
                f"  {_RED}Unexpected failures:{_RESET} {', '.join(other_failures)} "
                "— check network or run individually for details."
            )
    print()


if __name__ == "__main__":
    main()
