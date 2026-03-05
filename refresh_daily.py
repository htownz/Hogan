"""Daily data refresh script — run once per day before training or live trading.

Refreshes all external data sources in dependency order:
  1. Fear & Greed Index        (free, Alternative.me)
  2. CoinGecko macro data      (free demo key)
  3. GPR Index                 (free, academic download)
  4. Kraken Futures / Deriv.   (free public API)
  5. CryptoPanic news          (free token)
  6. CryptoQuant on-chain      (paid — skipped if CRYPTOQUANT_KEY absent)
  7. Glassnode                 (paid — skipped if GLASSNODE_KEY absent)
  8. Santiment                 (paid — skipped if SANTIMENT_KEY absent)
  9. SPY macro backfill        (yfinance, free)

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
    # pages=5 → ~100 recent posts; reads CRYPTOPANIC_KEY from env internally
    fetch_and_store(pages=5)


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


# ---------------------------------------------------------------------------
# Source registry
# ---------------------------------------------------------------------------

_SOURCES: list[tuple[str, str, Callable]] = [
    # free sources — run every day
    ("feargreed",    "Fear & Greed Index (Alternative.me, no key)",       _refresh_feargreed),
    ("coingecko",    "CoinGecko macro data (COINGECKO_KEY required)",     _refresh_coingecko),
    ("gpr",          "GPR Index (Caldara & Iacoviello, free download)",   _refresh_gpr),
    ("derivatives",  "Kraken Futures — funding rate + open interest",     _refresh_derivatives),
    ("spy",          "SPY daily macro candles (yfinance, free)",          _refresh_spy),
    # paid / key-gated sources
    ("news",         "CryptoPanic news sentiment (CRYPTOPANIC_KEY)",      _refresh_news_sentiment),
    ("onchain",      "CryptoQuant on-chain metrics (CRYPTOQUANT_KEY)",    _refresh_onchain),
    ("glassnode",    "Glassnode on-chain analytics (GLASSNODE_KEY)",      _refresh_glassnode),
    ("santiment",    "Santiment social/dev intelligence (SANTIMENT_KEY)", _refresh_santiment),
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
        _KEY_GATED = {"news", "onchain", "glassnode", "santiment", "coingecko"}
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
