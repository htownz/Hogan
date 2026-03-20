"""Unified data fetch orchestrator for Hogan's sentiment, macro, and on-chain data.

Runs all available fetchers in sequence with error isolation — one failing fetcher
never blocks the others.  Designed to be called at event_loop startup and then
periodically (e.g., every 4 hours) to keep DB tables fresh.

Usage::

    from hogan_bot.fetch_orchestrator import run_all_fetchers
    results = run_all_fetchers(db_path="data/hogan.db", symbols=["BTC/USD", "ETH/USD"])
    # results = {"feargreed": 1, "derivatives": {"funding_rate": 1, ...}, ...}

Or as standalone::

    python -m hogan_bot.fetch_orchestrator --db data/hogan.db --backfill
"""
from __future__ import annotations

import argparse
import logging
import os
import time
from typing import Any

logger = logging.getLogger(__name__)


def run_all_fetchers(
    db_path: str = "data/hogan.db",
    symbols: list[str] | None = None,
    backfill: bool = False,
) -> dict[str, Any]:
    """Run all available data fetchers with error isolation.

    Each fetcher runs in a try/except so failures are logged but never
    propagate.  Returns a dict mapping fetcher name to its result (or
    the exception string on failure).
    """
    if symbols is None:
        symbols = ["BTC/USD"]
    results: dict[str, Any] = {}
    primary_symbol = symbols[0] if symbols else "BTC/USD"

    _start = time.time()
    logger.info("FETCH_ORCHESTRATOR: starting data refresh (backfill=%s, symbols=%s)", backfill, symbols)

    # 1. Fear & Greed Index (free, no auth)
    try:
        from hogan_bot.fetch_feargreed import fetch_and_store as fetch_fg
        rows = fetch_fg(symbol=primary_symbol, db_path=db_path, backfill=backfill)
        results["feargreed"] = rows
        logger.info("FETCH feargreed: %d rows", rows)
    except Exception as exc:
        results["feargreed"] = f"ERROR: {exc}"
        logger.warning("FETCH feargreed failed: %s", exc)

    # 2. Derivatives — funding rate + OI (free, Kraken Futures public API)
    for sym in symbols:
        _key = f"derivatives_{sym}"
        try:
            from hogan_bot.fetch_derivatives import fetch_derivatives
            res = fetch_derivatives(symbol=sym, db_path=db_path)
            results[_key] = res
            logger.info("FETCH derivatives %s: %s", sym, res)
        except Exception as exc:
            results[_key] = f"ERROR: {exc}"
            logger.warning("FETCH derivatives %s failed: %s", sym, exc)

    # 3. Macro candles — VIX, DXY, SPY, Gold, etc. (free, yfinance)
    try:
        if backfill:
            from hogan_bot.fetch_macro_candles import backfill_macro_candles
            res = backfill_macro_candles(db_path=db_path)
        else:
            from hogan_bot.fetch_macro_candles import fetch_all_macro_candles
            res = fetch_all_macro_candles(db_path=db_path)
        results["macro_candles"] = res
        logger.info("FETCH macro_candles: %s", res)
    except Exception as exc:
        results["macro_candles"] = f"ERROR: {exc}"
        logger.warning("FETCH macro_candles failed: %s", exc)

    # 4. News sentiment (requires CRYPTOPANIC_KEY)
    _cp_key = os.getenv("CRYPTOPANIC_KEY", "")
    if _cp_key:
        try:
            from hogan_bot.fetch_news_sentiment import fetch_and_store as fetch_news
            rows = fetch_news(symbol=primary_symbol, db_path=db_path, pages=1)
            results["news_sentiment"] = rows
            logger.info("FETCH news_sentiment: %d rows", rows)
        except Exception as exc:
            results["news_sentiment"] = f"ERROR: {exc}"
            logger.warning("FETCH news_sentiment failed: %s", exc)
    else:
        results["news_sentiment"] = "SKIPPED (no CRYPTOPANIC_KEY)"
        logger.info("FETCH news_sentiment: skipped (no CRYPTOPANIC_KEY env var)")

    # 5. FRED macro data (free, no auth)
    try:
        from hogan_bot.fetch_fred import fetch_and_store as fetch_fred
        rows = fetch_fred(db_path=db_path)
        results["fred"] = rows
        logger.info("FETCH fred: %s", rows)
    except ImportError:
        results["fred"] = "SKIPPED (module not available)"
    except Exception as exc:
        results["fred"] = f"ERROR: {exc}"
        logger.warning("FETCH fred failed: %s", exc)

    # 6. CoinGecko market data (free, no auth)
    try:
        from hogan_bot.fetch_coingecko import fetch_and_store as fetch_cg
        rows = fetch_cg(db_path=db_path)
        results["coingecko"] = rows
        logger.info("FETCH coingecko: %s", rows)
    except ImportError:
        results["coingecko"] = "SKIPPED (module not available)"
    except Exception as exc:
        results["coingecko"] = f"ERROR: {exc}"
        logger.warning("FETCH coingecko failed: %s", exc)

    # 7. DeFi Llama TVL data (free, no auth)
    try:
        from hogan_bot.fetch_defillama import fetch_and_store as fetch_dl
        rows = fetch_dl(db_path=db_path)
        results["defillama"] = rows
        logger.info("FETCH defillama: %s", rows)
    except ImportError:
        results["defillama"] = "SKIPPED (module not available)"
    except Exception as exc:
        results["defillama"] = f"ERROR: {exc}"
        logger.warning("FETCH defillama failed: %s", exc)

    # 8. Blockchain/on-chain data (free where possible)
    try:
        from hogan_bot.fetch_blockchain import fetch_and_store as fetch_bc
        rows = fetch_bc(symbol=primary_symbol, db_path=db_path)
        results["blockchain"] = rows
        logger.info("FETCH blockchain: %s", rows)
    except ImportError:
        results["blockchain"] = "SKIPPED (module not available)"
    except Exception as exc:
        results["blockchain"] = f"ERROR: {exc}"
        logger.warning("FETCH blockchain failed: %s", exc)

    # 9. GPR (Geopolitical Risk) index
    try:
        from hogan_bot.fetch_gpr import fetch_and_store as fetch_gpr
        rows = fetch_gpr(db_path=db_path)
        results["gpr"] = rows
        logger.info("FETCH gpr: %s", rows)
    except ImportError:
        results["gpr"] = "SKIPPED (module not available)"
    except Exception as exc:
        results["gpr"] = f"ERROR: {exc}"
        logger.warning("FETCH gpr failed: %s", exc)

    _elapsed = time.time() - _start
    _ok = sum(1 for v in results.values() if not isinstance(v, str) or not v.startswith(("ERROR", "SKIPPED")))
    _fail = sum(1 for v in results.values() if isinstance(v, str) and v.startswith("ERROR"))
    _skip = sum(1 for v in results.values() if isinstance(v, str) and v.startswith("SKIPPED"))

    logger.info(
        "FETCH_ORCHESTRATOR: done in %.1fs — %d OK, %d failed, %d skipped",
        _elapsed, _ok, _fail, _skip,
    )
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description="Run all Hogan data fetchers")
    parser.add_argument("--db", default=os.getenv("HOGAN_DB_PATH", "data/hogan.db"))
    parser.add_argument("--symbols", default="BTC/USD,ETH/USD")
    parser.add_argument("--backfill", action="store_true")
    args = parser.parse_args()
    results = run_all_fetchers(
        db_path=args.db,
        symbols=args.symbols.split(","),
        backfill=args.backfill,
    )
    for k, v in results.items():
        print(f"  {k}: {v}")
