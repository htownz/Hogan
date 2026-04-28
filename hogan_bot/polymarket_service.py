"""Standalone Polymarket runtime for Hogan.

This entrypoint keeps Polymarket scanning operationally separate from the
exchange-trading event loop while reusing Hogan's shared data and models.
It does not place real Polymarket orders or load wallet credentials.
"""
from __future__ import annotations

import argparse
import logging
import os
import time

from hogan_bot.polymarket_alpha import (
    AlphaRunResult,
    RecommendationRunResult,
    print_recommendations,
    run_alpha_lab,
    run_recommendations_only,
)

logger = logging.getLogger(__name__)


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return float(raw)


def _print_scan_summary(result: AlphaRunResult) -> None:
    print(f"Report: {result.report_path}")
    print(f"Opportunities: {len(result.opportunities)}")
    print(f"Shadow trades opened: {result.shadow_opened}")
    print(f"Shadow trades closed: {result.shadow_ledger.closed}")
    print(f"Shadow unrealized PnL: {result.shadow_ledger.unrealized_pnl:.2f}")
    print(f"Hogan BTC probability: {result.btc_prob:.4f}" if result.btc_prob is not None else "Hogan BTC probability: n/a")
    print(f"Hogan ETH probability: {result.eth_prob:.4f}" if result.eth_prob is not None else "Hogan ETH probability: n/a")
    print(f"Arbitrage alerts: {result.arbitrage_alerts}")
    print(f"Promotion approved: {result.promotion_approved}")
    print(f"Authority mode: {result.authority_mode}")
    if result.promotion_reasons:
        print("Promotion blockers:", ", ".join(result.promotion_reasons))


def run_service_once(args: argparse.Namespace) -> AlphaRunResult | RecommendationRunResult:
    """Run one Polymarket service iteration."""
    include_clob = not args.no_clob
    use_long_horizon_model = not args.no_long_horizon_model
    if args.mode == "recommendations-only":
        result = run_recommendations_only(
            db_path=args.db,
            symbol=args.symbol,
            limit=args.limit,
            include_clob=include_clob,
            clob_limit=args.clob_limit,
            btc_prob=args.btc_prob,
            eth_prob=args.eth_prob,
            btc_long_prob=args.btc_long_prob,
            eth_long_prob=args.eth_long_prob,
            use_long_horizon_model=use_long_horizon_model,
            watchlist_ev_margin=args.watchlist_ev_margin,
        )
        print_recommendations(result, limit=args.recommendation_limit)
        return result

    result = run_alpha_lab(
        db_path=args.db,
        symbol=args.symbol,
        limit=args.limit,
        include_clob=include_clob,
        clob_limit=args.clob_limit,
        btc_prob=args.btc_prob,
        eth_prob=args.eth_prob,
        btc_long_prob=args.btc_long_prob,
        eth_long_prob=args.eth_long_prob,
        use_long_horizon_model=use_long_horizon_model,
        watchlist_ev_margin=args.watchlist_ev_margin,
        auto_shadow=not args.no_auto_shadow,
        authority_mode=args.authority_mode,
        max_open_shadow_trades=args.max_open_shadow_trades,
        max_open_shadow_exposure_usd=args.max_open_shadow_exposure,
        report_dir=args.report_dir,
    )
    _print_scan_summary(result)
    return result


def run_service(args: argparse.Namespace) -> int:
    """Run scan/recommendations once, or daemon loop until stopped."""
    if args.mode != "daemon":
        run_service_once(args)
        return 0

    iteration = 0
    while True:
        iteration += 1
        logger.info("Polymarket daemon iteration %d starting", iteration)
        run_service_once(args)
        if args.iterations is not None and iteration >= args.iterations:
            return 0
        time.sleep(max(0.0, float(args.interval_minutes)) * 60.0)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run the standalone Hogan Polymarket program",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--mode",
        choices=("scan", "daemon", "recommendations-only"),
        default=os.getenv("HOGAN_POLYMARKET_MODE", "scan"),
    )
    p.add_argument("--db", default=os.getenv("HOGAN_DB_PATH", "data/hogan.db"))
    p.add_argument("--symbol", default=os.getenv("HOGAN_POLYMARKET_SYMBOL", "BTC/USD"))
    p.add_argument("--limit", type=int, default=_env_int("HOGAN_POLYMARKET_LIMIT", 100))
    p.add_argument("--no-clob", action="store_true", default=not _env_bool("HOGAN_POLYMARKET_CLOB", True))
    p.add_argument("--clob-limit", type=int, default=_env_int("HOGAN_POLYMARKET_CLOB_LIMIT", 12))
    p.add_argument("--btc-prob", type=float, default=None)
    p.add_argument("--eth-prob", type=float, default=None)
    p.add_argument("--btc-long-prob", type=float, default=None)
    p.add_argument("--eth-long-prob", type=float, default=None)
    p.add_argument("--no-long-horizon-model", action="store_true", default=not _env_bool("HOGAN_POLYMARKET_LONG_HORIZON_MODEL", True))
    p.add_argument(
        "--authority-mode",
        choices=("research", "shadow", "advisory", "conditional"),
        default=os.getenv("HOGAN_POLYMARKET_AUTHORITY_MODE", "research"),
    )
    p.add_argument("--max-open-shadow-trades", type=int, default=_env_int("HOGAN_POLYMARKET_MAX_OPEN_SHADOW_TRADES", 10))
    p.add_argument("--max-open-shadow-exposure", type=float, default=_env_float("HOGAN_POLYMARKET_MAX_OPEN_SHADOW_EXPOSURE", 250.0))
    p.add_argument("--recommendation-limit", type=int, default=_env_int("HOGAN_POLYMARKET_RECOMMENDATION_LIMIT", 10))
    p.add_argument("--watchlist-ev-margin", type=float, default=_env_float("HOGAN_POLYMARKET_WATCHLIST_EV_MARGIN", 0.02))
    p.add_argument("--no-auto-shadow", action="store_true", default=not _env_bool("HOGAN_POLYMARKET_AUTO_SHADOW", True))
    p.add_argument("--report-dir", default=os.getenv("HOGAN_POLYMARKET_REPORT_DIR", "reports/polymarket"))
    p.add_argument("--interval-minutes", type=float, default=_env_float("HOGAN_POLYMARKET_INTERVAL_MINUTES", 30.0))
    p.add_argument("--iterations", type=int, default=None, help="Daemon loop limit for tests or bounded runs")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    raise SystemExit(run_service(parse_args(argv)))


if __name__ == "__main__":
    main()
