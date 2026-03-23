"""Paper test EUR/USD and GBP/USD via Oanda.

**EXPLORATORY** — this script is a standalone FX paper test loop.  It does
NOT use the canonical ``event_loop`` runtime path.  For production paper/live
FX trading, configure the Oanda adapter in ``.env`` and run::

    python -m hogan_bot.event_loop

This script is useful for quick iteration on FX-specific logic (session
filtering, pip-based sizing, spread modeling) before promoting changes into
the main runtime.

Prerequisites::

    1. Set up .env with Oanda credentials:
       OANDA_ACCESS_TOKEN=<token>
       OANDA_ACCOUNT_ID=<account-id>
       OANDA_ENVIRONMENT=practice

    2. Run:
       python scripts/paper_fx.py

    3. Monitor:
       python scripts/dashboards/dashboard.py
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hogan_bot.agent_pipeline import AgentPipeline
from hogan_bot.config import BotConfig, load_config
from hogan_bot.decision import (
    edge_gate,
    entry_quality_gate,
    estimate_spread_from_candles,
)
from hogan_bot.execution import FillSimConfig, RealisticPaperExecution
from hogan_bot.expectancy import ExpectancyTracker
from hogan_bot.fx_utils import (
    SessionFilter,
    current_session,
    fx_position_size,
    is_weekend,
    pip_size,
    pip_take_profit,
)
from hogan_bot.instrument_profiles import get_profile, spread_cost_bps
from hogan_bot.oanda_client import OandaClient
from hogan_bot.paper import PaperPortfolio
from hogan_bot.risk import DrawdownGuard
from hogan_bot.storage import (
    close_paper_trade,
    get_connection,
    open_paper_trade,
    record_equity,
)

logger = logging.getLogger(__name__)

_FX_SYMBOLS = ["EUR/USD", "GBP/USD"]
_PRIMARY_TF = "15m"
_CONTEXT_TF = "1h"
_POLL_INTERVAL_S = 60 * 15


def _fx_fee_rate(symbol: str) -> float:
    """Realistic FX cost model: spread-based, not a flat percentage.

    For EUR/USD, Oanda typical spread is ~1.0-1.5 pips.
    At price 1.0850, 1 pip = 0.0001, so spread = 0.0001/1.0850 ≈ 0.92 bps.
    We model this as the spread cost in fractional terms.
    """
    profile = get_profile(symbol)
    return profile.typical_spread_bps / 10_000.0


def _run_once(
    client: OandaClient,
    pipeline: AgentPipeline,
    portfolio: PaperPortfolio,
    executor: RealisticPaperExecution,
    guard: DrawdownGuard,
    expectancy: ExpectancyTracker,
    session_filter: SessionFilter,
    conn,
    config: BotConfig,
    symbols: list[str],
) -> None:
    """Single evaluation cycle across all FX symbols."""
    now = datetime.now(timezone.utc)

    if is_weekend(now):
        logger.info("FX markets closed (weekend) — skipping")
        return

    allowed, session_scale, reason = session_filter.should_trade(now)
    session = current_session(now)
    logger.info("Session check: %s (scale=%.2f) — %s", session, session_scale, reason)

    if not allowed:
        logger.info("Trading blocked: %s", reason)
        return

    for symbol in symbols:
        try:
            candles = client.fetch_candles(symbol, timeframe=_PRIMARY_TF, count=200)
            if candles.empty or len(candles) < 50:
                logger.warning("Not enough candles for %s (%d)", symbol, len(candles))
                continue

            px = float(candles["close"].iloc[-1])
            mark_prices = {symbol: px}
            equity = portfolio.total_equity(mark_prices)
            fee_rate = _fx_fee_rate(symbol)
            profile = get_profile(symbol)

            signal = pipeline.run(candles, symbol=symbol)
            action = signal.action

            spread_est = estimate_spread_from_candles(candles)
            atr_pct = signal.stop_distance_pct / max(2.5, 1.0)
            _edge_gd = edge_gate(
                action,
                atr_pct=atr_pct,
                take_profit_pct=profile.default_tp_pct,
                fee_rate=fee_rate,
                min_edge_multiple=config.min_edge_multiple,
                estimated_spread=spread_est,
            )
            action = _edge_gd.action

            tech_conf = signal.tech.confidence if signal.tech else None
            _quality_gd = entry_quality_gate(
                action,
                final_confidence=signal.confidence,
                tech_confidence=tech_conf,
                min_final_confidence=config.min_final_confidence,
                min_tech_confidence=config.min_tech_confidence,
            )
            action = _quality_gd.action
            quality_scale = _quality_gd.size_scale

            if action == "hold":
                logger.debug("HOLD %s | conf=%.2f session=%s", symbol, signal.confidence, session)
                continue

            stop_pips = profile.default_stop_pips
            tp_pips = profile.default_tp_pips
            size = fx_position_size(
                account_balance=equity,
                risk_pct=config.max_risk_per_trade * quality_scale * session_scale,
                stop_pips=stop_pips,
                symbol=symbol,
                price=px,
            )

            if size < profile.min_trade_size:
                logger.debug("Size too small for %s: %.0f units", symbol, size)
                continue

            now_ms = int(time.time() * 1000)

            if action == "buy":
                if symbol in portfolio.positions:
                    logger.debug("Already long %s", symbol)
                    continue
                # Cover any existing short first
                spos = portfolio.short_positions.get(symbol)
                if spos and spos.qty > 0:
                    executor.close_short(symbol, px, spos.qty, reason="flip_to_long")
                    close_paper_trade(conn, symbol, "short", px, spos.qty * px * fee_rate, now_ms,
                                      close_reason="flip_to_long")

                tp = pip_take_profit(symbol, px, "long", tp_pips)
                res = executor.open_long(
                    symbol, px, size,
                    trailing_stop_pct=stop_pips * pip_size(symbol) / px,
                    take_profit_pct=(tp - px) / px,
                )
                if res.ok:
                    fee = size * px * fee_rate
                    open_paper_trade(conn, symbol, "long", px, size, fee, now_ms,
                                     strategy_conf=signal.confidence)
                    logger.info(
                        "FX_BUY %s units=%.0f px=%.5f stop=%.0fpips tp=%.0fpips session=%s",
                        symbol, size, px, stop_pips, tp_pips, session,
                    )

            elif action == "sell":
                # Close any long first
                pos = portfolio.positions.get(symbol)
                if pos is not None:
                    qty = pos.qty
                    avg_entry = pos.avg_entry
                    res = executor.close_long(symbol, px, qty, reason="signal")
                    if res.ok:
                        fee = qty * px * fee_rate
                        close_paper_trade(conn, symbol, "long", px, fee, now_ms, close_reason="signal")
                        gross_pnl_pct = (px - avg_entry) / avg_entry if avg_entry else 0
                        expectancy.record_trade(
                            symbol=symbol, regime=session,
                            gross_pnl_pct=gross_pnl_pct,
                            net_pnl_pct=gross_pnl_pct - fee_rate * 2,
                            hold_bars=getattr(pos, "bars_held", 0),
                            close_reason="signal",
                            mae_pct=getattr(pos, "max_adverse_pct", 0.0),
                            mfe_pct=getattr(pos, "max_favorable_pct", 0.0),
                        )
                        logger.info(
                            "FX_SELL %s units=%.0f px=%.5f pnl=%.2f%% session=%s",
                            symbol, qty, px, gross_pnl_pct * 100, session,
                        )

                # Open a short if allowed
                if config.allow_shorts and symbol not in portfolio.short_positions:
                    tp = pip_take_profit(symbol, px, "short", tp_pips)
                    stop_pips * pip_size(symbol) / px
                    (px - tp) / px
                    portfolio.execute_short(symbol, px, size)
                    open_paper_trade(conn, symbol, "short", px, size, size * px * fee_rate, now_ms,
                                     strategy_conf=signal.confidence)
                    logger.info(
                        "FX_SHORT %s units=%.0f px=%.5f stop=%.0fpips tp=%.0fpips session=%s",
                        symbol, size, px, stop_pips, tp_pips, session,
                    )

            # Auto exits
            exits = portfolio.check_exits(mark_prices, max_hold_bars=96)
            for exit_sym, exit_reason in exits:
                epos = portfolio.positions.get(exit_sym)
                if epos is not None:
                    eqty = epos.qty
                    eavg = epos.avg_entry
                    res = executor.close_long(exit_sym, px, eqty, reason=exit_reason)
                    if res.ok:
                        fee = eqty * px * _fx_fee_rate(exit_sym)
                        close_paper_trade(conn, exit_sym, "long", px, fee, now_ms, close_reason=exit_reason)
                        gross = (px - eavg) / eavg if eavg else 0
                        expectancy.record_trade(
                            symbol=exit_sym, regime=session,
                            gross_pnl_pct=gross,
                            net_pnl_pct=gross - _fx_fee_rate(exit_sym) * 2,
                            hold_bars=getattr(epos, "bars_held", 0),
                            close_reason=exit_reason,
                        )
                        logger.info("FX_EXIT %s reason=%s px=%.5f", exit_sym, exit_reason, px)

            equity = portfolio.total_equity(mark_prices)
            dd = max(0, (guard.peak_equity - equity) / guard.peak_equity) if guard.peak_equity > 0 else 0
            record_equity(conn, now_ms, portfolio.cash_usd, equity, dd)
            guard.update_and_check(equity)

        except Exception as exc:
            logger.error("Error processing %s: %s", symbol, exc, exc_info=True)

    if expectancy._trades:
        report = expectancy.summary()
        overall = report.get("overall", {})
        logger.info(
            "FX EXPECTANCY [%d trades] win=%.1f%% net_edge=%.4f%% payoff=%.2f exp=%.4f%%",
            report["total_trades"],
            overall.get("win_rate", 0) * 100,
            overall.get("avg_net_edge_pct", 0),
            overall.get("payoff_ratio", 0),
            overall.get("expectancy_pct", 0),
        )
        for session_name, stats in report.get("by_regime", {}).items():
            logger.info(
                "  session=%s n=%d win=%.1f%% exp=%.4f%%",
                session_name, stats["n"], stats["win_rate"] * 100, stats["expectancy_pct"],
            )


def main():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
    )

    parser = argparse.ArgumentParser(description="Hogan FX Paper Trading (Exploratory)")
    parser.add_argument("--symbols", nargs="+", default=_FX_SYMBOLS)
    parser.add_argument("--balance", type=float, default=10_000.0)
    parser.add_argument("--db", default="data/hogan_fx.db")
    parser.add_argument("--once", action="store_true", help="Run once and exit")
    parser.add_argument("--interval", type=int, default=_POLL_INTERVAL_S)
    args = parser.parse_args()

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    client = OandaClient()
    logger.info("Connected to Oanda (%s) account=%s", client.environment, client.account_id)

    config = load_config()
    conn = get_connection(args.db)

    fx_spread_bps = spread_cost_bps(args.symbols[0]) if args.symbols else 1.0
    portfolio = PaperPortfolio(cash_usd=args.balance, fee_rate=fx_spread_bps / 10_000.0)
    fill_cfg = FillSimConfig(
        slippage_bps=0.5,
        spread_half_bps=fx_spread_bps / 2,
    )
    executor = RealisticPaperExecution(
        portfolio=portfolio, conn=conn, config=fill_cfg,
    )
    guard = DrawdownGuard(args.balance, config.max_drawdown)
    expectancy = ExpectancyTracker()
    session_filter = SessionFilter()

    pipeline = AgentPipeline(config, conn=conn)

    logger.info(
        "FX paper trading started (EXPLORATORY): symbols=%s balance=%.2f spread=%.1fbps",
        args.symbols, args.balance, fx_spread_bps,
    )

    if args.once:
        _run_once(client, pipeline, portfolio, executor, guard, expectancy, session_filter, conn, config, args.symbols)
    else:
        while True:
            try:
                _run_once(client, pipeline, portfolio, executor, guard, expectancy, session_filter, conn, config, args.symbols)
            except KeyboardInterrupt:
                break
            except Exception as exc:
                logger.error("Loop error: %s", exc, exc_info=True)

            logger.info("Sleeping %ds until next evaluation...", args.interval)
            time.sleep(args.interval)

    if expectancy._trades:
        logger.info("FINAL FX EXPECTANCY: %s", expectancy.summary())

    conn.close()
    logger.info("FX paper trading stopped.")


if __name__ == "__main__":
    main()
