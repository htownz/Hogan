"""Paper test EUR/USD and GBP/USD via Oanda.

Runs Hogan's core signal pipeline against FX pairs using Oanda's practice
environment.  No real money is at risk.

Prerequisites::

    1. Set up .env with Oanda credentials:
       OANDA_ACCESS_TOKEN=<token>
       OANDA_ACCOUNT_ID=<account-id>
       OANDA_ENVIRONMENT=practice

    2. Run:
       python scripts/paper_fx.py

    3. Monitor:
       python scripts/dashboards/dashboard.py

The script:
- Fetches 15m + 1h candles from Oanda for EUR/USD and GBP/USD
- Runs the agent pipeline with session-aware filters
- Simulates paper trades with realistic fills
- Logs expectancy metrics per session / per symbol
- Writes results to data/hogan_fx.db
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import time
from datetime import datetime, timezone

# Ensure hogan_bot is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from hogan_bot.agent_pipeline import AgentPipeline
from hogan_bot.config import BotConfig, load_config
from hogan_bot.decision import (
    edge_gate, entry_quality_gate,
    estimate_spread_from_candles,
)
from hogan_bot.expectancy import ExpectancyTracker
from hogan_bot.fx_utils import (
    SessionFilter, current_session, fx_position_size,
    pip_stop_loss, pip_take_profit, pip_size, is_weekend,
)
from hogan_bot.oanda_client import OandaClient
from hogan_bot.paper import PaperPortfolio
from hogan_bot.risk import DrawdownGuard
from hogan_bot.storage import get_connection, record_equity, open_paper_trade, close_paper_trade

logger = logging.getLogger(__name__)

_FX_SYMBOLS = ["EUR/USD", "GBP/USD"]
_PRIMARY_TF = "15m"
_CONTEXT_TF = "1h"
_POLL_INTERVAL_S = 60 * 15  # 15 minutes


def _run_once(
    client: OandaClient,
    pipeline: AgentPipeline,
    portfolio: PaperPortfolio,
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

            # Run the agent pipeline
            signal = pipeline.run(candles, symbol=symbol)
            action = signal.action

            # Spread-aware edge gate
            spread_est = estimate_spread_from_candles(candles)
            atr_pct = signal.stop_distance_pct / max(2.5, 1.0)
            action = edge_gate(
                action,
                atr_pct=atr_pct,
                take_profit_pct=config.take_profit_pct,
                fee_rate=0.00005,  # FX spread cost is ~0.5 pip ≈ ~0.5 bps
                estimated_spread=spread_est,
            )

            # Entry quality gate
            tech_conf = signal.tech.confidence if signal.tech else None
            action, quality_scale = entry_quality_gate(
                action,
                final_confidence=signal.confidence,
                tech_confidence=tech_conf,
            )

            if action == "hold":
                logger.debug("HOLD %s | conf=%.2f session=%s", symbol, signal.confidence, session)
                continue

            # Pip-based position sizing
            stop_pips = 30.0 if symbol == "EUR/USD" else 40.0
            size = fx_position_size(
                account_balance=equity,
                risk_pct=config.max_risk_per_trade * quality_scale * session_scale,
                stop_pips=stop_pips,
                symbol=symbol,
                price=px,
            )

            if size < 100:
                logger.debug("Size too small for %s: %.0f units", symbol, size)
                continue

            now_ms = int(time.time() * 1000)

            if action == "buy":
                if symbol in portfolio.positions:
                    logger.debug("Already long %s", symbol)
                    continue

                sl = pip_stop_loss(symbol, px, "long", stop_pips)
                tp = pip_take_profit(symbol, px, "long", stop_pips * 2)

                ok = portfolio.execute_buy(
                    symbol, px, size,
                    trailing_stop_pct=stop_pips * pip_size(symbol) / px,
                    take_profit_pct=(tp - px) / px,
                )
                if ok:
                    fee = size * px * 0.00005
                    open_paper_trade(conn, symbol, "long", px, size, fee, now_ms,
                                     ml_up_prob=None, strategy_conf=signal.confidence)
                    logger.info(
                        "FX_BUY %s units=%.0f px=%.5f sl=%.5f tp=%.5f session=%s",
                        symbol, size, px, sl, tp, session,
                    )

            elif action == "sell":
                pos = portfolio.positions.get(symbol)
                if pos is None:
                    logger.debug("No position to sell for %s", symbol)
                    continue

                qty = pos.qty
                avg_entry = pos.avg_entry
                ok = portfolio.execute_sell(symbol, px, qty)
                if ok:
                    fee = qty * px * 0.00005
                    close_paper_trade(conn, symbol, "long", px, fee, now_ms, close_reason="signal")
                    gross_pnl_pct = (px - avg_entry) / avg_entry if avg_entry else 0
                    expectancy.record_trade(
                        symbol=symbol,
                        regime=session,
                        gross_pnl_pct=gross_pnl_pct,
                        net_pnl_pct=gross_pnl_pct - 0.0001,
                        hold_bars=getattr(pos, "bars_held", 0),
                        close_reason="signal",
                        mae_pct=getattr(pos, "max_adverse_pct", 0.0),
                        mfe_pct=getattr(pos, "max_favorable_pct", 0.0),
                    )
                    logger.info(
                        "FX_SELL %s units=%.0f px=%.5f pnl=%.2f%% session=%s",
                        symbol, qty, px, gross_pnl_pct * 100, session,
                    )

            # Check auto exits
            exits = portfolio.check_exits(mark_prices, max_hold_bars=96)
            for exit_sym, exit_reason in exits:
                epos = portfolio.positions.get(exit_sym)
                if epos is None:
                    continue
                eqty = epos.qty
                eavg = epos.avg_entry
                portfolio.execute_sell(exit_sym, px, eqty)
                fee = eqty * px * 0.00005
                close_paper_trade(conn, exit_sym, "long", px, fee, now_ms, close_reason=exit_reason)
                gross = (px - eavg) / eavg if eavg else 0
                expectancy.record_trade(
                    symbol=exit_sym, regime=session,
                    gross_pnl_pct=gross, net_pnl_pct=gross - 0.0001,
                    hold_bars=getattr(epos, "bars_held", 0),
                    close_reason=exit_reason,
                )
                logger.info("FX_EXIT %s reason=%s px=%.5f", exit_sym, exit_reason, px)

            # Record equity
            equity = portfolio.total_equity(mark_prices)
            dd = max(0, (guard.peak_equity - equity) / guard.peak_equity) if guard.peak_equity > 0 else 0
            record_equity(conn, now_ms, portfolio.cash_usd, equity, dd)
            guard.update_and_check(equity)

        except Exception as exc:
            logger.error("Error processing %s: %s", symbol, exc, exc_info=True)

    # Periodic expectancy log
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

    parser = argparse.ArgumentParser(description="Hogan FX Paper Trading")
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
    portfolio = PaperPortfolio(cash_usd=args.balance, fee_rate=0.00005)
    guard = DrawdownGuard(args.balance, config.max_drawdown)
    expectancy = ExpectancyTracker()
    session_filter = SessionFilter()

    pipeline = AgentPipeline(config, conn=conn)

    logger.info(
        "FX paper trading started: symbols=%s balance=%.2f",
        args.symbols, args.balance,
    )

    if args.once:
        _run_once(client, pipeline, portfolio, guard, expectancy, session_filter, conn, config, args.symbols)
    else:
        while True:
            try:
                _run_once(client, pipeline, portfolio, guard, expectancy, session_filter, conn, config, args.symbols)
            except KeyboardInterrupt:
                break
            except Exception as exc:
                logger.error("Loop error: %s", exc, exc_info=True)

            logger.info("Sleeping %ds until next evaluation...", args.interval)
            time.sleep(args.interval)

    # Final report
    if expectancy._trades:
        logger.info("FINAL FX EXPECTANCY: %s", expectancy.summary())

    conn.close()
    logger.info("FX paper trading stopped.")


if __name__ == "__main__":
    main()
