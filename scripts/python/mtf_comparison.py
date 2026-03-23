"""4-way MTF comparison with clean short-side attribution.

Configs:
  A: 1h baseline (no pullback gate, no MTF)
  B: 1h + pullback gate
  C: 1h + MTF execution (no pullback gate)
  D: 1h + pullback gate + MTF execution

All with shorts enabled, 16h short hold, db_path for macro/sentiment.
"""
from __future__ import annotations

import sys

sys.path.insert(0, ".")

from hogan_bot.backtest import (
    diagnose_longs_by_confidence,
    diagnose_shorts_by_confidence,
)
from hogan_bot.backtest_cli import _run_single
from hogan_bot.config import load_config
from hogan_bot.storage import get_connection, load_candles


def _extract_metrics(r, label: str) -> dict:
    shorts = [t for t in r.closed_trades if t.get("side") == "short"]
    longs = [t for t in r.closed_trades if t.get("side") == "long"]
    n_shorts = len(shorts)
    n_longs = len(longs)
    short_wins = sum(1 for t in shorts if t.get("pnl_usd", 0) > 0)
    long_wins = sum(1 for t in longs if t.get("pnl_usd", 0) > 0)
    short_pnls = [t.get("pnl_pct", 0) for t in shorts]
    long_pnls = [t.get("pnl_pct", 0) for t in longs]
    short_total = sum(short_pnls)
    long_total = sum(long_pnls)

    all_pnls = short_pnls + long_pnls
    all_wins = sum(1 for p in all_pnls if p > 0)
    all_n = len(all_pnls)

    winner_pnls = [p for p in all_pnls if p > 0]
    loser_pnls = [p for p in all_pnls if p <= 0]
    avg_win = sum(winner_pnls) / len(winner_pnls) if winner_pnls else 0
    avg_loss = sum(loser_pnls) / len(loser_pnls) if loser_pnls else 0
    payoff = abs(avg_win / avg_loss) if avg_loss else float("inf")

    expectancy = avg_win * (all_wins / all_n) + avg_loss * (1 - all_wins / all_n) if all_n else 0

    funnel = r.signal_funnel or {}

    return {
        "label": label,
        "return": round(r.total_return_pct, 4),
        "max_dd": round(r.max_drawdown_pct, 4),
        "sharpe": round(r.sharpe_ratio, 3),
        "calmar": round(r.calmar_ratio, 3),
        "trades": all_n,
        "win_rate": round(all_wins / all_n, 3) if all_n else 0,
        "expectancy": round(expectancy, 3),
        "payoff": round(payoff, 2),
        "n_long": n_longs,
        "long_wr": round(long_wins / n_longs, 3) if n_longs else 0,
        "long_total": round(long_total, 3),
        "n_short": n_shorts,
        "short_wr": round(short_wins / n_shorts, 3) if n_shorts else 0,
        "short_total": round(short_total, 3),
        "pb_blocked": funnel.get("pullback_blocked", 0),
        "pb_halved": funnel.get("pullback_halved", 0),
        "pb_resistance": funnel.get("pullback_blocked_resistance", 0),
        "mtf_created": funnel.get("mtf_thesis_created", 0),
        "mtf_executed": funnel.get("mtf_thesis_executed", 0),
        "mtf_expired": funnel.get("mtf_thesis_expired", 0),
    }


def main() -> None:
    cfg = load_config()
    conn = get_connection("data/hogan.db")
    candles_1h = load_candles(conn, "BTC/USD", "1h", limit=5000)
    candles_15m = load_candles(conn, "BTC/USD", "15m")
    conn.close()
    print(f"Loaded {len(candles_1h)} 1h candles, {len(candles_15m)} 15m candles\n")

    configs = [
        ("A: baseline", {"enable_pullback_gate": False, "candles_15m": None}),
        ("B: +pullback", {"enable_pullback_gate": True, "candles_15m": None}),
        ("C: +MTF", {"enable_pullback_gate": False, "candles_15m": candles_15m}),
        ("D: +PB+MTF", {"enable_pullback_gate": True, "candles_15m": candles_15m}),
    ]

    results = []
    for label, kwargs in configs:
        print(f"Running {label}...", flush=True)
        r = _run_single(
            cfg, candles_1h, "BTC/USD", None,
            enable_shorts=True,
            short_max_hold_hours=16.0,
            db_path="data/hogan.db",
            enable_pullback_gate=kwargs["enable_pullback_gate"],
            candles_15m=kwargs["candles_15m"],
            mtf_thesis_max_age=6,
        )
        m = _extract_metrics(r, label)
        results.append((m, r))
        print(
            f"  -> return={m['return']:+.4f}%, trades={m['trades']}, "
            f"sharpe={m['sharpe']:.3f}, calmar={m['calmar']:.3f}",
            flush=True,
        )

    print()
    print("=" * 130)
    print("4-WAY MTF COMPARISON (all with shorts=True, hold=16h, db_path)")
    print("=" * 130)

    hdr = (
        f"{'config':<14s}  {'return%':>8s}  {'max_dd%':>8s}  {'sharpe':>7s}  "
        f"{'calmar':>7s}  {'trades':>6s}  {'win%':>5s}  {'expect':>7s}  "
        f"{'payoff':>6s}  {'longs':>5s}  {'l_wr':>5s}  {'l_tot%':>7s}  "
        f"{'shorts':>6s}  {'s_wr':>5s}  {'s_tot%':>7s}"
    )
    print(hdr)
    print("-" * 130)
    for m, _ in results:
        print(
            f"{m['label']:<14s}  {m['return']:>+8.4f}  "
            f"{m['max_dd']:>8.4f}  {m['sharpe']:>7.3f}  "
            f"{m['calmar']:>7.3f}  {m['trades']:>6d}  "
            f"{m['win_rate']:>5.1%}  {m['expectancy']:>+7.3f}  "
            f"{m['payoff']:>6.2f}  {m['n_long']:>5d}  "
            f"{m['long_wr']:>5.1%}  {m['long_total']:>+7.3f}  "
            f"{m['n_short']:>6d}  {m['short_wr']:>5.1%}  "
            f"{m['short_total']:>+7.3f}"
        )

    print()
    print("PULLBACK + MTF FUNNEL:")
    hdr2 = (
        f"{'config':<14s}  {'pb_blk':>6s}  {'pb_res':>6s}  {'pb_hlf':>6s}  "
        f"{'mtf_crt':>7s}  {'mtf_exe':>7s}  {'mtf_exp':>7s}"
    )
    print(hdr2)
    print("-" * 70)
    for m, _ in results:
        print(
            f"{m['label']:<14s}  {m['pb_blocked']:>6d}  "
            f"{m['pb_resistance']:>6d}  {m['pb_halved']:>6d}  "
            f"{m['mtf_created']:>7d}  {m['mtf_executed']:>7d}  "
            f"{m['mtf_expired']:>7d}"
        )

    print()
    print("LONG CONFIDENCE DETAIL:")
    for m, r in results:
        lc = diagnose_longs_by_confidence(r.closed_trades)
        if lc:
            by_rc = lc.get("by_regime_confidence", {})
            if by_rc:
                print(f"  {m['label']}:")
                for k, v in by_rc.items():
                    print(
                        f"    {k:<28s}  n={v['count']:>2d}  "
                        f"wr={v['win_rate']:>5.1%}  avg={v['avg_pnl_pct']:>+7.3f}%  "
                        f"total={v['total_pnl_pct']:>+7.3f}%"
                    )

    print()
    print("SHORT CONFIDENCE DETAIL:")
    for m, r in results:
        sc = diagnose_shorts_by_confidence(r.closed_trades)
        if sc:
            by_rc = sc.get("by_regime_confidence", {})
            if by_rc:
                print(f"  {m['label']}:")
                for k, v in by_rc.items():
                    print(
                        f"    {k:<28s}  n={v['count']:>2d}  "
                        f"wr={v['win_rate']:>5.1%}  avg={v['avg_pnl_pct']:>+7.3f}%  "
                        f"total={v['total_pnl_pct']:>+7.3f}%"
                    )

    # Best config
    best = max(results, key=lambda x: x[0]["calmar"])
    print(f"\nBest by Calmar: {best[0]['label']} "
          f"(Calmar={best[0]['calmar']:.3f}, Sharpe={best[0]['sharpe']:.3f}, "
          f"return={best[0]['return']:+.4f}%)")


if __name__ == "__main__":
    main()
