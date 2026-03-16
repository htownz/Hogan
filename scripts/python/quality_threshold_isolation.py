"""Isolate trending_down quality_final_mult: 0.50 (current) vs 0.80 (old).

Both configs use 16h short max hold. This tells us whether the quality
threshold change adds value or is just along for the ride.
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

from hogan_bot.config import load_config, DEFAULT_REGIME_CONFIGS
from hogan_bot.storage import get_connection, load_candles
from hogan_bot.backtest import diagnose_shorts_by_confidence
from hogan_bot.backtest_cli import _run_single


def _run_with_quality_mult(cfg, candles, mult: float) -> dict:
    """Run backtest after patching trending_down quality_final_mult."""
    original = DEFAULT_REGIME_CONFIGS["trending_down"].quality_final_mult
    try:
        DEFAULT_REGIME_CONFIGS["trending_down"].quality_final_mult = mult
        r = _run_single(
            cfg, candles, "BTC/USD", None,
            enable_shorts=True,
            short_max_hold_hours=16.0,
            db_path="data/hogan.db",
        )
    finally:
        DEFAULT_REGIME_CONFIGS["trending_down"].quality_final_mult = original

    shorts = [t for t in r.closed_trades if t.get("side") == "short"]
    longs = [t for t in r.closed_trades if t.get("side") == "long"]
    n_shorts = len(shorts)
    short_wins = sum(1 for t in shorts if t.get("pnl_usd", 0) > 0)
    short_pnls = [t.get("pnl_pct", 0) for t in shorts]
    short_total = sum(short_pnls)
    short_avg = short_total / n_shorts if n_shorts else 0

    long_wins = sum(1 for t in longs if t.get("pnl_usd", 0) > 0)

    td_shorts = [t for t in shorts if t.get("entry_regime") == "trending_down"]
    td_n = len(td_shorts)
    td_pnls = [t.get("pnl_pct", 0) for t in td_shorts]
    td_avg = sum(td_pnls) / td_n if td_n else 0

    winner_pnls = [p for p in short_pnls if p > 0]
    loser_pnls = [p for p in short_pnls if p <= 0]
    avg_win = sum(winner_pnls) / len(winner_pnls) if winner_pnls else 0
    avg_loss = sum(loser_pnls) / len(loser_pnls) if loser_pnls else 0
    payoff = abs(avg_win / avg_loss) if avg_loss else float("inf")

    conf = diagnose_shorts_by_confidence(r.closed_trades)
    conf_buckets = conf.get("by_regime_confidence", {}) if conf else {}

    funnel = r.signal_funnel or {}

    return {
        "mult": mult,
        "total_return": round(r.total_return_pct, 4),
        "max_dd": round(r.max_drawdown_pct, 4),
        "sharpe": round(r.sharpe_ratio, 3),
        "calmar": round(r.calmar_ratio, 3),
        "n_long": len(longs),
        "long_wr": round(long_wins / len(longs), 3) if longs else 0,
        "n_short": n_shorts,
        "short_wr": round(short_wins / n_shorts, 3) if n_shorts else 0,
        "short_avg": round(short_avg, 3),
        "short_total": round(short_total, 3),
        "short_payoff": round(payoff, 2),
        "td_n": td_n,
        "td_avg": round(td_avg, 3),
        "conf_buckets": {
            k: {"n": v["count"], "wr": v["win_rate"], "avg": v["avg_pnl_pct"]}
            for k, v in conf_buckets.items()
        },
        "post_quality_sell": funnel.get("post_quality_sell", 0),
        "post_edge_sell": funnel.get("post_edge_sell", 0),
        "pipeline_sell": funnel.get("pipeline_sell", 0),
        "executed_short": funnel.get("executed_short_entry", 0),
        "per_trade_shorts": [
            {
                "bar": t.get("entry_bar_idx"),
                "regime": t.get("entry_regime"),
                "conf": round(t.get("regime_confidence", 0) or 0, 3),
                "pnl_pct": round(t.get("pnl_pct", 0), 2),
                "exit": t.get("close_reason"),
            }
            for t in shorts
        ],
    }


def main() -> None:
    cfg = load_config()
    conn = get_connection("data/hogan.db")
    candles = load_candles(conn, "BTC/USD", "1h", limit=5000)
    conn.close()
    print(f"Loaded {len(candles)} candles\n")

    configs = [
        (0.80, "old (0.80)"),
        (0.50, "new (0.50)"),
    ]

    results = []
    for mult, label in configs:
        print(f"Running quality_final_mult={mult} ({label})...", flush=True)
        row = _run_with_quality_mult(cfg, candles, mult)
        results.append((label, row))
        print(
            f"  -> return={row['total_return']:+.4f}%, shorts={row['n_short']}, "
            f"short_total={row['short_total']:+.3f}%, "
            f"post_quality_sell={row['post_quality_sell']}",
            flush=True,
        )

    print()
    print("=" * 100)
    print("QUALITY THRESHOLD ISOLATION (both at 16h short hold)")
    print("=" * 100)

    hdr = (
        f"{'config':<12s}  {'return%':>8s}  {'max_dd%':>8s}  {'sharpe':>7s}  "
        f"{'calmar':>7s}  {'longs':>5s}  {'shorts':>6s}  {'s_wr':>5s}  "
        f"{'s_tot%':>7s}  {'payoff':>6s}  {'td_n':>4s}  {'td_avg%':>7s}  "
        f"{'q_sell':>6s}"
    )
    print(hdr)
    print("-" * 100)
    for label, row in results:
        print(
            f"{label:<12s}  {row['total_return']:>+8.4f}  "
            f"{row['max_dd']:>8.4f}  {row['sharpe']:>7.3f}  "
            f"{row['calmar']:>7.3f}  {row['n_long']:>5d}  "
            f"{row['n_short']:>6d}  {row['short_wr']:>5.1%}  "
            f"{row['short_total']:>+7.3f}  {row['short_payoff']:>6.2f}  "
            f"{row['td_n']:>4d}  {row['td_avg']:>+7.3f}  "
            f"{row['post_quality_sell']:>6d}"
        )

    print()
    print("FUNNEL COMPARISON (sell-side):")
    for label, row in results:
        print(
            f"  {label}: pipeline_sell={row['pipeline_sell']}  "
            f"post_edge_sell={row['post_edge_sell']}  "
            f"post_quality_sell={row['post_quality_sell']}  "
            f"executed_short={row['executed_short']}"
        )

    print()
    print("CONFIDENCE BUCKETS:")
    for label, row in results:
        print(f"  {label}:")
        for bk, bv in row["conf_buckets"].items():
            print(
                f"    {bk:<28s}  n={bv['n']:>2d}  "
                f"wr={bv['wr']:>5.1%}  avg_pnl={bv['avg']:>+7.3f}%"
            )

    print()
    print("PER-TRADE SHORT DETAIL:")
    for label, row in results:
        print(f"  {label}:")
        for t in row["per_trade_shorts"]:
            print(
                f"    bar={t['bar'] or 0:>5d}  regime={t['regime'] or '?':<14s}  "
                f"conf={t['conf']:>5.3f}  pnl={t['pnl_pct']:>+6.2f}%  "
                f"exit={t['exit'] or '?'}"
            )

    # Verdict
    old_row = results[0][1]
    new_row = results[1][1]
    print()
    print("=" * 100)
    if new_row["n_short"] > old_row["n_short"]:
        print(f"VERDICT: Lower threshold generates MORE shorts "
              f"({new_row['n_short']} vs {old_row['n_short']})")
        if new_row["calmar"] > old_row["calmar"]:
            print("  -> AND improves Calmar. The threshold change adds real value.")
        else:
            print("  -> BUT Calmar is worse. Extra shorts may be low-quality.")
    elif new_row["n_short"] == old_row["n_short"]:
        print(f"VERDICT: Same number of shorts ({new_row['n_short']}). "
              f"Threshold change has NO EFFECT on entry count.")
        if abs(new_row["short_total"] - old_row["short_total"]) < 0.01:
            print("  -> Short PnL also identical. The threshold is a passenger.")
        else:
            print(f"  -> Short PnL differs: old={old_row['short_total']:+.3f}% "
                  f"vs new={new_row['short_total']:+.3f}%")
    else:
        print(f"VERDICT: Higher threshold generates MORE shorts "
              f"({old_row['n_short']} vs {new_row['n_short']}) — "
              "lower threshold is too strict!")


if __name__ == "__main__":
    main()
