"""Short-hold sweep: compare short_max_hold_hours across 8h, 12h, 16h, 20h, 24h."""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

from hogan_bot.config import load_config
from hogan_bot.storage import get_connection, load_candles
from hogan_bot.backtest import diagnose_shorts_by_confidence
from hogan_bot.backtest_cli import _run_single


def main() -> None:
    cfg = load_config()
    conn = get_connection("data/hogan.db")
    candles = load_candles(conn, "BTC/USD", "1h", limit=5000)
    conn.close()
    print(f"Loaded {len(candles)} candles\n")

    hold_hours = [8, 12, 16, 20, 24]
    rows: list[dict] = []

    for h in hold_hours:
        print(f"Running short_max_hold_hours={h}...", flush=True)
        r = _run_single(
            cfg, candles, "BTC/USD", None,
            enable_shorts=True,
            short_max_hold_hours=float(h),
        )
        shorts = [t for t in r.closed_trades if t.get("side") == "short"]
        longs = [t for t in r.closed_trades if t.get("side") == "long"]
        n_shorts = len(shorts)
        short_wins = sum(1 for t in shorts if t.get("pnl_usd", 0) > 0)
        short_pnls = [t.get("pnl_pct", 0) for t in shorts]
        short_avg = sum(short_pnls) / n_shorts if n_shorts else 0
        short_total = sum(short_pnls)

        td_shorts = [t for t in shorts if t.get("entry_regime") == "trending_down"]
        td_n = len(td_shorts)
        td_wins = sum(1 for t in td_shorts if t.get("pnl_usd", 0) > 0)
        td_pnls = [t.get("pnl_pct", 0) for t in td_shorts]
        td_avg = sum(td_pnls) / td_n if td_n else 0

        [t.get("pnl_pct", 0) for t in longs]
        long_wins = sum(1 for t in longs if t.get("pnl_usd", 0) > 0)

        winner_pnls = [p for p in short_pnls if p > 0]
        loser_pnls = [p for p in short_pnls if p <= 0]
        avg_win = sum(winner_pnls) / len(winner_pnls) if winner_pnls else 0
        avg_loss = sum(loser_pnls) / len(loser_pnls) if loser_pnls else 0
        payoff = abs(avg_win / avg_loss) if avg_loss else float("inf")

        conf = diagnose_shorts_by_confidence(r.closed_trades)
        conf_buckets = conf.get("by_regime_confidence", {}) if conf else {}

        funnel = r.signal_funnel or {}

        rows.append({
            "hold_h": h,
            "total_return": round(r.total_return_pct, 4),
            "max_dd": round(r.max_drawdown_pct, 4),
            "sharpe": round(r.sharpe_ratio, 3),
            "calmar": round(r.calmar_ratio, 3),
            "n_long": len(longs),
            "long_wr": round(long_wins / len(longs), 3) if longs else 0,
            "n_short": n_shorts,
            "short_wins": short_wins,
            "short_wr": round(short_wins / n_shorts, 3) if n_shorts else 0,
            "short_avg_pnl": round(short_avg, 3),
            "short_total_pnl": round(short_total, 3),
            "short_payoff": round(payoff, 2),
            "td_n": td_n,
            "td_wins": td_wins,
            "td_avg_pnl": round(td_avg, 3),
            "conf_buckets": {
                k: {"n": v["count"], "wr": v["win_rate"], "avg": v["avg_pnl_pct"]}
                for k, v in conf_buckets.items()
            },
            "blocked_already_short": funnel.get("blocked_already_short", 0),
        })
        print(
            f"  -> return={r.total_return_pct:+.4f}%, "
            f"shorts={n_shorts}, short_pnl={short_total:+.3f}%, "
            f"blocked_already_short={funnel.get('blocked_already_short', 0)}",
            flush=True,
        )

    print()
    print("=" * 110)
    print("SHORT-HOLD SWEEP RESULTS")
    print("=" * 110)

    hdr = (
        f"{'hold_h':>6s}  {'return%':>8s}  {'max_dd%':>8s}  {'sharpe':>7s}  "
        f"{'calmar':>7s}  {'longs':>5s}  {'l_wr':>5s}  {'shorts':>6s}  "
        f"{'s_wr':>5s}  {'s_avg%':>7s}  {'s_tot%':>7s}  {'payoff':>6s}  "
        f"{'td_n':>4s}  {'td_avg%':>7s}  {'blk_s':>5s}"
    )
    print(hdr)
    print("-" * 110)
    for row in rows:
        round(row["td_wins"] / row["td_n"], 3) if row["td_n"] else 0
        print(
            f"{row['hold_h']:>6d}  {row['total_return']:>+8.4f}  "
            f"{row['max_dd']:>8.4f}  {row['sharpe']:>7.3f}  "
            f"{row['calmar']:>7.3f}  {row['n_long']:>5d}  "
            f"{row['long_wr']:>5.1%}  {row['n_short']:>6d}  "
            f"{row['short_wr']:>5.1%}  {row['short_avg_pnl']:>+7.3f}  "
            f"{row['short_total_pnl']:>+7.3f}  {row['short_payoff']:>6.2f}  "
            f"{row['td_n']:>4d}  {row['td_avg_pnl']:>+7.3f}  "
            f"{row['blocked_already_short']:>5d}"
        )

    print()
    print("CONFIDENCE BUCKET DETAIL:")
    for row in rows:
        print(f"  hold={row['hold_h']}h:")
        for bk, bv in row["conf_buckets"].items():
            print(
                f"    {bk:<28s}  n={bv['n']:>2d}  "
                f"wr={bv['wr']:>5.1%}  avg_pnl={bv['avg']:>+7.3f}%"
            )

    # Per-trade detail for the best config
    best = max(rows, key=lambda r: r["calmar"])
    print(f"\nBest config by Calmar: hold={best['hold_h']}h "
          f"(Calmar={best['calmar']:.3f}, return={best['total_return']:+.4f}%)")


if __name__ == "__main__":
    main()
