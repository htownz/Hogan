"""Deep long-confidence analysis across all 4 MTF configs.

Questions this answers:
1. Which long regime+confidence buckets are losing?
2. Are low-confidence longs worth taking?
3. Are ranging longs worth keeping at 0.25x size?
4. How does MTF change the picture per bucket?
5. Per-trade detail for every long across all configs.
"""
from __future__ import annotations

import sys
sys.path.insert(0, ".")

from hogan_bot.config import load_config
from hogan_bot.storage import get_connection, load_candles
from hogan_bot.backtest import diagnose_longs_by_confidence
from hogan_bot.backtest_cli import _run_single


def _run_config(cfg, candles_1h, candles_15m, label, pullback, mtf):
    r = _run_single(
        cfg, candles_1h, "BTC/USD", None,
        enable_shorts=True,
        short_max_hold_hours=16.0,
        db_path="data/hogan.db",
        enable_pullback_gate=pullback,
        candles_15m=candles_15m if mtf else None,
        mtf_thesis_max_age=6,
    )
    return label, r


def main() -> None:
    cfg = load_config()
    conn = get_connection("data/hogan.db")
    candles_1h = load_candles(conn, "BTC/USD", "1h", limit=5000)
    candles_15m = load_candles(conn, "BTC/USD", "15m")
    conn.close()
    print(f"Loaded {len(candles_1h)} 1h candles, {len(candles_15m)} 15m candles\n")

    configs = [
        ("A: baseline", False, False),
        ("B: +pullback", True, False),
        ("C: +MTF", False, True),
        ("D: +PB+MTF", True, True),
    ]

    results = []
    for label, pb, mtf in configs:
        print(f"Running {label}...", flush=True)
        lbl, r = _run_config(cfg, candles_1h, candles_15m, label, pb, mtf)
        results.append((lbl, r))

    # Collect all unique regime|confidence buckets across configs
    all_buckets: set[str] = set()
    config_bucket_data: list[tuple[str, dict[str, dict]]] = []

    for label, r in results:
        lc = diagnose_longs_by_confidence(r.closed_trades)
        by_rc = lc.get("by_regime_confidence", {}) if lc else {}
        all_buckets.update(by_rc.keys())
        config_bucket_data.append((label, by_rc))

    print()
    print("=" * 120)
    print("LONG TRADES BY REGIME x CONFIDENCE — CROSS-CONFIG COMPARISON")
    print("=" * 120)

    sorted_buckets = sorted(all_buckets)
    for bucket in sorted_buckets:
        print(f"\n  {bucket}:")
        print(f"    {'config':<14s}  {'n':>3s}  {'wins':>4s}  {'win%':>5s}  {'avg_pnl%':>8s}  {'total%':>7s}")
        print(f"    {'-'*14}  {'-'*3}  {'-'*4}  {'-'*5}  {'-'*8}  {'-'*7}")
        for label, by_rc in config_bucket_data:
            if bucket in by_rc:
                v = by_rc[bucket]
                print(
                    f"    {label:<14s}  {v['count']:>3d}  "
                    f"{v['wins']:>4d}  {v['win_rate']:>5.1%}  "
                    f"{v['avg_pnl_pct']:>+8.3f}  {v['total_pnl_pct']:>+7.3f}"
                )
            else:
                print(f"    {label:<14s}  {'—':>3s}  {'—':>4s}  {'—':>5s}  {'—':>8s}  {'—':>7s}")

    # Per-trade detail for each config
    print()
    print("=" * 120)
    print("PER-TRADE LONG DETAIL (all configs)")
    print("=" * 120)

    for label, r in results:
        longs = [t for t in r.closed_trades if t.get("side") == "long"]
        if not longs:
            print(f"\n  {label}: 0 longs")
            continue
        print(f"\n  {label} ({len(longs)} longs):")
        print(
            f"    {'bar':>5s}  {'regime':<14s}  {'conf':>5s}  {'bucket':<7s}  "
            f"{'pnl%':>7s}  {'hold':>4s}  {'exit':<22s}  {'MFE%':>6s}  {'MAE%':>6s}"
        )
        print(
            f"    {'-'*5}  {'-'*14}  {'-'*5}  {'-'*7}  "
            f"{'-'*7}  {'-'*4}  {'-'*22}  {'-'*6}  {'-'*6}"
        )
        for t in longs:
            conf = t.get("regime_confidence", 0) or 0
            bucket = "high" if conf >= 0.60 else ("medium" if conf >= 0.40 else "low")
            mfe = t.get("in_trade_mfe_pct", 0) or 0
            mae = t.get("in_trade_mae_pct", 0) or 0
            hold = (t.get("exit_bar_idx", 0) or 0) - (t.get("entry_bar_idx", 0) or 0)
            print(
                f"    {t.get('entry_bar_idx', 0) or 0:>5d}  "
                f"{(t.get('entry_regime') or '?'):<14s}  "
                f"{conf:>5.3f}  {bucket:<7s}  "
                f"{t.get('pnl_pct', 0):>+7.2f}  {hold:>4d}  "
                f"{(t.get('close_reason') or '?'):<22s}  "
                f"{mfe:>+6.2f}  {mae:>+6.2f}"
            )

    # Summary: what is the evidence for each regime?
    print()
    print("=" * 120)
    print("VERDICT BY REGIME (Config D = PB+MTF as reference)")
    print("=" * 120)

    _, best_r = results[3]  # Config D
    longs_d = [t for t in best_r.closed_trades if t.get("side") == "long"]
    by_regime: dict[str, list[dict]] = {}
    for t in longs_d:
        reg = t.get("entry_regime") or "unknown"
        by_regime.setdefault(reg, []).append(t)

    for reg in sorted(by_regime.keys()):
        trades = by_regime[reg]
        n = len(trades)
        wins = sum(1 for t in trades if t.get("pnl_usd", 0) > 0)
        pnls = [t.get("pnl_pct", 0) for t in trades]
        avg_pnl = sum(pnls) / n if n else 0
        total_pnl = sum(pnls)
        print(
            f"\n  {reg}: {n} trades, {wins} wins ({wins/n:.0%}), "
            f"avg={avg_pnl:+.3f}%, total={total_pnl:+.3f}%"
        )
        if total_pnl > 0:
            print("    -> KEEP: positive long contribution")
        elif total_pnl > -0.5:
            print("    -> BORDERLINE: tiny drag, might keep at reduced size")
        else:
            print("    -> CONSIDER BLOCKING: material drag on portfolio")


if __name__ == "__main__":
    main()
