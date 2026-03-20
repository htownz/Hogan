"""Diagnostic: analyze trade funnel and decision blocking."""
import sqlite3
import json
from datetime import datetime, timezone

DB = "data/hogan.db"


def main():
    conn = sqlite3.connect(DB)
    try:
        _run(conn)
    finally:
        conn.close()


def _run(conn):
    conn.row_factory = sqlite3.Row
    # 1. Paper trades
    print("=" * 60)
    print("PAPER TRADES (last 20)")
    print("=" * 60)
    rows = conn.execute(
        "SELECT * FROM paper_trades ORDER BY open_ts_ms DESC LIMIT 20"
    ).fetchall()
    if not rows:
        print("  (none)")
    for r in rows:
        d = dict(r)
        ts = d.get("open_ts_ms", 0)
        dt_str = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M") if ts else "?"
        close_px = d.get("close_price")
        close_reason = d.get("close_reason", "OPEN")
        pnl = ""
        if close_px and d.get("open_price"):
            side = d.get("side", "long")
            if side == "short":
                pnl_pct = (d["open_price"] - close_px) / d["open_price"] * 100
            else:
                pnl_pct = (close_px - d["open_price"]) / d["open_price"] * 100
            pnl = f" pnl={pnl_pct:+.2f}%"
        side = str(d.get("side") or "?")
        open_px = float(d.get("entry_price") or d.get("open_price") or 0)
        ml = d.get("ml_up_prob")
        sconf = d.get("strategy_conf")
        close_str = str(close_px) if close_px else "OPEN"
        reason_str = str(close_reason or "OPEN")
        print(f"  {dt_str} | {side:5s} | open={open_px:.2f} "
              f"close={close_str:>10} | reason={reason_str:15s} "
              f"| ml={ml} conf={sconf}{pnl}")

    # 2. Trade summary
    print()
    print("=" * 60)
    print("TRADE COUNTS BY STATUS")
    print("=" * 60)
    for r in conn.execute(
        "SELECT close_reason, COUNT(*) as n FROM paper_trades GROUP BY close_reason ORDER BY n DESC"
    ):
        print(f"  {r['close_reason'] or 'OPEN':20s}: {r['n']}")

    # 3. Decision log analysis
    print()
    print("=" * 60)
    print("DECISION LOG — LAST 50 DECISIONS")
    print("=" * 60)
    # Discover available columns
    cols_info = conn.execute("PRAGMA table_info(decision_log)").fetchall()
    all_cols = [c["name"] for c in cols_info]
    select_cols = ["ts_ms", "symbol"]
    for c in ["regime", "tech_action", "pipeline_action", "final_action",
              "final_confidence", "ml_up_prob", "conf_scale",
              "block_reasons", "direction_score", "quality_score", "size_score",
              "explanation"]:
        if c in all_cols:
            select_cols.append(c)
    rows = conn.execute(
        f"SELECT {','.join(select_cols)} FROM decision_log ORDER BY ts_ms DESC LIMIT 50"
    ).fetchall()
    action_counts = {}
    block_counts = {}
    regime_counts = {}
    for r in rows:
        d = dict(r)
        ts = d["ts_ms"]
        dt_str = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%m-%d %H:%M") if ts else "?"
        fa = d["final_action"]
        action_counts[fa] = action_counts.get(fa, 0) + 1
        regime = d.get("regime", "?")
        regime_counts[regime] = regime_counts.get(regime, 0) + 1

        blocks = d.get("block_reasons", "")
        if blocks:
            try:
                bl = json.loads(blocks) if blocks.startswith("[") else [blocks]
            except Exception:
                bl = [blocks]
            for b in bl:
                block_counts[b] = block_counts.get(b, 0) + 1

        if fa != "hold":
            print(f"  {dt_str} | {d['symbol']:10s} | regime={regime:15s} "
                  f"| tech={d.get('tech_action','?'):4s} pipe={d.get('pipeline_action','?'):4s} "
                  f"final={fa:4s} | conf={d.get('final_confidence',0):.3f} "
                  f"ml={d.get('ml_up_prob','?')} scale={d.get('conf_scale',0):.3f} "
                  f"| blocks={blocks}")

    print()
    print("--- Action distribution (last 50) ---")
    for a, n in sorted(action_counts.items(), key=lambda x: -x[1]):
        print(f"  {a:6s}: {n:3d} ({n/max(len(rows),1)*100:.0f}%)")

    print()
    print("--- Regime distribution (last 50) ---")
    for r, n in sorted(regime_counts.items(), key=lambda x: -x[1]):
        print(f"  {r:15s}: {n:3d}")

    print()
    print("--- Block reasons (last 50) ---")
    if not block_counts:
        print("  (none)")
    for b, n in sorted(block_counts.items(), key=lambda x: -x[1]):
        print(f"  {b:40s}: {n:3d}")

    # 4. Decision log total and pipeline-to-final conversion
    print()
    print("=" * 60)
    print("PIPELINE vs FINAL ACTION (last 200)")
    print("=" * 60)
    rows2 = conn.execute(
        """SELECT pipeline_action, final_action, COUNT(*) as n
           FROM decision_log
           GROUP BY pipeline_action, final_action
           ORDER BY n DESC LIMIT 20"""
    ).fetchall()
    for r in rows2:
        print(f"  pipeline={r['pipeline_action'] or '?':5s} -> final={r['final_action']:5s} : {r['n']}")

    # 5. Tech action vs final
    print()
    print("=" * 60)
    print("TECH ACTION vs FINAL ACTION (last 200)")
    print("=" * 60)
    if "tech_action" in all_cols:
        rows3 = conn.execute(
            """SELECT tech_action, final_action, COUNT(*) as n
               FROM decision_log
               GROUP BY tech_action, final_action
               ORDER BY n DESC LIMIT 20"""
        ).fetchall()
        for r in rows3:
            ta = r["tech_action"] or "?"
            fa = r["final_action"] or "?"
            print(f"  tech={ta:5s} -> final={fa:5s} : {r['n']}")
    else:
        print("  (tech_action column not in decision_log)")

    # 6b. Total decision count
    print()
    total = conn.execute("SELECT COUNT(*) as n FROM decision_log").fetchone()["n"]
    print(f"Total decisions logged: {total}")
    
    # 6c. Check for recent non-hold decisions
    print()
    print("=" * 60)
    print("NON-HOLD DECISIONS (all time)")
    print("=" * 60)
    nh_rows = conn.execute(
        "SELECT * FROM decision_log WHERE final_action != 'hold' ORDER BY ts_ms DESC LIMIT 20"
    ).fetchall()
    if not nh_rows:
        print("  (NO non-hold decisions found -- everything is being blocked!)")
    for r in nh_rows:
        d = dict(r)
        ts = d.get("ts_ms", 0)
        dt_str = datetime.fromtimestamp(ts / 1000, tz=timezone.utc).strftime("%m-%d %H:%M") if ts else "?"
        print(f"  {dt_str} | {d.get('symbol','?')} | final={d.get('final_action','?')} "
              f"| conf={d.get('final_confidence','?')} ml={d.get('ml_up_prob','?')} "
              f"| scale={d.get('conf_scale','?')}")

    # 6. Check .env for key settings
    print()
    print("=" * 60)
    print("KEY ENV / CONFIG")
    print("=" * 60)
    import os
    from dotenv import load_dotenv
    load_dotenv()
    keys = [
        "HOGAN_CHAMPION_MODE", "HOGAN_USE_ML_FILTER", "HOGAN_ML_AS_SIZER",
        "HOGAN_ENABLE_SHORTS", "HOGAN_MACRO_SITOUT", "HOGAN_USE_FUNDING_OVERLAY",
        "HOGAN_USE_POLICY_CORE", "HOGAN_SWARM_ENABLED", "HOGAN_PAPER_MODE",
        "HOGAN_USE_MACRO_FILTER", "HOGAN_ML_BUY_THRESHOLD", "HOGAN_ML_SELL_THRESHOLD",
        "HOGAN_MIN_FINAL_CONFIDENCE", "HOGAN_MIN_TECH_CONFIDENCE",
        "HOGAN_REALISTIC_PAPER",
    ]
    for k in keys:
        v = os.getenv(k, "(not set)")
        print(f"  {k:40s} = {v}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        traceback.print_exc()
