import sqlite3, time
from datetime import datetime, timezone

conn = sqlite3.connect("data/hogan.db")
conn.row_factory = sqlite3.Row

restart_ms = datetime(2026, 3, 18, 13, 12, 0, tzinfo=timezone.utc).timestamp() * 1000

total = conn.execute("SELECT COUNT(*) FROM decision_log").fetchone()[0]
recent = conn.execute(
    "SELECT COUNT(*) FROM decision_log WHERE ts_ms > ?", (restart_ms,)
).fetchone()[0]
print(f"Total decision_log: {total}")
print(f"Entries since restart: {recent}")

rows = conn.execute(
    "SELECT id, ts_ms, tech_action, pipeline_action, final_action, regime, "
    "ml_up_prob, conf_scale, position_size, block_reasons_json "
    "FROM decision_log WHERE ts_ms > ? ORDER BY id DESC LIMIT 10",
    (restart_ms,),
).fetchall()
if rows:
    print("\nLatest entries:")
    for r in rows:
        ts = datetime.fromtimestamp(r["ts_ms"] / 1000, tz=timezone.utc)
        print(f"  id={r['id']} ts={ts} tech={r['tech_action']} pipe={r['pipeline_action']} "
              f"final={r['final_action']} regime={r['regime']} "
              f"ml={r['ml_up_prob']:.4f} scale={r['conf_scale']:.4f} "
              f"size={r['position_size']:.6f} blocks={r['block_reasons_json']}")

    # Action distribution
    acts = conn.execute(
        "SELECT final_action, COUNT(*) as cnt FROM decision_log "
        "WHERE ts_ms > ? GROUP BY final_action ORDER BY cnt DESC",
        (restart_ms,),
    ).fetchall()
    print("\nAction distribution (post-restart):")
    for r in acts:
        print(f"  {r['final_action']}: {r['cnt']}")

pt = conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
print(f"\nPaper trades: {pt}")
if pt > 0:
    trades = conn.execute(
        "SELECT * FROM paper_trades ORDER BY rowid DESC LIMIT 5"
    ).fetchall()
    for t in trades:
        print(f"  {dict(t)}")

conn.close()
