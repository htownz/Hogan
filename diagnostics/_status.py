import sqlite3
from datetime import datetime, timezone

conn = sqlite3.connect("data/hogan.db")
c = conn.cursor()

# Paper trades
c.execute("SELECT COUNT(*) FROM paper_trades")
total_trades = c.fetchone()[0]
print(f"Paper trades: {total_trades}\n")

if total_trades > 0:
    c.execute("PRAGMA table_info(paper_trades)")
    cols = [r[1] for r in c.fetchall()]
    c.execute(f"SELECT * FROM paper_trades ORDER BY rowid DESC LIMIT 5")
    rows = c.fetchall()
    for r in rows:
        row_dict = dict(zip(cols, r))
        print(f"  Trade: {row_dict}")

# Recent decisions
restart_ms = int(datetime(2026, 3, 18, 13, 29, tzinfo=timezone.utc).timestamp() * 1000)
c.execute("SELECT COUNT(*) FROM decision_log WHERE ts_ms > ?", (restart_ms,))
recent = c.fetchone()[0]
print(f"\nDecisions since latest restart: {recent}")

c.execute(
    "SELECT id, ts_ms, tech_action, pipeline_action, final_action, regime, "
    "ml_up_prob, conf_scale, position_size, block_reasons_json "
    "FROM decision_log WHERE ts_ms > ? ORDER BY id DESC LIMIT 10",
    (restart_ms,)
)
rows = c.fetchall()
for r in rows:
    ts_str = datetime.fromtimestamp(r[1] / 1000, tz=timezone.utc).strftime("%H:%M:%S")
    ml = r[6] or 0
    conf = r[7] or 0
    size = r[8] or 0
    print(
        f"  id={r[0]:6d} {ts_str} tech={str(r[2]):5s} pipe={str(r[3]):5s} "
        f"final={str(r[4]):5s} regime={str(r[5]):15s} ml={ml:.4f} conf={conf:.4f} "
        f"size={size:.6f} blocks={r[9]}"
    )

conn.close()
