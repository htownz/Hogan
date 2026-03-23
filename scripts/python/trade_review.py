"""Quick trade performance review script."""
import os
import sqlite3

DB = os.path.join(os.path.dirname(__file__), "..", "data", "hogan.db")
conn = sqlite3.connect(DB)

print("=== RECENT CLOSED TRADES (last 30) ===")
rows = conn.execute("""
    SELECT symbol, side, entry_price, exit_price, qty, realized_pnl,
           close_reason,
           datetime(open_ts_ms/1000, 'unixepoch') as entry_time,
           datetime(close_ts_ms/1000, 'unixepoch') as close_time,
           ml_up_prob, strategy_conf, vol_ratio
    FROM paper_trades
    WHERE close_ts_ms IS NOT NULL
    ORDER BY open_ts_ms DESC LIMIT 30
""").fetchall()

for r in rows:
    pnl = r[5] or 0
    ml = r[9]
    conf = r[10]
    vol = r[11]
    win = "WIN " if pnl > 0 else "LOSS"
    ml_str = f"ml={ml:.3f}" if ml is not None else "ml=None"
    conf_str = f"conf={conf:.2f}" if conf is not None else ""
    vol_str = f"vol={vol:.2f}" if vol is not None else ""
    direction = ""
    if r[3] and r[2]:
        direction = f"  {'UP  ' if r[3] > r[2] else 'DOWN'} ({(r[3]-r[2])/r[2]*100:+.2f}%)"
    print(f"  [{win}] {r[7]} | {r[0]} {r[1]:5s} "
          f"entry={r[2]:.2f} exit={r[3]:.2f}{direction} "
          f"pnl={pnl:+.4f} | {ml_str} {conf_str} {vol_str} | close={r[6]}")

print()
print("=== WIN/LOSS BY SIDE ===")
summary = conn.execute("""
    SELECT side,
           COUNT(*) as trades,
           SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
           SUM(CASE WHEN realized_pnl <= 0 THEN 1 ELSE 0 END) as losses,
           ROUND(SUM(realized_pnl), 4) as total_pnl,
           ROUND(AVG(realized_pnl), 4) as avg_pnl,
           ROUND(AVG(ml_up_prob), 3) as avg_ml_prob,
           ROUND(AVG(strategy_conf), 3) as avg_conf
    FROM paper_trades WHERE realized_pnl IS NOT NULL
    GROUP BY side
""").fetchall()
for r in summary:
    win_rate = (r[2] / r[1] * 100) if r[1] > 0 else 0
    print(f"  side={r[0]:5s} trades={r[1]:3d} wins={r[2]:3d} losses={r[3]:3d} "
          f"win_rate={win_rate:.0f}% total_pnl={r[4]:+.4f} avg_pnl={r[5]:+.4f} "
          f"avg_ml={r[6]} avg_conf={r[7]}")

print()
print("=== WIN/LOSS BY SYMBOL ===")
by_sym = conn.execute("""
    SELECT symbol,
           COUNT(*) as trades,
           SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
           ROUND(SUM(realized_pnl), 4) as total_pnl,
           ROUND(AVG(realized_pnl), 4) as avg_pnl
    FROM paper_trades WHERE realized_pnl IS NOT NULL
    GROUP BY symbol ORDER BY total_pnl
""").fetchall()
for r in by_sym:
    win_rate = (r[2] / r[1] * 100) if r[1] > 0 else 0
    print(f"  {r[0]:15s} trades={r[1]:3d} wins={r[2]:3d} "
          f"win_rate={win_rate:.0f}% total={r[3]:+.4f} avg={r[4]:+.4f}")

print()
print("=== CLOSE REASON BREAKDOWN ===")
reasons = conn.execute("""
    SELECT close_reason, COUNT(*) as n,
           ROUND(AVG(realized_pnl), 4) as avg_pnl,
           SUM(CASE WHEN realized_pnl > 0 THEN 1 ELSE 0 END) as wins,
           SUM(CASE WHEN realized_pnl <= 0 THEN 1 ELSE 0 END) as losses
    FROM paper_trades WHERE realized_pnl IS NOT NULL
    GROUP BY close_reason ORDER BY n DESC
""").fetchall()
for r in reasons:
    print(f"  {str(r[0]):25s} n={r[1]:3d} avg_pnl={r[2]:+.4f} wins={r[3]} losses={r[4]}")

print()
print("=== OPEN POSITIONS ===")
open_pos = conn.execute("""
    SELECT symbol, side, entry_price, qty,
           datetime(open_ts_ms/1000, 'unixepoch') as entry_time,
           ml_up_prob, strategy_conf
    FROM paper_trades WHERE close_ts_ms IS NULL
""").fetchall()
for r in open_pos:
    print(f"  {r[4]} | {r[0]} {r[1]} entry={r[2]:.2f} qty={r[3]:.5f} ml={r[5]} conf={r[6]}")
if not open_pos:
    print("  (none)")

print()
total = conn.execute("SELECT COUNT(*) FROM paper_trades").fetchone()[0]
closed = conn.execute("SELECT COUNT(*) FROM paper_trades WHERE close_ts_ms IS NOT NULL").fetchone()[0]
print(f"=== TOTALS: {total} trades, {closed} closed, {total-closed} open ===")

# Check equity curve
print()
print("=== EQUITY JOURNAL (last 10) ===")
try:
    eq_rows = conn.execute("""
        SELECT datetime(ts_ms/1000, 'unixepoch'), equity, cash, symbol
        FROM equity_journal ORDER BY ts_ms DESC LIMIT 10
    """).fetchall()
    for r in eq_rows:
        print(f"  {r[0]} equity={r[1]:.2f} cash={r[2]:.2f} symbol={r[3]}")
except Exception as e:
    print(f"  (equity_journal not available: {e})")

conn.close()
