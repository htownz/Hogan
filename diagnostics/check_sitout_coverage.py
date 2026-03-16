"""Quick check: how many bars does the macro sitout filter affect?"""
import sqlite3
import pandas as pd
from hogan_bot.macro_sitout import MacroSitout, count_sitout_bars

conn = sqlite3.connect("data/hogan.db")

sitout = MacroSitout.from_db(conn)

df = pd.read_sql_query(
    "SELECT ts_ms, open, high, low, close, volume FROM candles "
    "WHERE symbol = 'BTC/USD' AND timeframe = '1h' ORDER BY ts_ms",
    conn,
)
conn.close()

df["timestamp"] = pd.to_datetime(df["ts_ms"], unit="ms", utc=True)
print(f"Total candles: {len(df)}")
print(f"Date range: {df['timestamp'].iloc[0]} to {df['timestamp'].iloc[-1]}")

stats = count_sitout_bars(sitout, df)
print(f"\nSitout stats:")
for k, v in stats.items():
    pct = v / stats["total_bars"] * 100 if stats["total_bars"] else 0
    print(f"  {k:<20} {v:>6}  ({pct:.1f}%)")

print("\nPer-window breakdown:")
windows = [
    ("W0", "2024-06-02", "2024-10-10"),
    ("W1", "2024-10-10", "2025-02-17"),
    ("W2", "2025-02-17", "2025-06-27"),
    ("W3", "2025-06-27", "2025-11-03"),
    ("W4", "2025-11-03", "2026-03-13"),
]
for name, start, end in windows:
    mask = (df["timestamp"] >= start) & (df["timestamp"] <= end)
    window_df = df[mask]
    if window_df.empty:
        print(f"  {name}: no data")
        continue
    ws = count_sitout_bars(sitout, window_df)
    total = ws["total_bars"]
    print(f"  {name} ({start} to {end}): "
          f"sitout={ws['sitout_bars']} ({ws['sitout_bars']/total*100:.1f}%)  "
          f"scaled={ws['scaled_bars']} ({ws['scaled_bars']/total*100:.1f}%)  "
          f"clear={ws['clear_bars']} ({ws['clear_bars']/total*100:.1f}%)")
