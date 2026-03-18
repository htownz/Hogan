import sqlite3
from datetime import datetime, timezone

conn = sqlite3.connect("data/hogan.db")
rows = conn.execute(
    "SELECT ts_ms, close FROM candles WHERE symbol='BTC/USD' AND timeframe='1h' "
    "ORDER BY ts_ms DESC LIMIT 5"
).fetchall()
print("Latest candles:")
for r in rows:
    print(f"  {datetime.fromtimestamp(r[0]/1000, tz=timezone.utc)} close={r[1]}")

total = conn.execute(
    "SELECT COUNT(*) FROM candles WHERE symbol='BTC/USD' AND timeframe='1h'"
).fetchone()[0]
print(f"Total 1h candles: {total}")
conn.close()
