import sqlite3

conn = sqlite3.connect("data/hogan.db")

tables = conn.execute("SELECT name FROM sqlite_master WHERE type='table'").fetchall()
print("Tables:", [t[0] for t in tables])

try:
    cur = conn.execute(
        "SELECT COUNT(*), MIN(date), MAX(date) FROM onchain_metrics WHERE metric = 'fear_greed_value'"
    )
    print("Fear & Greed:", cur.fetchone())
except Exception as e:
    print("No onchain_metrics table:", e)

try:
    cur = conn.execute(
        "SELECT COUNT(*), MIN(ts_ms), MAX(ts_ms) FROM candles WHERE symbol = 'VIX/USD'"
    )
    print("VIX candles:", cur.fetchone())
except Exception as e:
    print("No VIX data:", e)

try:
    cur = conn.execute(
        "SELECT symbol, COUNT(*) FROM candles GROUP BY symbol"
    )
    print("Candle symbols:", cur.fetchall())
except Exception as e:
    print("Error:", e)

conn.close()
