"""Build 15m candles from existing 5m data for BTC/USD.

Aggregates every 3 consecutive 5m bars aligned to 15m boundaries.
"""
import pandas as pd
from hogan_bot.storage import get_connection, upsert_candles

conn = get_connection("data/hogan.db")

symbol = "BTC/USD"

print("Loading 5m candles...")
rows = conn.execute(
    "SELECT ts_ms, open, high, low, close, volume "
    "FROM candles WHERE symbol=? AND timeframe='5m' ORDER BY ts_ms",
    (symbol,),
).fetchall()
df5 = pd.DataFrame(rows, columns=["ts_ms", "open", "high", "low", "close", "volume"])
print(f"  Loaded {len(df5)} 5m bars")

df5["ts_15m"] = (df5["ts_ms"] // 900_000) * 900_000

agg = df5.groupby("ts_15m").agg(
    open=("open", "first"),
    high=("high", "max"),
    low=("low", "min"),
    close=("close", "last"),
    volume=("volume", "sum"),
    bar_count=("open", "count"),
).reset_index()

complete = agg[agg["bar_count"] == 3].copy()
complete["timestamp"] = pd.to_datetime(complete["ts_15m"], unit="ms", utc=True)
complete = complete[["timestamp", "open", "high", "low", "close", "volume"]]
print(f"  Aggregated to {len(complete)} complete 15m bars")
print(f"  Range: {complete['timestamp'].iloc[0]} -> {complete['timestamp'].iloc[-1]}")

upsert_candles(conn, symbol, "15m", complete)
conn.close()
print("Done.")
