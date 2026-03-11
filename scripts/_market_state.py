"""Check current market state for signal readiness."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from hogan_bot.config import load_config, symbol_config
from hogan_bot.storage import load_candles, get_connection

config = load_config()
conn = get_connection(config.db_path)
candles = load_candles(conn, "BTC/USD", "1h", limit=100)
cfg = symbol_config(config, "BTC/USD")

close = candles["close"].astype(float)
short_ma = close.rolling(cfg.short_ma_window).mean()
long_ma = close.rolling(cfg.long_ma_window).mean()

sma = float(short_ma.iloc[-1])
lma = float(long_ma.iloc[-1])
px = float(close.iloc[-1])
spread = (sma - lma) / lma * 100

trend = "BULLISH" if sma > lma else "BEARISH"
print(f"Price: {px:.2f}")
print(f"MA{cfg.short_ma_window}: {sma:.2f}")
print(f"MA{cfg.long_ma_window}: {lma:.2f}")
print(f"MA spread: {spread:.3f}%")
print(f"Trend: {trend}")
print(f"Price vs MA22: {(px - sma)/sma*100:.3f}%")

vol = candles["volume"].astype(float)
avg_vol = vol.rolling(cfg.volume_window).mean()
vr = float(vol.iloc[-1] / max(avg_vol.iloc[-1], 1e-9))
print(f"Current vol ratio: {vr:.3f}")
print(f"Vol confirmed: {vr >= cfg.volume_threshold}")

in_uptrend = sma > lma
strong = abs(spread) > 1.0
bars_below = sum(1 for j in range(-4, -1) if float(close.iloc[j]) < float(short_ma.iloc[j]))
bars_above = sum(1 for j in range(-4, -1) if float(close.iloc[j]) > float(short_ma.iloc[j]))
reclaimed_up = px > sma
reclaimed_down = px < sma

print(f"\nTrend strong (>1%): {strong} ({abs(spread):.3f}%)")
print(f"Bars below MA22 (last 3): {bars_below}/3")
print(f"Bars above MA22 (last 3): {bars_above}/3")

if in_uptrend:
    pullback_ready = strong and bars_below >= 2 and reclaimed_up and vr >= cfg.volume_threshold
    print(f"Pullback buy conditions met: {pullback_ready}")
else:
    pullback_ready = strong and bars_above >= 2 and reclaimed_down and vr >= cfg.volume_threshold
    print(f"Pullback sell conditions met: {pullback_ready}")

# Next crossover distance
diff = sma - lma
print(f"\nMA distance to crossover: {abs(diff):.2f} ({abs(spread):.3f}%)")

# When was the last crossover?
for i in range(len(candles) - 2, 0, -1):
    c = short_ma.iloc[i] > long_ma.iloc[i]
    p = short_ma.iloc[i-1] > long_ma.iloc[i-1]
    if c != p:
        bars_ago = len(candles) - 1 - i
        cross_type = "GOLDEN" if c else "DEATH"
        print(f"Last crossover: {cross_type} cross, {bars_ago} bars ago (~{bars_ago}h)")
        break

conn.close()
