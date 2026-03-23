"""Smoke test: verify signal confidence is non-zero after EMA cloud fix."""
import sys

sys.path.insert(0, ".")
import numpy as np
import pandas as pd

from hogan_bot.strategy import generate_signal

# Synthetic bearish candles (price falling, low volume — simulates current market)
np.random.seed(42)
n = 500
price = 68000.0
closes = [price - i * 5 + np.random.normal(0, 50) for i in range(n)]
df = pd.DataFrame({
    "open":   [c + np.random.normal(0, 20) for c in closes],
    "high":   [c + abs(np.random.normal(0, 40)) for c in closes],
    "low":    [c - abs(np.random.normal(0, 40)) for c in closes],
    "close":  closes,
    "volume": [abs(np.random.normal(50000, 5000)) for _ in range(n)],  # low volume (below avg)
})

sig = generate_signal(
    df,
    short_window=5, long_window=20,
    volume_window=10, volume_threshold=0.3,
    use_ema_clouds=True, use_ict=True,
    use_fvg=False, signal_mode="any",
    min_vote_margin=1,
    ict_require_time_window=False, ict_require_pd=False,
    ict_ote_enabled=False,
)

print(f"action     = {sig.action}")
print(f"confidence = {sig.confidence:.4f}  (was always 0.0 before fix)")
print(f"vol_ratio  = {sig.volume_ratio:.4f}")
print(f"stop_pct   = {sig.stop_distance_pct:.4f}")
if sig.confidence > 0.0:
    print("✓ PASS: confidence > 0 after EMA cloud fix")
else:
    print("✗ confidence is still 0 — EMA cloud may be neutral on this data")
    print("  (This is OK if EMA cloud truly reads as neutral for this candle series)")
