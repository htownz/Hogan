"""Test: can SignalEvaluator process candles from Kraken REST warmup?"""
import sys
sys.path.insert(0, ".")
import ccxt
import pandas as pd
from hogan_bot.config import load_config
from hogan_bot.champion import apply_champion_mode
from hogan_bot.ml import load_model

cfg = load_config()
cfg = apply_champion_mode(cfg)

# Fetch candles the same way the REST warmup does
exchange = ccxt.kraken()
raw = exchange.fetch_ohlcv("BTC/USD", "1h", limit=200)
print(f"Fetched {len(raw)} candles")

rows = []
for ohlcv in raw:
    ts_ms, o, h, l, c, v = ohlcv
    rows.append({"ts_ms": ts_ms, "open": o, "high": h, "low": l, "close": c, "volume": v})

candles = pd.DataFrame(rows).sort_values("ts_ms").reset_index(drop=True)
print(f"DataFrame: {len(candles)} rows, columns: {list(candles.columns)}")
print(f"Latest candle: ts={candles.iloc[-1]['ts_ms']} close={candles.iloc[-1]['close']}")

# Check the length gate
long_ma = max(cfg.long_ma_window, 20)
print(f"\nLength check: {len(candles)} >= {long_ma} = {len(candles) >= long_ma}")

# Try to run the signal evaluation
try:
    ml_model = load_model(cfg.ml_model_path)
    print(f"ML model loaded: {type(ml_model)}")
except Exception as e:
    ml_model = None
    print(f"ML model load failed: {e}")

from hogan_bot.event_loop import SignalEvaluator
from hogan_bot.storage import get_connection

conn = get_connection()
evaluator = SignalEvaluator(cfg, ml_model, conn=conn)

px = float(candles["close"].iloc[-1])
equity = 10000.0

try:
    sig = evaluator.evaluate(
        "BTC/USD", candles, equity,
        recent_whipsaw_count=0,
        mtf_candles=None,
        peak_equity=equity,
    )
    print(f"\nSignal evaluation SUCCESS:")
    print(f"  action={sig.action}")
    print(f"  tech_action={sig.raw_tech_action}")
    print(f"  pipeline_action={sig.pipeline_action}")
    print(f"  size={sig.size:.6f}")
    print(f"  final_confidence={sig.final_confidence:.4f}")
    print(f"  ml_up_prob={sig.up_prob}")
    print(f"  regime={sig.regime}")
    print(f"  block_reasons={sig.block_reasons}")
    print(f"  eff_allow_shorts={sig.eff_allow_shorts}")
except Exception as e:
    import traceback
    print(f"\nSignal evaluation FAILED:")
    traceback.print_exc()
