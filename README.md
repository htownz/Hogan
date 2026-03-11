# Hogan

Hogan is a **BTC-first, 1h-led market operating system** — an agent-routed, forecast-and-risk-driven trading system for BTC/USD and major pairs on 110+ exchanges.

## Important safety notes

- This build is **paper mode only** (no live order routing yet).
- If you ever exposed API keys in chat/logs/repo, **rotate them immediately** in Kraken.
- No strategy guarantees profit; use strict risk controls and human oversight.

## What’s implemented now

- **AgentPipeline** — decision brain: Technical + Sentiment + Macro agents combined by MetaWeigher
- Agent-routed, forecast-and-risk-driven flow (not signal combinator)
- Default timeframe **1h** (BTC-first)
- Kraken OHLCV ingestion via `ccxt`
- Ripster EMA clouds (fast 8/9, slow 34/50) — cloud direction per bar
- Dynamic stop-distance proxy from recent candle range
- Position sizing bounded by risk-per-trade and aggressive allocation cap
- Portfolio simulation with fees (`PaperPortfolio`)
- Max drawdown guard to halt trading on breach
- Optional ML probability filter (logistic model) for buy/sell gating
- Backtest engine + CLI to evaluate strategy/ML settings on historical candles
- 24/5 behavior toggle (`HOGAN_TRADE_WEEKENDS=false` by default)
- `--max-loops` option for finite test runs
- **ICT/FVG** — experimental, quarantined (not a core feature)

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
python -m hogan_bot.main --max-loops 2
```

## ML enhancement workflow (recommended)

The ML pipeline uses 43 features (momentum, RSI, volatility, candle microstructure, volume, EMA cloud direction/width, OBV, Stochastic RSI, VWAP deviation, MACD, Bollinger Bands, ATR, multi-timeframe features, on-chain/macro signals) and supports two classifier types.

### 1. Validate with walk-forward cross-validation first

```bash
python -m hogan_bot.train --symbol BTC/USD --timeframe 1h --limit 5000 --cv --cv-splits 5
```

Output includes per-fold accuracy and ROC-AUC, plus mean values across folds.

### 2. Train a model

Logistic regression (scaled, default):

```bash
python -m hogan_bot.train --symbol BTC/USD --timeframe 1h --limit 5000 --horizon-bars 3 --model-path models/hogan_logreg.pkl
```

Random forest (includes feature importances in output):

```bash
python -m hogan_bot.train --symbol BTC/USD --timeframe 1h --limit 5000 --model-type random_forest --model-path models/hogan_rf.pkl
```

Both commands now report `accuracy`, `roc_auc`, `precision`, `recall`, and `f1`.

### 3. Enable ML filter in `.env`

```env
HOGAN_USE_ML_FILTER=true
HOGAN_ML_MODEL_PATH=models/hogan_logreg.pkl
HOGAN_ML_BUY_THRESHOLD=0.55
HOGAN_ML_SELL_THRESHOLD=0.45
```

### 4. Backtest with and without ML before long paper runs

```bash
python -m hogan_bot.backtest_cli --symbol BTC/USD --limit 5000
python -m hogan_bot.backtest_cli --symbol BTC/USD --limit 5000 --use-ml
```

### 5. Run paper trading

```bash
python -m hogan_bot.main --max-loops 200
```

## Environment config

`.env.example`:

```env
KRAKEN_API_KEY=
KRAKEN_API_SECRET=
HOGAN_STARTING_BALANCE=1800
HOGAN_AGGRESSIVE_ALLOCATION=0.75
HOGAN_MAX_RISK_PER_TRADE=0.03
HOGAN_MAX_DRAWDOWN=0.15
HOGAN_SYMBOLS=BTC/USD,ETH/USD
HOGAN_TIMEFRAME=1h
HOGAN_OHLCV_LIMIT=500
HOGAN_SHORT_MA=20
HOGAN_LONG_MA=50
HOGAN_VOLUME_WINDOW=20
HOGAN_VOLUME_THRESHOLD=1.2
HOGAN_FEE_RATE=0.0026
HOGAN_SLEEP_SECONDS=30
HOGAN_TRADE_WEEKENDS=false
HOGAN_PAPER_MODE=true
HOGAN_USE_ML_FILTER=false
HOGAN_ML_MODEL_PATH=models/hogan_logreg.pkl
HOGAN_ML_BUY_THRESHOLD=0.55
HOGAN_ML_SELL_THRESHOLD=0.45
HOGAN_USE_EMA_CLOUDS=false
HOGAN_EMA_FAST_SHORT=8
HOGAN_EMA_FAST_LONG=9
HOGAN_EMA_SLOW_SHORT=34
HOGAN_EMA_SLOW_LONG=50
HOGAN_USE_FVG=false
HOGAN_FVG_MIN_GAP_PCT=0.001
HOGAN_SIGNAL_MODE=any
```

### Signal indicator env vars

| Variable | Default | Description |
|---|---|---|
| `HOGAN_USE_EMA_CLOUDS` | `false` | Enable Ripster EMA cloud signal |
| `HOGAN_EMA_FAST_SHORT` | `8` | Fast cloud short EMA span |
| `HOGAN_EMA_FAST_LONG` | `9` | Fast cloud long EMA span |
| `HOGAN_EMA_SLOW_SHORT` | `34` | Slow cloud short EMA span |
| `HOGAN_EMA_SLOW_LONG` | `50` | Slow cloud long EMA span |
| `HOGAN_USE_FVG` | `false` | *(Experimental)* Enable ICT Fair-Value Gap signal |
| `HOGAN_FVG_MIN_GAP_PCT` | `0.001` | Minimum gap size (fraction of price) to record an FVG |
| `HOGAN_SIGNAL_MODE` | `any` | `ma_only` / `any` / `all` — legacy signal combinator (secondary to AgentPipeline) |

## EMA clouds workflow

Enable EMA clouds in `.env` and backtest before paper-trading:

```env
HOGAN_USE_EMA_CLOUDS=true
```

Backtest with clouds enabled:

```bash
python -m hogan_bot.backtest_cli --symbol BTC/USD --limit 5000
```

Retrain the ML model to pick up cloud features:

```bash
python -m hogan_bot.train --symbol BTC/USD --timeframe 1h --limit 5000 --horizon-bars 3 --model-path models/hogan_logreg.pkl
```

Then backtest with clouds and ML combined:

```bash
python -m hogan_bot.backtest_cli --symbol BTC/USD --limit 5000 --use-ml
```

**ICT/FVG** is experimental and quarantined. Use `HOGAN_USE_FVG=true` only for testing; it is not part of the core AgentPipeline decision flow.

## Multi-exchange support

Hogan is powered by [CCXT](https://docs.ccxt.com/) (110+ exchanges).  The active exchange is
controlled by a single environment variable — no code changes needed.

### Switching exchange

```env
HOGAN_EXCHANGE=binance      # or bybit, coinbase, okx, huobi, kucoin, …
```

Run the bot or backtest on a different venue:

```bash
# Backtest BTC/USDT on Binance
HOGAN_EXCHANGE=binance python -m hogan_bot.backtest_cli --symbol BTC/USDT --limit 5000

# Fetch Binance data into local DB
python -m hogan_bot.fetch_data --exchange binance --symbol BTC/USDT ETH/USDT --limit 10000
```

### Using your forked CCXT

You have a fork at <https://github.com/htownz/ccxt>.  To install it:

```bash
pip uninstall ccxt -y
pip install git+https://github.com/htownz/ccxt.git@master#egg=ccxt
```

Keep your fork in sync with upstream:

```bash
git remote add upstream https://github.com/ccxt/ccxt.git
git fetch upstream
git merge upstream/master
git push origin master
```

### Multi-exchange data aggregation (`hogan_bot.multi_exchange`)

```python
from hogan_bot.multi_exchange import (
    fetch_multi_ohlcv,    # parallel OHLCV fetch
    vwap_composite,       # volume-weighted average across venues
    price_spread,         # cross-exchange price divergence
    fetch_funding_rates,  # perpetual funding rates
    fetch_open_interests, # open interest snapshot
    fetch_tickers,        # latest tickers
    composite_last_price, # volume-weighted last price
)

# Fetch BTC/USDT 1h bars from three exchanges in parallel
dfs = fetch_multi_ohlcv("BTC/USDT", ["binance", "bybit", "okx"], timeframe="1h")

# Build a composite OHLCV series (volume-weighted)
composite = vwap_composite(dfs)

# Monitor cross-exchange spread (useful for arbitrage awareness)
spread_df = price_spread(dfs)

# Check perpetual funding rates (high positive rate = crowded longs = bearish bias)
rates = fetch_funding_rates("BTC/USDT:USDT", ["binance", "bybit"])
for eid, r in rates.items():
    if r:
        print(f"{eid}: {r['fundingRate']:.4%}")
```

### Available CCXT endpoints

| Method | Description |
|---|---|
| `fetch_ohlcv_df()` | OHLCV bars as a pandas DataFrame |
| `fetch_ticker()` | Latest bid/ask/last/volume snapshot |
| `fetch_order_book()` | Level-2 order book (configurable depth) |
| `fetch_trades()` | Most recent public trades |
| `fetch_funding_rate()` | Current perpetual funding rate (derivatives) |
| `fetch_open_interest()` | Open interest snapshot (derivatives) |
| `fetch_funding_rate_history()` | Historical funding rates |
| `list_symbols()` | All active symbols (optionally filtered by quote) |
| `list_timeframes()` | Supported bar intervals |
| `market_info()` | Price/qty precision, min order size, fees |

Endpoints that are not supported by a specific exchange return `None` gracefully
— you never need to check `exchange.has` manually.

## Walk-forward retraining (`hogan_bot.retrain`)

The bot adapts to changing market regimes by periodically retraining on the
most recent N bars (rolling window).  A new model only **replaces** the
production model when it improves the chosen metric by at least
`--min-improvement` — a stale or regressing model is never promoted.

### One-shot retrain

```bash
# Fetch the latest 5 000 bars from Kraken, train logreg, promote if better
python -m hogan_bot.retrain

# Load candles from local SQLite DB (offline, faster)
python -m hogan_bot.retrain --from-db

# Evaluate without touching anything
python -m hogan_bot.retrain --dry-run
```

### Scheduled retrain (blocking loop)

```bash
# Retrain every 24 hours (daily)
python -m hogan_bot.retrain --schedule 24

# Every 6 hours on Binance with XGBoost
python -m hogan_bot.retrain --exchange binance --symbol BTC/USDT \
    --model-type xgboost --schedule 6
```

### Scheduling on Windows (Task Scheduler)

1. Open **Task Scheduler → Create Basic Task**.
2. Set the trigger to **Daily** (or your preferred interval).
3. Set the action to **Start a Program**:
   - Program: `C:\path\to\Hogan\.venv\Scripts\python.exe`
   - Arguments: `-m hogan_bot.retrain --from-db --window-bars 5000`
   - Start in: `C:\path\to\Hogan`

### Scheduling on Linux/macOS (cron)

```cron
# Retrain daily at 02:00 UTC, load from local DB
0 2 * * * cd /path/to/Hogan && .venv/bin/python -m hogan_bot.retrain --from-db >> logs/retrain.log 2>&1
```

### How it works

```
Cycle N-1   ├───────────── window ──────────────┤
Cycle N         ├───────────── window ──────────────┤
Cycle N+1           ├───────────── window ──────────────┤
                                                      ↑ "now"
```

1. Fetch the `window_bars` most recent candles (live or from SQLite).
2. Train a **candidate** model to a timestamped temp path.
3. Compare candidate `roc_auc` (or `--promotion-metric`) against the registry best.
4. If improvement ≥ `--min-improvement`: copy candidate → production path + log to registry.
5. Delete the candidate file regardless of outcome (no orphaned artifacts).

### Key options

| Flag | Default | Description |
|---|---|---|
| `--window-bars` | 5000 | Rolling window size (number of bars) |
| `--model-type` | `logreg` | `logreg` / `random_forest` / `xgboost` / `lightgbm` |
| `--promotion-metric` | `roc_auc` | Metric compared against registry best |
| `--min-improvement` | `0.005` | Minimum gain over registry best to promote |
| `--schedule HOURS` | — | Run continuously every N hours |
| `--from-db` | — | Load from local SQLite instead of live fetch |
| `--dry-run` | — | Train + evaluate, no writes at all |

### Environment variables (`.env`)

```env
HOGAN_RETRAIN_WINDOW_BARS=5000
HOGAN_RETRAIN_MODEL_TYPE=logreg
HOGAN_RETRAIN_MIN_IMPROVEMENT=0.005
HOGAN_RETRAIN_PROMOTION_METRIC=roc_auc
HOGAN_RETRAIN_SCHEDULE_HOURS=24
```

## How to further enhance ML abilities next

- Funding rates and open interest (now available via `multi_exchange`) can be
  added as ML features — high positive funding is a crowded-long signal.
- Probability calibration (Platt scaling): `python -m hogan_bot.train --calibrate`.
- Confidence-based position sizing: `HOGAN_ML_CONFIDENCE_SIZING=true`.
- Try gradient boosted trees: `--model-type xgboost` or `--model-type lightgbm`.
- Promote only models that beat baseline after fees/slippage using the model registry.
