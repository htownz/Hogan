# Hogan

Hogan is a **paper-trading research bot** for BTC/USD, ETH/USD and any trading pair on 110+ exchanges, aimed at day-trading and short-term strategy iteration.

## Important safety notes

- This build is **paper mode only** (no live order routing yet).
- If you ever exposed API keys in chat/logs/repo, **rotate them immediately** in Kraken.
- No strategy guarantees profit; use strict risk controls and human oversight.

## What’s implemented now

- Kraken OHLCV ingestion via `ccxt`
- MA crossover + volume confirmation signal
- **Ripster EMA clouds** (fast 8/9, slow 34/50) — bullish/bearish/neutral cloud direction per bar
- **ICT Fair-Value Gap (FVG) detector** — identifies bullish and bearish gaps, tracks fill status, and generates entry signals when price re-enters an active zone
- Configurable signal combinator (`HOGAN_SIGNAL_MODE`):
  - `ma_only` — original MA crossover only
  - `any` — buy/sell fires if any enabled signal agrees (default)
  - `all` — buy/sell fires only when every enabled signal agrees
- Dynamic stop-distance proxy from recent candle range
- Position sizing bounded by:
  - risk-per-trade
  - aggressive allocation cap
- Portfolio simulation with fees (`PaperPortfolio`)
- Max drawdown guard to halt trading on breach
- Optional ML probability filter that gates buy/sell signals using a trained logistic model
  - ML feature set now includes cloud direction and FVG activity columns
- Backtest engine + CLI to evaluate strategy/ML settings on historical candles
- 24/5 behavior toggle (`HOGAN_TRADE_WEEKENDS=false` by default)
- `--max-loops` option for finite test runs

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
python -m hogan_bot.main --max-loops 2
```

## ML enhancement workflow (recommended)

The ML pipeline uses 20 features (momentum, RSI, volatility, candle microstructure, volume, EMA cloud direction and width, FVG counts) and supports two classifier types.

### 1. Validate with walk-forward cross-validation first

```bash
python -m hogan_bot.train --symbol BTC/USD --timeframe 5m --limit 5000 --cv --cv-splits 5
```

Output includes per-fold accuracy and ROC-AUC, plus mean values across folds.

### 2. Train a model

Logistic regression (scaled, default):

```bash
python -m hogan_bot.train --symbol BTC/USD --timeframe 5m --limit 5000 --horizon-bars 3 --model-path models/hogan_logreg.pkl
```

Random forest (includes feature importances in output):

```bash
python -m hogan_bot.train --symbol BTC/USD --timeframe 5m --limit 5000 --model-type random_forest --model-path models/hogan_rf.pkl
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
HOGAN_TIMEFRAME=5m
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
| `HOGAN_USE_FVG` | `false` | Enable ICT Fair-Value Gap signal |
| `HOGAN_FVG_MIN_GAP_PCT` | `0.001` | Minimum gap size (fraction of price) to record an FVG |
| `HOGAN_SIGNAL_MODE` | `any` | `ma_only` / `any` / `all` — how multiple signals are combined |

## EMA clouds + FVG workflow

Enable one or both new signal layers in `.env` and backtest before paper-trading:

```env
HOGAN_USE_EMA_CLOUDS=true
HOGAN_USE_FVG=true
HOGAN_SIGNAL_MODE=any
```

Backtest with clouds + FVGs enabled:

```bash
python -m hogan_bot.backtest_cli --symbol BTC/USD --limit 5000
```

Retrain the ML model to pick up the new cloud/FVG features:

```bash
python -m hogan_bot.train --symbol BTC/USD --timeframe 5m --limit 5000 --horizon-bars 3 --model-path models/hogan_logreg.pkl
```

Then backtest with both new signals and ML combined:

```bash
python -m hogan_bot.backtest_cli --symbol BTC/USD --limit 5000 --use-ml
```

Signal combinator guide:

| `HOGAN_SIGNAL_MODE` | Behaviour |
|---|---|
| `ma_only` | Only the MA crossover fires; EMA clouds and FVGs are ignored even if enabled |
| `any` | A buy/sell fires if the MA crossover, cloud direction, **or** FVG entry agrees |
| `all` | A buy/sell fires only when every enabled signal agrees simultaneously |

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

## How to further enhance ML abilities next

- Add walk-forward retraining (e.g., daily rolling retrain on latest bars).
- Funding rates and open interest (now available via `multi_exchange`) can be
  added as ML features — high positive funding is a crowded-long signal.
- Add calibration (Platt scaling or isotonic regression): `python -m hogan_bot.train --calibrate`.
- Add confidence bands: only trade when predicted probability is far from 0.5
  (`HOGAN_ML_CONFIDENCE_SIZING=true`).
- Try gradient boosted trees: `--model-type xgboost` or `--model-type lightgbm`.
- Promote only models that beat baseline after fees/slippage using the model registry.
