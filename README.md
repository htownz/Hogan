# Hogan

Hogan is a **BTC-first, 1h-led market operating system** — an agent-routed, forecast-and-risk-driven trading system for BTC/USD and major pairs on 110+ exchanges.

## Important Safety Notes

- **Paper mode by default.** Live trading requires explicit opt-in (see `.env.example`).
- Oanda FX live execution and CCXT crypto execution paths are available but must be activated deliberately.
- If you ever exposed API keys in chat/logs/repo, **rotate them immediately**.
- No strategy guarantees profit; use strict risk controls and human oversight.

## Architecture Overview

Hogan follows a multi-layer architecture:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    AGENT PIPELINE (Decision Brain)                   │
│  TechnicalAgent · SentimentAgent · MacroAgent → MetaWeigher          │
└─────────────────────────────────────────────────────────────────────┘
         │  weighted decision
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    ML PROBABILITY FILTER (Optional)                  │
│  66 features · XGBoost/LightGBM/LogReg · Walk-forward retraining    │
└─────────────────────────────────────────────────────────────────────┘
         │  calibrated probability
         ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    RISK MANAGEMENT + EXECUTION                       │
│  Position sizing · Drawdown guard · Paper/Live execution            │
└─────────────────────────────────────────────────────────────────────┘
```

## What's Implemented Now

### Core Trading System
- **AgentPipeline** — decision brain: Technical + Sentiment + Macro agents combined by MetaWeigher
- Agent-routed, forecast-and-risk-driven flow
- Default timeframe **1h** (BTC-first)
- Multi-exchange support via [CCXT](https://docs.ccxt.com/) (110+ exchanges)

### ML Pipeline
- **66 ML features** (59 core + 7 experimental): momentum, RSI, volatility, candle microstructure, volume, EMA cloud direction/width, OBV, Stochastic RSI, VWAP deviation, MACD, Bollinger Bands, ATR, multi-timeframe features, on-chain/macro signals
- Walk-forward cross-validation with champion/challenger model promotion
- Online learning from paper trade outcomes
- Model types: LogReg, Random Forest, XGBoost, LightGBM

### Risk Management
- Position sizing bounded by risk-per-trade and aggressive allocation cap
- Dynamic stop-distance proxy from recent candle range
- Max drawdown guard to halt trading on breach
- Portfolio simulation with fees (`PaperPortfolio`)

### Infrastructure
- Backtest engine + CLI to evaluate strategy/ML settings on historical candles
- SQLite storage for candles and trade journal
- Ripster EMA clouds (fast 8/9, slow 34/50) — cloud direction per bar
- 24/5 behavior toggle (`HOGAN_TRADE_WEEKENDS=false` by default)
- `--max-loops` option for finite test runs
- Dashboard and Discord notifications

### Experimental Features (Quarantined)
- **ICT/FVG** — experimental, quarantined from the core AgentPipeline decision flow

## Project Structure

```
/Hogan
├── README.md              # This file
├── requirements.txt       # Python dependencies
├── .env.example           # Environment variable template
├── data/                  # SQLite databases and caches
├── deployment/            # Service files (systemd, Windows Task Scheduler)
├── docs/                  # Architecture docs, planning, command reference
├── hogan_bot/             # Core trading bot Python package
├── logs/                  # Runtime logs
├── models/                # ML models (.pkl) and optimization configs
├── monitoring/            # Prometheus, Grafana, Docker configs
├── scripts/
│   ├── dashboards/        # Dashboard HTML and Python servers
│   ├── data/              # Data refresh and management scripts
│   ├── python/            # Python utility scripts (analysis, diagnostics)
│   └── windows/           # PowerShell automation scripts
└── tests/                 # Unit tests
```

## Quick Start

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
# Canonical runtime (async event loop):
python -m hogan_bot.event_loop
# Or with a finite run:
python -m hogan_bot.event_loop --max-events 100
```

> **Note:** `main.py` and `trader_service.py` are deprecated legacy runtimes.
> Always use `hogan_bot.event_loop` for paper and live trading.

## ML Enhancement Workflow (Recommended)

The ML pipeline uses **66 features** (59 core + 7 experimental ICT features) covering momentum, RSI, volatility, candle microstructure, volume, EMA cloud direction/width, OBV, Stochastic RSI, VWAP deviation, MACD, Bollinger Bands, ATR, multi-timeframe features, and on-chain/macro signals. It supports multiple classifier types: LogReg, Random Forest, XGBoost, and LightGBM.

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
python -m hogan_bot.event_loop --max-events 500
```

## Environment Config

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

## EMA Clouds Workflow

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

## Multi-Exchange Support

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

## Walk-Forward Retraining (`hogan_bot.retrain`)

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

## Future Enhancements

- Funding rates and open interest (now available via `multi_exchange`) can be
  added as ML features — high positive funding is a crowded-long signal.
- Probability calibration (Platt scaling): `python -m hogan_bot.train --calibrate`.
- Confidence-based position sizing: `HOGAN_ML_CONFIDENCE_SIZING=true`.
- Try gradient boosted trees: `--model-type xgboost` or `--model-type lightgbm`.
- Promote only models that beat baseline after fees/slippage using the model registry.

## Roadmap

See `docs/` for detailed architecture and planning documents:

| Phase | Focus | Status |
|-------|-------|--------|
| **Phase A** | Simplify live path — quarantine ICT, 1h default | ✅ Complete |
| **Phase B** | Build market-state feature spine with event-time index | 🔄 In Progress |
| **Phase C** | Build model stack (regime, forecast, risk, execution) | 📋 Planned |
| **Phase D** | Build local AI research swarm (7 agents) | 📋 Planned |
| **Phase E** | Continuous learning with champion/challenger | 📋 Planned |

### Key Architecture Documents

- `docs/HOGAN_2_ARCHITECTURE.md` — Full Hogan 2.0 vision and design
- `docs/NEXT_STEPS_PLAN.md` — Prioritized enhancement roadmap
- `docs/QUARANTINE_LIST.md` — What's quarantined vs. active in champion path
- `docs/VALIDATION_COMMANDS.md` — Commands to validate system health

## License

This project is proprietary. All rights reserved.
