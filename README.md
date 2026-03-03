# Hogan

Hogan is a **paper-trading research bot** for BTC/USD and ETH/USD on Kraken, aimed at day-trading and short-term strategy iteration.

## Important safety notes

- This build is **paper mode only** (no live order routing yet).
- If you ever exposed API keys in chat/logs/repo, **rotate them immediately** in Kraken.
- No strategy guarantees profit; use strict risk controls and human oversight.

## What’s implemented now

- Kraken OHLCV ingestion via `ccxt`
- MA crossover + volume confirmation signal
- Dynamic stop-distance proxy from recent candle range
- Position sizing bounded by:
  - risk-per-trade
  - aggressive allocation cap
- Portfolio simulation with fees (`PaperPortfolio`)
- Max drawdown guard to halt trading on breach
- Optional ML probability filter that gates buy/sell signals using a trained logistic model
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

1. Train a baseline directional model on BTC or ETH data:

```bash
python -m hogan_bot.train --symbol BTC/USD --timeframe 5m --limit 5000 --horizon-bars 3 --model-path models/hogan_logreg.pkl
```

2. Enable ML filter in `.env`:

```env
HOGAN_USE_ML_FILTER=true
HOGAN_ML_MODEL_PATH=models/hogan_logreg.pkl
HOGAN_ML_BUY_THRESHOLD=0.55
HOGAN_ML_SELL_THRESHOLD=0.45
```

3. Run paper trading and compare with ML disabled (A/B test):

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
```

## How to further enhance ML abilities next

- Add walk-forward retraining (e.g., daily rolling retrain on latest bars).
- Add calibration and confidence bands before allowing trades.
- Add regime features (volatility bucket, trend regime, funding/open-interest if available).
- Add model registry + experiment tracking (metrics, params, model hash).
- Promote only models that beat baseline after fees/slippage.
