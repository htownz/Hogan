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
```

## Next recommended steps

1. Add local persistence (SQLite) for candles, signals, fills, equity curve.
2. Add offline backtester module (walk-forward + fee/slippage stress).
3. Add feature store for volume regimes and volatility regimes.
4. Train baseline ML classifier for directional filter (not direct execution).
5. Add execution safeguards before enabling any live trading.
