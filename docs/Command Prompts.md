# Hogan 2.0 Command Reference

## Startup

```powershell
cd C:\Users\15125\Documents\Hogan\Hogan
.\.venv\Scripts\Activate.ps1
```

---

## Daily Operations

### Start the bot (paper mode)

```powershell
python -m hogan_bot.main
```

### Start the trader service

```powershell
python -m hogan_bot.trader_service
```

### Dashboard

```powershell
streamlit run dashboard.py
```

### Daily data refresh (all free sources)

```powershell
python refresh_daily.py
```

---

## Data

### Fetch latest candles (1h default)

```powershell
python -m hogan_bot.fetch_data --symbol BTC/USD --timeframe 1h --limit 720
```

### Backfill deep history (Kraken, paginated)

```powershell
python -m hogan_bot.fetch_data --symbol BTC/USD --timeframe 1h --backfill --target-bars 20000
python -m hogan_bot.fetch_data --symbol BTC/USD --timeframe 30m --backfill --target-bars 40000
```

### Backfill from Yahoo Finance (2 years 1h, no API key)

```powershell
python -m hogan_bot.backfill --symbol BTC/USD ETH/USD SOL/USD --timeframe 1h --period 2y
```

### Full multi-source backfill (Alpaca + macro)

```powershell
.\scripts\backfill_mtf.ps1
```

### Combined Kraken + Yahoo backfill

```powershell
.\scripts\backfill_kraken_yahoo.ps1
```

### Check DB candle counts

```powershell
python scripts/list_candles.py --db data/hogan.db
```

### Continuous candle accumulation (background)

```powershell
while ($true) {
    python -m hogan_bot.fetch_data --symbol BTC/USD --timeframe 1h --limit 720
    Start-Sleep -Seconds 3600
}
```

### Individual data sources

```powershell
python -m hogan_bot.fetch_feargreed         # no key
python -m hogan_bot.fetch_coingecko         # COINGECKO_KEY
python -m hogan_bot.fetch_gpr               # no key (cached 30 days)
python -m hogan_bot.fetch_derivatives       # Kraken Futures, no key
python -m hogan_bot.fetch_news_sentiment    # CRYPTOPANIC_KEY (free)
python -m hogan_bot.fetch_messari           # MESSARI_KEY
python -m hogan_bot.fetch_dune              # DUNE_API_KEY
python -m hogan_bot.fetch_oanda             # OANDA_ACCESS_TOKEN
python -m hogan_bot.fetch_glassnode         # GLASSNODE_KEY (paid)
python -m hogan_bot.fetch_santiment         # SANTIMENT_KEY (paid)
```

---

## Backtesting

### Standard backtest (1h BTC, from DB)

```powershell
python -m hogan_bot.backtest_cli --symbol BTC/USD --timeframe 1h --from-db --limit 8000
```

### With ML filter

```powershell
python -m hogan_bot.backtest_cli --symbol BTC/USD --timeframe 1h --from-db --limit 8000 --use-ml
```

### With RL vote

```powershell
python -m hogan_bot.backtest_cli --symbol BTC/USD --timeframe 1h --from-db --limit 8000 --use-rl
```

### Compare 5 signal configs side-by-side

```powershell
python -m hogan_bot.backtest_cli --symbol BTC/USD --timeframe 1h --from-db --limit 8000 --compare
```

### With ICT (experimental only)

```powershell
python -m hogan_bot.backtest_cli --symbol BTC/USD --timeframe 1h --from-db --limit 8000 --use-ict
```

### OOS validation

```powershell
python scripts/validate_oos.py --symbol BTC/USD --timeframe 1h --auto-split
```

### Full OOS with explicit splits

```powershell
python scripts/validate_oos.py --symbol BTC/USD --timeframe 1h --train-bars 8000 --val-bars 3000 --test-bars 3000
```

---

## Optimization

### Default (no ICT in search space)

```powershell
python -m hogan_bot.optimize --symbol BTC/USD --timeframe 1h --from-db --trials 200 --metric sharpe --max-drawdown 20
```

### With seed for reproducibility

```powershell
python -m hogan_bot.optimize --symbol BTC/USD --timeframe 1h --from-db --trials 200 --metric sharpe --max-drawdown 20 --seed 42
```

### Experimental (includes ICT in search)

```powershell
python -m hogan_bot.optimize --symbol BTC/USD --timeframe 1h --from-db --trials 200 --metric sharpe --experimental
```

---

## ML Training

### Retrain (walk-forward, from DB)

```powershell
python -m hogan_bot.retrain --from-db --window-bars 8000 --model-type xgboost --horizon-bars 12 --force-promote
```

### Train ML model

```powershell
python -m hogan_bot.train --symbol BTC/USD --timeframe 1h --from-db
```

### Massive retrain (multi-symbol, multi-horizon)

```powershell
python scripts/massive_retrain.py
```

---

## RL Training

### Smoke test (verify pipeline)

```powershell
python -m hogan_bot.rl_train --symbol BTC/USD --timeframe 1h --from-db --timesteps 5000 --verbose 0
```

### Full training

```powershell
python -m hogan_bot.rl_train --symbol BTC/USD --timeframe 1h --from-db --timesteps 500000
```

### RL hyperparameter tuning

```powershell
python -m hogan_bot.rl_tune --symbol BTC/USD --timeframe 1h --from-db
```

### Ubuntu / WSL (for GPU)

```bash
source ~/hogan-venv/bin/activate
cd /mnt/c/Users/15125/Documents/Hogan/Hogan
python3 -m hogan_bot.rl_train --symbol BTC/USD --timeframe 1h --from-db --timesteps 500000
```

---

## Tests

```powershell
$env:PYTHONPATH = "c:\Users\15125\Documents\Hogan\Hogan"
pytest tests/ -q
```

### ICT tests (should still pass after quarantine)

```powershell
python -m pytest tests/test_ict.py -v
```

---

## Validation (Phase A)

### Verify config defaults

```powershell
python -c "from hogan_bot.config import load_config; c = load_config(); print(f'timeframe={c.timeframe}, execution_tf={c.execution_timeframe}')"
# Expected: timeframe=1h, execution_tf=15m
```

### Verify feature counts

```powershell
python -c "from hogan_bot.ml import _FEATURE_COLUMNS, _EXPERIMENTAL_FEATURES; print(f'Core: {len(_FEATURE_COLUMNS)}, Experimental: {len(_EXPERIMENTAL_FEATURES)}')"
# Expected: Core: 59, Experimental: 7
```

---

## Discord Bot Commands

| Command | What it shows |
|---------|--------------|
| `!balance` | Equity, cash, invested, unrealized P&L, total return |
| `!positions` | Open trades: entry price, qty, current price, P&L % |
| `!pnl` | Realized P&L from fill history |
| `!fills` | Last 10 executed trades |
| `!status` | Mode, model, features enabled, ML thresholds |
| `!market` | Fear & Greed, BTC dominance, yield curve, DeFi TVL, funding |
| `!signals` | Live ML up-probability for BTC/USD and ETH/USD |
| `!help` | Full command list |

---

## Recommended Terminal Layout

| Terminal | Purpose |
|----------|---------|
| 1 | Candle accumulation loop (fetch_data every hour) |
| 2 | Bot (`python -m hogan_bot.main`) |
| 3 | Free for commands, backtests, training |
