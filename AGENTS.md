# Hogan — Agent Context

This file helps AI assistants understand the Hogan project when working in this codebase.

## What Hogan Is
A BTC-first, 1h-led trading system. Paper mode by default. Live trading requires explicit opt-in. Uses CCXT for exchanges (Kraken default), ML for probability filtering, and an agent pipeline for decisions.

## How to Run
```bash
python -m hogan_bot.main
# Or: python -m hogan_bot.trader_service (alias)
```

## How to Verify the Bot Is Running
1. **Process check**: `Get-Process python*` (Windows) or `ps aux | grep python` (Linux)
2. **Command line**: `wmic process where "ProcessId=<PID>" get CommandLine` — should show `hogan_bot.main` or `hogan_bot.trader_service`
3. **Logs**: Look for "Event loop starting", "REST polling active", or "Loaded ML model"
4. **Metrics**: `http://localhost:8000` (if HOGAN_METRICS_PORT=8000)

## Key Config Files
- `.env` — API keys, HOGAN_* vars (load_dotenv in config.py)
- `hogan_bot/config.py` — BotConfig, load_config, symbol_config

## Champion Path
- `HOGAN_CHAMPION_MODE=true` locks experimental features
- 15-feature subset in `feature_registry.CHAMPION_FEATURE_COLUMNS`
- Train: `python -m hogan_bot.train --champion`

## Data & Exchanges
- `HOGAN_USE_REST_DATA=1` — REST polling during Kraken maintenance
- `HOGAN_EXCHANGE` — kraken, binance, bybit, oanda, etc.
- `HOGAN_USE_OANDA=true` — uses OandaDataEngine (REST candles) + OandaExecution
  - Works in both paper and live mode
  - Requires OANDA_ACCESS_TOKEN + OANDA_ACCOUNT_ID in .env

## Testing
```bash
pytest tests/test_champion.py tests/test_ml.py tests/test_exchange.py -v
```
