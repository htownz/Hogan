# Hogan 2.0 — Validation Commands & Metrics

## Phase A Validation

After applying the Phase A changes, run these commands to verify correctness.

### 1. Verify ICT is quarantined from default path

```bash
# Default optimizer search should NOT include ICT
python -m hogan_bot.optimize --symbol BTC/USD --timeframe 1h --trials 5 --from-db --quiet
# Inspect output: all configs should have use_ict=false

# Experimental mode should include ICT in search
python -m hogan_bot.optimize --symbol BTC/USD --timeframe 1h --trials 5 --from-db --quiet --experimental
# Inspect output: some configs may have use_ict=true
```

### 2. Verify default timeframe is 1h

```bash
python -c "from hogan_bot.config import load_config; c = load_config(); print(f'timeframe={c.timeframe}, execution_tf={c.execution_timeframe}')"
# Expected: timeframe=1h, execution_tf=15m
```

### 3. Verify ML uses 59 features (no ICT in default set)

```bash
python -c "from hogan_bot.ml import _FEATURE_COLUMNS, _EXPERIMENTAL_FEATURES; print(f'Core: {len(_FEATURE_COLUMNS)}, Experimental: {len(_EXPERIMENTAL_FEATURES)}')"
# Expected: Core: 59, Experimental: 7
```

### 4. Verify strategy works without MA gatekeeper

```bash
# Backtest on 1h BTC — should generate trades even when MA doesn't cross
python -m hogan_bot.backtest_cli --symbol BTC/USD --timeframe 1h --from-db --limit 8000
```

### 5. Verify ICT tests still pass

```bash
python -m pytest tests/test_ict.py -v
```

### 6. OOS validation on 1h data

```bash
python scripts/validate_oos.py --symbol BTC/USD --timeframe 1h --auto-split
```

### 7. Data coverage check

```bash
python scripts/list_candles.py --db data/hogan.db
# Should show 17k+ 1h bars for BTC/USD, ETH/USD, SOL/USD
```

---

## Key Metrics to Track

### Before/After Phase A

| Metric | Measure | Target |
|--------|---------|--------|
| OOS Sharpe (1h BTC) | `scripts/validate_oos.py` | > 0.0 (profitable after fees) |
| OOS Sortino (1h BTC) | `scripts/validate_oos.py` | > 0.0 |
| Max Drawdown | Backtest | < 20% |
| Trade Count (1h, 1yr) | Backtest | > 30 |
| Win Rate | Backtest | > 45% |
| Feature Count | `len(_FEATURE_COLUMNS)` | 59 (core) |
| ICT in search space | Default optimizer | No |
| Default timeframe | `load_config().timeframe` | "1h" |

### Ongoing Health Checks

```bash
# Quick smoke test — runs backtest with default config
python -m hogan_bot.backtest_cli --symbol BTC/USD --timeframe 1h --from-db --limit 5000

# Full OOS validation
python scripts/validate_oos.py --symbol BTC/USD --timeframe 1h --train-bars 8000 --val-bars 3000 --test-bars 3000

# Compare experimental ICT vs champion (no ICT)
python -m hogan_bot.backtest_cli --symbol BTC/USD --timeframe 1h --from-db --compare
```

---

## Phase B+ Validation (future)

### Feature Spine

```bash
# Verify all features have as-of timestamps
python -c "from hogan_bot.feature_registry import registry; [print(f.name, f.latency_class) for f in registry]"

# Check for lookahead bias
python -m hogan_bot.lookahead_check --timeframe 1h
```

### Model Stack

```bash
# Regime model accuracy
python -c "from hogan_bot.models.regime_model import evaluate; evaluate()"

# Forecast model calibration (Brier score)
python -c "from hogan_bot.models.forecast_model import evaluate; evaluate()"
```

### Champion/Challenger

```bash
# Shadow evaluation report
python scripts/shadow_report.py

# Promotion gate check
python scripts/promotion_check.py --challenger models/challenger.pkl
```
