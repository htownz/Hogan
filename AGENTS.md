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

## Swarm vs gate chain
- `HOGAN_SWARM_MODE=active` / `conditional_active`: swarm runs **after** ML + edge/quality/ranging/pullback gates.
- Default `swarm_active_allow_new_signals=false`: swarm cannot promote a gated `hold` into `buy`/`sell` (only veto, align, or size-scale when the gated action is already directional). Set `HOGAN_SWARM_ACTIVE_ALLOW_NEW_SIGNALS=true` only for experiments.
- `conditional_active` always honors vetoes (hold + zero size when any agent vetoes).
- Swarm weight learning and `auto_quarantine_check` are **disabled** when `config._backtest` is true (backtest must not write promotions/quarantine to the shared DB).
- If agents are stuck `advisory_only` and fusion scores stay at zero: `python scripts/reset_swarm_agent_modes.py` (use `--dry-run` first).

## Enhancement backlog (short)
- **Swarm**: per-regime `swarm_min_agreement` / margin tuning (see `docs/SWARM_CONDITIONAL_TUNING.md`); merge-block + veto metrics: dashboard Swarm tab + Prometheus `hogan_swarm_*` counters.
- **Swarm authority**: `hogan_bot/swarm_authority.py` — startup config validation + shadow/active drift detection. Wired into event_loop, shadow_report, and promotion_check.
- **Execution health**: `hogan_bot/execution_health.py` — tracks order outcomes, slippage, fill rates, circuit-breaker state. Wired into event_loop. Operator runbook: `docs/EXECUTION_RUNBOOK.md`.
- **Exit lifecycle**: `hogan_bot/exit_lifecycle.py` — tail-loss, time-in-trade-by-regime, exit quality analytics. Wired into walk-forward JSON + CLI.
- **Quarantine**: optional minimum calendar age before demoting `pipeline_v1`; auto-quarantine counts only directional agents that appear in the accuracy table so a phantom default-active `volatility_regime_v1` cannot unlock demoting the last real voter (`pipeline_v1`).
- **ML**: silence sklearn feature-name warning in tests or align `LogisticRegression` with feature names.
- **Certification**: use `python scripts/swarm_certification.py --scratch-db` to avoid writing swarm rows to production (add `--keep-scratch-db` to inspect the copy).

## Champion Path
- `HOGAN_CHAMPION_MODE=true` locks experimental features
- 8-feature subset in `feature_registry.CHAMPION_FEATURE_COLUMNS`
- Train: `python -m hogan_bot.train --champion`

## Data & Exchanges
- `HOGAN_USE_REST_DATA=1` — REST polling during Kraken maintenance
- `HOGAN_EXCHANGE` — kraken, binance, bybit, oanda, etc.
- `HOGAN_USE_OANDA=true` — uses OandaDataEngine (REST candles) + OandaExecution
  - Works in both paper and live mode
  - Requires OANDA_ACCESS_TOKEN + OANDA_ACCOUNT_ID in .env
- `HOGAN_ENABLE_SHORTS=true` — enables regime-gated short selling in event loop
  - Shorts allowed in: `volatile` (short_size_scale=0.50), `trending_down` (short_size_scale=1.0)
  - Shorts blocked in: `ranging`, `trending_up`
- `HOGAN_CLOSE_AND_REVERSE=false` — close-and-reverse disabled by default (no benefit in BTC attribution tests)

## Regime Logic — Responsibility Boundaries
Each regime-aware component has a clearly defined role. Avoid double-counting.
- **MetaWeigher** (`agent_pipeline.py`): direction and vote-level regime adaptation. Uses `meta_*_delta` and `meta_*_threshold` from `RegimeConfig`.
- **entry_quality_gate** (`decision.py`): minimum setup cleanliness / confidence sufficiency. Uses `quality_final_mult` / `quality_tech_mult` from `RegimeConfig`.
- **ranging_gate** (`decision.py`): chop-specific suppression only (active in ranging). Soft mode reduces size, hard mode blocks.
- **effective_thresholds** (`regime.py`): execution economics — ML gates, TP/SL, position scale. Three confidence tiers (0.25/0.35/0.50) adjust execution parameters by regime confidence.
- **Side gating** (`config.py` `RegimeConfig`): `allow_longs` / `allow_shorts` per regime. Volatile and trending_down allow shorts; ranging and trending_up block them.

## Canonical Profile
- Defined in `hogan_bot/profiles.py` → `CANONICAL_PROFILE`
- BTC/USD, 1h, ML on (threshold 0.51), shorts enabled, pullback gate on, close-and-reverse off
- Short max hold: 12h (from sweep optimization)
- Use via backtest CLI: `--profile canonical`

## Macro Sitout Filter
- `HOGAN_MACRO_SITOUT=true` in `.env` enables the macro event sit-out filter
- Blocks trades on FOMC/CPI/NFP event days, scales down during extreme greed
- Asymmetric: does NOT penalize extreme fear (strategy thrives in volatile crash-recovery)
- Walk-forward validated: mean return improved from -0.14% to -0.01%

## Promotion & validation docs
- **What counts as success** (profitability, monthly framing, OOS + drawdown): `docs/SUCCESS_DEFINITION.md`
- **Operator checklist** (before changing `HOGAN_SWARM_MODE` / `HOGAN_SWARM_PHASE`): `docs/PROMOTION_CHECKLIST.md`
- **Conditional swarm + regime weights**: `docs/SWARM_CONDITIONAL_TUNING.md`
- **Walk-forward gate for strategy changes**: `docs/STRATEGY_CHANGE_GATE.md`
- **Archived runs**: `python scripts/run_validation_battery.py --db data/hogan.db` → `reports/validation/`

## Validation & Testing
```bash
# Fast sanity (subset, ~10s) — pass the listed paths as one pytest invocation
python -m pytest tests/test_champion.py tests/test_ml.py tests/test_exchange.py tests/test_agent_quarantine.py tests/test_swarm_certification.py::TestSwarmGatedMerge tests/test_swarm_certification.py::TestShadowParity tests/test_decision_parity.py::TestPolicyCoreEquivalence tests/test_observability_scripts.py::TestConfigDefaults tests/test_swarm_policy_observability.py -q

# Lint (CI: E,F,I,B007,B904 on hogan_bot, tests, scripts, diagnostics; on Windows use python -m ruff)
python -m ruff check hogan_bot/ tests/ scripts/ diagnostics/ --select E,F,I,B007,B904 --ignore E501

# Unit tests (CI runs these on push/PR)
pytest tests/ -v

# Walk-forward validation (rolling OOS with promotion gate)
python -m hogan_bot.walk_forward --db data/hogan.db --n-splits 5

# Walk-forward without ML (technical pipeline baseline)
python -m hogan_bot.walk_forward --db data/hogan.db --n-splits 5 --no-ml

# Walk-forward with macro sitout (recommended diagnostic)
python -m hogan_bot.walk_forward --db data/hogan.db --n-splits 5 --no-ml --macro-sitout

# Feature importance audit (permutation importance on champion features)
python -m hogan_bot.feature_importance --db data/hogan.db
```

## CI/CD
- `.github/workflows/ci.yml` runs all tests + ruff (`E,F,I,B007,B904` on `hogan_bot/`, `tests/`, `scripts/`, `diagnostics/`) on push/PR to main/develop
