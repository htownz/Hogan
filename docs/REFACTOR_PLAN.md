# Hogan 2.0 — File-by-File Refactor Plan

## Phase A: Simplify Current Live Path

### A1. Quarantine ICT from champion path

| File | Action | Details |
|------|--------|---------|
| `hogan_bot/config.py` | **Edit** | `use_ict` default stays `False`. Add `use_ict` to a new `_EXPERIMENTAL` set. Change `timeframe` default from `"5m"` to `"1h"`. Add `execution_timeframe: str = "15m"`. |
| `hogan_bot/strategy.py` | **Edit** | Remove MA gatekeeper (`if ma_action == "hold": return hold`). Replace with signal-provider interface: each provider returns `(action, confidence)`, policy layer aggregates. ICT import becomes conditional. |
| `hogan_bot/optimize.py` | **Edit** | Move ICT params from `_PARAM_SPACE` to `_EXPERIMENTAL_SPACE`. Default search no longer includes ICT. Add `--experimental` flag to opt-in. |
| `hogan_bot/main.py` | **Edit** | Stop passing ICT params to `generate_signal()`. Use new signal interface. |
| `hogan_bot/trader_service.py` | **Edit** | Same as main.py — remove ICT param forwarding. |
| `hogan_bot/backtest.py` | **Edit** | ICT params become optional/experimental. Default backtest uses 1h. |
| `hogan_bot/backtest_cli.py` | **Edit** | `--use-ict` moves under `--experimental` group. Default `--timeframe` changes to `1h`. |
| `hogan_bot/ml.py` | **Edit** | ICT structural features (7) move to an `_EXPERIMENTAL_FEATURES` list. They're still computed but not included in the default feature set. ML training uses `_CORE_FEATURES` by default. |
| `hogan_bot/ict.py` | **Keep as-is** | No code changes. Module stays importable for `--experimental` mode. |
| `hogan_bot/agent_pipeline.py` | **Keep as-is** | Already doesn't pass ICT params. |
| `tests/test_ict.py` | **Keep as-is** | Tests still run; ICT isn't deleted. |
| `.env.example` | **Edit** | Move ICT env vars to an `# EXPERIMENTAL` section with comment. |
| `README.md` | **Edit** | Move ICT references to an "Experimental / Lab" section. Update default examples to 1h. |
| `Command Prompts.md` | **Edit** | Update examples to 1h, mark ICT commands as experimental. |

### A2. Remove MA as mandatory gatekeeper

| File | Action | Details |
|------|--------|---------|
| `hogan_bot/strategy.py` | **Edit** | Replace `generate_signal()` with `generate_signal_v2()` that uses a `SignalProvider` interface. MA becomes one provider among several, not a gate. Each provider independently votes. The aggregation logic stays (any/all mode). |
| `hogan_bot/strategy.py` | **New class** | `SignalProvider(Protocol)` with `def evaluate(candles, config) -> SignalVote`. |
| `hogan_bot/signals/` | **New package** | Create `hogan_bot/signals/__init__.py`, `ma.py`, `ema_cloud.py`, `fvg.py`, `ml_filter.py`, `regime.py`, `forecast.py`, `risk.py`. Each implements `SignalProvider`. |
| `hogan_bot/signals/regime.py` | **New file** | Placeholder for 1h regime classification. Initially wraps existing `regime.py` detection. |
| `hogan_bot/signals/forecast.py` | **New file** | Placeholder for forward-return forecasting. Initially wraps ML `predict_up_probability`. |
| `hogan_bot/signals/risk.py` | **New file** | Placeholder for risk model. Initially wraps existing ATR-based stop logic. |

### A3. Default to 1h BTC

| File | Action | Details |
|------|--------|---------|
| `hogan_bot/config.py` | **Edit** | `timeframe` default: `"5m"` → `"1h"`. Keep `HOGAN_TIMEFRAME` env override. Add `execution_timeframe` default `"15m"`. |
| `hogan_bot/rl_env.py` | **Edit** | Verify annualization uses `timeframe_utils` (already done in prior patch). |
| `hogan_bot/retrain.py` | **Edit** | Default `RETRAIN_WINDOW_BARS` appropriate for 1h (8000 = ~11 months). |

---

## Phase B: Build Market-State Feature Spine

| File | Action | Details |
|------|--------|---------|
| `hogan_bot/feature_spine.py` | **New file** | Canonical event-time index builder. Takes a bar series, joins all feature layers with as-of semantics. Outputs a single DataFrame with freshness columns. |
| `hogan_bot/feature_registry.py` | **New file** | `FeatureMeta` dataclass. Registry of all features with `name`, `layer`, `source`, `latency_class`, `staleness_max`, `fill_policy`, `as_of_column`. |
| `hogan_bot/feature_layers/` | **New package** | `microstructure.py`, `derivatives.py`, `macro.py`, `onchain_sentiment.py`. Each builds its feature group from raw data. |
| `hogan_bot/ml.py` | **Edit** | `_feature_frame()` delegates to `feature_layers/microstructure.py`. `add_macro_features()` delegates to `feature_spine.py`. |
| `hogan_bot/macro_features.py` | **Edit** | Fix `_compute_extended_features` to use as-of joins (not broadcast latest). |
| `hogan_bot/features_mtf.py` | **Edit** | Align with feature spine. Point-in-time semantics already partially correct here. |

---

## Phase C: Build Model Stack

| File | Action | Details |
|------|--------|---------|
| `hogan_bot/models/regime_model.py` | **New file** | HGB/LightGBM classifier for 1h regime labels (trend_up, trend_down, mean_revert, breakout, panic, grind, risk_on, risk_off). |
| `hogan_bot/models/forecast_model.py` | **New file** | Calibrated P(return > threshold) at 4h/12h/24h. Triple-barrier labels. |
| `hogan_bot/models/risk_model.py` | **New file** | Quantile regression for vol, classifier for stop-hit, MAE estimator. |
| `hogan_bot/models/execution_model.py` | **New file** | 15m entry/exit timing. PPO or rule-based initially. |
| `hogan_bot/policy.py` | **New file** | Regime router + EV threshold + risk gate. Replaces `generate_signal()` as the decision center. |
| `hogan_bot/retrain.py` | **Edit** | Support retraining regime/forecast/risk models with walk-forward. |
| `hogan_bot/auto_promote.py` | **Edit** | Multi-model promotion: each model type has its own champion/challenger. |

---

## Phase D: Build Local AI Swarm

| File | Action | Details |
|------|--------|---------|
| `hogan_bot/swarm/__init__.py` | **New package** | |
| `hogan_bot/swarm/base.py` | **New file** | `ResearchAgent` base class with structured JSON output, LLM client abstraction. |
| `hogan_bot/swarm/data_custodian.py` | **New file** | Gap detection, staleness audit, outlier flagging. |
| `hogan_bot/swarm/market_cartographer.py` | **New file** | Regime labeling, structural breaks, analog identification. |
| `hogan_bot/swarm/feature_scientist.py` | **New file** | Feature proposals, interaction terms. |
| `hogan_bot/swarm/forecast_lab.py` | **New file** | Train/evaluate candidate models. |
| `hogan_bot/swarm/validator_judge.py` | **New file** | Walk-forward, leakage tests, fee realism. |
| `hogan_bot/swarm/risk_steward.py` | **New file** | Size, stop width, kill-switch. |
| `hogan_bot/swarm/memory_agent.py` | **New file** | Historical analog retrieval (RAG-backed). |
| `hogan_bot/swarm/orchestrator.py` | **New file** | Ray-based coordinator. Runs swarm tasks on schedule. |

---

## Phase E: Continuous Learning

| File | Action | Details |
|------|--------|---------|
| `hogan_bot/retrain.py` | **Edit** | Nightly challenger retrain loop. Shadow paper evaluation. |
| `hogan_bot/auto_promote.py` | **Edit** | Multi-gate promotion: OOS Sharpe, max_dd, min_trades, walk-forward consistency, shadow PnL. |
| `hogan_bot/online_learner.py` | **Edit** | Stage 1: recalibrate probabilities only. Stage 2+: ensemble weight updates. |
| `scripts/validate_oos.py` | **Edit** | Support multi-model validation (regime + forecast + risk). |

---

## Files That Don't Change

These files are unaffected by the refactor:

- `hogan_bot/exchange.py` — CCXT wrapper, already generic
- `hogan_bot/storage.py` — SQLite store, already generic
- `hogan_bot/paper.py` — Paper portfolio mechanics
- `hogan_bot/execution.py` — Paper/Live execution interface
- `hogan_bot/risk.py` — Position sizing and drawdown guard
- `hogan_bot/notifier.py` — Notification layer
- `hogan_bot/security.py` — Key storage
- `hogan_bot/metrics.py` — Prometheus metrics
- `hogan_bot/data_engine.py` — Data streaming
- `hogan_bot/data_watchdog.py` — Staleness checks
- `hogan_bot/timeframe_utils.py` — Already correct
- All `fetch_*.py` modules — Data fetchers are independent
- `hogan_bot/labeler.py` — Trade labeling
- `hogan_bot/registry.py` — Model registry
- `dashboard.py` — Streamlit dashboard (minor config updates later)
- `refresh_daily.py` — Daily refresh orchestrator

---

## Migration Order

```
Phase A (this PR):
  A1 → quarantine ICT
  A2 → remove MA gate, create signal providers
  A3 → default to 1h

Phase B (next PR):
  feature_registry → feature_spine → feature_layers → ml.py migration

Phase C (next):
  regime_model → forecast_model → risk_model → policy.py → execution_model

Phase D (parallel with C):
  swarm base → individual agents → orchestrator

Phase E (after C+D stable):
  retrain loop → promotion gates → online learning
```
