# Hogan Enhancement Plan — Next Level

A prioritized roadmap based on deep codebase audit. Current state: bot is paper-trading,
59 ML features (8 champion), online learning wired, Optuna optimizer running, multi-symbol training
across BTC/ETH/SOL, XGBoost promoted as best model.

---

## Phase 1: Correctness (DONE — Merged)

These bugs were identified and fixed. Listed here for audit trail.

### 1.1 ~~Fix duplicate portfolio mutations~~ DONE
**Problem:** `PaperExecution.buy()` mutated the portfolio, then `main.py`/`event_loop.py`
called `portfolio.execute_buy()` again — doubling position sizes.
**Fix:** Executor is now the sole owner of portfolio state. `LiveExecution` also tracks
the portfolio (receives it via constructor). No caller touches `portfolio.execute_*`
when an executor is present.

### 1.2 ~~Fix paper journal side mismatch~~ DONE
**Problem:** `event_loop.py` journaled trades with `side="buy"` but `close_paper_trade`
queries by `side="long"` and computes P&L only when `side == "long"`. Result: P&L
computation was wrong and no journal rows ever matched on close.
**Fix:** Normalized all journal sides to `"long"` / `"short"` across `event_loop.py`.

### 1.3 ~~Fix retrain multi-symbol fall-through~~ DONE
**Problem:** `retrain.py` multi-symbol path trained a model, then fell through into the
single-symbol block and overwrote the result. Multi-symbol models were silently replaced
by single-symbol models.
**Fix:** Added early `return` after multi-symbol `_train_from_xy()`. Registry now logs
the full symbol set (`"BTC/USD,ETH/USD,SOL/USD"`) instead of the primary symbol.

### 1.4 ~~Fix online learning label bug~~ DONE
**Problem:** `_label_buffer_from_trade` used `mark_prices.get(symbol, px)` as entry price.
Since `mark_prices` holds current prices, P&L was always ~0 and every trade got label=0.
Also: features were written on both buy AND sell, creating orphaned unlabeled rows.
Auto-exits never labeled the training buffer at all.
**Fix:** Entry price captured from `pos.avg_entry` before the sell mutates state. Features
only written on buy (entry). Auto-exit path now labels the buffer with real P&L.

### 1.5 ~~Guard unfinished feature flags~~ DONE
**Problem:** `--use-extended-mtf` logged a warning but silently trained a standard model
without extended features. Users thought they had MTF but didn't.
**Fix:** Hard `RuntimeError` if flag is used with standard ML training. Clear message
directs to `rl_train.py --ext-features`.

---

## Phase 2: Strategy Evolution (This Week)

### 2.1 Regime-Aware Strategy Switching
**What:** Hogan detects regimes (trending/ranging/volatile) but uses one parameter set
for all. The Optuna optimizer should produce separate configs per regime.
**How:**
1. Run Optuna 3x: once with only trending candles, once ranging, once volatile.
2. Store top config per regime in `models/regime_configs.json`.
3. In `trader_service.py`, load the regime-specific config at runtime.

### 2.2 Multi-Strategy Ensemble
**What:** Use the top 5–10 Optuna results as a portfolio of strategies. Allocate capital
to whichever is performing best in a rolling window.
**How:**
1. After Optuna finishes, save top 10 configs.
2. In each loop, run all configs on the latest candle window. Weight by recent Sharpe.
3. Blend signals: majority vote or confidence-weighted.

### 2.3 Test 1h Timeframe
**Why:** 5m moves are often 0.05–0.30%; round-trip fee is 0.52%. You need larger moves
to overcome fees. 1h typical moves: 0.5–2%.
**How:**
1. Backfill 1h candles: `python -m hogan_bot.fetch_data --symbol BTC/USD --timeframe 1h --limit 5000`
2. Retrain on 1h: `python -m hogan_bot.retrain --from-db --symbol BTC/USD --timeframe 1h`
3. If backtest improves, set `HOGAN_TIMEFRAME=1h` in `.env`.

### 2.4 ~~Short-Entry Live Parity~~ DONE
**What:** Backtest had full short-entry logic but the live event loop did not.
**Fix:** Added `open_short` to `PaperExecution` and `RealisticPaperExecution`.
Wired comprehensive short-entry (on sell signals) and buy-covers-short (on buy signals)
logic into `event_loop.py` with regime gating, size scaling, risk parameter application,
and journaling. (Commit e54fc30)

### 2.5 Signal Funnel Diagnosis & Trade Frequency Optimization
**What:** Full-dataset funnel diagnostic revealed ML filter kills 91.4% of buy signals
and 84.6% of the market was regime-blocked from shorting.
**Fix:** Enabled shorts in volatile regime (`allow_shorts=True`, `short_size_scale=0.50`).
Trade count: 22 → 72 (10 longs, 26 shorts). Edge preserved (positive return, positive Sharpe).
Created canonical profile (`hogan_bot/profiles.py`) and `--profile` CLI flag.
Short max hold set to 12h from sweep optimization.

### 2.6 Paper Labels for Multi-Symbol Retrain
**Problem:** `--use-paper-labels` is skipped for multi-symbol training.
**Fix:** For each symbol, call `make_paper_trade_labels(db_path, candles_sym, sym)`,
concatenate the extra (X, y) rows, and pass to `_train_from_xy`.

---

## Phase 3: Infrastructure (2–4 Weeks)

### 3.1 Freqtrade-Style Evaluation Discipline
**What:** Import Freqtrade's promotion gate: parameter sweeps, automatic lookahead checks,
walk-forward validation, promotion only when challenger beats incumbent after
fees/slippage/drawdown.
**How:** The retrain pipeline already has promotion gating and walk-forward. Add:
- Lookahead-bias checker (verify no future data leaks in features)
- Automatic A/B test: challenger must outperform on out-of-sample window

### 3.2 ~~Event-Driven Architecture Cleanup~~ DONE
**What:** Ensure backtest and live share one event model.
**Status:** `main.py` now delegates to `event_loop.py`, which holds the full
implementation. One brain, one signal flow. Live/backtest parity is maintained
through `SignalResult` carrying regime-adjusted execution fields.

### 3.3 MLflow Model Governance
**What:** Every retrain, promotion, and live decision gets traced. Use MLflow for
experiment tracking instead of JSONL registry.
**How:**
1. `pip install mlflow`
2. Wrap `retrain.py` in `mlflow.start_run()`, log metrics/artifacts.
3. Replace JSONL registry reads with MLflow model registry queries.

### 3.4 OpenBB as Analyst Cockpit
**What:** Expose Hogan as a custom agent inside OpenBB. Answer: "What regime are we in?",
"Why did we skip this trade?", "Show champion vs challenger."
**How:** Use `mcp_server` module as the agent surface. OpenBB's AI SDK can query it.

---

## Phase 4: Advanced (1–2 Months)

### 4.1 Reinforcement Learning Agent
**What:** PPO policy as additional vote alongside MA/EMA/ICT. Learns entry/exit timing
from reward signals (actual P&L).
**Status:** RL scaffolding exists (`rl_train.py`, `rl_policy.py`). Needs: training on
sufficient data, hyperparameter tuning, integration as a signal voter.

### 4.2 Triple-Barrier Labeling
**What:** Label trades by profit target, stop loss, and max hold time — not just
"price up/down in N bars." Produces cleaner training labels.

### 4.3 Self-Evaluation Loop
**What:** Hogan periodically backtests its own recent performance, detects when strategy
is degrading, and triggers a retrain or parameter re-optimization automatically.

---

## Summary Checklist

| Priority | Task | Status | Impact |
|---------|------|--------|--------|
| P0 | Fix duplicate portfolio mutations | DONE | Critical |
| P0 | Fix paper journal side mismatch | DONE | Critical |
| P0 | Fix retrain multi-symbol fall-through | DONE | Critical |
| P0 | Fix online learning label bug | DONE | Critical |
| P0 | Guard unfinished feature flags | DONE | High |
| P1 | Regime-aware strategy switching | Next | High |
| P1 | Multi-strategy ensemble | Next | High |
| P1 | Test 1h timeframe | **DONE** | High |
| P1 | Short-entry live parity | **DONE** | Critical |
| P1 | Signal funnel diagnosis + gate tuning | **DONE** | High |
| P1 | Canonical profile (`--profile canonical`) | **DONE** | High |
| P1 | GitHub Actions CI pipeline | **DONE** | High |
| P1 | Walk-forward validation harness | **DONE** | Critical |
| P1 | Feature importance audit tool | **DONE** | High |
| P2 | Paper labels for multi-symbol | Planned | Medium |
| P2 | Freqtrade evaluation discipline | Planned | High |
| P2 | Regime-specific Optuna optimization | Next | High |
| P2 | Enhanced triple-barrier labels for champion | Next | High |
| P2 | Multi-model ensemble voting | Next | High |
| P3 | Event-driven architecture cleanup | **DONE** | Medium |
| P3 | Champion-specific retrain path (BTC 1h only) | Next | High |
| P3 | Data freshness enforcement in live loop | Next | Medium |
| P3 | Containerization + VPS Compose baseline | **DONE** | Medium |
| P3 | Docker image publishing + deploy automation | Planned | Medium |
| P3 | Live reconciliation loop | Planned | High |
| P3 | MLflow model governance | Planned | Medium |
| P3 | OpenBB analyst cockpit | Planned | Medium |
| P4 | RL agent training | Planned | High |
| P4 | Triple-barrier labeling | **DONE** | High |
| P4 | Self-evaluation loop | **DONE** | High |
| P4 | Adaptive ML confidence | **DONE** | High |
| P4 | Performance tracker + auto-weight tuning | **DONE** | High |
| P4 | Cross-asset regime signals | Planned | Medium |
| P4 | Adaptive MetaWeigher auto-promotion | Planned | Medium |
| P4 | Execution timing optimization (15m entry) | Planned | Medium |

---

*Updated after deep codebase audit. P0 correctness fixes are merged — all paper trade
data from this point forward has accurate accounting.*

*Phase 4 update: Self-evaluation loop, enhanced triple-barrier labeling, adaptive ML
confidence, and performance-based MetaWeigher tuning are now implemented and wired into
the agent pipeline.*

*Phase 2.4–2.5 update (2026-03-16): Short-entry live parity wired in (commit e54fc30).
Signal funnel diagnosis on full 17,587-candle dataset revealed ML filter kills 91.4% of
buys and regime blocks 98% of shorts. Enabled volatile shorts (short_size_scale=0.50),
trade count 22→72 with positive edge. Canonical profile locked in `hogan_bot/profiles.py`.
Short max hold optimized to 12h.*

*Tier 1 update (2026-03-16): GitHub Actions CI pipeline (`.github/workflows/ci.yml`),
walk-forward validation harness (`hogan_bot/walk_forward.py`), and feature importance
audit (`hogan_bot/feature_importance.py`) are complete. Walk-forward provides rolling
N-fold OOS validation with a structured promotion gate (min Sharpe, max drawdown,
min positive windows). Feature importance uses permutation importance to rank the 15
champion features and flag noise/harmful ones for removal.*

*VPS update (2026-04-24): Dockerfile, root Compose stack (bot + TimescaleDB +
Prometheus + Grafana), Docker deployment runbook, and CI Docker build/Compose
validation are in place. CI now publishes immutable GHCR images on pushes to
`main`, `docker-compose.prod.yml` supports VPS image-pull deployments, and
`python -m hogan_bot.healthcheck` provides container/runtime preflight checks.
Timescale candle migration now has dry-run and post-copy verification modes for
safe VPS backfills before switching `HOGAN_STORAGE_BACKEND=timescale`.
The default container skips optional advanced modeling and RL training
dependencies; build with `INSTALL_MODELING=true` for
XGBoost/LightGBM/Optuna/MLflow images and `INSTALL_RL=true` for PPO
training/inference images. Automated VPS deploy orchestration remains planned
follow-up work.*
