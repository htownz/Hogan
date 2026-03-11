# Hogan 2.0 Architecture

## Vision

A BTC-first, 1h-led market intelligence and trading system with:

- **1h regime model** — classifies market state
- **15m execution model** — times entries/exits
- **Global market-state features** — four layers, time-aligned
- **Local multi-agent research swarm** — LLMs as research staff, never direct traders
- **Champion/challenger continuous learning** — OOS promotion only
- **No ICT dependency in the core path**

---

## Core Principles

1. **LLMs are research agents**, not direct trade executors.
2. **Numeric models make every trade decision.**
3. **Every feature must be time-aligned** with as-of semantics and freshness tracking.
4. **All promotion must be out-of-sample** and shadow-validated.
5. **BTC only** until the system proves itself.

---

## Architecture Overview

```
┌─────────────────────────────────────────────────────────────────────┐
│                        RESEARCH SWARM (async)                       │
│  DataCustodian · MarketCartographer · FeatureScientist              │
│  ForecastLab · ValidatorJudge · RiskSteward · MemoryAgent           │
│                                                                     │
│  Outputs: structured JSON hypotheses, diagnostics, feature          │
│  proposals, validation reports. Never submits orders.               │
└─────────────────────────────────────────────────────────────────────┘
        │  feature proposals, regime labels, diagnostics
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    FEATURE SPINE (event-time index)                  │
│                                                                     │
│  Layer 1: Market Microstructure                                     │
│    OHLCV 1m/5m/15m/1h, realized vol, range expansion, trend        │
│    persistence, volume profile, session effects                     │
│                                                                     │
│  Layer 2: Derivatives State                                         │
│    Funding, OI, basis, perp/spot premium, liquidation intensity,    │
│    long/short crowding                                              │
│                                                                     │
│  Layer 3: Cross-Asset / Macro                                       │
│    DXY, VIX, rates (10Y, 2Y, FFR), SPY/NQ, gold, oil,            │
│    Fed-event proximity, risk-on/risk-off composite                  │
│                                                                     │
│  Layer 4: On-Chain / Sentiment                                      │
│    Exchange flows, realized-price metrics, stablecoin supply,       │
│    fear/greed, news sentiment, social bursts                        │
│                                                                     │
│  Requirements per feature:                                          │
│    ✓ as-of timestamp       ✓ freshness flag                        │
│    ✓ latency class         ✓ missingness policy                    │
└─────────────────────────────────────────────────────────────────────┘
        │  time-aligned feature matrix
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                        MODEL STACK                                  │
│                                                                     │
│  ┌────────────────┐  ┌─────────────────┐  ┌──────────────────┐    │
│  │ Regime Model   │  │ Forecast Model  │  │ Risk Model       │    │
│  │ (1h)           │  │ (1h)            │  │ (1h)             │    │
│  │                │  │                 │  │                  │    │
│  │ Classifies:    │  │ Outputs:        │  │ Predicts:        │    │
│  │ trend-up       │  │ P(return > x)   │  │ realized vol     │    │
│  │ trend-down     │  │ for 4h/12h/24h  │  │ max adverse exc  │    │
│  │ mean-revert    │  │ horizons        │  │ stop-hit prob    │    │
│  │ breakout       │  │                 │  │ expected hold    │    │
│  │ panic          │  │                 │  │                  │    │
│  │ grind/carry    │  │                 │  │                  │    │
│  │ risk-on/off    │  │                 │  │                  │    │
│  └────────┬───────┘  └────────┬────────┘  └────────┬─────────┘    │
│           │                   │                     │              │
│           ▼                   ▼                     ▼              │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │              REGIME ROUTER / POLICY LAYER                  │    │
│  │                                                            │    │
│  │  Trade only when:                                          │    │
│  │    EV after fees/slippage > threshold                      │    │
│  │    AND risk model says size is acceptable                  │    │
│  │    AND regime router selects a strategy family             │    │
│  │                                                            │    │
│  │  Strategy families:                                        │    │
│  │    • Trend follow                                          │    │
│  │    • Breakout / volatility expansion                       │    │
│  │    • Carry / funding dislocation                           │    │
│  │    • Mean reversion in compressed ranges                   │    │
│  └────────────────────────────────────────────────────────────┘    │
│           │                                                        │
│           ▼                                                        │
│  ┌────────────────────────────────────────────────────────────┐    │
│  │            15m EXECUTION MODEL                             │    │
│  │  Times entries and exits within the 1h decision window     │    │
│  │  Inputs: 15m microstructure, order book, regime context    │    │
│  └────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
        │  sized order with stop/target
        ▼
┌─────────────────────────────────────────────────────────────────────┐
│                    EXECUTION + RISK MANAGEMENT                      │
│  Position sizing · Drawdown guard · Kill switch · Paper/Live        │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Feature Spine Design

### Event-Time Index

Every feature row is keyed by a canonical event timestamp (bar close time).
All joins are **as-of / backward-looking only**.

### Feature Metadata Registry

Each registered feature carries:

| Field           | Description                                           |
|-----------------|-------------------------------------------------------|
| `name`          | Unique feature name                                   |
| `layer`         | microstructure / derivatives / macro / onchain_sent    |
| `source`        | fetch module that produces it                         |
| `latency_class` | `realtime` / `hourly` / `daily` / `weekly`            |
| `staleness_max` | Maximum acceptable age before marking stale           |
| `fill_policy`   | `ffill` / `zero` / `drop_row` / `median`             |
| `as_of_column`  | Column used for point-in-time join                    |

### Freshness Tracking

```python
class FeatureSlot:
    name: str
    value: float
    as_of_ts: int          # when this value was produced
    fetched_ts: int        # when we last tried to update
    is_stale: bool         # fetched_ts - as_of_ts > staleness_max
    is_missing: bool       # no value available
```

---

## Model Stack Details

### Regime Model (1h)

- **Input:** 1h feature spine (microstructure + derivatives + macro + sentiment)
- **Output:** probability distribution over regime labels
- **Labels:** derived from rolling realized vol, trend persistence, drawdown depth
- **Model:** HGB or LightGBM classifier, trained on 8000+ 1h bars
- **Retrain:** weekly on rolling window

### Forecast Model (1h)

- **Input:** same feature spine + regime embedding
- **Output:** calibrated P(return > fee_threshold) at 4h, 12h, 24h horizons
- **Labels:** triple-barrier (profit / stop / timeout)
- **Model:** ensemble of HGB + LightGBM + optional PPO value head
- **Retrain:** weekly, challenger must beat champion on OOS Brier score

### Risk Model (1h)

- **Input:** feature spine + regime + recent trade history
- **Output:** realized vol forecast, max adverse excursion, stop-hit probability
- **Used by:** position sizer, drawdown guard
- **Model:** quantile regression (HGB) for vol, classifier for stop-hit

### Execution Model (15m)

- **Input:** 15m microstructure, 1h regime context, order-book features
- **Output:** entry/exit timing within the 1h decision window
- **Model:** PPO or rule-based initially; upgradeable
- **Purpose:** reduce slippage, improve fill quality

---

## Research Swarm Agents

| Agent               | Role                                                    | Output Format     |
|---------------------|---------------------------------------------------------|-------------------|
| DataCustodianAgent  | Gaps, stale features, outliers, freshness audit         | `DiagnosticReport`|
| MarketCartographer  | Regime labels, structural breaks, analog identification | `RegimeLabel`     |
| FeatureScientist    | New candidate features, interaction terms               | `FeatureProposal` |
| ForecastLabAgent    | Train/evaluate candidate models                         | `ModelCard`       |
| ValidatorJudgeAgent | Walk-forward, leakage tests, fee/slippage realism       | `ValidationReport`|
| RiskStewardAgent    | Size, stop width, kill-switch decisions                 | `RiskDirective`   |
| MemoryAgent         | Similar historical states, strategy behavior there      | `AnalogReport`    |

All agents output structured JSON. No agent submits orders.

### Swarm Stack

- **Orchestration:** Ray actors (stateful, concurrent)
- **LLM serving:** Ollama (local) or vLLM (OpenAI-compatible server)
- **Structured output:** JSON schema enforcement via Ollama/vLLM
- **Feature store:** Feast (historical + online serving)
- **Model registry:** MLflow (lineage, versioning, aliases, champion/challenger)

---

## Continuous Learning Pipeline

### Stage 1 — Calibration (current target)

- Retrain weekly on rolling windows
- Online updates only recalibrate probabilities and ensemble weights
- No live self-rewriting of core alpha

### Stage 2 — Shadow Evaluation

- Challengers train in shadow mode
- Paper-trade beside the champion
- Promote only on OOS + rolling-window superiority

### Stage 3 — Automated Feature Discovery

- Swarm proposes new features / model families
- Every proposal goes through ValidatorJudge first
- Promotion requires: OOS Sharpe > champion, max_dd < threshold, min trades

### Promotion Gate

```
promote_challenger() requires ALL of:
  1. OOS Sharpe ratio > champion_sharpe + min_improvement
  2. OOS max_drawdown < max_allowed_drawdown
  3. OOS trade_count >= min_trades
  4. Walk-forward consistency across >= 3 folds
  5. Shadow paper PnL positive over >= 7 days
```

---

## Phase Plan

### Phase A — Simplify Current Live Path

- Quarantine ICT from champion path
- Refactor strategy flow: remove MA gatekeeper
- Create clean signal provider interface (regime, forecast, risk, execution)
- Default to 1h BTC

### Phase B — Build Market-State Feature Spine

- Canonical event-time index
- As-of joins for all external data
- Freshness columns and missingness flags
- Feature metadata registry

### Phase C — Build Model Stack

- Regime classifier
- Forward return forecaster (4h / 12h / 24h)
- Risk model (vol, MAE, stop-hit)
- 15m execution timing

### Phase D — Build Local AI Swarm

- Seven research agents
- Structured JSON output
- Ray orchestration
- Local LLM integration

### Phase E — Continuous Learning

- Nightly challenger retrains
- Rolling walk-forward validation
- Shadow paper evaluation
- Automated promotion gates

---

## Migration Strategy

The system transitions incrementally:

1. **Phase A** makes the current bot tradeable without ICT and without MA gating
2. **Phase B** replaces the ad-hoc feature assembly with a proper spine
3. **Phase C** replaces `generate_signal()` with model outputs
4. **Phase D** adds research automation
5. **Phase E** closes the learning loop

Each phase is independently deployable. The bot remains functional throughout.
