# Hogan Swarm Command Center

Operational guide for the swarm observability, replay, and promotion tooling.

## What It Answers

| # | Question | Where |
|---|----------|-------|
| 1 | What is the swarm seeing right now? | Dashboard → Swarm → Live Snapshot |
| 2 | Why did it decide that? | Dashboard → Swarm → Replay / Decision Story |
| 3 | What happened after? | Dashboard → Swarm → Replay (outcomes) |
| 4 | Is the swarm improving or drifting? | Dashboard → Swarm → Learning & Drift |
| 5 | Is the swarm ready to graduate? | Dashboard → Swarm → Promotion Readiness |

## Architecture

```
┌─ Event Loop (event_loop.py) ──────────────┐
│  LiveDataEngine → SignalEvaluator          │
│       ↓                                    │
│  policy_core.decide() → SwarmController    │
│       ↓                                    │
│  SQLite (storage.py)                       │
│    swarm_decisions                          │
│    swarm_agent_votes                        │
│    swarm_outcomes                           │
│    swarm_weight_snapshots                   │
│    swarm_promotion_reports                  │
└────────────────────────────────────────────┘
        ↓ (read-only via WAL)
┌─ Command Center (dashboard.py) ───────────┐
│  swarm_observability.py  ← query helpers   │
│  swarm_metrics.py        ← computed stats  │
│  swarm_replay.py         ← story renderer  │
└────────────────────────────────────────────┘
```

## SQLite Tables

### `swarm_decisions`
One row per evaluated bar. Contains: `final_action`, `final_conf`, `agreement`, `entropy`, `vetoed`, `block_reasons_json`, `decision_json`, `mode` (shadow/active).

### `swarm_agent_votes`
One row per agent per decision. Contains: `agent_id`, `action`, `confidence`, `size_scale`, `veto`, `block_reasons_json`, linked by `decision_id`.

### `swarm_outcomes` (new)
Forward markouts and attribution per decision:
- `forward_5m_bps`, `forward_15m_bps`, `forward_30m_bps`, `forward_60m_bps`
- `mae_bps` / `mfe_bps` — maximum adverse/favorable excursion
- `was_trade_taken`, `was_veto_correct`, `was_skip_correct`
- `outcome_label` (win/loss/scratch)

### `swarm_weight_snapshots`
Weight and calibration history per regime.

### `swarm_promotion_reports` (new)
Persisted promotion check results:
- `phase`, `recommendation`, `blockers_json`, `warnings_json`, `gates_json`
- `metrics_json`, `summary`

## Modules

### `hogan_bot/swarm_observability.py`
Reusable query helpers — no Streamlit dependency.

| Function | Returns |
|----------|---------|
| `load_latest_swarm_decision(conn, symbol, tf)` | 1-row DataFrame |
| `load_swarm_votes(conn, decision_id?, symbol?, tf?)` | votes DataFrame |
| `load_swarm_outcomes(conn, decision_id?, symbol?)` | outcomes DataFrame |
| `load_swarm_weight_history(conn, symbol?, tf?, days)` | weights DataFrame |
| `load_swarm_promotion_status(conn, symbol?, tf?)` | latest report row |
| `load_swarm_decisions(conn, symbol?, tf?, mode?, limit)` | bulk decisions |
| `load_veto_ledger(conn, symbol?, limit)` | veto events with reasons |
| `load_decision_detail(conn, decision_id)` | decision + votes + baseline + outcome |
| `load_swarm_loss_clusters(conn, symbol?, days)` | loss cluster data |
| `load_swarm_score_calibration(conn, symbol?, days)` | score vs return data |

### `hogan_bot/swarm_metrics.py`
Pure functions — take DataFrames, return dicts or DataFrames.

| Function | Returns |
|----------|---------|
| `compute_veto_precision(decisions, outcomes)` | precision, avg PnL vetoed vs allowed |
| `compute_no_trade_rate(decisions, baseline?)` | no-trade rate, skip rate vs baseline |
| `compute_trade_density(decisions, bucket_hours)` | trades/holds per time bucket |
| `compute_agent_leaderboard(votes, outcomes?)` | per-agent vote/veto/confidence stats |
| `compute_opportunity_monotonicity(cal_df)` | bin analysis, correlation, monotonic bool |
| `compute_disagreement_stats(decisions)` | mean agreement/entropy, high-disagreement % |

### `hogan_bot/swarm_replay.py`
Deterministic decision explainer — no LLM calls.

| Function | Returns |
|----------|---------|
| `render_decision_story(decision, votes?, baseline?)` | Markdown string |
| `build_replay_frame(decision, votes?, baseline?, outcome?, candles?)` | structured dict |
| `compute_baseline_vs_swarm_delta(decisions, baseline)` | match/mismatch stats |

## Dashboard Sections (Swarm Tab)

### 1. Live Swarm Snapshot
Top-level metrics: decision count, mode, latest action/confidence/agreement, veto state, last update age.

### 2. Consensus Over Time + Weight History
Existing panels: agreement/entropy line chart, stacked weight area chart.

### 3. Veto Analysis
Existing panels: top veto reasons bar chart, Agent Voting Board heatmap, Veto Ledger dataframe.

### 4. Replay by Decision
Select any decision → metrics, per-agent votes, baseline comparison, Decision Story (plain English), raw JSON.

### 5. Promotion Readiness
Shadow sample progress bars, READY/COLLECTING badge, mean agreement metric.

### 6. Learning & Drift
New section with:
- Disagreement statistics (mean agreement, entropy, high-disagreement %)
- Trade density chart (24h buckets, stacked bars)
- Agent leaderboard table (votes, vetoes, confidence)
- Score calibration chart (mean return by confidence bin, correlation, monotonicity)

### 7. Persisted Promotion Reports
Latest report from `swarm_promotion_reports` table: phase, recommendation, blocker count, full gates JSON.

## CLI Tools

### Shadow Report
```bash
python scripts/shadow_report.py --db data/hogan.db --json
```

### Promotion Check
```bash
python scripts/promotion_check.py --db data/hogan.db --json
```

## Operator Workflow

### Daily
1. Open `streamlit run scripts/dashboards/dashboard.py`
2. Check Live Snapshot — is the swarm processing?
3. Review vetoes — are they blocking losers or winners?
4. Replay best and worst decision of the day
5. Check Learning & Drift — any disagreement spikes?
6. Confirm logging health (decision count increasing)

### Weekly
1. Review agent leaderboard — any agent consistently wrong?
2. Review score calibration — still monotonic?
3. Run `promotion_check.py` — check gates
4. Review no-trade rate — paralysis creeping in?
5. Decide: hold phase or advance?

## Testing

```bash
# Observability, metrics, replay (26 tests)
python -m pytest tests/test_swarm_observability.py -v

# Shadow report + promotion check (25 tests)
python -m pytest tests/test_observability_scripts.py -v

# Full certification suite (39 tests)
python -m pytest tests/test_champion.py -v
```
