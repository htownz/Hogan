# Swarm Promotion Checklist

Operational reference for graduating the Hogan swarm through its authority phases.
Each phase has hard go/no-go gates.  **Never advance without all gates passing.**

## Prerequisites

| Item | Required | Check |
|------|----------|-------|
| `use_policy_core` | `True` (default) | Swarm is inert on the legacy path |
| `HOGAN_SWARM_ENABLED` | `true` | Enables swarm logging |
| `HOGAN_SWARM_MODE` | `shadow` initially | Controls execution authority |
| `HOGAN_SWARM_PHASE` | Set per phase below | Pins the operator-declared phase |
| Certification tests | All 39+ green | `python -m pytest tests/test_swarm_certification.py -v` |

## Phase 0 — Certification

**Goal:** Prove the implementation is real, stable, and not split-brained.

| Gate | Required | CLI check |
|------|----------|-----------|
| Deterministic replay | Same output on repeated runs | `test_swarm_certification.py` |
| Live-sim / backtest parity | Same decisions on same candles | `TestPolicyCoreParity` |
| Full swarm record per bar | Votes, score, plan, blockers logged | `TestLoggingCompleteness` |
| No PIT/leakage failures | `check_lookahead()` passes | `TestLookaheadBias` |
| No exceptions in long sim | 500+ bar backtest clean | `TestLongSimStability` |
| `use_policy_core=True` | Verified in config | `promotion_check.py` |

**Env:**
```
HOGAN_SWARM_PHASE=certification
HOGAN_SWARM_ENABLED=true
HOGAN_SWARM_MODE=shadow
```

**Advance when:** All certification tests pass, `promotion_check.py` returns "advance".

---

## Phase 1 — Shadow Proving

**Goal:** Collect evidence that the swarm adds value.  Zero execution authority.

| Gate | Required | Measured by |
|------|----------|-------------|
| Shadow decisions | >= 300 | `shadow_report.py` |
| Would-trade events | >= 100 | `shadow_report.py` |
| Veto events | >= 50 | `shadow_report.py` |
| Regime coverage | >= 3 distinct | `shadow_report.py` |
| Veto accuracy | Vetoed PnL < allowed PnL | `shadow_report.py` |
| No-trade rate | <= 30% of baseline | `shadow_report.py` |

**Env:**
```
HOGAN_SWARM_PHASE=shadow
HOGAN_SWARM_ENABLED=true
HOGAN_SWARM_MODE=shadow
```

**Run:**
```bash
python scripts/shadow_report.py --db data/hogan.db --symbol BTC/USD
python scripts/promotion_check.py --db data/hogan.db --symbol BTC/USD
```

**Advance when:** All shadow gates pass.  `shadow_report.py` exits 0.

---

## Phase 2 — Veto-Only Paper

**Goal:** Swarm can block weak trades but does not control size or entry style.

| Gate | Required |
|------|----------|
| Active veto events | >= 50 |
| Allowed paper trades | >= 50 |
| Paper PnL | >= 0 (non-inferior) |
| Max drawdown improvement | >= 10% vs baseline |
| Trade count preserved | >= 70% of baseline |

**Env:**
```
HOGAN_SWARM_PHASE=paper_veto
HOGAN_SWARM_MODE=active
```

**Advance when:** `promotion_check.py` returns "advance".

---

## Phase 3 — Size & Entry Authority

**Goal:** Swarm controls posture (full/reduced/probe/skip) and entry routing (market/limit/confirm).

| Gate | Required |
|------|----------|
| Paper trades | >= 75 |
| Probe/reduced examples | >= 20 |
| Size-adjusted expectancy | Improves >= 10% vs veto-only |
| Slippage | Within modeled tolerance |
| Paper PnL | > 0 |

**Env:**
```
HOGAN_SWARM_PHASE=paper_routing
HOGAN_SWARM_MODE=active
```

---

## Phase 4 — Learning (Shadow Weights)

**Goal:** Weight proposals only.  No immediate authority.

| Gate | Required |
|------|----------|
| Weight proposal cycles | >= 2 |
| Paper trades under fixed logic | >= 100 |
| Daily weight shifts | Small and bounded (< 5%) |
| Calibration | No degradation |

**Env:**
```
HOGAN_SWARM_PHASE=learning
HOGAN_SWARM_WEIGHT_UPDATE_MODE=shadow
```

---

## Phase 5 — Adaptive Paper

**Goal:** Promoted weights run in paper.

| Gate | Required |
|------|----------|
| Paper trades | >= 200 |
| Paper PnL | > 0 across 2+ windows |
| Win rate | >= 40% |
| Edge vs realized | No major drift |

**Env:**
```
HOGAN_SWARM_PHASE=adaptive_paper
HOGAN_SWARM_WEIGHT_UPDATE_MODE=active
```

---

## Phase 6 — Micro-Live

**Goal:** Tiny live size, one symbol, hard kill switches.

| Gate | Required |
|------|----------|
| Micro-live trades | >= 30 |
| Paper PnL | > 0 |
| Win rate | >= 40% |
| Slippage | Stable |
| No operational incidents | 0 parity drifts, 0 stale-data executions |

**Kill switches — stop immediately if:**
- One confirmed parity drift
- Two unexplained execution mismatches
- Stale-data or missing-vote execution
- Realized slippage repeatedly exceeds tolerance
- Daily loss cap hit
- Dashboard/replay gap on any live trade

**Env:**
```
HOGAN_SWARM_PHASE=micro_live
HOGAN_SWARM_MODE=active
HOGAN_LIVE_ACK=true
```

---

## Tools

| Tool | Purpose |
|------|---------|
| `python scripts/promotion_check.py --db data/hogan.db` | Phase + gate + recommendation |
| `python scripts/shadow_report.py --db data/hogan.db` | Shadow evaluation report |
| `streamlit run scripts/dashboards/dashboard.py` | Live dashboard with Swarm tab |
| `python -m pytest tests/test_swarm_certification.py -v` | Certification suite |

## Operating Cadence

**Daily:** Review top opportunities, inspect every veto, replay best/worst call, confirm logging health.

**Weekly:** Agent leaderboard, loss clusters, no-trade rate, weight proposals, decide if next step is earned.

## Rules

1. No phase skip.  The sequence is: Certification → Shadow → Veto → Size/Entry → Learning → Adaptive → Micro-Live.
2. No authority increase without explicit operator ack (set `HOGAN_SWARM_PHASE` manually).
3. No trust of paper PnL without replay and attribution review.
4. No adaptive weights until fixed system proves itself.
5. No live authority because the dashboard looks good — require the evidence.
