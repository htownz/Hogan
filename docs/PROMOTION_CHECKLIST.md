# Operator checklist — swarm mode / phase promotion

Run this **before** changing `HOGAN_SWARM_MODE`, `HOGAN_SWARM_PHASE`, or enabling `HOGAN_SWARM_ACTIVE_ALLOW_NEW_SIGNALS` in any environment that matters.

## 1. Validation battery (archives artifacts)

From the repo root:

```bash
python scripts/run_validation_battery.py --db data/hogan.db
```

Optional stress profile (ML sizer + macro sitout on walk-forward):

```bash
python scripts/run_validation_battery.py --db data/hogan.db --wf-extra
```

Outputs under `reports/validation/`:

- `manifest_<UTC>.json` — exit codes and command lines
- `wf_<UTC>.json` / `wf_<UTC>.log` — walk-forward report + log
- `cert_<UTC>.log` — swarm certification log (`--scratch-db` so production SQLite is not polluted)

## 2. Walk-forward only (manual)

```bash
python -m hogan_bot.walk_forward --db data/hogan.db --n-splits 5
python -m hogan_bot.walk_forward --db data/hogan.db --n-splits 5 --ml-sizer --macro-sitout
```

## 3. Swarm certification (scratch DB)

```bash
python scripts/swarm_certification.py --db data/hogan.db --scratch-db
```

Add `--keep-scratch-db` if you need to inspect the temp copy.

## 4. Promotion gate (JSON)

```bash
python scripts/promotion_check.py --db data/hogan.db --symbol BTC/USD --json
```

Review `recommendation`, `blockers`, and `gates` before advancing phase.

## 5. Shadow report

```bash
python scripts/shadow_report.py --db data/hogan.db --symbol BTC/USD --json
```

Use for veto / no-trade behavior vs baseline before moving from shadow-heavy modes.

## 6. Rollback

- Revert `.env` (`HOGAN_SWARM_MODE`, `HOGAN_SWARM_PHASE`, experimental flags).
- If agent modes are stuck: `python scripts/reset_swarm_agent_modes.py --dry-run` then without `--dry-run`.

## 7. Observability

- **Dashboard** (Streamlit): Swarm tab → “Policy / gated merge blocks”, “Dominant veto agent”.
- **Metrics** (if `HOGAN_METRICS_PORT` is set): `hogan_swarm_merge_blocks_total`, `hogan_swarm_final_veto_total`, `hogan_swarm_dominant_veto_agent_total`.

## Related docs

- [`SWARM_CONDITIONAL_TUNING.md`](SWARM_CONDITIONAL_TUNING.md) — `conditional_active` thresholds and regime weights.
- [`STRATEGY_CHANGE_GATE.md`](STRATEGY_CHANGE_GATE.md) — strategy / exit / ML changes.
