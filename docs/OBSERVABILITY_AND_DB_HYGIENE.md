# Observability and DB Hygiene

How to see **why Hogan didn't trade**, detect stale agent modes, and keep
the database healthy.

## Why we didn't trade — decision transparency

Every bar that runs through `policy_core.decide()` records block reasons in
`decision_log.block_reasons_json`. These reasons span **all** gate layers:

| Reason prefix | Gate | What it means |
|---------------|------|---------------|
| `ml_filter` | ML gate | ML probability below threshold |
| `edge_gate` | Edge gate | Expected edge < ATR friction |
| `quality_gate` | Quality gate | Setup quality below minimum |
| `ranging_gate` | Ranging gate | Chop detection triggered |
| `pullback_gate` | Pullback gate | No pullback in trend |
| `sell_pullback_gate` | Sell pullback | No bounce to sell into |
| `macro_sitout` | Macro sitout | FOMC/CPI/NFP event day |
| `macro_filter_block_longs` | Macro filter | SPY/DXY/VIX correlation block |
| `freshness_critical_block` | Freshness | Features too stale to trust |
| `ml_blind_blocks_shorts` | ML blind | Model conviction too low for shorts |
| `swarm_blocked_unsigned_signal` | Swarm merge | Swarm blocked a new signal |
| `swarm_direction_clash` | Swarm merge | Swarm vs pipeline direction conflict |

### Prometheus metrics

- `hogan_block_reasons_total{reason}` — counter per gate hit per bar (ALL reasons)
- `hogan_hold_no_reason_total` — holds with empty block_reasons (logging gap signal)
- `hogan_swarm_merge_blocks_total{reason}` — swarm-specific merge blocks
- `hogan_swarm_final_veto_total{swarm_mode}` — swarm veto events
- `hogan_swarm_dominant_veto_agent_total{agent, swarm_mode}` — which agent vetoed

### CLI: full block-reason report

```bash
python scripts/db_hygiene.py --db data/hogan.db --report --hours 24
```

Shows counts of every block reason, grouped by regime, plus action distribution.
Add `--json` for CI consumption.

### Dashboard

The Swarm tab in `scripts/dashboards/dashboard.py` shows:
- Veto reasons and dominant veto agents
- `swarm_*` policy merge blocks
- Shadow vs baseline comparison

For **non-swarm** block reasons (ML, edge, macro), use the DB hygiene report
or query `decision_log.block_reasons_json` directly.

## Agent mode hygiene

Agent modes (`swarm_agent_modes` table) control whether each swarm agent can
vote, veto, or is quarantined. Stale modes can silently degrade behavior.

### Staleness detection

```bash
python scripts/db_hygiene.py --db data/hogan.db
```

Flags agents stuck in `advisory_only` or `quarantined` for >7 days.

### Manual reset

```bash
python scripts/reset_swarm_agent_modes.py --db data/hogan.db --dry-run
python scripts/reset_swarm_agent_modes.py --db data/hogan.db
```

Resets all non-active agents back to `active`. Use after long shadow runs
when accuracy-based quarantine may be outdated.

### Automated check

`observability_health_report()` includes stale agent alerts. The DB hygiene
CLI reports them automatically.

## DB hygiene

High-volume tables (`decision_log`, `swarm_decisions`, `swarm_agent_votes`,
`swarm_weight_snapshots`, `swarm_stall_alerts`) grow continuously.

### Table stats

```bash
python scripts/db_hygiene.py --db data/hogan.db
```

Shows row counts, oldest/newest timestamps for all key tables.

### Retention pruning

```bash
# Dry run (see what would be deleted)
python scripts/db_hygiene.py --db data/hogan.db --prune --retain-days 90

# Execute pruning + vacuum
python scripts/db_hygiene.py --db data/hogan.db --prune --retain-days 90 --execute --vacuum
```

Default retention: 90 days. Pruned tables:
- `decision_log`
- `swarm_decisions`
- `swarm_agent_votes`
- `swarm_weight_snapshots`
- `swarm_stall_alerts`

**Does NOT prune**: `candles`, `paper_trades`, `equity_snapshots` (needed for
backtesting and performance tracking).

### VACUUM

After pruning, run `--vacuum` to reclaim disk space. This rewrites the entire
database file and may take a few seconds for large DBs.

## Architecture reference

- `hogan_bot/observability.py` — all observability utilities
- `hogan_bot/metrics.py` — `BLOCK_REASONS`, `HOLD_NO_REASON` counters
- `hogan_bot/policy_core.py` — increments block reason counters per bar
- `scripts/db_hygiene.py` — CLI for all operations above
- `scripts/reset_swarm_agent_modes.py` — manual agent mode reset
