# Strategy change gate (walk-forward)

Changes that materially affect **entries, exits, sizing, or ML routing** must pass the **walk-forward harness** before merge to main or before enabling in live-like paper.

## In scope

- `hogan_bot/exit_model.py` — `ExitEvaluator`, trailing / panic / stagnation parameters.
- `hogan_bot/decision.py` — edge / quality / ranging / pullback gates, sizers.
- `hogan_bot/ml.py` — feature sets, regime-routed models (`--regime-models`), thresholds.
- `hogan_bot/policy_core.py` — any change to `decide()` or `merge_swarm_with_gated_action`.
- New swarm authority (modes, thresholds, `HOGAN_SWARM_ACTIVE_ALLOW_NEW_SIGNALS`).

## Required commands

Baseline champion / canonical-style check:

```bash
python -m hogan_bot.walk_forward --db data/hogan.db --n-splits 5
```

Recommended diagnostics (pick what your change touches):

```bash
python -m hogan_bot.walk_forward --db data/hogan.db --n-splits 5 --ml-sizer --macro-sitout
python -m hogan_bot.walk_forward --db data/hogan.db --n-splits 5 --regime-models
python -m hogan_bot.walk_forward --db data/hogan.db --n-splits 5 --no-ml
```

## Exit lifecycle review

Walk-forward JSON output now includes an `exit_lifecycle` section with:
- **Tail losses**: worst single loss, percentile breakpoints, loss-beyond-threshold counts.
- **Time in trade by regime**: avg/median bars held, PnL, and worst loss per regime.
- **Exit quality by regime**: PnL breakdown by (regime, exit_reason) to identify bleeders.
- **Hold duration vs PnL**: trade outcomes bucketed by hold duration.

The CLI also prints a summary table. Review this **before and after** any exit/hold parameter change to verify the change actually helped where it was supposed to.

## Archive

Use the validation battery to store logs + manifest:

```bash
python scripts/run_validation_battery.py --db data/hogan.db --wf-extra
```

Artifacts: `reports/validation/`.

## Swarm-specific

If the change affects swarm behavior, also run:

```bash
python scripts/swarm_certification.py --db data/hogan.db --scratch-db
python scripts/promotion_check.py --db data/hogan.db --symbol BTC/USD --json
```

See [`PROMOTION_CHECKLIST.md`](PROMOTION_CHECKLIST.md).
