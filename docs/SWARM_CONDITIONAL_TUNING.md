# Swarm `conditional_active` and regime weights

Default swarm mode is **`conditional_active`** (`HOGAN_SWARM_MODE`). In this mode the swarm only applies the same merge rules as **`active`** when **both** gates pass:

- `agreement >= swarm_conditional_min_agreement`
- `final_confidence >= swarm_conditional_min_confidence`

Otherwise the **gated** policy action and size are left unchanged (vetoes still force hold).

## Environment variables (see `hogan_bot/config.py`)

| Variable | `BotConfig` field | Default | Role |
|----------|-------------------|---------|------|
| `HOGAN_SWARM_MODE` | `swarm_mode` | `conditional_active` | `shadow` / `conditional_active` / `active` |
| `HOGAN_SWARM_PHASE` | `swarm_phase` | `certification` | Roadmap phase label (used by promotion tooling) |
| `HOGAN_SWARM_CONDITIONAL_MIN_AGREEMENT` | `swarm_conditional_min_agreement` | `0.70` | Minimum fused agreement to apply active-style merge |
| `HOGAN_SWARM_CONDITIONAL_MIN_CONFIDENCE` | `swarm_conditional_min_confidence` | `0.60` | Minimum fused confidence to apply active-style merge |
| `HOGAN_SWARM_USE_REGIME_WEIGHTS` | `swarm_use_regime_weights` | `false` | Use DB-backed regime-specific weight snapshots when promoting |
| `HOGAN_SWARM_WEIGHT_UPDATE_MODE` | `swarm_weight_update_mode` | `shadow` | Weight proposal logging mode |
| `HOGAN_SWARM_WEIGHT_LEARNING` | `swarm_weight_learning_enabled` | `true` | Periodic proposals (disabled in backtest) |
| `HOGAN_SWARM_WEIGHT_MIN_TRADES` | `swarm_weight_min_trades` | `50` | Min trades before weight shift is considered |
| `HOGAN_SWARM_WEIGHT_MAX_DAILY_SHIFT` | `swarm_weight_max_daily_shift` | `0.05` | Cap on per-update weight movement |
| `HOGAN_SWARM_WEIGHT_AUTO_PROMOTE` | `swarm_weight_auto_promote` | `true` | Auto-apply proposals when stable criteria fail / pass per learner |

---

## Mode transition rules (safety invariants)

The valid progression is **shadow -> conditional_active -> active**. Each step requires evidence.

### shadow (observe only)

- Swarm votes are logged but **never** influence the gated action.
- **Required before any other mode:** >= 300 shadow decisions, >= 3 regimes, veto accuracy positive.
- Use `python scripts/shadow_report.py` and `python scripts/promotion_check.py` to check readiness.

### conditional_active (default)

- Swarm merge applies **only** when `agreement >= min_agreement` AND `confidence >= min_confidence`.
- Vetoes always force hold regardless of thresholds.
- Cannot promote gated `hold` to `buy`/`sell` (unless `allow_new_signals=true`, which is experimental).
- **Required before advancing:** shadow evidence gates passed, drift within bounds (trade count drift <= 30%, veto rate drift <= 20%), paper PnL non-negative.

### active (full authority)

- Same as conditional but without the agreement/confidence gate -- swarm merge always applies.
- **Required:** all conditional gates passed, >= 50 active veto events, paper PnL positive, drift check.
- **Not recommended** as default; `conditional_active` with tuned thresholds is safer.

### Invariants (always enforced)

1. **Gate chain first**: ML, edge, quality, ranging, pullback run before swarm in all modes.
2. **No new signals by default**: `HOGAN_SWARM_ACTIVE_ALLOW_NEW_SIGNALS=false` means the swarm cannot promote a gated `hold` into `buy`/`sell`.
3. **Backtest isolation**: weight learning and auto-quarantine never write to DB during backtest (`config._backtest=True`).
4. **Startup validation**: `hogan_bot/swarm_authority.py` runs at event loop startup and warns about dangerous configs (allow_new_signals in live, regime weights without evidence, active without shadow history, loose conditional thresholds).

---

## Safe tuning loop

1. Keep **`HOGAN_SWARM_WEIGHT_UPDATE_MODE=shadow`** (or equivalent) until walk-forward + certification look good.
2. Use **shadow** swarm mode first; compare trade count and block reasons via `python scripts/shadow_report.py` and the dashboard Swarm tab.
3. Run `python scripts/promotion_check.py --db data/hogan.db --symbol BTC/USD --json` and verify all gates pass before advancing.
4. Tighten **conditional** thresholds (higher agreement/confidence) -> swarm intervenes **less**; loosen -> intervenes **more** (more alignment with fused direction when gates pass).
5. Monitor **drift** between shadow and active behavior (trade count, veto rate) -- `shadow_report.py` now includes a drift section.
6. Enable **`HOGAN_SWARM_USE_REGIME_WEIGHTS=true`** only after per-regime stability shows up in shadow logs and promotion checks; see `hogan_bot/swarm_decision/weight_learner.py`. Startup will warn if enabled without sufficient regime weight snapshots.

---

## Drift detection

`hogan_bot/swarm_authority.py` provides `compute_shadow_active_drift()` which compares:

- **Trade count drift**: `|active_trades - shadow_trades| / shadow_trades` -- flags when > 30%.
- **Veto rate drift**: `|active_veto_rate - shadow_veto_rate| / shadow_veto_rate` -- flags when > 20%.

Both `shadow_report.py` and `promotion_check.py` now include drift in their output and use it as a promotion gate in Phase 2+.

---

## Experimental

- `HOGAN_SWARM_ACTIVE_ALLOW_NEW_SIGNALS=true` -- allows promoting gated `hold` -> `buy`/`sell`. **Not** a production default; use scratch DB / paper only. Startup validation will log an **error** if this is enabled outside backtest mode.
