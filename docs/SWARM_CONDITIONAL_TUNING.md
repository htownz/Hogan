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

## Safe tuning loop

1. Keep **`HOGAN_SWARM_WEIGHT_UPDATE_MODE=shadow`** (or equivalent) until walk-forward + certification look good.
2. Use **shadow** swarm mode first; compare trade count and block reasons via `python scripts/shadow_report.py` and the dashboard Swarm tab.
3. Tighten **conditional** thresholds (higher agreement/confidence) → swarm intervenes **less**; loosen → intervenes **more** (more alignment with fused direction when gates pass).
4. Enable **`HOGAN_SWARM_USE_REGIME_WEIGHTS=true`** only after per-regime stability shows up in shadow logs and promotion checks; see `hogan_bot/swarm_decision/weight_learner.py`.

## Experimental

- `HOGAN_SWARM_ACTIVE_ALLOW_NEW_SIGNALS=true` — allows promoting gated `hold` → `buy`/`sell`. **Not** a production default; use scratch DB / paper only.
