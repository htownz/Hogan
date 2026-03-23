# Validation run archive

This directory stores outputs from [`scripts/run_validation_battery.py`](../../scripts/run_validation_battery.py).

- `wf_*.json` — Walk-forward report JSON (same schema as `hogan_bot.walk_forward` `--output`).
- `wf_*.log` — Captured stdout/stderr from the walk-forward CLI.
- `cert_*.log` — Captured stdout/stderr from swarm certification (`--scratch-db`).
- `manifest_*.json` — Run metadata: timestamps, exit codes, command lines, paths.

**Do not commit large logs** if your repo excludes `reports/`; add `reports/validation/*.log` to `.gitignore` if needed, or commit only `manifest_*.json` and `wf_*.json` summaries.

See [`docs/PROMOTION_CHECKLIST.md`](../../docs/PROMOTION_CHECKLIST.md) for when to run the battery.
