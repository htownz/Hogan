# GHCR Release And Rollback

CI publishes Hogan images to GitHub Container Registry:

- `ghcr.io/<owner>/<repo>:sha-<commit>` on pushes to `main`
- `ghcr.io/<owner>/<repo>:latest` on pushes to `main`
- `ghcr.io/<owner>/<repo>:vX.Y.Z` on pushed tags matching `v*`

Prefer immutable `sha-<commit>` tags for VPS deployments.

## Release

```bash
git tag vYYYY.MM.DD
git push origin vYYYY.MM.DD
```

After CI publishes the tag, deploy it:

```bash
export HOGAN_BOT_IMAGE=ghcr.io/<owner>/<repo>:vYYYY.MM.DD
python scripts/deploy_vps.py --image "$HOGAN_BOT_IMAGE"
```

## Rollback

1. Pick the previous known-good image tag from GHCR or the deploy log.
2. Set `HOGAN_BOT_IMAGE` to that immutable tag.
3. Redeploy through the same path:

```bash
export HOGAN_BOT_IMAGE=ghcr.io/<owner>/<repo>:sha-<previous-good-commit>
python scripts/deploy_vps.py --image "$HOGAN_BOT_IMAGE"
```

The deploy script runs a backup before replacing the container unless
`--skip-backup` is set. Use `scripts/runtime_backup.py restore` only when the
state itself must be rolled back, not for normal image rollback.
