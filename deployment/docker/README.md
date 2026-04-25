# Hogan Docker/VPS Deployment

This Compose stack runs:

- `hogan-bot`: `python -m hogan_bot.main`
- `timescaledb`: TimescaleDB/Postgres for candle history experiments
- `prometheus`: scrapes Hogan's Prometheus metrics endpoint
- `grafana`: dashboards provisioned from `monitoring/grafana/provisioning`

## First Boot

```bash
cp .env.example .env  # if you maintain one; otherwise create .env manually
mkdir -p data models reports
docker compose build
docker compose up -d timescaledb prometheus grafana
docker compose up -d hogan-bot
```

Keep the first VPS boot in paper mode:

```env
HOGAN_PAPER_MODE=true
HOGAN_LIVE_MODE=false
HOGAN_LIVE_ACK=false
HOGAN_METRICS_PORT=8000
```

Timescale is available to the bot when you opt in:

```env
HOGAN_STORAGE_BACKEND=timescale
HOGAN_DATABASE_URL=postgresql://hogan:hogan@timescaledb:5432/hogan
```

SQLite remains the default local/runtime backend until the candle migration is
validated.

## Ports

- Hogan metrics: `8000`
- Prometheus: `9090`
- Grafana: `3000`
- Timescale/Postgres: `5432`

Bind these behind a firewall on a VPS. Do not expose exchange credentials,
database credentials, or live trading endpoints to the public internet.

## Volumes and Backups

- `./data`: SQLite DB, runtime lock, logs
- `./models`: model artifacts, mounted read-only by the bot
- `timescale_data`: TimescaleDB persistent data
- `prometheus_data`, `grafana_data`: monitoring state

Backup example:

```bash
docker exec hogan-timescaledb pg_dump -U hogan -d hogan > hogan_timescale.sql
tar czf hogan_runtime.tgz data models reports
```

## Operations

```bash
docker compose ps
docker compose logs -f hogan-bot
docker compose restart hogan-bot
docker compose down
```

Run exactly one `hogan-bot` container per account/strategy unless the runtime
lock and execution ownership model are redesigned.
