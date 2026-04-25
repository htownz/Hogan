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
docker compose up -d
```

Keep the first VPS boot in paper mode:

```env
HOGAN_PAPER_MODE=true
HOGAN_LIVE_MODE=false
HOGAN_LIVE_ACK=false
HOGAN_METRICS_PORT=8000
POSTGRES_PASSWORD=<strong-unique-password>
GRAFANA_ADMIN_PASSWORD=<strong-unique-password>
HOGAN_TIMESCALE_IMAGE=timescale/timescaledb:2.16.1-pg16
HOGAN_PROMETHEUS_IMAGE=prom/prometheus:v3.4.0
HOGAN_GRAFANA_IMAGE=grafana/grafana:11.5.2
```

Timescale is available to the bot when you opt in:

```env
HOGAN_STORAGE_BACKEND=timescale
HOGAN_DATABASE_URL=postgresql://hogan:<strong-unique-password>@timescaledb:5432/hogan
```

SQLite remains the default local/runtime backend until the candle migration is
validated.

## Image Versions

The Compose files use pinned defaults for TimescaleDB, Prometheus, and Grafana
so a routine restart does not silently pull a new major/minor release. Override
these only during an intentional upgrade:

```env
HOGAN_TIMESCALE_IMAGE=timescale/timescaledb:2.16.1-pg16
HOGAN_PROMETHEUS_IMAGE=prom/prometheus:v3.4.0
HOGAN_GRAFANA_IMAGE=grafana/grafana:11.5.2
```

## Ports

- Hogan metrics: `8000`
- Prometheus: `9090`
- Grafana: `3000`
- Timescale/Postgres: `5432` on `127.0.0.1`

The Compose file binds service ports to `127.0.0.1` by default. Keep them
behind SSH tunnels, a private VPN, or a reverse proxy with authentication. Do
not expose exchange credentials, database credentials, metrics, or live trading
endpoints to the public internet.

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

## Monitoring-Only Stack

Use the root `docker-compose.yml` for VPS deployment: it runs the bot,
TimescaleDB, Prometheus, and Grafana on one Docker network, and Prometheus
scrapes `hogan-bot:8000`.

Use `monitoring/docker-compose.monitoring.yml` only when the bot is running on
the host instead of in Docker. That stack uses `monitoring/prometheus.yml`,
which scrapes `host.docker.internal:8000`, and binds Prometheus/Grafana to
`127.0.0.1` for local SSH-tunnel or VPN access.

## Models

The image intentionally does not bake in `models/`; the bot mounts `./models`
read-only. Copy champion/challenger artifacts to the VPS before starting any
ML-enabled profile, or keep `HOGAN_USE_ML_FILTER=false` / `HOGAN_ML_AS_SIZER=false`
until models are trained in-place.
