"""Runtime health checks for container and VPS deployments."""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import urllib.error
import urllib.request
from pathlib import Path

from hogan_bot.config import load_config

LIVE_ACK = "I_UNDERSTAND_LIVE_TRADING"


def _check_metrics(port: int, timeout: float) -> list[str]:
    url = f"http://127.0.0.1:{port}"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            response.read(256)
            if response.status >= 400:
                return [f"metrics endpoint returned HTTP {response.status}"]
    except (OSError, urllib.error.URLError) as exc:
        return [f"metrics endpoint unavailable at {url}: {exc}"]
    return []


def _check_paths(db_path: str, model_paths: list[str]) -> list[str]:
    errors: list[str] = []
    db_parent = Path(db_path).expanduser().parent
    if db_parent and not db_parent.exists():
        errors.append(f"database directory does not exist: {db_parent}")

    missing_models = [path for path in model_paths if path and not Path(path).exists()]
    if missing_models:
        errors.append("required model file(s) missing: " + ", ".join(missing_models))
    return errors


def _check_sqlite(db_path: str) -> list[str]:
    path = Path(db_path).expanduser()
    if not path.exists():
        return []
    try:
        conn = sqlite3.connect(path)
        try:
            conn.execute("PRAGMA quick_check").fetchone()
        finally:
            conn.close()
    except sqlite3.Error as exc:
        return [f"SQLite health check failed for {path}: {exc}"]
    return []


def _check_timescale(database_url: str) -> list[str]:
    try:
        import psycopg
    except ImportError as exc:
        return [f"Timescale health check requires psycopg: {exc}"]
    try:
        with psycopg.connect(database_url, connect_timeout=3) as conn:
            with conn.cursor() as cur:
                cur.execute("SELECT 1")
                cur.fetchone()
    except Exception as exc:
        return [f"Timescale health check failed: {exc}"]
    return []


def _check_model_artifact(path: str) -> list[str]:
    if not path:
        return []
    model_path = Path(path)
    if not model_path.exists():
        return [f"required model file missing: {path}"]
    try:
        from hogan_bot.feature_registry import (
            CHAMPION_FEATURE_COLUMNS,
            EXPERIMENTAL_FEATURE_COLUMNS,
            get_feature_columns,
        )
        from hogan_bot.ml import load_model
        artifact = load_model(path)
    except Exception as exc:
        return [f"model artifact load failed for {path}: {exc}"]

    artifact_features = list(getattr(artifact, "feature_columns", []) or [])
    allowed = set(CHAMPION_FEATURE_COLUMNS)
    allowed.update(get_feature_columns(False))
    allowed.update(EXPERIMENTAL_FEATURE_COLUMNS)
    unknown = sorted(set(artifact_features) - allowed)
    if unknown:
        return [f"model artifact has unknown feature columns: {', '.join(unknown)}"]
    return []


def run_healthcheck(
    *,
    check_metrics: bool = True,
    check_db: bool = False,
    strict_models: bool = False,
    timeout: float = 3.0,
) -> list[str]:
    """Return a list of healthcheck failures; an empty list means healthy."""
    config = load_config()
    errors = list(config.validate())

    if config.live_mode and os.getenv("HOGAN_LIVE_ACK", "") != LIVE_ACK:
        errors.append(f"live_mode requires HOGAN_LIVE_ACK={LIVE_ACK}")

    model_paths: list[str] = []
    if strict_models and (config.use_ml_filter or config.use_ml_as_sizer):
        model_paths.append(config.ml_model_path)
    if strict_models and config.use_trade_quality:
        model_paths.append(config.trade_quality_model_path)
    if strict_models and config.use_rl_agent:
        model_paths.append(config.rl_model_path)
    errors.extend(_check_paths(config.db_path, model_paths))

    if check_db:
        errors.extend(_check_sqlite(config.db_path))
        backend = str(getattr(config, "storage_backend", "sqlite")).lower()
        if backend in {"timescale", "postgres", "postgresql"}:
            errors.extend(_check_timescale(config.database_url))

    if strict_models and (config.use_ml_filter or config.use_ml_as_sizer):
        errors.extend(_check_model_artifact(config.ml_model_path))

    if check_metrics:
        errors.extend(_check_metrics(config.metrics_port, timeout))

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Check Hogan runtime health")
    parser.add_argument(
        "--no-metrics",
        action="store_true",
        help="Skip metrics endpoint check; useful before the event loop is running.",
    )
    parser.add_argument(
        "--strict-models",
        action="store_true",
        help="Fail if enabled ML/trade-quality/RL model files are missing.",
    )
    parser.add_argument(
        "--check-db",
        action="store_true",
        help="Run SQLite quick_check and Timescale connectivity checks when configured.",
    )
    parser.add_argument("--timeout", type=float, default=3.0)
    args = parser.parse_args(argv)

    errors = run_healthcheck(
        check_metrics=not args.no_metrics,
        check_db=args.check_db,
        strict_models=args.strict_models,
        timeout=args.timeout,
    )
    if errors:
        for error in errors:
            print(f"unhealthy: {error}", file=sys.stderr)
        return 1
    print("healthy")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
