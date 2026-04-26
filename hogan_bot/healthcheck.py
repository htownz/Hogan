"""Runtime health checks for container and VPS deployments."""
from __future__ import annotations

import argparse
import os
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


def run_healthcheck(
    *,
    check_metrics: bool = True,
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
    parser.add_argument("--timeout", type=float, default=3.0)
    args = parser.parse_args(argv)

    errors = run_healthcheck(
        check_metrics=not args.no_metrics,
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
