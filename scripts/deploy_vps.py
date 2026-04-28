"""Deploy a published Hogan GHCR image with backup and health checks."""
from __future__ import annotations

import argparse
import os
import subprocess

COMPOSE_FILES = ["-f", "docker-compose.yml", "-f", "docker-compose.prod.yml"]


def run_command(cmd: list[str], *, dry_run: bool = False, env: dict[str, str] | None = None) -> None:
    print("+ " + " ".join(cmd))
    if dry_run:
        return
    subprocess.run(cmd, check=True, env=env)


def compose_cmd(*args: str) -> list[str]:
    return ["docker", "compose", *COMPOSE_FILES, *args]


def deploy(args) -> int:
    image = args.image or os.getenv("HOGAN_BOT_IMAGE", "")
    if not image:
        raise SystemExit("--image or HOGAN_BOT_IMAGE is required")
    env = os.environ.copy()
    env["HOGAN_BOT_IMAGE"] = image

    run_command(compose_cmd("config", "--quiet"), dry_run=args.dry_run, env=env)
    if not args.skip_backup:
        backup_cmd = [
            "python",
            "scripts/runtime_backup.py",
            "backup",
            "--backup-dir",
            args.backup_dir,
        ]
        if args.include_timescale_backup:
            backup_cmd.append("--include-timescale")
        run_command(backup_cmd, dry_run=args.dry_run, env=env)

    run_command(compose_cmd("pull", "hogan-bot"), dry_run=args.dry_run, env=env)
    run_command(compose_cmd("up", "-d", "--remove-orphans"), dry_run=args.dry_run, env=env)
    run_command(compose_cmd("ps"), dry_run=args.dry_run, env=env)
    if not args.skip_healthcheck:
        run_command(
            compose_cmd(
                "exec",
                "-T",
                "hogan-bot",
                "python",
                "-m",
                "hogan_bot.healthcheck",
                "--timeout",
                str(args.health_timeout),
            ),
            dry_run=args.dry_run,
            env=env,
        )
    print(f"deploy complete: {image}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", help="Published image tag, e.g. ghcr.io/owner/repo:sha-<commit>")
    parser.add_argument("--backup-dir", default="backups")
    parser.add_argument("--include-timescale-backup", action="store_true")
    parser.add_argument("--skip-backup", action="store_true")
    parser.add_argument("--skip-healthcheck", action="store_true")
    parser.add_argument("--health-timeout", type=float, default=5.0)
    parser.add_argument("--dry-run", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return deploy(args)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
