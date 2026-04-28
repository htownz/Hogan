"""Backup and restore Hogan runtime state for VPS deployments."""
from __future__ import annotations

import argparse
import os
import shutil
import sqlite3
import subprocess
import tarfile
from datetime import UTC, datetime, timedelta
from pathlib import Path


def timestamp_utc() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")


def run_command(cmd: list[str], *, dry_run: bool = False, stdout_path: Path | None = None) -> None:
    print("+ " + " ".join(cmd))
    if dry_run:
        return
    if stdout_path is not None:
        with stdout_path.open("wb") as fh:
            subprocess.run(cmd, check=True, stdout=fh)
    else:
        subprocess.run(cmd, check=True)


def backup_sqlite(src: Path, dst: Path, *, dry_run: bool = False) -> None:
    print(f"backup sqlite: {src} -> {dst}")
    if dry_run:
        return
    if not src.exists():
        raise FileNotFoundError(f"SQLite DB not found: {src}")
    dst.parent.mkdir(parents=True, exist_ok=True)
    source = sqlite3.connect(src)
    try:
        target = sqlite3.connect(dst)
        try:
            source.backup(target)
        finally:
            target.close()
    finally:
        source.close()


def archive_dirs(root: Path, names: list[str], dst: Path, *, dry_run: bool = False) -> None:
    existing = [name for name in names if (root / name).exists()]
    print(f"archive dirs: {', '.join(existing) if existing else '(none)'} -> {dst}")
    if dry_run:
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(dst, "w:gz") as tar:
        for name in existing:
            tar.add(root / name, arcname=name)


def backup_timescale(args, backup_dir: Path, *, dry_run: bool = False) -> None:
    out = backup_dir / "timescale.sql"
    cmd = [
        "docker",
        "exec",
        args.timescale_container,
        "pg_dump",
        "-U",
        args.postgres_user,
        "-d",
        args.postgres_db,
    ]
    run_command(cmd, dry_run=dry_run, stdout_path=out)


def verify_backup(path: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"backup directory missing: {path}"]
    for name in ("hogan.db", "runtime.tgz"):
        file_path = path / name
        if not file_path.exists() or file_path.stat().st_size == 0:
            errors.append(f"missing or empty backup artifact: {file_path}")
    return errors


def prune_backups(backup_root: Path, retention_days: int, *, dry_run: bool = False) -> list[Path]:
    cutoff = datetime.now(UTC) - timedelta(days=retention_days)
    removed: list[Path] = []
    if retention_days <= 0 or not backup_root.exists():
        return removed
    for child in backup_root.iterdir():
        if not child.is_dir():
            continue
        try:
            stamp = datetime.strptime(child.name, "%Y%m%dT%H%M%SZ").replace(tzinfo=UTC)
        except ValueError:
            continue
        if stamp >= cutoff:
            continue
        print(f"prune backup: {child}")
        removed.append(child)
        if not dry_run:
            shutil.rmtree(child)
    return removed


def backup(args) -> int:
    root = Path(args.root).resolve()
    backup_root = Path(args.backup_dir).resolve()
    backup_dir = backup_root / (args.name or timestamp_utc())
    if not args.dry_run:
        backup_dir.mkdir(parents=True, exist_ok=True)
    backup_sqlite(root / args.sqlite_db, backup_dir / "hogan.db", dry_run=args.dry_run)
    archive_dirs(root, args.archive_dirs.split(","), backup_dir / "runtime.tgz", dry_run=args.dry_run)
    if args.include_timescale:
        backup_timescale(args, backup_dir, dry_run=args.dry_run)
    if not args.dry_run:
        errors = verify_backup(backup_dir)
        if errors:
            raise SystemExit("\n".join(errors))
    prune_backups(backup_root, args.retention_days, dry_run=args.dry_run)
    print(f"backup complete: {backup_dir}")
    return 0


def restore(args) -> int:
    if not args.confirm_restore:
        raise SystemExit("restore requires --confirm-restore")
    root = Path(args.root).resolve()
    backup_dir = Path(args.backup).resolve()
    errors = verify_backup(backup_dir)
    if errors:
        raise SystemExit("\n".join(errors))
    sqlite_dst = root / args.sqlite_db
    print(f"restore sqlite: {backup_dir / 'hogan.db'} -> {sqlite_dst}")
    if not args.dry_run:
        sqlite_dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(backup_dir / "hogan.db", sqlite_dst)
        with tarfile.open(backup_dir / "runtime.tgz", "r:gz") as tar:
            tar.extractall(root)
    if args.restore_timescale:
        sql_path = backup_dir / "timescale.sql"
        if not sql_path.exists():
            raise SystemExit(f"Timescale dump missing: {sql_path}")
        cmd = [
            "docker",
            "exec",
            "-i",
            args.timescale_container,
            "psql",
            "-U",
            args.postgres_user,
            "-d",
            args.postgres_db,
        ]
        print("+ " + " ".join(cmd) + f" < {sql_path}")
        if not args.dry_run:
            with sql_path.open("rb") as fh:
                subprocess.run(cmd, check=True, stdin=fh)
    print("restore complete")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--root", default=".")
    common.add_argument("--sqlite-db", default=os.getenv("HOGAN_DB_PATH", "data/hogan.db"))
    common.add_argument("--backup-dir", default="backups")
    common.add_argument("--timescale-container", default="hogan-timescaledb")
    common.add_argument("--postgres-user", default=os.getenv("POSTGRES_USER", "hogan"))
    common.add_argument("--postgres-db", default=os.getenv("POSTGRES_DB", "hogan"))
    common.add_argument("--dry-run", action="store_true")

    p_backup = sub.add_parser("backup", parents=[common])
    p_backup.add_argument("--name", help="Override timestamped backup directory name")
    p_backup.add_argument("--archive-dirs", default="data,models,reports")
    p_backup.add_argument("--include-timescale", action="store_true")
    p_backup.add_argument("--retention-days", type=int, default=14)
    p_backup.set_defaults(func=backup)

    p_restore = sub.add_parser("restore", parents=[common])
    p_restore.add_argument("--backup", required=True)
    p_restore.add_argument("--confirm-restore", action="store_true")
    p_restore.add_argument("--restore-timescale", action="store_true")
    p_restore.set_defaults(func=restore)

    p_verify = sub.add_parser("verify", parents=[common])
    p_verify.add_argument("--backup", required=True)
    p_verify.set_defaults(func=lambda args: 0 if not verify_backup(Path(args.backup)) else 1)

    p_prune = sub.add_parser("prune", parents=[common])
    p_prune.add_argument("--retention-days", type=int, default=14)
    p_prune.set_defaults(
        func=lambda args: 0 if prune_backups(Path(args.backup_dir), args.retention_days, dry_run=args.dry_run) is not None else 1
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
