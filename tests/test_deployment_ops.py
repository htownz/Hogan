from __future__ import annotations

import argparse
import sqlite3


def test_runtime_backup_sqlite_round_trip(tmp_path):
    from scripts.runtime_backup import backup_sqlite

    src = tmp_path / "source.db"
    conn = sqlite3.connect(src)
    conn.execute("CREATE TABLE t (id INTEGER PRIMARY KEY, value TEXT)")
    conn.execute("INSERT INTO t (value) VALUES ('ok')")
    conn.commit()
    conn.close()

    dst = tmp_path / "backup" / "hogan.db"
    backup_sqlite(src, dst)

    copied = sqlite3.connect(dst)
    try:
        assert copied.execute("SELECT value FROM t").fetchone()[0] == "ok"
    finally:
        copied.close()


def test_runtime_backup_pg_dump_command(monkeypatch, tmp_path):
    from scripts import runtime_backup

    calls: list[list[str]] = []

    def _fake_run(cmd, *, dry_run=False, stdout_path=None):
        calls.append(cmd)
        assert stdout_path == tmp_path / "timescale.sql"

    monkeypatch.setattr(runtime_backup, "run_command", _fake_run)
    args = argparse.Namespace(
        timescale_container="hogan-timescaledb",
        postgres_user="hogan",
        postgres_db="hogan",
    )

    runtime_backup.backup_timescale(args, tmp_path)

    assert calls == [[
        "docker",
        "exec",
        "hogan-timescaledb",
        "pg_dump",
        "-U",
        "hogan",
        "-d",
        "hogan",
    ]]


def test_deploy_vps_dry_run_command_order(monkeypatch):
    from scripts import deploy_vps

    calls: list[list[str]] = []

    def _fake_run(cmd, *, dry_run=False, env=None):
        calls.append(cmd)
        assert dry_run is True
        assert env["HOGAN_BOT_IMAGE"] == "ghcr.io/example/hogan:sha-test"

    monkeypatch.setattr(deploy_vps, "run_command", _fake_run)
    args = argparse.Namespace(
        image="ghcr.io/example/hogan:sha-test",
        backup_dir="backups",
        include_timescale_backup=False,
        skip_backup=False,
        skip_healthcheck=False,
        health_timeout=5.0,
        dry_run=True,
    )

    deploy_vps.deploy(args)

    assert calls[0] == ["docker", "compose", "-f", "docker-compose.yml", "-f", "docker-compose.prod.yml", "config", "--quiet"]
    assert calls[1][:3] == ["python", "scripts/runtime_backup.py", "backup"]
    assert calls[-1][:7] == [
        "docker",
        "compose",
        "-f",
        "docker-compose.yml",
        "-f",
        "docker-compose.prod.yml",
        "exec",
    ]


def test_timescale_smoke_fixture_creates_candles(tmp_path):
    from scripts.timescale_smoke import create_sqlite_fixture

    db_path = tmp_path / "fixture.db"
    create_sqlite_fixture(db_path)

    conn = sqlite3.connect(db_path)
    try:
        count = conn.execute("SELECT COUNT(*) FROM candles").fetchone()[0]
    finally:
        conn.close()

    assert count == 3
