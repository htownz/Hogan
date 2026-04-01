"""Single-instance runtime lock for Hogan processes."""
from __future__ import annotations

import json
import logging
import os
import uuid
from ctypes import wintypes
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


class InstanceLockError(RuntimeError):
    """Raised when runtime lock acquisition fails."""


def _pid_exists(pid: int) -> bool:
    """Best-effort process existence check."""
    if pid <= 0:
        return False
    if os.name == "nt":
        try:
            import ctypes

            process_query_limited_information = 0x1000
            kernel32 = ctypes.windll.kernel32
            kernel32.OpenProcess.argtypes = (
                wintypes.DWORD,
                wintypes.BOOL,
                wintypes.DWORD,
            )
            kernel32.OpenProcess.restype = wintypes.HANDLE
            handle = kernel32.OpenProcess(process_query_limited_information, False, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        except Exception:
            return False
    try:
        os.kill(pid, 0)
    except PermissionError:
        return True
    except SystemError:
        return False
    except OSError:
        return False
    return True


def _read_lock(lock_path: Path) -> dict:
    """Read lock metadata, returning {} if unreadable."""
    try:
        return json.loads(lock_path.read_text(encoding="utf-8"))
    except Exception:
        return {}


@dataclass
class RuntimeInstanceLock:
    """Owns a lock file so only one runtime can execute."""

    lock_path: Path
    _token: str = field(default_factory=lambda: uuid.uuid4().hex)
    _held: bool = False

    def __post_init__(self) -> None:
        self.lock_path = Path(self.lock_path)

    def acquire(self) -> None:
        """Acquire lock or fail fast if another process owns it."""
        self.lock_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "pid": os.getpid(),
            "token": self._token,
            "argv": os.sys.argv,
        }
        for _ in range(2):
            try:
                fd = os.open(str(self.lock_path), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                with os.fdopen(fd, "w", encoding="utf-8") as handle:
                    json.dump(payload, handle)
                self._held = True
                return
            except FileExistsError:
                existing = _read_lock(self.lock_path)
                owner_pid = int(existing.get("pid", 0) or 0)
                if owner_pid and _pid_exists(owner_pid):
                    raise InstanceLockError(
                        f"another Hogan runtime is active (pid={owner_pid}, lock='{self.lock_path}')"
                    )
                # Stale lock (dead pid or unreadable metadata): remove then retry.
                self.lock_path.unlink(missing_ok=True)
        raise InstanceLockError(f"unable to acquire runtime lock at '{self.lock_path}'")

    def release(self) -> None:
        """Release lock only when still owned by this process/token."""
        if not self._held:
            return
        existing = _read_lock(self.lock_path)
        if existing.get("token") != self._token:
            logger.warning("Lock token mismatch for %s; skipping release", self.lock_path)
            self._held = False
            return
        self.lock_path.unlink(missing_ok=True)
        self._held = False

    def __enter__(self) -> "RuntimeInstanceLock":
        self.acquire()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.release()
