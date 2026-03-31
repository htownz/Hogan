from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any


@dataclass
class TaskOutcome:
    name: str
    ok: bool
    result: Any = None
    error: str | None = None


class RuntimeTaskSupervisor:
    """Lightweight async task supervisor for event-loop side jobs."""

    def __init__(self, logger, *, namespace: str = "runtime") -> None:
        self._logger = logger
        self._namespace = namespace
        self._tasks: dict[str, asyncio.Task] = {}

    def running(self, name: str) -> bool:
        t = self._tasks.get(name)
        return t is not None and not t.done()

    def start(self, name: str, coro) -> bool:
        """Start *coro* under *name* if not already running."""
        if self.running(name):
            return False
        task_name = f"{self._namespace}:{name}"
        self._tasks[name] = asyncio.create_task(coro, name=task_name)
        return True

    def poll(self) -> list[TaskOutcome]:
        """Collect completed task outcomes and clear them from supervision."""
        done_names = [n for n, t in self._tasks.items() if t.done()]
        out: list[TaskOutcome] = []
        for n in done_names:
            t = self._tasks.pop(n)
            if t.cancelled():
                out.append(TaskOutcome(name=n, ok=False, error="cancelled"))
                continue
            exc = t.exception()
            if exc is None:
                out.append(TaskOutcome(name=n, ok=True, result=t.result()))
            else:
                out.append(TaskOutcome(name=n, ok=False, error=str(exc)))
        return out

    async def shutdown(self, timeout_s: float = 2.0) -> None:
        """Cancel any running tasks and wait briefly for teardown."""
        pending = [t for t in self._tasks.values() if not t.done()]
        for t in pending:
            t.cancel()
        if pending:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*pending, return_exceptions=True),
                    timeout=timeout_s,
                )
            except Exception as exc:
                self._logger.debug("RuntimeTaskSupervisor shutdown timeout/error: %s", exc)
        self._tasks.clear()
