"""Legacy polling service — redirects to ``hogan_bot.event_loop``.

This module is a thin alias for ``hogan_bot.event_loop``.  Running it
invokes the event loop with no deprecation warning.

Canonical command::

    python -m hogan_bot.event_loop

This module remains supported for backward compatibility.
"""
from __future__ import annotations

import asyncio
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


def run(max_loops: int | None = None) -> None:
    """Redirect to ``event_loop.run_event_loop()``.

    The ``max_loops`` parameter is accepted for backward compatibility
    but is not forwarded (event_loop runs until interrupted).
    """
    from hogan_bot.event_loop import run_event_loop
    asyncio.run(run_event_loop())


if __name__ == "__main__":
    run()
