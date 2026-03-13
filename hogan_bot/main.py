"""Legacy polling loop — DEPRECATED.

This module is superseded by ``hogan_bot.event_loop``, which is the
canonical runtime path for Hogan.  All logic formerly here has been
removed.  Running this module now redirects to ``event_loop``.

Use instead::

    python -m hogan_bot.event_loop
"""
from __future__ import annotations

import asyncio
import logging
import warnings

warnings.warn(
    "hogan_bot.main is deprecated and now redirects to hogan_bot.event_loop. "
    "Use `python -m hogan_bot.event_loop` directly.",
    DeprecationWarning,
    stacklevel=2,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def run(max_loops: int | None = None) -> None:
    """Thin redirect to ``event_loop.run_event_loop()``.

    The ``max_loops`` parameter is accepted for backward compatibility
    but is not forwarded (event_loop runs until interrupted).
    """
    logger.warning(
        "main.run() is deprecated — redirecting to event_loop.run_event_loop(). "
        "Switch to `python -m hogan_bot.event_loop` to avoid this warning."
    )
    from hogan_bot.event_loop import run_event_loop
    asyncio.run(run_event_loop())


if __name__ == "__main__":
    run()
