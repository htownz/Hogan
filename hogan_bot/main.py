"""Champion runtime — the single canonical entry point for Hogan.

Run::

    python -m hogan_bot.main

Or with a finite event count::

    python -m hogan_bot.main --max-events 100

This module delegates to the async event loop in ``event_loop``.
All trading logic, agent pipeline, ML filter, and execution live there.
main is the champion runtime; event_loop is the implementation.
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys

logger = logging.getLogger(__name__)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler("data/hogan_bot.log", encoding="utf-8"),
    ],
)


def run(max_loops: int | None = None) -> None:
    """Run the Hogan event loop. Entry point for the champion runtime."""
    from hogan_bot.event_loop import run_event_loop
    parser = argparse.ArgumentParser(description="Hogan champion runtime")
    parser.add_argument("--max-events", type=int, default=None, help="Stop after N candle events")
    args, _ = parser.parse_known_args()
    max_events = args.max_events if args.max_events is not None else max_loops
    try:
        asyncio.run(run_event_loop(max_events=max_events))
    except KeyboardInterrupt:
        logger.info("Shutdown requested (KeyboardInterrupt).")
    except Exception as exc:
        logger.critical("Fatal error in event loop: %s — process exiting.", exc, exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    run()
