"""Backward-compatible alias for the champion runtime.

    python -m hogan_bot.trader_service

runs the same as::

    python -m hogan_bot.main

Use main as the canonical entry point.
"""
from __future__ import annotations

from hogan_bot.main import run


if __name__ == "__main__":
    run()
