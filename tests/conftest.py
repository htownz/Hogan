"""Pytest configuration: ensure project root is on Python path."""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest

# Add project root so "from hogan_bot.X import Y" works when running pytest
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))


@pytest.fixture(autouse=True)
def _isolate_env():
    """Snapshot os.environ before each test and restore it afterwards.

    Prevents load_dotenv() or os.environ mutations in one test from
    leaking into subsequent tests (e.g. HOGAN_ATR_MIN_PCT=0 from .env).
    """
    snapshot = os.environ.copy()
    yield
    os.environ.clear()
    os.environ.update(snapshot)
