"""Pytest configuration: ensure project root is on Python path."""
from __future__ import annotations

import sys
from pathlib import Path

# Add project root so "from hogan_bot.X import Y" works when running pytest
_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))
