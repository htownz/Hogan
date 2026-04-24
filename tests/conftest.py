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


@pytest.fixture(autouse=True)
def _isolate_regime_history():
    """Clear :data:`hogan_bot.regime._REGIME_HISTORY` before and after each test.

    ``detect_regime`` stores a per-symbol rolling buffer of observed regimes
    at module level to drive hysteresis (N-bar confirmation before a regime
    switch). When one test runs ``policy_core.decide`` several times it pushes
    entries into that buffer — and any **subsequent** test that depends on a
    freshly-warmed regime path then sees residual history.

    Concrete failure this prevents:
    ``test_correctness_patches.TestSwarmBacktestDBIsolation::test_live_still_writes_swarm_rows``
    ran ``decide`` 5× on a BTC/USD tape that ended in the ranging regime, which
    pre-loaded the hysteresis buffer before
    ``test_decision_parity.TestPolicyCoreEquivalence::test_decide_deterministic``.
    Between that test's two ``decide()`` calls the buffer grew one more entry,
    flipping the smoothed regime on the second call and producing a different
    confidence (0.127 vs 0.099). The individual tests were each correct; only
    their shared module-level state was broken.

    Importing ``reset_regime_history`` is cheap and safe — the function is a
    one-line ``dict.clear``.
    """
    try:
        from hogan_bot.regime import reset_regime_history
    except Exception:
        # regime module not importable in this collection (e.g. bare unit tests
        # that only touch utility modules) — fixture becomes a no-op.
        yield
        return
    reset_regime_history()
    yield
    reset_regime_history()
