"""Signal providers for Hogan 2.0.

Each provider implements the :class:`SignalProvider` protocol and returns a
:class:`SignalVote` with an independent opinion on the current bar.  The
policy layer aggregates votes — no single provider is a mandatory gatekeeper.

Phase A providers (wrappers around existing logic):
    - ``ma`` — MA crossover
    - ``ema_cloud`` — Ripster EMA cloud trend

Phase C providers (coming):
    - ``regime`` — 1h regime classification
    - ``forecast`` — forward-return probability
    - ``risk`` — risk model output
    - ``execution`` — 15m execution timing
"""

from hogan_bot.strategy import SignalProvider, SignalVote

__all__ = ["SignalProvider", "SignalVote"]
