"""Shared retry utilities for transient network failures.

Provides a generic ``retry`` helper with exponential back-off and jitter,
plus pre-configured wrappers for CCXT and urllib call sites.

Design rules
------------
* **Read-only / idempotent** calls should always be retried.
* **Order-creation** (POST) calls should NOT be retried automatically —
  a timeout does not mean the order was rejected.
* Back-off uses *full jitter*: ``uniform(0, base_delay * 2^attempt)``
  capped at *max_delay*.  This prevents thundering-herd retries when
  multiple symbols poll the same exchange concurrently.
"""
from __future__ import annotations

import functools
import logging
import random
import socket
import time
from typing import Any, Callable, TypeVar
from urllib.error import URLError

logger = logging.getLogger(__name__)

T = TypeVar("T")

# ── CCXT transient exceptions ─────────────────────────────────────────────
try:
    import ccxt
    CCXT_TRANSIENT: tuple[type[Exception], ...] = (
        ccxt.RequestTimeout,
        ccxt.NetworkError,
        ccxt.ExchangeNotAvailable,
    )
except ImportError:
    CCXT_TRANSIENT = ()

URLLIB_TRANSIENT: tuple[type[Exception], ...] = (
    URLError,
    socket.timeout,
    ConnectionError,
    OSError,
)


def retry(
    fn: Callable[..., T],
    *args: Any,
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable: tuple[type[Exception], ...] = (),
    label: str = "",
    **kwargs: Any,
) -> T:
    """Call *fn* with retry on transient exceptions.

    Parameters
    ----------
    fn:           Callable to invoke.
    max_attempts: Total attempts (including the first).
    base_delay:   Initial back-off seed in seconds.
    max_delay:    Cap on the per-attempt sleep.
    retryable:    Exception types that trigger a retry.
    label:        Human-readable name for log messages.
    """
    last_exc: Exception | None = None
    tag = label or getattr(fn, "__qualname__", str(fn))
    for attempt in range(max_attempts):
        try:
            return fn(*args, **kwargs)
        except retryable as exc:
            last_exc = exc
            if attempt < max_attempts - 1:
                cap = min(base_delay * (2 ** attempt), max_delay)
                delay = random.uniform(0, cap)
                logger.warning(
                    "%s: attempt %d/%d failed (%s), retry in %.1fs",
                    tag, attempt + 1, max_attempts, exc, delay,
                )
                time.sleep(delay)
            else:
                logger.error(
                    "%s: all %d attempts exhausted, raising: %s",
                    tag, max_attempts, exc,
                )
    raise last_exc  # type: ignore[misc]


def with_retry(
    max_attempts: int = 3,
    base_delay: float = 1.0,
    max_delay: float = 30.0,
    retryable: tuple[type[Exception], ...] = (),
) -> Callable:
    """Decorator form of :func:`retry`.

    Usage::

        @with_retry(retryable=CCXT_TRANSIENT)
        def fetch_ticker(self, symbol):
            return self._exchange.fetch_ticker(symbol)
    """
    def decorator(fn: Callable[..., T]) -> Callable[..., T]:
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> T:
            return retry(
                fn, *args,
                max_attempts=max_attempts,
                base_delay=base_delay,
                max_delay=max_delay,
                retryable=retryable,
                label=fn.__qualname__,
                **kwargs,
            )
        return wrapper
    return decorator


# ── Pre-configured decorators ─────────────────────────────────────────────

ccxt_retry = with_retry(
    max_attempts=3,
    base_delay=1.0,
    max_delay=15.0,
    retryable=CCXT_TRANSIENT,
)
"""Decorator for read-only CCXT calls (fetch_ticker, fetch_balance, …)."""

urllib_retry = with_retry(
    max_attempts=3,
    base_delay=1.0,
    max_delay=15.0,
    retryable=URLLIB_TRANSIENT,
)
"""Decorator for urllib-based REST calls (Oanda, Fear & Greed, …)."""
