# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: MIT

"""Circuit breaker pattern for resilient endpoint calls.

Tracks consecutive failures and opens the circuit after a threshold,
preventing repeated attempts to failing endpoints.
"""

from __future__ import annotations

import asyncio
import time
from enum import Enum

from nemocode.core.logging_config import StructuredLogger

logger = StructuredLogger(__name__)


class CircuitState(Enum):
    """States of a circuit breaker."""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreakerOpen(Exception):
    """Raised when the circuit breaker is open and requests are blocked."""

    def __init__(self, cooldown_remaining: float = 0.0) -> None:
        self.cooldown_remaining = cooldown_remaining
        super().__init__(f"Circuit breaker is open, retry in {cooldown_remaining:.1f}s")


class CircuitBreaker:
    """Circuit breaker that tracks consecutive failures for an endpoint.

    States:
        CLOSED: Normal operation. Failures are counted.
        OPEN: Failing fast. Requests are blocked until cooldown expires.
        HALF_OPEN: Testing. One request is allowed through; success closes
            the circuit, failure re-opens it.

    Args:
        failure_threshold: Number of consecutive failures before opening
            the circuit. Defaults to 3.
        cooldown_seconds: Seconds to wait before transitioning from OPEN
            to HALF_OPEN. Defaults to 30.

    Example:
        >>> cb = CircuitBreaker(failure_threshold=3, cooldown_seconds=30)
        >>> if cb.can_execute():
        ...     try:
        ...         result = await call_endpoint()
        ...         cb.record_success()
        ...     except Exception:
        ...         cb.record_failure()
    """

    def __init__(
        self,
        failure_threshold: int = 3,
        cooldown_seconds: float = 30.0,
    ) -> None:
        self._failure_threshold = failure_threshold
        self._cooldown_seconds = cooldown_seconds
        self._failure_count = 0
        self._state = CircuitState.CLOSED
        self._opened_at: float | None = None
        self._retry_after_until: float | None = None
        self._lock = asyncio.Lock()

    @property
    def state(self) -> CircuitState:
        """Current circuit state, accounting for automatic OPEN→HALF_OPEN transition."""
        if self._state == CircuitState.OPEN and self._opened_at is not None:
            if time.monotonic() - self._opened_at >= self._cooldown_seconds:
                return CircuitState.HALF_OPEN
        return self._state

    @property
    def failure_count(self) -> int:
        """Number of consecutive failures."""
        return self._failure_count

    @property
    def cooldown_remaining(self) -> float:
        """Seconds remaining until the circuit transitions to HALF_OPEN, or 0."""
        if self._state != CircuitState.OPEN or self._opened_at is None:
            return 0.0
        elapsed = time.monotonic() - self._opened_at
        return max(0.0, self._cooldown_seconds - elapsed)

    async def can_execute(self) -> bool:
        """Check whether a request is allowed through.

        Returns:
            True if the circuit is CLOSED or HALF_OPEN (and no retry-after
            cooldown is active), False if OPEN.
        """
        async with self._lock:
            current = self.state

            if current == CircuitState.CLOSED:
                return True

            if current == CircuitState.HALF_OPEN:
                # Transition to half-open: allow one test request
                self._state = CircuitState.HALF_OPEN
                return True

            # OPEN — check if a Retry-After override has expired
            if self._retry_after_until is not None:
                if time.monotonic() >= self._retry_after_until:
                    self._retry_after_until = None
                    self._state = CircuitState.HALF_OPEN
                    return True

            return False

    async def record_success(self) -> None:
        """Record a successful request, closing the circuit."""
        async with self._lock:
            was_open = self._state != CircuitState.CLOSED
            self._failure_count = 0
            self._state = CircuitState.CLOSED
            self._opened_at = None
            self._retry_after_until = None
            if was_open:
                logger.info("circuit_closed", extra={"failure_count": 0})

    async def record_failure(self) -> None:
        """Record a failed request.

        Opens the circuit after reaching the failure threshold.
        """
        async with self._lock:
            self._failure_count += 1
            if self._failure_count >= self._failure_threshold:
                self._state = CircuitState.OPEN
                self._opened_at = time.monotonic()
                self._retry_after_until = None
                logger.warning(
                    "circuit_opened",
                    extra={
                        "failure_count": self._failure_count,
                        "cooldown_seconds": self._cooldown_seconds,
                    },
                )

    async def record_failure_with_retry_after(self, retry_after: float) -> None:
        """Record a failure with a server-suggested retry-after delay.

        Opens the circuit and sets a minimum wait time based on the
        Retry-After header value.

        Args:
            retry_after: Seconds the server suggests waiting before
                retrying.
        """
        async with self._lock:
            self._failure_count += 1
            self._state = CircuitState.OPEN
            self._opened_at = time.monotonic()
            self._retry_after_until = time.monotonic() + retry_after
            logger.warning(
                "circuit_opened",
                extra={
                    "failure_count": self._failure_count,
                    "retry_after_s": retry_after,
                },
            )
