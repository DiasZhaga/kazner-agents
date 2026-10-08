"""Retries with exponential backoff, shared by the LLM client and the external tools."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import TypeVar

from kazner_agents.errors import TransientError

T = TypeVar("T")


def call_with_retries(
    fn: Callable[[], T],
    max_retries: int,
    base_delay: float,
    sleep: Callable[[float], None] = time.sleep,
) -> T:
    """Call fn(). If it raises TransientError, wait and try again, at most max_retries times.

    The waits are base_delay * 1, 2, 4, ... (exponential backoff). Any other exception is not
    retried. After the last failed retry the TransientError is raised to the caller.
    """
    retries_done = 0
    while True:
        try:
            return fn()
        except TransientError:
            if retries_done >= max_retries:
                raise
            sleep(base_delay * 2**retries_done)
            retries_done += 1
