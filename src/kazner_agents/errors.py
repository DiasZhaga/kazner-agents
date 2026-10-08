"""Exception types. The orchestrator decides what to do by the type of error.

- TransientError: may succeed if tried again (network, timeout, rate limit) -> retried.
- LimitExceeded: a run limit was hit (steps, time, LLM calls) -> the whole run stops.
- Any other KnaError: this batch cannot be finished -> batch `failed`, run continues.
"""


class KnaError(Exception):
    """Base class for all errors raised by kazner-agents."""


class TransientError(KnaError):
    """A temporary failure; the same call may succeed when retried."""


class LLMError(KnaError):
    """The LLM answered, but the answer cannot be used (refusal, cut off, invalid)."""


class AgentError(KnaError):
    """An agent could not produce a valid output for its input."""


class LoopDetected(KnaError):
    """The same message was routed too many times (loop protection)."""


class LimitExceeded(KnaError):
    """A run-level limit was reached: max steps, wall-clock time or max LLM calls."""
