"""RunContext: what every agent gets during a run (settings, log, LLM client, state store).

`call_tool` is how agents use a tool: it runs the function, optionally retries it, and writes
one `tool` record to the run log. This is what `kna load-report` counts.
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

from kazner_agents.config import Settings
from kazner_agents.retry import call_with_retries
from kazner_agents.run_log import RunLogger

if TYPE_CHECKING:
    from kazner_agents.llm import LLMClient
    from kazner_agents.state import StateStore


@dataclass
class RunContext:
    run_id: str
    settings: Settings
    logger: RunLogger
    llm: LLMClient
    state: StateStore
    output_dir: Path

    def call_tool(
        self, agent: str, name: str, fn: Callable[..., Any], *args, retry: bool = False, **kwargs
    ) -> Any:
        """Call a tool function and log it. With retry=True, TransientErrors are retried."""
        attempts = 0

        def one_attempt():
            nonlocal attempts
            attempts += 1
            return fn(*args, **kwargs)

        start = time.perf_counter()
        status, error = "error", None
        try:
            if retry:
                result = call_with_retries(
                    one_attempt, self.settings.max_retries, self.settings.retry_base_delay
                )
            else:
                result = one_attempt()
            status = "ok"
            return result
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.logger.log(
                agent, "tool", name, status, (time.perf_counter() - start) * 1000,
                attempts=attempts, error=error,
            )
