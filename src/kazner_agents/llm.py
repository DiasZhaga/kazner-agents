"""The one module through which every LLM call goes.

LLMClient adds, around any backend: the per-run cap on calls, the time budget, a timeout per
attempt, retries with exponential backoff, and one log record per call with token usage.

Backends:
- OpenAIBackend: the real OpenAI Responses API with Structured Outputs.
- FakeLLM: deterministic and offline, for tests and demos. It cannot read the prompt, so it
  answers with a `responder` function that gets the structured input from request.metadata.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Protocol

from pydantic import BaseModel, ValidationError

from kazner_agents.errors import KnaError, LimitExceeded, LLMError, TransientError
from kazner_agents.retry import call_with_retries
from kazner_agents.run_log import RunLogger

MIN_SECONDS_FOR_ATTEMPT = 1.0


@dataclass
class LLMRequest:
    instructions: str  # the system part of the prompt
    user: str  # the input text of the prompt
    schema: type[BaseModel]  # the answer must match this model (Structured Outputs)
    metadata: dict[str, Any] = field(default_factory=dict)  # used only by FakeLLM


@dataclass
class LLMResult:
    parsed: BaseModel
    input_tokens: int
    output_tokens: int


class LLMBackend(Protocol):
    model_name: str

    def complete(self, request: LLMRequest, timeout: float) -> LLMResult: ...


class LLMClient:
    def __init__(
        self,
        backend: LLMBackend,
        logger: RunLogger,
        max_calls: int,
        timeout_seconds: float,
        max_retries: int = 3,
        retry_base_delay: float = 1.0,
        deadline: float | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ):
        self.backend = backend
        self.logger = logger
        self.max_calls = max_calls
        self.timeout_seconds = timeout_seconds
        self.max_retries = max_retries
        self.retry_base_delay = retry_base_delay
        self.deadline = deadline  # time.monotonic() value when the run must stop
        self.sleep = sleep
        self.calls_made = 0  # every attempt counts: every attempt may cost money
        self.input_tokens = 0
        self.output_tokens = 0

    def complete(self, agent: str, name: str, request: LLMRequest) -> BaseModel:
        """Ask the LLM; return the parsed answer or raise a KnaError."""
        attempts = 0
        input_tokens = 0
        output_tokens = 0

        def one_attempt() -> BaseModel:
            nonlocal attempts, input_tokens, output_tokens
            timeout = self._timeout_for_next_attempt()
            attempts += 1
            self.calls_made += 1
            result = self.backend.complete(request, timeout=timeout)
            input_tokens += result.input_tokens
            output_tokens += result.output_tokens
            return result.parsed

        start = time.monotonic()
        status, error = "error", None
        try:
            parsed = call_with_retries(
                one_attempt, self.max_retries, self.retry_base_delay, sleep=self.sleep
            )
            status = "ok"
            return parsed
        except KnaError as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self.input_tokens += input_tokens
            self.output_tokens += output_tokens
            self.logger.log(
                agent, "llm", name, status, (time.monotonic() - start) * 1000,
                model=self.backend.model_name, attempts=attempts,
                input_tokens=input_tokens, output_tokens=output_tokens, error=error,
            )

    def _timeout_for_next_attempt(self) -> float:
        """Check the call cap and the time budget; return the timeout for the next attempt."""
        if self.calls_made >= self.max_calls:
            raise LimitExceeded(f"LLM call limit reached ({self.max_calls} calls per run)")
        timeout = self.timeout_seconds
        if self.deadline is not None:
            remaining = self.deadline - time.monotonic()
            if remaining < MIN_SECONDS_FOR_ATTEMPT:
                raise LimitExceeded("run time limit reached before an LLM call")
            timeout = min(timeout, remaining)
        return timeout


class FakeLLM:
    """Deterministic offline backend. `fail_times` makes the first N attempts fail."""

    model_name = "fake"

    def __init__(
        self,
        responder: Callable[[LLMRequest], dict],
        fail_times: int = 0,
        failure: type[Exception] = TransientError,
    ):
        self.responder = responder
        self.fail_times = fail_times
        self.failure = failure
        self.calls = 0

    def complete(self, request: LLMRequest, timeout: float) -> LLMResult:
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.failure(f"fake failure {self.calls} of {self.fail_times}")
        answer = self.responder(request)
        try:
            parsed = request.schema.model_validate(answer)
        except ValidationError as exc:
            raise LLMError(f"fake answer does not match the schema: {exc}") from exc
        # Rough token estimate (about 4 characters per token), so cost reports work offline.
        prompt_chars = len(request.instructions) + len(request.user)
        answer_chars = len(json.dumps(answer, ensure_ascii=False))
        return LLMResult(parsed, prompt_chars // 4, answer_chars // 4)


class OpenAIBackend:
    """OpenAI Responses API with Structured Outputs (strict JSON schema from a pydantic model)."""

    def __init__(self, api_key: str, model: str, reasoning_effort: str):
        import openai  # imported here so that tests and FakeLLM runs never need it

        if not api_key:
            raise LLMError("OPENAI_API_KEY is not set; put it in .env (see .env.example)")
        self.openai = openai
        # max_retries=0: LLMClient does the retries, the SDK must not retry on its own as well.
        self.client = openai.OpenAI(api_key=api_key, max_retries=0)
        self.model_name = model
        self.reasoning_effort = reasoning_effort

    def complete(self, request: LLMRequest, timeout: float) -> LLMResult:
        openai = self.openai
        try:
            response = self.client.responses.parse(
                model=self.model_name,
                instructions=request.instructions,
                input=request.user,
                text_format=request.schema,
                reasoning={"effort": self.reasoning_effort},
                timeout=timeout,
            )
        except (
            openai.APITimeoutError,
            openai.APIConnectionError,
            openai.RateLimitError,
            openai.InternalServerError,
        ) as exc:
            raise TransientError(f"OpenAI temporary error: {type(exc).__name__}") from exc
        except openai.AuthenticationError as exc:
            # Our own message: the SDK's text can contain part of the key.
            raise LLMError("OpenAI rejected the API key (401); check OPENAI_API_KEY") from exc
        except openai.APIStatusError as exc:
            raise LLMError(f"OpenAI API error {exc.status_code}: {exc.message}") from exc
        except ValidationError as exc:
            raise LLMError(f"OpenAI answer does not match the schema: {exc}") from exc

        if response.status == "incomplete":
            reason = getattr(response.incomplete_details, "reason", "unknown")
            raise LLMError(f"OpenAI answer was cut off ({reason})")
        if response.output_parsed is None:
            raise LLMError("the model refused or returned no parsable answer")
        usage = response.usage
        return LLMResult(response.output_parsed, usage.input_tokens, usage.output_tokens)
