import time

import pytest
from pydantic import BaseModel

from kazner_agents.errors import LimitExceeded, LLMError, TransientError
from kazner_agents.llm import FakeLLM, LLMClient, LLMRequest
from kazner_agents.run_log import RunLogger, read_log


class Answer(BaseModel):
    value: int


def make_client(tmp_path, backend, max_calls=40, deadline=None):
    waits = []
    logger = RunLogger(tmp_path / "run.jsonl", "r1")
    client = LLMClient(
        backend, logger, max_calls=max_calls, timeout_seconds=60, max_retries=3,
        retry_base_delay=1.0, deadline=deadline, sleep=waits.append,
    )
    return client, waits


REQUEST = LLMRequest(instructions="Return 7.", user="now", schema=Answer)


def test_success_is_logged_with_tokens(tmp_path):
    client, waits = make_client(tmp_path, FakeLLM(lambda r: {"value": 7}))
    assert client.complete("LLMAnnotator", "annotate", REQUEST) == Answer(value=7)
    [record] = read_log(tmp_path / "run.jsonl")
    assert record["kind"] == "llm" and record["status"] == "ok"
    assert record["attempts"] == 1 and record["input_tokens"] > 0
    assert waits == []


def test_retries_with_exponential_backoff_then_succeeds(tmp_path):
    backend = FakeLLM(lambda r: {"value": 1}, fail_times=2)
    client, waits = make_client(tmp_path, backend)
    assert client.complete("A", "x", REQUEST).value == 1
    assert waits == [1.0, 2.0]
    assert client.calls_made == 3
    [record] = read_log(tmp_path / "run.jsonl")
    assert record["attempts"] == 3 and record["status"] == "ok"


def test_gives_up_after_three_retries(tmp_path):
    backend = FakeLLM(lambda r: {"value": 1}, fail_times=10)
    client, waits = make_client(tmp_path, backend)
    with pytest.raises(TransientError):
        client.complete("A", "x", REQUEST)
    assert backend.calls == 4  # 1 attempt + 3 retries
    assert waits == [1.0, 2.0, 4.0]
    [record] = read_log(tmp_path / "run.jsonl")
    assert record["status"] == "error" and "TransientError" in record["error"]


def test_permanent_errors_are_not_retried(tmp_path):
    backend = FakeLLM(lambda r: {"value": 1}, fail_times=1, failure=LLMError)
    client, waits = make_client(tmp_path, backend)
    with pytest.raises(LLMError):
        client.complete("A", "x", REQUEST)
    assert backend.calls == 1 and waits == []


def test_call_cap_counts_every_attempt(tmp_path):
    backend = FakeLLM(lambda r: {"value": 1}, fail_times=10)
    client, _ = make_client(tmp_path, backend, max_calls=2)
    with pytest.raises(LimitExceeded):
        client.complete("A", "x", REQUEST)
    assert backend.calls == 2


def test_no_attempt_when_time_budget_is_used_up(tmp_path):
    backend = FakeLLM(lambda r: {"value": 1})
    client, _ = make_client(tmp_path, backend, deadline=time.monotonic() + 0.5)
    with pytest.raises(LimitExceeded):
        client.complete("A", "x", REQUEST)
    assert backend.calls == 0


def test_invalid_fake_answer_fails_validation(tmp_path):
    client, _ = make_client(tmp_path, FakeLLM(lambda r: {"value": "many"}))
    with pytest.raises(LLMError):
        client.complete("A", "x", REQUEST)
