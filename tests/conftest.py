from pathlib import Path

import pytest

from kazner_agents.config import Settings
from kazner_agents.context import RunContext
from kazner_agents.llm import FakeLLM, LLMClient
from kazner_agents.run_log import RunLogger


def make_settings(home: Path) -> Settings:
    # No waiting between retries in tests.
    return Settings(home=home, retry_base_delay=0.0)


@pytest.fixture
def ctx(tmp_path: Path) -> RunContext:
    """A run context with FakeLLM; no network, no state store."""
    settings = make_settings(tmp_path)
    logger = RunLogger(tmp_path / "logs" / "test-run.jsonl", "test-run")
    llm = LLMClient(FakeLLM(lambda request: {}), logger, max_calls=10, timeout_seconds=5)
    return RunContext("test-run", settings, logger, llm, None, tmp_path / "outputs" / "test-run")
