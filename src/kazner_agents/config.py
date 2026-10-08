"""Run settings, read from environment variables and the project's `.env` file.

The API key is read here but never printed: `Settings.__repr__` hides it.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import dotenv_values

# src/kazner_agents/config.py -> parents[2] is the repository root (editable install).
DEFAULT_HOME = Path(__file__).resolve().parents[2]


@dataclass
class Settings:
    home: Path
    openai_api_key: str = field(default="", repr=False)
    model: str = "gpt-6-luna"
    reasoning_effort: str = "low"
    price_input_per_1m: float = 0.10
    price_output_per_1m: float = 0.50
    max_steps: int = 60
    max_seconds: float = 300.0
    llm_timeout_seconds: float = 60.0
    max_llm_calls: int = 40
    max_retries: int = 3
    retry_base_delay: float = 1.0
    user_agent: str = "kazner-agents/0.1 (course project)"

    @property
    def logs_dir(self) -> Path:
        return self.home / "logs"

    @property
    def outputs_dir(self) -> Path:
        return self.home / "outputs"

    @property
    def state_path(self) -> Path:
        return self.home / "state" / "kna.sqlite3"


def load_settings(home: Path | None = None) -> Settings:
    """Build Settings from `<home>/.env`, overridden by real environment variables."""
    home = Path(home or os.environ.get("KNA_HOME") or DEFAULT_HOME).resolve()
    env_file = home / ".env"
    values = dotenv_values(env_file) if env_file.exists() else {}
    values.update(os.environ)

    def get(name: str, default: str) -> str:
        value = values.get(name)
        return value if value else default

    return Settings(
        home=home,
        openai_api_key=get("OPENAI_API_KEY", ""),
        model=get("KNA_MODEL", "gpt-6-luna"),
        reasoning_effort=get("KNA_REASONING_EFFORT", "low"),
        price_input_per_1m=float(get("KNA_PRICE_INPUT_PER_1M", "0.10")),
        price_output_per_1m=float(get("KNA_PRICE_OUTPUT_PER_1M", "0.50")),
        max_steps=int(get("KNA_MAX_STEPS", "60")),
        max_seconds=float(get("KNA_MAX_SECONDS", "300")),
        llm_timeout_seconds=float(get("KNA_LLM_TIMEOUT_SECONDS", "60")),
        max_llm_calls=int(get("KNA_MAX_LLM_CALLS", "40")),
        user_agent=get("KNA_USER_AGENT", "kazner-agents/0.1 (course project)"),
    )
