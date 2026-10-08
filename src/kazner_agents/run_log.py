"""JSONL run log: one line per agent step, tool call or LLM call.

Records hold metadata only (names, status, durations, token counts) and never the text being
annotated, so the log of a run can be shared as evidence of the load balance.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path


class RunLogger:
    def __init__(self, path: Path, run_id: str):
        self.path = Path(path)
        self.run_id = run_id
        self.step = 0  # set by the orchestrator before each delivered message
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(
        self, agent: str, kind: str, name: str, status: str, duration_ms: float, **extra
    ) -> dict:
        """Append one record. kind is 'llm', 'tool' or 'message'."""
        record = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
            "run_id": self.run_id,
            "step": self.step,
            "agent": agent,
            "kind": kind,
            "name": name,
            "status": status,
            "duration_ms": round(duration_ms, 1),
        }
        record.update({key: value for key, value in extra.items() if value is not None})
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
        return record


def read_log(path: Path) -> list[dict]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]
