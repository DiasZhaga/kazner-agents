"""Load report: each agent's share of all LLM and tool calls in a run, read from its log.

The course requirement: no agent may do more than 40% of the LLM and tool calls in a typical
scenario. Message records are not counted (see DESIGN.md section 8).
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

COUNTED_KINDS = ("llm", "tool")
MAX_SHARE = 0.40


@dataclass
class AgentLoad:
    agent: str
    llm: int = 0
    tool: int = 0

    @property
    def total(self) -> int:
        return self.llm + self.tool


def compute_load(records: list[dict]) -> list[AgentLoad]:
    loads: dict[str, AgentLoad] = {}
    for record in records:
        if record["kind"] not in COUNTED_KINDS:
            continue
        load = loads.setdefault(record["agent"], AgentLoad(record["agent"]))
        if record["kind"] == "llm":
            load.llm += 1
        else:
            load.tool += 1
    return sorted(loads.values(), key=lambda load: (-load.total, load.agent))


def overloaded(loads: list[AgentLoad], max_share: float = MAX_SHARE) -> list[str]:
    total = sum(load.total for load in loads)
    return [load.agent for load in loads if total and load.total / total > max_share]


def format_load_report(run_id: str, loads: list[AgentLoad], max_share: float = MAX_SHARE) -> str:
    total = sum(load.total for load in loads)
    lines = [
        f"Load report for run {run_id} (LLM + tool calls; limit {max_share:.0%} per agent)",
        "",
        f"{'Agent':<20}{'LLM':>5}{'Tool':>6}{'Total':>7}{'Share':>8}",
    ]
    for load in loads:
        share = load.total / total if total else 0.0
        flag = "  <-- over the limit" if share > max_share else ""
        lines.append(
            f"{load.agent:<20}{load.llm:>5}{load.tool:>6}{load.total:>7}{share:>8.1%}{flag}"
        )
    lines.append(f"{'Total':<20}{sum(l.llm for l in loads):>5}{sum(l.tool for l in loads):>6}{total:>7}")
    bad = overloaded(loads, max_share)
    lines.append("")
    lines.append(f"FAIL: over {max_share:.0%}: {', '.join(bad)}" if bad else "OK: no agent over the limit")
    return "\n".join(lines)


def find_log(logs_dir: Path, run_id: str) -> Path:
    """The log file of a run; 'latest' means the newest run (run ids start with the time)."""
    if run_id == "latest":
        logs = sorted(Path(logs_dir).glob("*.jsonl"))
        if not logs:
            raise FileNotFoundError(f"no run logs in {logs_dir}")
        return logs[-1]
    path = Path(logs_dir) / f"{run_id}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"no log for run {run_id} in {logs_dir}")
    return path
