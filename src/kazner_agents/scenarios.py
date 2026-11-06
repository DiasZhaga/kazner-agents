"""Test scenarios: runs with a known input and expected results, checked automatically.

Each scenario is a JSON file in `scenarios/`:

    {"name": ..., "description": ...,
     "task": {CollectRequest fields},
     "annotators": ["mbert", "llm"],
     "settings": {Settings fields to override, e.g. "max_steps": 10},
     "fakes": {how the offline stand-ins behave},
     "expect": {the expected results}}

Scenarios run offline: FakeLLM, FakeTagger, and a fake Wikipedia that serves a fixture text.
`kna scenarios` prints a pass/fail table; pytest runs the same files.
"""

from __future__ import annotations

import dataclasses
import json
import tempfile
from pathlib import Path

from kazner_agents.agents.collector import Collector
from kazner_agents.agents.llm_annotator import LLMAnnotator
from kazner_agents.agents.verifier import AdjudicationAnswer
from kazner_agents.config import Settings
from kazner_agents.errors import AgentError, LLMError, TransientError
from kazner_agents.fakes import fake_responder, fake_tagger
from kazner_agents.llm import FakeLLM, LLMRequest
from kazner_agents.load_report import compute_load
from kazner_agents.messages import CollectRequest
from kazner_agents.orchestrator import RunSummary, default_agents, run_pipeline
from kazner_agents.run_log import read_log
from kazner_agents.state import StateStore
from kazner_agents.wikipedia import Article

SCENARIOS_DIR = Path(__file__).resolve().parents[2] / "scenarios"


def load_scenarios(directory: Path = SCENARIOS_DIR) -> list[dict]:
    return [json.loads(p.read_text(encoding="utf-8")) for p in sorted(directory.glob("*.json"))]


class LoopingAnnotator(LLMAnnotator):
    """A broken annotator that sends the batch back to itself, to test loop protection."""

    def handle(self, payload, ctx):
        return [payload]


def make_fake_llm(fakes: dict) -> FakeLLM:
    """FakeLLM as the scenario asks: N transient failures first, or a failing adjudicator."""

    def responder(request: LLMRequest) -> dict:
        if fakes.get("adjudicator_fails") and request.schema is AdjudicationAnswer:
            raise LLMError("fake adjudicator is down")
        return fake_responder(request)

    return FakeLLM(responder, fail_times=fakes.get("llm_fail_times", 0), failure=TransientError)


def make_agents(fakes: dict, directory: Path) -> dict:
    article_file = directory / "fixtures" / fakes.get("wikipedia_text", "abai.txt")

    def fetch(title: str, user_agent: str) -> Article:
        if fakes.get("wikipedia") == "missing":
            raise AgentError(f"Kazakh Wikipedia has no article titled {title!r}")
        return Article(title=title, url=f"https://kk.wikipedia.org/wiki/{title}",
                       text=article_file.read_text(encoding="utf-8"))

    agents = default_agents(fake_tagger)
    agents["Collector"] = Collector(fetch_article=fetch)
    if fakes.get("looping_annotator"):
        agents["LLMAnnotator"] = LoopingAnnotator()
    return agents


def run_scenario(scenario: dict, directory: Path = SCENARIOS_DIR) -> tuple[RunSummary, list[tuple[str, bool, str]]]:
    """Run one scenario in a temporary folder; return the summary and the checks."""
    with tempfile.TemporaryDirectory() as tmp:
        overrides = dict(scenario.get("settings", {}))
        if "kaznerd_dir" in overrides:
            overrides["kaznerd_dir"] = directory / overrides["kaznerd_dir"]
        settings = dataclasses.replace(Settings(home=Path(tmp), retry_base_delay=0.0), **overrides)
        task_fields = dict(scenario["task"])
        if task_fields.get("path"):
            task_fields["path"] = str(directory / task_fields["path"])
        fakes = scenario.get("fakes", {})
        summary = run_pipeline(
            settings, CollectRequest(**task_fields), make_fake_llm(fakes),
            agents=make_agents(fakes, directory), run_id="scenario",
            annotators=tuple(scenario.get("annotators", ["mbert", "llm"])),
        )
        records = read_log(summary.log_path)
        state = StateStore(settings.state_path)
        review_items = state.get_review_items("scenario")
        state.close()
        checks = check_expectations(scenario["expect"], summary, records, review_items)
    return summary, checks


def check_expectations(expect: dict, summary: RunSummary, records: list[dict],
                       review_items: list[dict]) -> list[tuple[str, bool, str]]:
    """Compare a finished run with the scenario's expected results."""
    checks = []

    def check(name: str, ok: bool, detail: str) -> None:
        checks.append((name, bool(ok), detail))

    if "status" in expect:
        check("status", summary.status == expect["status"], summary.status)
    if "reason_contains" in expect:
        reason = summary.reason or ""
        check("reason", expect["reason_contains"] in reason, reason)
    if "batches" in expect:
        check("batches", summary.batches == expect["batches"], str(summary.batches))
    if "batch_error_contains" in expect:
        errors = " | ".join(summary.batch_errors.values())
        check("batch error", expect["batch_error_contains"] in errors, errors[:120])
    report = summary.run_report
    if "mode" in expect:
        check("mode", report is not None and report.mode == expect["mode"],
              report.mode if report else "no run report")
    if "annotators_scored" in expect:
        scored = sorted(report.per_annotator) if report else []
        check("annotators scored", scored == sorted(expect["annotators_scored"]), str(scored))
    if "final_f1_at_least" in expect:
        f1 = report.f1 if report and report.f1 is not None else -1.0
        check("final F1", f1 >= expect["final_f1_at_least"], f"{f1:.3f}")
    if expect.get("agreement_reported"):
        check("agreement", report is not None and report.agreement_rate is not None,
              str(report.agreement_rate if report else None))
    if "review_items_at_least" in expect:
        check("review queue", len(review_items) >= expect["review_items_at_least"], str(len(review_items)))
    if "review_items" in expect:
        check("review queue", len(review_items) == expect["review_items"], str(len(review_items)))
    if "decisions_at_least" in expect and report is not None:
        for decision, minimum in expect["decisions_at_least"].items():
            n = report.counts.get(f"decision:{decision}", 0)
            check(f"decisions {decision}", n >= minimum, str(n))
    if "llm_attempts_in_one_call" in expect:
        attempts = max((r.get("attempts", 0) for r in records if r["kind"] == "llm"), default=0)
        check("LLM retries", attempts == expect["llm_attempts_in_one_call"], str(attempts))
    if "rejected_messages_at_least" in expect:
        rejected = sum(r["status"] == "rejected" for r in records)
        check("loop protection", rejected >= expect["rejected_messages_at_least"], str(rejected))
    if "agents_without_messages" in expect:
        busy = {r["agent"] for r in records}
        idle = [a for a in expect["agents_without_messages"] if a not in busy]
        check("idle agents", idle == expect["agents_without_messages"], f"active: {sorted(busy)}")
    if "max_agent_share" in expect:
        loads = compute_load(records)
        total = sum(load.total for load in loads)
        share = max((load.total / total for load in loads), default=0.0)
        check("load balance", share <= expect["max_agent_share"], f"max share {share:.1%}")
    return checks
