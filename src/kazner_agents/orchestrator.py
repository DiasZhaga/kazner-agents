"""Orchestrator: plans the batches, routes every message, stores state and enforces limits.

It does no domain work itself. Each agent returns payloads; the orchestrator wraps each payload
in an Envelope, finds the recipients in the routing table, checks the limits and the loop
protection, saves and logs the message, and delivers it.

The plan of a run says which annotators work (mBERT, the LLM, or both); the routing table is
built from that plan (DESIGN.md section 11).
"""

from __future__ import annotations

import secrets
import time
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from kazner_agents.agents.base import Agent
from kazner_agents.agents.collector import Collector
from kazner_agents.agents.evaluator import Evaluator
from kazner_agents.agents.llm_annotator import LLMAnnotator
from kazner_agents.agents.normalizer import BoundaryNormalizer
from kazner_agents.agents.pre_annotator import PreAnnotator, Tagger, load_mbert
from kazner_agents.agents.specs import ORCHESTRATOR
from kazner_agents.agents.verifier import Verifier
from kazner_agents.config import Settings
from kazner_agents.context import RunContext
from kazner_agents.errors import AgentError, KnaError, LimitExceeded, LoopDetected
from kazner_agents.llm import LLMBackend, LLMClient
from kazner_agents.messages import (
    CollectRequest,
    Envelope,
    EvaluationReport,
    Message,
    ReportRequest,
)
from kazner_agents.run_log import RunLogger
from kazner_agents.state import StateStore

NAME = ORCHESTRATOR.name
ANNOTATOR_AGENTS = {"mbert": "PreAnnotator", "llm": "LLMAnnotator"}
DEFAULT_ANNOTATORS = ("mbert", "llm")


def build_routes(annotators: tuple[str, ...]) -> dict[str, list[str]]:
    """Which agents receive which payload type, for a run with these annotators."""
    two_annotators = len(annotators) == 2
    return {
        "CollectRequest": ["Collector"],
        "DocumentBatch": [ANNOTATOR_AGENTS[a] for a in annotators],
        "Annotation": ["BoundaryNormalizer"],
        "NormalizedAnnotation": ["Verifier" if two_annotators else "Evaluator"],
        "VerificationResult": ["Evaluator"],
        "ReportRequest": ["Evaluator"],
        "EvaluationReport": [NAME],
    }


ROUTES = build_routes(DEFAULT_ANNOTATORS)  # the default plan, shown in docs/messages.md

# The same (batch, type, sender, recipient) may be delivered at most this many times.
MAX_REPEATS = 2


@dataclass
class RunSummary:
    run_id: str
    status: str  # completed | completed_with_failures | stopped | failed
    reason: str | None
    batches: dict[str, str] = field(default_factory=dict)  # batch_id -> done/failed/skipped
    batch_errors: dict[str, str] = field(default_factory=dict)
    reports: list[EvaluationReport] = field(default_factory=list)  # one per finished batch
    run_report: EvaluationReport | None = None  # the Evaluator's report on the whole run
    annotators: tuple[str, ...] = DEFAULT_ANNOTATORS
    steps: int = 0
    llm_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    output_dir: Path | None = None
    log_path: Path | None = None


class Orchestrator:
    spec = ORCHESTRATOR
    name = NAME

    def __init__(
        self,
        ctx: RunContext,
        agents: dict[str, Agent],
        annotators: tuple[str, ...] = DEFAULT_ANNOTATORS,
    ):
        if not annotators or any(a not in ANNOTATOR_AGENTS for a in annotators):
            raise AgentError(f"unknown annotators {annotators}; choose from {list(ANNOTATOR_AGENTS)}")
        self.ctx = ctx
        self.agents = agents
        self.annotators = annotators
        self.routes = build_routes(annotators)
        self.step = 0
        self.repeats: Counter[tuple] = Counter()
        self.reports: list[EvaluationReport] = []
        self.deadline = time.monotonic() + ctx.settings.max_seconds
        ctx.llm.deadline = self.deadline  # the LLM client respects the same time budget

    # --- the plan --------------------------------------------------------------------------

    def run(self, task: CollectRequest) -> RunSummary:
        ctx = self.ctx
        ctx.call_tool(self.name, "state_create_run", ctx.state.create_run, ctx.run_id, task)
        summary = RunSummary(ctx.run_id, "completed", None, annotators=self.annotators)

        try:
            document_batches = self.deliver(self.name, "Collector", task)
        except KnaError as exc:
            # Without text there is nothing to annotate: the run fails with a readable reason.
            return self.finish(summary, "failed", f"collecting text failed: {exc}")

        stop_reason = None
        for batch in document_batches:
            status, reason = "done", None
            if stop_reason:
                status, reason = "skipped", stop_reason
            else:
                try:
                    self.process_batch(sender="Collector", first_payload=batch)
                except LimitExceeded as exc:
                    status, reason = "failed", str(exc)
                    stop_reason = f"run stopped: {exc}"
                except KnaError as exc:
                    status, reason = "failed", f"{type(exc).__name__}: {exc}"
            summary.batches[batch.batch_id] = status
            if reason:
                summary.batch_errors[batch.batch_id] = reason
            ctx.call_tool(
                self.name, "state_batch_status", ctx.state.set_batch_status,
                ctx.run_id, batch.batch_id, status, reason,
            )
        summary.reports = list(self.reports)

        done = [batch_id for batch_id, status in summary.batches.items() if status == "done"]
        if done and not stop_reason:
            try:  # one more round: the Evaluator reports on the whole run
                self.process_batch(sender=self.name, first_payload=ReportRequest(batch_ids=done))
                summary.run_report = self.reports[-1]
            except KnaError as exc:
                summary.batch_errors["run-report"] = f"{type(exc).__name__}: {exc}"

        if stop_reason:
            return self.finish(summary, "stopped", stop_reason)
        if "failed" in summary.batches.values():
            return self.finish(summary, "completed_with_failures", "some batches failed")
        return self.finish(summary, "completed", None)

    def process_batch(self, sender: str, first_payload: Message) -> None:
        """Route messages until nothing is left to deliver (the EvaluationReport comes back)."""
        queue = [(sender, first_payload)]
        while queue:
            sender, payload = queue.pop(0)
            recipients = self.routes.get(type(payload).__name__)
            if recipients is None:
                raise AgentError(f"no route for message type {type(payload).__name__}")
            for recipient in recipients:  # a DocumentBatch goes to every planned annotator
                for output in self.deliver(sender, recipient, payload):
                    queue.append((recipient, output))

    def finish(self, summary: RunSummary, status: str, reason: str | None) -> RunSummary:
        ctx = self.ctx
        ctx.call_tool(
            self.name, "state_finish_run", ctx.state.finish_run, ctx.run_id, status, reason
        )
        summary.status, summary.reason = status, reason
        summary.steps = self.step
        summary.llm_calls = ctx.llm.calls_made
        summary.input_tokens = ctx.llm.input_tokens
        summary.output_tokens = ctx.llm.output_tokens
        summary.output_dir = ctx.output_dir
        summary.log_path = ctx.logger.path
        return summary

    # --- one step: deliver one message -----------------------------------------------------

    def check_limits(self) -> None:
        settings = self.ctx.settings
        if self.step >= settings.max_steps:
            raise LimitExceeded(f"step limit reached ({settings.max_steps} messages per run)")
        if time.monotonic() > self.deadline:
            raise LimitExceeded(f"time limit reached ({settings.max_seconds:g} s per run)")

    def deliver(self, sender: str, recipient: str, payload: Message) -> list[Message]:
        """One step: wrap, check, save, log and hand the payload to its recipient."""
        ctx = self.ctx
        type_name = type(payload).__name__
        batch_id = getattr(payload, "batch_id", None)

        self.check_limits()
        self.step += 1
        ctx.logger.step = self.step
        envelope = Envelope(
            run_id=ctx.run_id, step=self.step, sender=sender, recipient=recipient,
            type=type_name, payload=payload,
        )

        key = (batch_id, type_name, sender, recipient)
        self.repeats[key] += 1
        if self.repeats[key] > MAX_REPEATS:
            ctx.logger.log(
                recipient, "message", type_name, "rejected", 0.0,
                sender=sender, recipient=recipient, msg_id=envelope.msg_id, batch_id=batch_id,
            )
            raise LoopDetected(
                f"{type_name} from {sender} to {recipient} for batch {batch_id} repeated more"
                f" than {MAX_REPEATS} times"
            )

        ctx.state.save_message(envelope, batch_id)
        start = time.perf_counter()
        status, error = "error", None
        try:
            if recipient == self.name:  # an EvaluationReport: this batch (or the run) is finished
                self.reports.append(payload)
                outputs = []
            else:
                agent = self.agents.get(recipient)
                if agent is None:
                    raise AgentError(f"agent {recipient} is not available")
                outputs = agent.handle(payload, ctx)
            status = "ok"
            return outputs
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            ctx.logger.log(
                recipient, "message", type_name, status, (time.perf_counter() - start) * 1000,
                sender=sender, recipient=recipient, msg_id=envelope.msg_id, batch_id=batch_id,
                error=error,
            )


# --- building and running a whole run ------------------------------------------------------


def new_run_id() -> str:
    return datetime.now().strftime("%Y%m%d-%H%M%S") + "-" + secrets.token_hex(2)


def default_agents(make_tagger: Callable[[Settings], Tagger] = load_mbert) -> dict[str, Agent]:
    """New agent objects for one run (agents keep per-run state, e.g. the loaded model)."""
    agents = [
        Collector(), PreAnnotator(make_tagger), LLMAnnotator(), BoundaryNormalizer(),
        Verifier(), Evaluator(),
    ]
    return {agent.name: agent for agent in agents}


def run_pipeline(
    settings: Settings,
    task: CollectRequest,
    backend: LLMBackend,
    agents: dict[str, Agent] | None = None,
    run_id: str | None = None,
    annotators: tuple[str, ...] = DEFAULT_ANNOTATORS,
    make_tagger: Callable[[Settings], Tagger] = load_mbert,
) -> RunSummary:
    """Create the run context, run the orchestrator, and close the state store."""
    run_id = run_id or new_run_id()
    logger = RunLogger(settings.logs_dir / f"{run_id}.jsonl", run_id)
    llm = LLMClient(
        backend, logger,
        max_calls=settings.max_llm_calls,
        timeout_seconds=settings.llm_timeout_seconds,
        max_retries=settings.max_retries,
        retry_base_delay=settings.retry_base_delay,
    )
    state = StateStore(settings.state_path)
    ctx = RunContext(run_id, settings, logger, llm, state, settings.outputs_dir / run_id)
    try:
        orchestrator = Orchestrator(ctx, agents or default_agents(make_tagger), annotators)
        return orchestrator.run(task)
    finally:
        state.close()
