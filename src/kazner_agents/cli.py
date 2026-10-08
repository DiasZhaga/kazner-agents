"""Command-line interface: `kna run`, `kna load-report`, `kna docs`."""

from __future__ import annotations

import argparse
import sys

from kazner_agents import __version__
from kazner_agents.agents.llm_annotator import fake_annotation_responder
from kazner_agents.config import Settings, load_settings
from kazner_agents.docs_gen import outdated_docs, write_docs
from kazner_agents.errors import KnaError
from kazner_agents.llm import FakeLLM, OpenAIBackend
from kazner_agents.load_report import compute_load, find_log, format_load_report, overloaded
from kazner_agents.messages import CollectRequest
from kazner_agents.orchestrator import RunSummary, run_pipeline
from kazner_agents.run_log import read_log


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kna", description="Kazakh NER annotation agents")
    parser.add_argument("--version", action="version", version=f"kna {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="annotate a Wikipedia article or a text file")
    run.add_argument("--source", choices=["wikipedia", "file"], required=True)
    run.add_argument("--title", help="Kazakh Wikipedia article title (source wikipedia)")
    run.add_argument("--path", help="UTF-8 text file (source file)")
    run.add_argument("--max-sentences", type=int, default=20)
    run.add_argument("--batch-size", type=int, default=10)
    run.add_argument("--llm", choices=["fake", "openai"], default="fake")

    report = commands.add_parser("load-report", help="each agent's share of LLM + tool calls")
    report.add_argument("run_id", help="a run id, or 'latest'")

    docs = commands.add_parser("docs", help="regenerate docs/agents.md and docs/messages.md")
    docs.add_argument("--check", action="store_true", help="only check that they are current")
    return parser


def make_backend(settings: Settings, llm: str):
    if llm == "openai":
        return OpenAIBackend(settings.openai_api_key, settings.model, settings.reasoning_effort)
    return FakeLLM(fake_annotation_responder)


def print_summary(summary: RunSummary, settings: Settings, llm: str) -> None:
    print(f"run {summary.run_id}: {summary.status}" + (f" ({summary.reason})" if summary.reason else ""))
    for batch_id, status in summary.batches.items():
        error = summary.batch_errors.get(batch_id)
        print(f"  {batch_id}: {status}" + (f" - {error}" if error else ""))
    entities = sum(report.counts.get("entities", 0) for report in summary.reports)
    review = sum(report.counts.get("review_sentences", 0) for report in summary.reports)
    print(f"steps: {summary.steps}, entities: {entities}, sentences for review: {review}")
    cost = (
        summary.input_tokens * settings.price_input_per_1m
        + summary.output_tokens * settings.price_output_per_1m
    ) / 1_000_000
    model = settings.model if llm == "openai" else "FakeLLM (estimated tokens, no cost)"
    print(
        f"LLM: {summary.llm_calls} calls to {model}, tokens in/out: "
        f"{summary.input_tokens}/{summary.output_tokens}"
        + (f", estimated cost: ${cost:.4f}" if llm == "openai" else "")
    )
    print(f"outputs: {summary.output_dir}")
    print(f"log: {summary.log_path}")


def cmd_run(args, settings: Settings) -> int:
    try:
        task = CollectRequest(
            source=args.source, title=args.title, path=args.path,
            max_sentences=args.max_sentences, batch_size=args.batch_size,
        )
        backend = make_backend(settings, args.llm)
    except (ValueError, KnaError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    summary = run_pipeline(settings, task, backend)
    print_summary(summary, settings, args.llm)
    print()
    print(format_load_report(summary.run_id, compute_load(read_log(summary.log_path))))
    return 0 if summary.status == "completed" else 1


def cmd_load_report(args, settings: Settings) -> int:
    try:
        path = find_log(settings.logs_dir, args.run_id)
    except FileNotFoundError as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    loads = compute_load(read_log(path))
    print(format_load_report(path.stem, loads))
    return 1 if overloaded(loads) else 0


def cmd_docs(args, settings: Settings) -> int:
    docs_dir = settings.home / "docs"
    if args.check:
        stale = outdated_docs(docs_dir)
        for path in stale:
            print(f"out of date: {path.name} (run `kna docs`)")
        return 1 if stale else 0
    for path in write_docs(docs_dir):
        print(f"wrote {path}")
    return 0


def main(argv: list[str] | None = None) -> int:
    # Kazakh text must print on a Windows console too.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    args = build_parser().parse_args(argv)
    settings = load_settings()
    commands = {"run": cmd_run, "load-report": cmd_load_report, "docs": cmd_docs}
    return commands[args.command](args, settings)


if __name__ == "__main__":
    sys.exit(main())
