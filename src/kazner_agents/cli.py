"""Command-line interface: `kna run`, `kna review`, `kna scenarios`, `kna load-report`, `kna docs`."""

from __future__ import annotations

import argparse
import sys

from kazner_agents import __version__
from kazner_agents.agents.pre_annotator import load_mbert
from kazner_agents.config import Settings, load_settings
from kazner_agents.docs_gen import outdated_docs, write_docs
from kazner_agents.errors import KnaError
from kazner_agents.fakes import fake_responder, fake_tagger
from kazner_agents.llm import FakeLLM, OpenAIBackend
from kazner_agents.load_report import compute_load, find_log, format_load_report, overloaded
from kazner_agents.messages import CollectRequest
from kazner_agents.orchestrator import ANNOTATOR_AGENTS, RunSummary, run_pipeline
from kazner_agents.run_log import read_log
from kazner_agents.scenarios import load_scenarios, run_scenario
from kazner_agents.state import StateStore


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="kna", description="Kazakh NER annotation agents")
    parser.add_argument("--version", action="version", version=f"kna {__version__}")
    commands = parser.add_subparsers(dest="command", required=True)

    run = commands.add_parser("run", help="annotate a Wikipedia article, a text file or a KazNERD sample")
    run.add_argument("--source", choices=["wikipedia", "file", "kaznerd_test"], required=True)
    run.add_argument("--title", help="Kazakh Wikipedia article title (source wikipedia)")
    run.add_argument("--path", help="UTF-8 text file (source file)")
    run.add_argument("--max-sentences", type=int, default=20)
    run.add_argument("--batch-size", type=int, default=10)
    run.add_argument("--seed", type=int, default=13, help="sample seed (source kaznerd_test)")
    run.add_argument("--annotators", default="mbert,llm", help="mbert, llm, or mbert,llm (default)")
    run.add_argument("--llm", choices=["fake", "openai"], default="fake")
    run.add_argument("--tagger", choices=["fake", "mbert"], default="mbert",
                     help="the PreAnnotator's model: the fine-tuned mBERT or the offline FakeTagger")

    review = commands.add_parser("review", help="show the human-review queue of a run")
    review.add_argument("run_id")

    scenarios = commands.add_parser("scenarios", help="run the test scenarios offline")
    scenarios.add_argument("--only", help="run only scenarios whose name contains this text")

    report = commands.add_parser("load-report", help="each agent's share of LLM + tool calls")
    report.add_argument("run_id", help="a run id, or 'latest'")

    docs = commands.add_parser("docs", help="regenerate docs/agents.md and docs/messages.md")
    docs.add_argument("--check", action="store_true", help="only check that they are current")
    return parser


def make_backend(settings: Settings, llm: str):
    if llm == "openai":
        return OpenAIBackend(settings.openai_api_key, settings.model, settings.reasoning_effort)
    return FakeLLM(fake_responder)


def print_summary(summary: RunSummary, settings: Settings, llm: str) -> None:
    print(f"run {summary.run_id}: {summary.status}" + (f" ({summary.reason})" if summary.reason else ""))
    print(f"annotators: {', '.join(summary.annotators)}")
    for batch_id, status in summary.batches.items():
        error = summary.batch_errors.get(batch_id)
        print(f"  {batch_id}: {status}" + (f" - {error}" if error else ""))
    if "run-report" in summary.batch_errors:
        print(f"  run report: {summary.batch_errors['run-report']}")
    report = summary.run_report
    if report is not None:
        counts = report.counts
        print(f"steps: {summary.steps}, sentences: {counts.get('sentences', 0)}, "
              f"entities: {counts.get('entities', 0)}, sentences for review: "
              f"{counts.get('review_sentences', 0)}")
        decisions = {k.split(":")[1]: v for k, v in counts.items() if k.startswith("decision:")}
        if decisions:
            print("verifier decisions: " + ", ".join(f"{k} {v}" for k, v in sorted(decisions.items())))
        if report.mode == "gold":
            if report.f1 is not None and len(report.per_annotator) > 1:
                print(f"gold F1  final {report.f1:.3f}  (P {report.precision:.3f}, R {report.recall:.3f})")
            for name, m in sorted(report.per_annotator.items()):
                print(f"gold F1  {name:<5} {m['f1']:.3f}  (P {m['precision']:.3f}, R {m['recall']:.3f})")
        elif report.mode == "agreement":
            print(f"agreement of mBERT and LLM: entity F1 {report.agreement_rate:.3f}")
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
    annotators = tuple(a.strip() for a in args.annotators.split(",") if a.strip())
    try:
        task = CollectRequest(
            source=args.source, title=args.title, path=args.path, seed=args.seed,
            max_sentences=args.max_sentences, batch_size=args.batch_size,
        )
        if not annotators or any(a not in ANNOTATOR_AGENTS for a in annotators):
            raise ValueError(f"--annotators: choose from {', '.join(ANNOTATOR_AGENTS)}")
        if "mbert" in annotators and args.tagger == "mbert" and not settings.mbert_path.exists():
            raise ValueError(f"mBERT checkpoint not found at {settings.mbert_path} (set KNA_MBERT_PATH)")
        backend = make_backend(settings, args.llm)
    except (ValueError, KnaError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    make_tagger = load_mbert if args.tagger == "mbert" else fake_tagger
    summary = run_pipeline(settings, task, backend, annotators=annotators, make_tagger=make_tagger)
    print_summary(summary, settings, args.llm)
    print()
    print(format_load_report(summary.run_id, compute_load(read_log(summary.log_path))))
    return 0 if summary.status == "completed" else 1


def cmd_review(args, settings: Settings) -> int:
    state = StateStore(settings.state_path)
    items = state.get_review_items(args.run_id)
    state.close()
    if not items:
        print(f"no open review items for run {args.run_id}")
        return 0
    print(f"{len(items)} regions for human review in run {args.run_id}:")
    for item in items:
        print(f"\n{item['sentence_id']} words {item['start_word']}-{item['end_word']}: {item['words']}")
        print(f"  A (mbert): {item['mbert']}")
        print(f"  B (llm):   {item['llm']}")
        print(f"  why: {item['reason']}")
    return 0


def cmd_scenarios(args, settings: Settings) -> int:
    scenarios = [s for s in load_scenarios() if not args.only or args.only in s["name"]]
    failed = 0
    for scenario in scenarios:
        _, checks = run_scenario(scenario)
        ok = all(passed for _, passed, _ in checks)
        failed += not ok
        print(f"{'PASS' if ok else 'FAIL'}  {scenario['name']}: {scenario['description']}")
        for name, passed, detail in checks:
            print(f"      {'ok  ' if passed else 'FAIL'} {name}: {detail}")
    print(f"\n{len(scenarios) - failed} of {len(scenarios)} scenarios passed")
    return 1 if failed else 0


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
    commands = {
        "run": cmd_run, "review": cmd_review, "scenarios": cmd_scenarios,
        "load-report": cmd_load_report, "docs": cmd_docs,
    }
    return commands[args.command](args, settings)


if __name__ == "__main__":
    sys.exit(main())
