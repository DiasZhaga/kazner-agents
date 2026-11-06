"""Evaluator: measure each batch, export the dataset, and report on the whole run.

Modes, chosen from the data:
- gold: the sentences have gold labels (kaznerd_test) -> precision/recall/F1 of the final labels
  and of each annotator;
- agreement: two annotators, no gold -> how much mBERT and the LLM agree;
- summary: one annotator, no gold -> counts only.
A batch report carries the mode and counts. Scores are computed once, when the ReportRequest
after the last batch asks for the run report: from the exported sentences.jsonl, so they are
micro-averaged over all sentences, not averaged batch scores.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from kazner_agents.agents.base import Agent
from kazner_agents.agents.specs import EVALUATOR
from kazner_agents.context import RunContext
from kazner_agents.iob2 import iob2_to_spans
from kazner_agents.messages import (
    DocumentBatch,
    EvaluationReport,
    Message,
    NormalizedAnnotation,
    ReportRequest,
    VerificationResult,
)
from kazner_agents.metrics import agreement, compute_metrics
from kazner_agents.state import StateStore

IOB2_FILE = "annotations.iob2"  # KazNERD format: "word LABEL" per line, blank line between
SENTENCES_FILE = "sentences.jsonl"  # one sentence per line: labels, gold, both annotators, provenance
REVIEW_FILE = "review.jsonl"  # sentences a person should check (human-review list)
REPORT_JSON, REPORT_MD = "report.json", "report.md"


def load_batch(state: StateStore, run_id: str, batch_id: str):
    """Tool (state store): the DocumentBatch and every NormalizedAnnotation of a batch."""
    document = state.load_document_batch(run_id, batch_id)
    normalized = state.load_payloads(run_id, batch_id, "NormalizedAnnotation")
    return document, normalized


def build_records(document: DocumentBatch, normalized: list[NormalizedAnnotation],
                  verified: VerificationResult | None) -> list[dict]:
    """One record per sentence: words, final labels, gold, each annotator, review reasons."""
    by_annotator = {n.annotator: {s.sentence_id: s for s in n.sentences} for n in normalized}
    verdicts = {v.sentence_id: v for v in verified.sentences} if verified else {}
    records = []
    for sentence in document.sentences:
        sid = sentence.sentence_id
        annotations = {name: rows[sid].labels for name, rows in by_annotator.items()}
        reasons = [f"{name}: {r}" for name, rows in by_annotator.items() for r in rows[sid].repairs]
        decisions = Counter()
        if verified:
            final = verdicts[sid].labels
            for d in verdicts[sid].decisions:
                decisions[d.decision] += 1
                if d.decision == "escalated":
                    reasons.append(f"escalated words {d.start_word}-{d.end_word}: {d.rationale}")
        else:
            final = next(iter(annotations.values()))
        records.append({
            "sentence_id": sid,
            "doc_id": document.doc_id,
            "source_url": document.source_url,
            "licence": document.licence,
            "words": sentence.words,
            "labels": final,
            "gold": sentence.gold_labels,
            "annotations": annotations,
            "decisions": dict(decisions),
            "review_reasons": reasons,
        })
    return records


def summarise(records: list[dict]) -> dict:
    """Mode and counts of a list of sentence records (no scores: those are computed once per run)."""
    annotators = sorted(records[0]["annotations"]) if records else []
    counts = Counter()
    for r in records:
        counts["sentences"] += 1
        counts["words"] += len(r["words"])
        counts["review_sentences"] += 1 if r["review_reasons"] else 0
        for name, n in r["decisions"].items():
            counts[f"decision:{name}"] += n
        for _, _, etype in iob2_to_spans(r["labels"]):
            counts["entities"] += 1
            counts[f"type:{etype}"] += 1
    if records and all(r["gold"] is not None for r in records):
        mode = "gold"
    elif len(annotators) == 2:
        mode = "agreement"
    else:
        mode = "summary"
    return {"mode": mode, "annotators": annotators, "counts": dict(counts),
            "per_annotator": {}, "final": None, "agreement": None}


def score_exported(output_dir: Path) -> dict:
    """Tool (scorer): score every sentence exported in this run (micro over all sentences)."""
    path = output_dir / SENTENCES_FILE
    records = []
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()
        records = [json.loads(line) for line in lines if line.strip()]
    result = summarise(records)
    if result["mode"] == "gold":
        gold = [r["gold"] for r in records]
        result["final"] = compute_metrics(gold, [r["labels"] for r in records])
        for name in result["annotators"]:
            result["per_annotator"][name] = compute_metrics(gold, [r["annotations"][name] for r in records])
    elif result["mode"] == "agreement":
        first, second = result["annotators"]
        result["agreement"] = agreement([r["annotations"][first] for r in records],
                                        [r["annotations"][second] for r in records])
    return result


def export_batch(output_dir: Path, records: list[dict]) -> list[str]:
    """Tool (file exporter): append the batch to the export files; return the files written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written = [IOB2_FILE, SENTENCES_FILE]
    with (output_dir / IOB2_FILE).open("a", encoding="utf-8", newline="\n") as f:
        for r in records:
            for word, label in zip(r["words"], r["labels"]):
                f.write(f"{word} {label}\n")
            f.write("\n")
    with (output_dir / SENTENCES_FILE).open("a", encoding="utf-8", newline="\n") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    to_review = [r for r in records if r["review_reasons"]]
    if to_review:
        written.append(REVIEW_FILE)
        with (output_dir / REVIEW_FILE).open("a", encoding="utf-8", newline="\n") as f:
            for r in to_review:
                item = {"sentence_id": r["sentence_id"], "words": r["words"], "labels": r["labels"],
                        "reasons": r["review_reasons"]}
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
    return written


def write_report(output_dir: Path, run_id: str, scores: dict) -> list[str]:
    """Tool (file exporter): report.json and a short report.md for people."""
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / REPORT_JSON).write_text(json.dumps(scores, indent=2, ensure_ascii=False) + "\n",
                                          encoding="utf-8", newline="\n")
    lines = [f"# Run {run_id}", "", f"Mode: **{scores['mode']}**", ""]
    if scores["mode"] == "gold":
        lines += ["| Labels | Precision | Recall | F1 | Entities (gold / predicted) |", "|---|---|---|---|---|"]
        rows = [("final (after Verifier)", scores["final"])] if len(scores["per_annotator"]) > 1 else []
        rows += list(scores["per_annotator"].items())
        for name, m in rows:
            lines.append(f"| {name} | {m['precision']:.3f} | {m['recall']:.3f} | {m['f1']:.3f} "
                         f"| {m['n_gold_entities']} / {m['n_pred_entities']} |")
    elif scores["mode"] == "agreement":
        a = scores["agreement"]
        lines.append(f"Agreement of mBERT and LLM: entity F1 {a['entity_f1']:.3f}, "
                     f"token agreement {a['token_agreement']:.3f}.")
    lines += ["", "| Count | Value |", "|---|---|"]
    lines += [f"| {k} | {v} |" for k, v in sorted(scores["counts"].items())]
    (output_dir / REPORT_MD).write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")
    return [REPORT_JSON, REPORT_MD]


def to_report(batch_id: str, scores: dict, files: list[str]) -> EvaluationReport:
    final = scores["final"] or {}
    return EvaluationReport(
        batch_id=batch_id,
        mode=scores["mode"],
        precision=final.get("precision"),
        recall=final.get("recall"),
        f1=final.get("f1"),
        agreement_rate=(scores["agreement"] or {}).get("entity_f1"),
        per_annotator={
            name: {k: m[k] for k in ("precision", "recall", "f1")}
            for name, m in scores["per_annotator"].items()
        },
        counts=scores["counts"],
        export_paths=files,
    )


class Evaluator(Agent):
    spec = EVALUATOR

    def handle(self, payload: Message, ctx: RunContext) -> list[EvaluationReport]:
        if isinstance(payload, ReportRequest):
            return [self.run_report(ctx)]
        verified = payload if isinstance(payload, VerificationResult) else None
        document, normalized = ctx.call_tool(
            self.name, "state_store", load_batch, ctx.state, ctx.run_id, payload.batch_id
        )
        records = build_records(document, normalized, verified)
        files = ctx.call_tool(self.name, "file_exporter", export_batch, ctx.output_dir, records)
        return [to_report(payload.batch_id, summarise(records), files)]

    def run_report(self, ctx: RunContext) -> EvaluationReport:
        scores = ctx.call_tool(self.name, "scorer", score_exported, ctx.output_dir)
        files = ctx.call_tool(self.name, "file_exporter", write_report, ctx.output_dir, ctx.run_id, scores)
        return to_report("run", scores, files)
