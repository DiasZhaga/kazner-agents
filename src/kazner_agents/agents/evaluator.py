"""Evaluator: measure a batch and export the dataset files.

A2 implements the `summary` mode (one annotator, no gold): counts per entity type, number of
repairs, and the human-review list. The `gold` and `agreement` modes come in Assignment 4.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from kazner_agents.agents.base import Agent
from kazner_agents.agents.specs import EVALUATOR
from kazner_agents.context import RunContext
from kazner_agents.iob2 import iob2_to_spans
from kazner_agents.messages import DocumentBatch, EvaluationReport, NormalizedAnnotation

IOB2_FILE = "annotations.iob2"  # KazNERD format: "word LABEL" per line, blank line between
SENTENCES_FILE = "sentences.jsonl"  # one sentence per line, with provenance
REVIEW_FILE = "review.jsonl"  # sentences a person should check (human-review list)


def summarise(normalized: NormalizedAnnotation) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for sentence in normalized.sentences:
        counts["sentences"] += 1
        counts["words"] += len(sentence.words)
        counts["repairs"] += len(sentence.repairs)
        counts["review_sentences"] += 1 if sentence.repairs else 0
        for _, _, etype in iob2_to_spans(sentence.labels):
            counts["entities"] += 1
            counts[f"type:{etype}"] += 1
    return dict(counts)


def export_batch(
    output_dir: Path, document: DocumentBatch, normalized: NormalizedAnnotation
) -> list[str]:
    """Tool: append the batch to the export files; return the names of the files written."""
    output_dir.mkdir(parents=True, exist_ok=True)
    written = [IOB2_FILE, SENTENCES_FILE]
    with (output_dir / IOB2_FILE).open("a", encoding="utf-8", newline="\n") as f:
        for sentence in normalized.sentences:
            for word, label in zip(sentence.words, sentence.labels):
                f.write(f"{word} {label}\n")
            f.write("\n")
    with (output_dir / SENTENCES_FILE).open("a", encoding="utf-8", newline="\n") as f:
        for sentence in normalized.sentences:
            record = {
                "sentence_id": sentence.sentence_id,
                "doc_id": document.doc_id,
                "source_url": document.source_url,
                "licence": document.licence,
                "annotator": normalized.annotator,
                "words": sentence.words,
                "labels": sentence.labels,
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    to_review = [s for s in normalized.sentences if s.repairs]
    if to_review:
        written.append(REVIEW_FILE)
        with (output_dir / REVIEW_FILE).open("a", encoding="utf-8", newline="\n") as f:
            for sentence in to_review:
                record = {
                    "sentence_id": sentence.sentence_id,
                    "words": sentence.words,
                    "labels": sentence.labels,
                    "reasons": sentence.repairs,
                }
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
    return written


class Evaluator(Agent):
    spec = EVALUATOR

    def handle(self, payload: NormalizedAnnotation, ctx: RunContext) -> list[EvaluationReport]:
        # Provenance (source URL, licence) is in the DocumentBatch kept in the state store.
        document = ctx.call_tool(
            self.name, "state_store", ctx.state.load_document_batch, ctx.run_id, payload.batch_id
        )
        files = ctx.call_tool(
            self.name, "file_exporter", export_batch, ctx.output_dir, document, payload
        )
        return [
            EvaluationReport(
                batch_id=payload.batch_id,
                mode="summary",
                counts=summarise(payload),
                export_paths=files,
            )
        ]
