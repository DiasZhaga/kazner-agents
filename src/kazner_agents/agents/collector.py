"""Collector: fetch the text, split it into sentences and words, cut it into batches."""

from __future__ import annotations

import random
import re
from pathlib import Path

from kazner_agents import wikipedia
from kazner_agents.agents.base import Agent
from kazner_agents.agents.specs import COLLECTOR
from kazner_agents.context import RunContext
from kazner_agents.errors import AgentError
from kazner_agents.messages import CollectRequest, DocumentBatch, Sentence
from kazner_agents.text_split import split_sentences

MIN_WORDS = 3  # shorter "sentences" are usually list items or fragments

KAZNERD_URL = (
    "https://github.com/IS2AI/KazNERD/tree/bd4333d0f5952b9fafb2ef2ac2fefa0ad3c0333f/KazNERD"
)
KAZNERD_LICENCE = "CC BY 4.0"
KAZNERD_TEST_FILE = "IOB2_test.txt"


def read_text_file(path: str) -> str:
    """Tool: read a UTF-8 text file."""
    try:
        return Path(path).read_text(encoding="utf-8")
    except OSError as exc:
        raise AgentError(f"cannot read file {path!r}: {exc.strerror or exc}") from exc


def read_kaznerd_sample(path: Path, size: int, seed: int) -> list[tuple[int, list[str], list[str]]]:
    """Tool: a deterministic sample of KazNERD sentences as (index in file, words, gold labels).

    The same seed always gives the same sentences; they are returned in file order.
    """
    try:
        lines = Path(path).read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        raise AgentError(
            f"cannot read KazNERD file {str(path)!r} (set KNA_KAZNERD_DIR): {exc.strerror or exc}"
        ) from exc
    sentences, words, labels = [], [], []
    for line in lines + [""]:  # the extra empty line closes the last sentence
        parts = line.split()
        if not parts:
            if words:
                sentences.append((words, labels))
            words, labels = [], []
        elif len(parts) >= 2:
            words.append(parts[0])
            labels.append("O" if parts[-1] == "0" else parts[-1])  # the pilot wrote O as 0
    picked = sorted(random.Random(seed).sample(range(len(sentences)), min(size, len(sentences))))
    return [(index, *sentences[index]) for index in picked]


def slug(text: str) -> str:
    return re.sub(r"\W+", "-", text.lower()).strip("-") or "doc"


class Collector(Agent):
    spec = COLLECTOR

    def __init__(self, fetch_article=wikipedia.fetch_article):
        self.fetch_article = fetch_article  # replaceable in tests (no network)

    def handle(self, payload: CollectRequest, ctx: RunContext) -> list[DocumentBatch]:
        if payload.source == "kaznerd_test":
            sample = ctx.call_tool(
                self.name, "kaznerd_reader", read_kaznerd_sample,
                ctx.settings.kaznerd_dir / KAZNERD_TEST_FILE, payload.max_sentences, payload.seed,
            )
            doc_id, source_url, licence = "kaznerd-test", KAZNERD_URL, KAZNERD_LICENCE
            sentences = [
                Sentence(sentence_id=f"{doc_id}-s{index:05d}", words=words, gold_labels=labels)
                for index, words, labels in sample
            ]
        else:
            if payload.source == "wikipedia":
                article = ctx.call_tool(
                    self.name, "wikipedia_api", self.fetch_article,
                    payload.title, ctx.settings.user_agent, retry=True,
                )
                text, source_url, licence = article.text, article.url, wikipedia.LICENCE
                doc_id = "kkwiki-" + slug(article.title)
            else:  # file
                text = ctx.call_tool(self.name, "file_reader", read_text_file, payload.path)
                name = Path(payload.path).name  # only the file name: no local paths in exports
                source_url, licence = f"file:{name}", "unknown (local file)"
                doc_id = "file-" + slug(Path(name).stem)

            all_sentences = ctx.call_tool(self.name, "sentence_splitter", split_sentences, text)
            kept = [words for words in all_sentences if len(words) >= MIN_WORDS]
            kept = kept[: payload.max_sentences]
            sentences = [
                Sentence(sentence_id=f"{doc_id}-s{index:04d}", words=words)
                for index, words in enumerate(kept)
            ]

        if not sentences:
            raise AgentError("no sentences with at least 3 words were found in the text")
        batches = []
        for start in range(0, len(sentences), payload.batch_size):
            batches.append(
                DocumentBatch(
                    batch_id=f"b{len(batches) + 1:03d}",
                    doc_id=doc_id,
                    source_url=source_url,
                    licence=licence,
                    sentences=sentences[start : start + payload.batch_size],
                )
            )
        return batches
