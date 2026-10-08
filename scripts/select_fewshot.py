"""Select few-shot examples for the LLMAnnotator from the KazNERD training split.

Deterministic greedy cover: go through the 25 entity types from the rarest to the most frequent;
if no chosen sentence has that type yet, add the shortest valid sentence (7-20 words) that
contains it. The result covers all 25 types with a small number of sentences.
Sentences with phone-number-like tokens (7+ digits) are skipped, so that no personal contact
data ends up in the repository.

KazNERD (Yeshpanov, Khassanov, Varol, LREC 2022) is released under CC BY 4.0.

Usage:
    python scripts/select_fewshot.py [--train ../ner-project/data/IOB2_train.txt]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from kazner_agents.iob2 import validate_iob2
from kazner_agents.labels import ENTITY_TYPES

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_TRAIN = ROOT.parent / "ner-project" / "data" / "IOB2_train.txt"
DEFAULT_OUT = ROOT / "src" / "kazner_agents" / "data" / "fewshot_kaznerd.jsonl"


def read_iob2(path: Path) -> list[tuple[list[str], list[str]]]:
    sentences, words, labels = [], [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            if words:
                sentences.append((words, labels))
            words, labels = [], []
        elif len(parts) == 2:
            words.append(parts[0])
            labels.append(parts[1])
    if words:
        sentences.append((words, labels))
    return sentences


def types_in(labels: list[str]) -> set[str]:
    return {label[2:] for label in labels if label.startswith("B-")}


def looks_like_phone_number(word: str) -> bool:
    return sum(ch.isdigit() for ch in word) >= 7


def select(sentences, min_words: int = 7, max_words: int = 20) -> list[int]:
    frequency = Counter(t for _, labels in sentences for t in types_in(labels))
    covered: set[str] = set()
    chosen: list[int] = []
    for etype in sorted(ENTITY_TYPES, key=lambda t: (frequency[t], t)):
        if etype in covered:
            continue
        candidates = [
            i for i, (words, labels) in enumerate(sentences)
            if etype in types_in(labels)
            and not validate_iob2(labels)
            and not any(looks_like_phone_number(word) for word in words)
        ]
        in_range = [i for i in candidates if min_words <= len(sentences[i][0]) <= max_words]
        pool = in_range or candidates
        best = min(pool, key=lambda i: (len(sentences[i][0]), i))
        chosen.append(best)
        covered |= types_in(sentences[best][1])
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--train", type=Path, default=DEFAULT_TRAIN)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    sentences = read_iob2(args.train)
    chosen = select(sentences)
    with args.out.open("w", encoding="utf-8", newline="\n") as f:
        for index in chosen:
            words, labels = sentences[index]
            record = {"kaznerd_train_index": index, "words": words, "labels": labels}
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    print(f"wrote {len(chosen)} examples to {args.out}")


if __name__ == "__main__":
    main()
