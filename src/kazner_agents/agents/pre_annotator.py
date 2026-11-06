"""PreAnnotator: label every word with the fine-tuned mBERT from the pilot study.

The tagger is behind a small interface so that tests use the deterministic FakeTagger and never
load torch or a model:

- MBertTagger: the real model (optional extra `pip install -e ".[mbert]"`).
- FakeTagger: a crude capitalisation/number heuristic, NOT real NER, different on purpose from
  the FakeLLM heuristic so that offline runs give the Verifier real disagreements to settle.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Protocol

from kazner_agents.agents.base import Agent
from kazner_agents.agents.specs import PRE_ANNOTATOR
from kazner_agents.config import Settings
from kazner_agents.context import RunContext
from kazner_agents.errors import AgentError
from kazner_agents.messages import Annotation, DocumentBatch, SentenceAnnotation

MAX_TOKENS = 512  # BERT's position limit; words beyond it get O


class Tagger(Protocol):
    def tag(self, sentences: list[list[str]]) -> list[tuple[list[str], int]]:
        """IOB2 labels for every word, and how many words were cut off (labelled O)."""
        ...


class FakeTagger:
    """Deterministic offline stand-in for mBERT. Not real NER."""

    def tag(self, sentences: list[list[str]]) -> list[tuple[list[str], int]]:
        return [(fake_labels(words), 0) for words in sentences]


def fake_labels(words: list[str]) -> list[str]:
    """Numbers: a 4-digit number -> DATE, others -> CARDINAL. Runs of capitalised words, the
    first word of the sentence included: two or more -> ORGANISATION, one -> GPE."""
    labels = ["O"] * len(words)
    i = 0
    while i < len(words):
        word = words[i]
        if word.isdigit():
            labels[i] = "B-DATE" if len(word) == 4 else "B-CARDINAL"
            i += 1
        elif word[0].isupper() and word.replace("-", "").isalpha():
            end = i
            while end + 1 < len(words) and words[end + 1][0].isupper() and words[end + 1].isalpha():
                end += 1
            etype = "ORGANISATION" if end > i else "GPE"
            labels[i] = f"B-{etype}"
            for j in range(i + 1, end + 1):
                labels[j] = f"I-{etype}"
            i = end + 1
        else:
            i += 1
    return labels


class MBertTagger:
    """The fine-tuned mBERT checkpoint (a Hugging Face token-classification model)."""

    def __init__(self, path: Path):
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer
            from transformers.utils import logging as hf_logging
        except ImportError as exc:
            raise AgentError('mBERT needs torch and transformers: pip install -e ".[mbert]"') from exc
        hf_logging.disable_progress_bar()  # no download/loading bars in the CLI output
        hf_logging.set_verbosity_error()
        if not Path(path).exists():
            raise AgentError(f"mBERT checkpoint not found at {path} (set KNA_MBERT_PATH)")
        self.torch = torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = AutoModelForTokenClassification.from_pretrained(path).to(self.device).eval()
        self.id2label = self.model.config.id2label

    def tag(self, sentences: list[list[str]]) -> list[tuple[list[str], int]]:
        encoded = self.tokenizer(
            sentences, is_split_into_words=True, truncation=True, max_length=MAX_TOKENS,
            padding=True, return_tensors="pt",
        )
        with self.torch.no_grad():
            logits = self.model(**encoded.to(self.device)).logits
        predictions = logits.argmax(dim=-1).cpu().tolist()
        results = []
        for i, words in enumerate(sentences):
            labels = ["O"] * len(words)
            labelled = set()
            # As in training: the label of a word is the label of its FIRST sub-token.
            for position, word_id in enumerate(encoded.word_ids(i)):
                if word_id is None or word_id in labelled:
                    continue
                labelled.add(word_id)
                labels[word_id] = self.id2label[predictions[i][position]]
            results.append((labels, len(words) - len(labelled)))
        return results


def load_mbert(settings: Settings) -> Tagger:
    return MBertTagger(settings.mbert_path)


class PreAnnotator(Agent):
    spec = PRE_ANNOTATOR

    def __init__(self, make_tagger: Callable[[Settings], Tagger] = load_mbert):
        self.make_tagger = make_tagger
        self.tagger: Tagger | None = None  # loaded once per run (agents are per run)

    def handle(self, payload: DocumentBatch, ctx: RunContext) -> list[Annotation]:
        if self.tagger is None:
            self.tagger = ctx.call_tool(self.name, "mbert_load", self.make_tagger, ctx.settings)
        results = ctx.call_tool(
            self.name, "mbert_inference", self.tagger.tag, [s.words for s in payload.sentences]
        )
        sentences = []
        for sentence, (labels, cut_off) in zip(payload.sentences, results):
            notes = []
            if cut_off:
                notes.append(f"{cut_off} words beyond {MAX_TOKENS} sub-tokens were labelled O")
            sentences.append(SentenceAnnotation(
                sentence_id=sentence.sentence_id, words=sentence.words, labels=labels, notes=notes,
            ))
        return [Annotation(batch_id=payload.batch_id, annotator="mbert", sentences=sentences)]
