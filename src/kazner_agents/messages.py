"""Typed messages exchanged by the agents (DESIGN.md section 5).

Every message is an `Envelope` whose `payload` is one of the payload models below. All models
are pydantic v2: they validate their fields on creation and convert to/from JSON.
Models for agents that come in Assignment 4 (VerificationResult) are defined already.
"""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Literal, Union

from pydantic import BaseModel, ConfigDict, Field, model_validator

from kazner_agents.labels import EntityType, is_valid_label


class Message(BaseModel):
    """Base for all message models: unknown fields are an error, not silently ignored."""

    model_config = ConfigDict(extra="forbid")


# --- building blocks -----------------------------------------------------------------------


class Sentence(Message):
    sentence_id: str = Field(description="Unique id, e.g. 'kk-wiki-abai-s0003'")
    words: list[str] = Field(min_length=1, description="The sentence split into words")
    gold_labels: list[str] | None = Field(
        default=None, description="Gold IOB2 labels (only for the kaznerd_test source, A4)"
    )

    @model_validator(mode="after")
    def gold_matches_words(self) -> Sentence:
        if self.gold_labels is not None:
            if len(self.gold_labels) != len(self.words):
                raise ValueError("gold_labels must have one label per word")
            bad = [label for label in self.gold_labels if not is_valid_label(label)]
            if bad:
                raise ValueError(f"unknown gold labels: {bad}")
        return self


class Span(Message):
    start_word: int = Field(ge=0, description="Index of the first word (inclusive)")
    end_word: int = Field(ge=0, description="Index of the last word (inclusive)")
    type: EntityType = Field(description="One of the 25 KazNERD entity types")
    text: str | None = Field(default=None, description="The words of the span, for checking")


class SentenceAnnotation(Message):
    sentence_id: str
    words: list[str] = Field(min_length=1)
    entities: list[Span] | None = Field(default=None, description="Spans (LLM annotator)")
    labels: list[str] | None = Field(default=None, description="IOB2 labels (mBERT, A4)")
    notes: list[str] = Field(default_factory=list, description="Problems seen by the annotator")

    @model_validator(mode="after")
    def exactly_one_kind(self) -> SentenceAnnotation:
        if (self.entities is None) == (self.labels is None):
            raise ValueError("exactly one of 'entities' or 'labels' must be given")
        if self.labels is not None and len(self.labels) != len(self.words):
            raise ValueError("labels must have one label per word")
        return self


class SentenceLabels(Message):
    sentence_id: str
    words: list[str] = Field(min_length=1)
    labels: list[str] = Field(description="Valid IOB2 labels, one per word")
    repairs: list[str] = Field(default_factory=list, description="What the normaliser fixed")

    @model_validator(mode="after")
    def labels_are_valid(self) -> SentenceLabels:
        if len(self.labels) != len(self.words):
            raise ValueError("labels must have one label per word")
        bad = [label for label in self.labels if not is_valid_label(label)]
        if bad:
            raise ValueError(f"unknown labels: {bad}")
        return self


class Decision(Message):
    start_word: int = Field(ge=0)
    end_word: int = Field(ge=0)
    decision: Literal["agreed", "adjudicated", "escalated"]
    rationale: str


class SentenceVerdict(Message):
    sentence_id: str
    words: list[str] = Field(min_length=1)
    labels: list[str] = Field(description="Final IOB2 labels")
    decisions: list[Decision] = Field(default_factory=list)


# --- payloads ------------------------------------------------------------------------------


class CollectRequest(Message):
    """Orchestrator -> Collector: what text to fetch and how to batch it."""

    source: Literal["wikipedia", "file", "kaznerd_test"]
    title: str | None = Field(default=None, description="Wikipedia article title")
    path: str | None = Field(default=None, description="Path of a UTF-8 text file")
    max_sentences: int = Field(default=20, ge=1, le=1000)
    batch_size: int = Field(default=10, ge=1, le=50)

    @model_validator(mode="after")
    def source_has_its_argument(self) -> CollectRequest:
        if self.source == "wikipedia" and not self.title:
            raise ValueError("source 'wikipedia' needs a title")
        if self.source == "file" and not self.path:
            raise ValueError("source 'file' needs a path")
        return self


class DocumentBatch(Message):
    """Collector -> annotators: a batch of sentences with provenance."""

    batch_id: str
    doc_id: str
    source_url: str
    licence: str
    sentences: list[Sentence] = Field(min_length=1)


class Annotation(Message):
    """Annotator -> BoundaryNormalizer: entities for every sentence of a batch."""

    batch_id: str
    annotator: Literal["mbert", "llm"]
    sentences: list[SentenceAnnotation]


class NormalizedAnnotation(Message):
    """BoundaryNormalizer -> Evaluator (A4: Verifier): valid IOB2 labels per word."""

    batch_id: str
    annotator: Literal["mbert", "llm"]
    sentences: list[SentenceLabels]
    repairs_applied: int = Field(ge=0, description="Number of repairs in the whole batch")


class VerificationResult(Message):
    """Verifier -> Evaluator (A4): final labels and a decision for every disagreement."""

    batch_id: str
    sentences: list[SentenceVerdict]


class EvaluationReport(Message):
    """Evaluator -> Orchestrator: quality numbers for a batch and the exported files."""

    batch_id: str
    mode: Literal["gold", "agreement", "summary"]
    precision: float | None = None
    recall: float | None = None
    f1: float | None = None
    agreement_rate: float | None = None
    counts: dict[str, int] = Field(default_factory=dict)
    export_paths: list[str] = Field(default_factory=list)


Payload = Union[
    CollectRequest,
    DocumentBatch,
    Annotation,
    NormalizedAnnotation,
    VerificationResult,
    EvaluationReport,
]

PAYLOAD_TYPES: dict[str, type[Message]] = {
    cls.__name__: cls
    for cls in (
        CollectRequest,
        DocumentBatch,
        Annotation,
        NormalizedAnnotation,
        VerificationResult,
        EvaluationReport,
    )
}


def _now() -> datetime:
    return datetime.now(timezone.utc)


def _new_id() -> str:
    return str(uuid.uuid4())


class Envelope(Message):
    """The wrapper around every payload: who sent what to whom, and when."""

    msg_id: str = Field(default_factory=_new_id)
    run_id: str
    step: int = Field(ge=0)
    sender: str
    recipient: str
    type: str = Field(description="Class name of the payload")
    created_at: datetime = Field(default_factory=_now)
    payload: Payload

    @model_validator(mode="before")
    @classmethod
    def parse_payload_by_type(cls, data):
        # When reading JSON, the payload is a plain dict: build it with the class named by
        # `type`, so that e.g. an Annotation can never be mistaken for another payload.
        if isinstance(data, dict) and isinstance(data.get("payload"), dict):
            payload_cls = PAYLOAD_TYPES.get(data.get("type"))
            if payload_cls is None:
                raise ValueError(f"unknown message type: {data.get('type')!r}")
            data = {**data, "payload": payload_cls.model_validate(data["payload"])}
        return data

    @model_validator(mode="after")
    def type_matches_payload(self) -> Envelope:
        if type(self.payload).__name__ != self.type:
            raise ValueError(
                f"type {self.type!r} does not match payload {type(self.payload).__name__}"
            )
        return self
