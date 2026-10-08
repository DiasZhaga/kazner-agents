"""LLMAnnotator: label the sentences of a batch with an LLM (one call per batch).

The prompt follows the structure from the course lecture on prompt engineering:
Role + Task + Context (entity types) + numbered Steps + Examples (few-shot) + Output format,
with XML-like delimiters around the sentences. The answer is forced into a JSON schema
(`LLMAnswer`) by OpenAI Structured Outputs.
"""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from kazner_agents.agents.base import Agent
from kazner_agents.agents.specs import LLM_ANNOTATOR
from kazner_agents.context import RunContext
from kazner_agents.iob2 import iob2_to_spans
from kazner_agents.labels import DESCRIPTIONS, ENTITY_TYPES, EntityType
from kazner_agents.llm import LLMRequest
from kazner_agents.messages import Annotation, DocumentBatch, SentenceAnnotation, Span

FEWSHOT_PATH = Path(__file__).resolve().parents[1] / "data" / "fewshot_kaznerd.jsonl"


# --- the schema of the LLM answer (separate from the messages; strict: all fields required) --


class LLMEntity(BaseModel):
    model_config = ConfigDict(extra="forbid")
    start_word: int
    end_word: int
    type: EntityType
    text: str


class LLMSentence(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sentence_id: str
    entities: list[LLMEntity]


class LLMAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    sentences: list[LLMSentence]


# --- prompt --------------------------------------------------------------------------------


def load_examples(path: Path = FEWSHOT_PATH) -> list[dict]:
    """Tool: the few-shot example store (KazNERD training sentences, CC BY 4.0)."""
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def numbered(words: list[str]) -> str:
    return " ".join(f"{i}:{word}" for i, word in enumerate(words))


def build_instructions(examples: list[dict]) -> str:
    types = "\n".join(f"- {t}: {DESCRIPTIONS[t]}" for t in ENTITY_TYPES)
    shots = []
    for number, example in enumerate(examples, start=1):
        words = example["words"]
        entities = [
            {"start_word": s, "end_word": e, "type": t, "text": " ".join(words[s : e + 1])}
            for s, e, t in iob2_to_spans(example["labels"])
        ]
        answer = {"sentence_id": f"ex{number}", "entities": entities}
        shots.append(
            f'<sentence id="ex{number}">{numbered(words)}</sentence>\n'
            f"<answer>{json.dumps(answer, ensure_ascii=False)}</answer>"
        )
    return f"""## Role
You are an expert annotator of named entities in Kazakh text. You follow the KazNERD annotation
guidelines.

## Task
For every sentence you are given, find all named entities and return each one as a span of
word indices with its entity type.

## Entity types (KazNERD, 25 types)
{types}

## Steps
1. Read the sentence. Every word is shown with its index: "3:Семейде" means word 3 is "Семейде".
2. Find each entity. A span covers whole words; Kazakh suffixes are part of the word
   ("Семейде" is one GPE word).
3. Give start_word and end_word (both inclusive), using only the indices shown.
4. Copy the words of the span into "text" exactly as written, separated by spaces.
5. Spans must not overlap. Words outside entities are not listed.
6. Return every sentence_id you were given, with an empty list if it has no entities.

## Examples
{chr(10).join(shots)}

## Output format
JSON matching the schema: {{"sentences": [{{"sentence_id": ..., "entities": [{{"start_word": ...,
"end_word": ..., "type": ..., "text": ...}}]}}]}}"""


def build_user_prompt(batch: DocumentBatch) -> str:
    lines = [
        f'<sentence id="{s.sentence_id}">{numbered(s.words)}</sentence>' for s in batch.sentences
    ]
    return "<sentences>\n" + "\n".join(lines) + "\n</sentences>"


# --- FakeLLM answers -----------------------------------------------------------------------


def heuristic_entities(words: list[str]) -> list[dict]:
    """A crude, deterministic stand-in for the LLM, for offline runs and tests. Not real NER:

    a 4-digit number before 'жыл...' -> DATE; a number before '%' -> PERCENTAGE; any other
    number -> CARDINAL; capitalised words not at the start of the sentence -> PERSON if two or
    more in a row, otherwise GPE.
    """
    entities = []
    i = 0
    while i < len(words):
        word, end, etype = words[i], i, None
        following = words[i + 1] if i + 1 < len(words) else ""
        if word.isdigit() and len(word) == 4 and following.lower().startswith("жыл"):
            end, etype = i + 1, "DATE"
        elif word.isdigit() and following == "%":
            end, etype = i + 1, "PERCENTAGE"
        elif word.isdigit():
            etype = "CARDINAL"
        elif i > 0 and word[0].isupper() and word.replace("-", "").isalpha():
            while end + 1 < len(words) and words[end + 1][0].isupper() and words[end + 1].isalpha():
                end += 1
            etype = "PERSON" if end > i else "GPE"
        if etype:
            text = " ".join(words[i : end + 1])
            entities.append({"start_word": i, "end_word": end, "type": etype, "text": text})
        i = end + 1
    return entities


def fake_annotation_responder(request: LLMRequest) -> dict:
    """FakeLLM responder: reads the sentences from request.metadata, not from the prompt."""
    return {
        "sentences": [
            {"sentence_id": s["sentence_id"], "entities": heuristic_entities(s["words"])}
            for s in request.metadata["sentences"]
        ]
    }


# --- the agent -----------------------------------------------------------------------------


class LLMAnnotator(Agent):
    spec = LLM_ANNOTATOR

    def __init__(self, examples_path: Path = FEWSHOT_PATH):
        self.examples_path = examples_path
        self.examples: list[dict] | None = None  # loaded once per run (agents are per run)

    def handle(self, payload: DocumentBatch, ctx: RunContext) -> list[Annotation]:
        if self.examples is None:
            self.examples = ctx.call_tool(
                self.name, "fewshot_store", load_examples, self.examples_path
            )
        request = LLMRequest(
            instructions=build_instructions(self.examples),
            user=build_user_prompt(payload),
            schema=LLMAnswer,
            metadata={
                "sentences": [
                    {"sentence_id": s.sentence_id, "words": s.words} for s in payload.sentences
                ]
            },
        )
        answer: LLMAnswer = ctx.llm.complete(self.name, "annotate_batch", request)

        answered = {s.sentence_id: s for s in answer.sentences}
        sentences = []
        for sentence in payload.sentences:
            found = answered.get(sentence.sentence_id)
            notes, spans = [], []
            if found is None:
                notes.append("sentence missing from the LLM answer")
            else:
                for entity in found.entities:
                    if entity.start_word < 0 or entity.end_word < 0:
                        notes.append(f"negative word index in {entity.type} span dropped")
                        continue
                    spans.append(
                        Span(
                            start_word=entity.start_word,
                            end_word=entity.end_word,
                            type=entity.type,
                            text=entity.text,
                        )
                    )
            sentences.append(
                SentenceAnnotation(
                    sentence_id=sentence.sentence_id,
                    words=sentence.words,
                    entities=spans,
                    notes=notes,
                )
            )
        return [Annotation(batch_id=payload.batch_id, annotator="llm", sentences=sentences)]
