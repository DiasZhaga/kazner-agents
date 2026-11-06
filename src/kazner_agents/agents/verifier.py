"""Verifier: compare the mBERT and LLM annotations of a batch; agree, adjudicate or escalate.

Per sentence:
1. entities found by both annotators, (start, end, type) equal, are AGREED;
2. the other entities form disagreement REGIONS: groups of overlapping spans;
3. GUIDELINE RULES settle a region when both spans have the same type and differ only by
   punctuation at the edge (KazNERD leaves quotes and brackets around a name outside it);
4. the rest goes to ONE LLM call per batch; the options are shown as "A" and "B" without saying
   which annotator wrote them, so the LLM does not simply pick its own annotation;
5. "unsure", or a failed adjudication call, ESCALATES the region to the human-review queue; until
   a person decides, the mBERT version is kept (it is trained on the KazNERD guidelines).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from pydantic import BaseModel, ConfigDict

from kazner_agents.agents.base import Agent
from kazner_agents.agents.llm_annotator import numbered
from kazner_agents.agents.specs import VERIFIER
from kazner_agents.context import RunContext
from kazner_agents.errors import AgentError, KnaError, LimitExceeded
from kazner_agents.iob2 import iob2_to_spans
from kazner_agents.labels import DESCRIPTIONS, ENTITY_TYPES
from kazner_agents.llm import LLMRequest
from kazner_agents.messages import (
    Decision,
    NormalizedAnnotation,
    SentenceVerdict,
    VerificationResult,
)

SpanT = tuple[int, int, str]  # (start word, end word inclusive, entity type)
PROVISIONAL = "mbert"  # whose labels an escalated region keeps until a person decides


@dataclass
class Region:
    region_id: str
    sentence_index: int
    start: int
    end: int
    mbert: list[SpanT] = field(default_factory=list)
    llm: list[SpanT] = field(default_factory=list)


# --- comparing the two annotations ---------------------------------------------------------


def compare(sentence_index: int, mbert_labels: list[str], llm_labels: list[str]):
    """Return (agreed spans, disagreement regions) for one sentence."""
    mbert_spans = set(iob2_to_spans(mbert_labels))
    llm_spans = set(iob2_to_spans(llm_labels))
    agreed = sorted(mbert_spans & llm_spans)
    rest = sorted([(s, "mbert") for s in mbert_spans - llm_spans] + [(s, "llm") for s in llm_spans - mbert_spans])
    regions: list[Region] = []
    for span, owner in rest:
        start, end, _ = span
        if regions and start <= regions[-1].end:  # overlaps the current region: join it
            region = regions[-1]
            region.end = max(region.end, end)
        else:
            region = Region(f"s{sentence_index}r{len(regions)}", sentence_index, start, end)
            regions.append(region)
        (region.mbert if owner == "mbert" else region.llm).append(span)
    return agreed, regions


def _is_punctuation(word: str) -> bool:
    return not any(ch.isalnum() for ch in word)


def _strip_punctuation(words: list[str], start: int, end: int) -> tuple[int, int]:
    while start < end and _is_punctuation(words[start]):
        start += 1
    while end > start and _is_punctuation(words[end]):
        end -= 1
    return start, end


def apply_rules(sentences_words: list[list[str]], regions: list[Region]) -> dict[str, str]:
    """Tool: the guideline rules. Returns {region id: 'mbert' | 'llm'} for the regions they settle.

    Rule: the two spans have the same type and the same words once punctuation at their edges is
    removed -> keep the span that has no punctuation at its edges.
    """
    settled = {}
    for region in regions:
        if len(region.mbert) != 1 or len(region.llm) != 1:
            continue
        (s1, e1, t1), (s2, e2, t2) = region.mbert[0], region.llm[0]
        words = sentences_words[region.sentence_index]
        core = _strip_punctuation(words, s1, e1)
        if t1 != t2 or core != _strip_punctuation(words, s2, e2):
            continue
        if (s1, e1) == core:
            settled[region.region_id] = "mbert"
        elif (s2, e2) == core:
            settled[region.region_id] = "llm"
    return settled


# --- LLM adjudication ----------------------------------------------------------------------


class Ruling(BaseModel):
    model_config = ConfigDict(extra="forbid")
    region_id: str
    choice: Literal["A", "B", "neither", "unsure"]
    rationale: str


class AdjudicationAnswer(BaseModel):
    model_config = ConfigDict(extra="forbid")
    rulings: list[Ruling]


def describe(words: list[str], spans: list[SpanT]) -> str:
    if not spans:
        return "no entity"
    return "; ".join(f"{s}-{e} {t} \"{' '.join(words[s : e + 1])}\"" for s, e, t in spans)


def build_adjudication_prompt(sentences_words: list[list[str]], regions: list[Region]) -> tuple[str, str]:
    types = "\n".join(f"- {t}: {DESCRIPTIONS[t]}" for t in ENTITY_TYPES)
    instructions = f"""## Role
You are the senior adjudicator of a Kazakh named-entity annotation project that follows the
KazNERD guidelines.

## Task
Two annotators, A and B, disagree on some parts of some sentences. For every region, decide which
annotation is correct.

## Entity types (KazNERD, 25 types)
{types}

## Steps
1. Read the whole sentence; every word is shown with its index.
2. Look at the region (its word indices) and at the two options.
3. Answer "A" or "B" if that option is correct, "neither" if the region contains no named
   entity, and "unsure" if you cannot decide: a person will then review it.
4. Give a one-sentence rationale.
5. Answer every region_id you were given.

## Output format
JSON matching the schema: {{"rulings": [{{"region_id": ..., "choice": ..., "rationale": ...}}]}}"""
    blocks = []
    for region in regions:
        words = sentences_words[region.sentence_index]
        blocks.append(
            f'<region id="{region.region_id}" words="{region.start}-{region.end}">\n'
            f"<sentence>{numbered(words)}</sentence>\n"
            f"A: {describe(words, region.mbert)}\nB: {describe(words, region.llm)}\n</region>"
        )
    return instructions, "<regions>\n" + "\n".join(blocks) + "\n</regions>"


def fake_adjudication_responder(request: LLMRequest) -> dict:
    """FakeLLM responder: deterministic rulings from the region data in request.metadata.

    Same words but different types -> "unsure" (escalated); only one option has an entity ->
    that option; otherwise -> "B".
    """
    rulings = []
    for region in request.metadata["regions"]:
        a, b = region["mbert"], region["llm"]
        if len(a) == 1 and len(b) == 1 and a[0][:2] == b[0][:2]:
            choice = "unsure"
        elif a and not b:
            choice = "A"
        else:
            choice = "B"
        rulings.append({"region_id": region["region_id"], "choice": choice, "rationale": "fake rule"})
    return {"rulings": rulings}


# --- the agent -----------------------------------------------------------------------------


class Verifier(Agent):
    spec = VERIFIER

    def __init__(self):
        self.waiting: dict[str, NormalizedAnnotation] = {}  # batch id -> the first annotation

    def handle(self, payload: NormalizedAnnotation, ctx: RunContext) -> list[VerificationResult]:
        first = self.waiting.pop(payload.batch_id, None)
        if first is None:  # the other annotation of this batch has not arrived yet
            self.waiting[payload.batch_id] = payload
            return []
        by_annotator = {first.annotator: first, payload.annotator: payload}
        if set(by_annotator) != {"mbert", "llm"}:
            raise AgentError(f"batch {payload.batch_id}: Verifier needs one mbert and one llm annotation")
        mbert, llm = by_annotator["mbert"], by_annotator["llm"]
        if [s.sentence_id for s in mbert.sentences] != [s.sentence_id for s in llm.sentences]:
            raise AgentError(f"batch {payload.batch_id}: the two annotations cover different sentences")

        sentences_words = [s.words for s in mbert.sentences]
        agreed_per_sentence, regions = [], []
        for index, (m, l) in enumerate(zip(mbert.sentences, llm.sentences)):
            agreed, sentence_regions = compare(index, m.labels, l.labels)
            agreed_per_sentence.append(agreed)
            regions += sentence_regions

        by_rule = ctx.call_tool(self.name, "guideline_rules", apply_rules, sentences_words, regions)
        open_regions = [r for r in regions if r.region_id not in by_rule]
        rulings, failure = self.adjudicate(open_regions, sentences_words, ctx)

        verdicts, escalations = [], []
        for index, sentence in enumerate(mbert.sentences):
            words = sentence.words
            labels = ["O"] * len(words)
            decisions = []
            for start, end, etype in agreed_per_sentence[index]:
                _place(labels, [(start, end, etype)])
                decisions.append(Decision(start_word=start, end_word=end, decision="agreed",
                                          kept="both", rationale="both annotators agree"))
            for region in [r for r in regions if r.sentence_index == index]:
                decision, kept, rationale = self.settle(region, by_rule, rulings, failure)
                _place(labels, region.mbert if kept == "mbert" else region.llm if kept == "llm" else [])
                decisions.append(Decision(start_word=region.start, end_word=region.end,
                                          decision=decision, kept=kept, rationale=rationale))
                if decision == "escalated":
                    escalations.append({
                        "sentence_id": sentence.sentence_id,
                        "start_word": region.start,
                        "end_word": region.end,
                        "words": " ".join(words[region.start : region.end + 1]),
                        "mbert": describe(words, region.mbert),
                        "llm": describe(words, region.llm),
                        "reason": rationale,
                    })
            verdicts.append(SentenceVerdict(sentence_id=sentence.sentence_id, words=words,
                                            labels=labels, decisions=decisions))

        if escalations:
            ctx.call_tool(self.name, "review_queue", ctx.state.add_review_items,
                          ctx.run_id, payload.batch_id, escalations)
        return [VerificationResult(batch_id=payload.batch_id, sentences=verdicts)]

    def adjudicate(self, regions: list[Region], sentences_words: list[list[str]], ctx: RunContext):
        """One LLM call for all open regions. Returns ({region id: Ruling}, failure reason)."""
        if not regions:
            return {}, None
        instructions, user = build_adjudication_prompt(sentences_words, regions)
        metadata = {"regions": [{"region_id": r.region_id, "mbert": r.mbert, "llm": r.llm} for r in regions]}
        request = LLMRequest(instructions=instructions, user=user, schema=AdjudicationAnswer, metadata=metadata)
        try:
            answer = ctx.llm.complete(self.name, "adjudicate", request)
        except LimitExceeded:
            raise  # a run limit stops the whole run
        except KnaError as exc:
            return {}, f"adjudication failed: {exc}"  # escalate instead of failing the batch
        return {ruling.region_id: ruling for ruling in answer.rulings}, None

    @staticmethod
    def settle(region: Region, by_rule: dict[str, str], rulings: dict, failure: str | None):
        """(decision, kept, rationale) for one region."""
        if region.region_id in by_rule:
            return "adjudicated", by_rule[region.region_id], "guideline rule: no punctuation at entity edges"
        ruling = rulings.get(region.region_id)
        if ruling is None:
            return "escalated", PROVISIONAL, failure or "the adjudicator gave no ruling"
        if ruling.choice == "unsure":
            return "escalated", PROVISIONAL, f"adjudicator unsure: {ruling.rationale}"
        kept = {"A": "mbert", "B": "llm", "neither": "neither"}[ruling.choice]
        return "adjudicated", kept, f"LLM: {ruling.rationale}"


def _place(labels: list[str], spans: list[SpanT]) -> None:
    for start, end, etype in spans:
        labels[start] = f"B-{etype}"
        for i in range(start + 1, end + 1):
            labels[i] = f"I-{etype}"
