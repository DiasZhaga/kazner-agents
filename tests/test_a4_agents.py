import importlib.util

import pytest

from kazner_agents.agents.pre_annotator import FakeTagger, PreAnnotator, fake_labels
from kazner_agents.agents.verifier import (
    AdjudicationAnswer,
    Region,
    Verifier,
    apply_rules,
    compare,
)
from kazner_agents.errors import LLMError
from kazner_agents.fakes import fake_responder
from kazner_agents.llm import FakeLLM, LLMClient
from kazner_agents.messages import DocumentBatch, NormalizedAnnotation, Sentence, SentenceLabels
from kazner_agents.metrics import agreement, compute_metrics
from kazner_agents.orchestrator import build_routes
from kazner_agents.run_log import read_log


class MemoryState:
    """Just enough of the state store for the Verifier's review queue."""

    def __init__(self):
        self.items = []

    def add_review_items(self, run_id, batch_id, items):
        self.items += items
        return len(items)


def normalized(annotator, rows):
    sentences = [
        SentenceLabels(sentence_id=f"s{i}", words=words, labels=labels)
        for i, (words, labels) in enumerate(rows)
    ]
    return NormalizedAnnotation(batch_id="b001", annotator=annotator, sentences=sentences, repairs_applied=0)


# --- Verifier ------------------------------------------------------------------------------


def test_compare_splits_agreed_entities_from_disagreement_regions():
    mbert = ["B-PERSON", "I-PERSON", "O", "B-GPE", "O"]
    llm = ["B-PERSON", "I-PERSON", "O", "B-LOCATION", "I-LOCATION"]
    agreed, regions = compare(0, mbert, llm)
    assert agreed == [(0, 1, "PERSON")]
    assert len(regions) == 1
    assert (regions[0].start, regions[0].end) == (3, 4)
    assert regions[0].mbert == [(3, 3, "GPE")] and regions[0].llm == [(3, 4, "LOCATION")]


def test_guideline_rule_keeps_the_span_without_edge_punctuation():
    words = ["«", "Атырау", "»", "ықшамауданы"]
    regions = [Region("r0", 0, 0, 2, mbert=[(1, 1, "LOCATION")], llm=[(0, 2, "LOCATION")])]
    assert apply_rules([words], regions) == {"r0": "mbert"}
    other_type = [Region("r1", 0, 0, 2, mbert=[(1, 1, "LOCATION")], llm=[(0, 2, "GPE")])]
    assert apply_rules([words], other_type) == {}


def run_verifier(ctx, mbert_rows, llm_rows, responder=fake_responder):
    ctx.llm = LLMClient(FakeLLM(responder), ctx.logger, max_calls=10, timeout_seconds=5)
    ctx.state = MemoryState()
    verifier = Verifier()
    assert verifier.handle(normalized("mbert", mbert_rows), ctx) == []  # waits for the pair
    [result] = verifier.handle(normalized("llm", llm_rows), ctx)
    return result


WORDS = ["Бұл", "кітапты", "Мұхтар", "Әуезов", "1897", "жылы", "жазды", "."]


def test_verifier_agrees_adjudicates_and_escalates(ctx):
    mbert = ["O", "O", "B-ORGANISATION", "I-ORGANISATION", "B-DATE", "O", "O", "O"]
    llm = ["O", "O", "B-PERSON", "I-PERSON", "B-DATE", "I-DATE", "O", "O"]
    result = run_verifier(ctx, [(WORDS, mbert)], [(WORDS, llm)])
    decisions = {(d.start_word, d.end_word): d for d in result.sentences[0].decisions}
    # same words, different type -> the fake adjudicator is unsure -> escalated, mBERT kept
    assert decisions[(2, 3)].decision == "escalated" and decisions[(2, 3)].kept == "mbert"
    # different boundaries -> adjudicated by the (fake) LLM, which picks B = the LLM's span
    assert decisions[(4, 5)].decision == "adjudicated" and decisions[(4, 5)].kept == "llm"
    assert result.sentences[0].labels == [
        "O", "O", "B-ORGANISATION", "I-ORGANISATION", "B-DATE", "I-DATE", "O", "O",
    ]
    assert len(ctx.state.items) == 1 and ctx.state.items[0]["words"] == "Мұхтар Әуезов"
    calls = [(r["kind"], r["name"]) for r in read_log(ctx.logger.path)]
    assert ("llm", "adjudicate") in calls and ("tool", "review_queue") in calls


def test_verifier_makes_no_llm_call_when_annotators_agree(ctx):
    labels = ["O", "O", "B-PERSON", "I-PERSON", "B-DATE", "I-DATE", "O", "O"]
    result = run_verifier(ctx, [(WORDS, labels)], [(WORDS, labels)])
    assert result.sentences[0].labels == labels
    assert {d.decision for d in result.sentences[0].decisions} == {"agreed"}
    assert all(r["kind"] != "llm" for r in read_log(ctx.logger.path))


def test_failed_adjudication_escalates_instead_of_failing(ctx):
    def broken(request):
        if request.schema is AdjudicationAnswer:
            raise LLMError("adjudicator down")
        return fake_responder(request)

    mbert = ["O", "O", "B-ORGANISATION", "I-ORGANISATION", "B-DATE", "O", "O", "O"]
    llm = ["O", "O", "B-PERSON", "I-PERSON", "B-DATE", "I-DATE", "O", "O"]
    result = run_verifier(ctx, [(WORDS, mbert)], [(WORDS, llm)], responder=broken)
    assert {d.decision for d in result.sentences[0].decisions} == {"escalated"}
    assert result.sentences[0].labels == mbert  # provisional: the mBERT version
    assert "adjudication failed" in ctx.state.items[0]["reason"]


# --- PreAnnotator --------------------------------------------------------------------------


def test_fake_tagger_is_deterministic_and_labels_every_word():
    words = ["Астана", "қаласында", "Назарбаев", "Университеті", "2010", "жылы", "ашылды", "."]
    assert fake_labels(words) == [
        "B-GPE", "O", "B-ORGANISATION", "I-ORGANISATION", "B-DATE", "O", "O", "O",
    ]
    assert FakeTagger().tag([words]) == [(fake_labels(words), 0)]


def test_pre_annotator_loads_once_and_notes_cut_off_words(ctx):
    class ShortTagger:
        def tag(self, sentences):
            return [(["O"] * len(words), 2) for words in sentences]

    loads = []

    def make(settings):
        loads.append(1)
        return ShortTagger()

    batch = DocumentBatch(batch_id="b001", doc_id="d", source_url="x", licence="x",
                          sentences=[Sentence(sentence_id="s0", words=["a", "b", "c"])])
    agent = PreAnnotator(make)
    agent.handle(batch, ctx)
    [annotation] = agent.handle(batch, ctx)
    assert len(loads) == 1
    assert annotation.annotator == "mbert" and annotation.sentences[0].labels == ["O", "O", "O"]
    assert "2 words beyond 512" in annotation.sentences[0].notes[0]


# --- scoring and routing -------------------------------------------------------------------


def test_metrics_are_entity_level_and_strict():
    gold = [["B-PERSON", "I-PERSON", "O", "B-GPE"]]
    pred = [["B-PERSON", "O", "O", "B-GPE"]]  # PERSON boundary wrong -> not counted
    scores = compute_metrics(gold, pred)
    assert scores["precision"] == 0.5 and scores["recall"] == 0.5
    assert compute_metrics([["O"]], [["O"]])["f1"] == 1.0
    assert agreement(gold, gold) == {"entity_f1": 1.0, "token_agreement": 1.0}


@pytest.mark.skipif(importlib.util.find_spec("kazner") is None, reason="kazner not installed")
def test_metrics_match_the_kazner_evaluator():
    from kazner.evaluate import compute_metrics as kazner_metrics

    gold = [["B-PERSON", "I-PERSON", "O", "B-GPE"], ["B-DATE", "I-DATE", "O"]]
    pred = [["B-PERSON", "O", "O", "B-GPE"], ["B-DATE", "I-DATE", "B-ORDINAL"]]
    assert compute_metrics(gold, pred) == kazner_metrics(gold, pred)


def test_routes_follow_the_plan():
    both = build_routes(("mbert", "llm"))
    assert both["DocumentBatch"] == ["PreAnnotator", "LLMAnnotator"]
    assert both["NormalizedAnnotation"] == ["Verifier"]
    single = build_routes(("llm",))
    assert single["DocumentBatch"] == ["LLMAnnotator"]
    assert single["NormalizedAnnotation"] == ["Evaluator"]
