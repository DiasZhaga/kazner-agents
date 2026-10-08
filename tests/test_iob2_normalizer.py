from kazner_agents.agents.normalizer import BoundaryNormalizer
from kazner_agents.agents.specs import ALL_SPECS
from kazner_agents.iob2 import iob2_to_spans, repair_iob2, spans_to_iob2, validate_iob2
from kazner_agents.messages import Annotation, SentenceAnnotation, Span
from kazner_agents.run_log import read_log

WORDS = ["Абай", "Құнанбайұлы", "1845", "жылы", "Семейде", "туған", "."]


def span(start, end, etype, text=None):
    return Span(start_word=start, end_word=end, type=etype, text=text)


def test_spans_become_iob2():
    labels, repairs = spans_to_iob2(
        WORDS, [span(0, 1, "PERSON"), span(2, 3, "DATE"), span(4, 4, "GPE")]
    )
    assert labels == ["B-PERSON", "I-PERSON", "B-DATE", "I-DATE", "B-GPE", "O", "O"]
    assert repairs == []


def test_span_with_wrong_index_is_moved_to_its_text():
    labels, repairs = spans_to_iob2(WORDS, [span(5, 5, "GPE", text="Семейде")])
    assert labels[4] == "B-GPE" and labels[5] == "O"
    assert repairs and "moved" in repairs[0]


def test_text_match_ignores_spaces_and_case():
    labels, repairs = spans_to_iob2(WORDS, [span(0, 1, "PERSON", text="абай  құнанбайұлы")])
    assert labels[:2] == ["B-PERSON", "I-PERSON"] and repairs == []


def test_span_with_unknown_text_is_dropped():
    labels, repairs = spans_to_iob2(WORDS, [span(0, 0, "PERSON", text="Шоқан")])
    assert labels == ["O"] * len(WORDS)
    assert "dropped" in repairs[0]


def test_reversed_out_of_range_and_overlapping_spans_are_repaired():
    labels, repairs = spans_to_iob2(
        WORDS,
        [span(1, 0, "PERSON"), span(1, 2, "DATE"), span(5, 9, "GPE"), span(10, 12, "ORGANISATION")],
    )
    assert labels == ["B-PERSON", "I-PERSON", "O", "O", "O", "B-GPE", "I-GPE"]
    assert len(repairs) == 4
    assert validate_iob2(labels) == []


def test_invalid_iob2_sequence_is_repaired():
    labels, repairs = repair_iob2(["I-PERSON", "I-PERSON", "O", "B-GPE", "I-DATE", "X-FOO"])
    assert labels == ["B-PERSON", "I-PERSON", "O", "B-GPE", "B-DATE", "O"]
    assert len(repairs) == 3
    assert validate_iob2(labels) == []


def test_validator_finds_problems():
    assert validate_iob2(["O", "I-GPE"]) == ["word 1: I-GPE after O"]
    assert validate_iob2(["B-PER"]) == ["word 0: unknown label 'B-PER'"]


def test_iob2_to_spans():
    labels = ["B-PERSON", "I-PERSON", "B-DATE", "B-DATE", "I-DATE", "O", "B-GPE"]
    assert iob2_to_spans(labels) == [(0, 1, "PERSON"), (2, 2, "DATE"), (3, 4, "DATE"), (6, 6, "GPE")]


def test_normalizer_agent_handles_spans_and_labels(ctx):
    annotation = Annotation(
        batch_id="b001",
        annotator="llm",
        sentences=[
            SentenceAnnotation(sentence_id="s0", words=WORDS, entities=[span(0, 1, "PERSON")]),
            SentenceAnnotation(
                sentence_id="s1", words=["Астана", "қаласы"], labels=["I-GPE", "O"],
                notes=["sentence missing from the LLM answer"],
            ),
        ],
    )
    [result] = BoundaryNormalizer().handle(annotation, ctx)
    assert result.sentences[0].labels[:2] == ["B-PERSON", "I-PERSON"]
    assert result.sentences[1].labels == ["B-GPE", "O"]
    assert result.repairs_applied == 1
    assert result.sentences[1].repairs[0].startswith("annotator:")
    [record] = read_log(ctx.logger.path)
    assert record["agent"] == "BoundaryNormalizer" and record["kind"] == "tool"


def test_every_agent_has_at_most_five_tools():
    assert len(ALL_SPECS) == 7
    assert all(len(spec.tools) <= 5 for spec in ALL_SPECS)
