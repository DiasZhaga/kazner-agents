import pytest
from pydantic import ValidationError

from kazner_agents.messages import (
    Annotation,
    CollectRequest,
    DocumentBatch,
    Envelope,
    EvaluationReport,
    NormalizedAnnotation,
    Sentence,
    SentenceAnnotation,
    SentenceLabels,
    Span,
)


def make_batch() -> DocumentBatch:
    return DocumentBatch(
        batch_id="b001",
        doc_id="doc",
        source_url="https://kk.wikipedia.org/wiki/Абай",
        licence="CC BY-SA 4.0",
        sentences=[Sentence(sentence_id="doc-s0000", words=["Абай", "Семейде", "оқыды", "."])],
    )


def test_envelope_round_trips_through_json():
    env = Envelope(
        run_id="r1", step=1, sender="Collector", recipient="LLMAnnotator",
        type="DocumentBatch", payload=make_batch(),
    )
    restored = Envelope.model_validate_json(env.model_dump_json())
    assert restored == env
    assert isinstance(restored.payload, DocumentBatch)


def test_envelope_type_must_match_payload():
    with pytest.raises(ValidationError):
        Envelope(
            run_id="r1", step=1, sender="a", recipient="b", type="Annotation",
            payload=make_batch(),
        )


def test_envelope_rejects_unknown_type_in_json():
    with pytest.raises(ValidationError):
        Envelope.model_validate(
            {"run_id": "r", "step": 0, "sender": "a", "recipient": "b",
             "type": "Nope", "payload": {}}
        )


def test_unknown_fields_are_rejected():
    with pytest.raises(ValidationError):
        CollectRequest(source="file", path="x.txt", colour="red")


def test_collect_request_needs_title_or_path():
    with pytest.raises(ValidationError):
        CollectRequest(source="wikipedia")
    with pytest.raises(ValidationError):
        CollectRequest(source="file")
    assert CollectRequest(source="wikipedia", title="Абай").batch_size == 10


def test_span_type_must_be_a_kaznerd_type():
    Span(start_word=0, end_word=0, type="PERSON")
    with pytest.raises(ValidationError):
        Span(start_word=0, end_word=0, type="PER")
    with pytest.raises(ValidationError):
        Span(start_word=-1, end_word=0, type="PERSON")


def test_sentence_annotation_has_exactly_one_kind():
    words = ["Абай", "ақын"]
    SentenceAnnotation(sentence_id="s", words=words, entities=[])
    SentenceAnnotation(sentence_id="s", words=words, labels=["B-PERSON", "O"])
    with pytest.raises(ValidationError):
        SentenceAnnotation(sentence_id="s", words=words)
    with pytest.raises(ValidationError):
        SentenceAnnotation(sentence_id="s", words=words, entities=[], labels=["O", "O"])
    with pytest.raises(ValidationError):
        SentenceAnnotation(sentence_id="s", words=words, labels=["O"])


def test_sentence_labels_must_be_valid_iob2_labels():
    SentenceLabels(sentence_id="s", words=["Абай"], labels=["B-PERSON"])
    with pytest.raises(ValidationError):
        SentenceLabels(sentence_id="s", words=["Абай"], labels=["B-PER"])
    with pytest.raises(ValidationError):
        SentenceLabels(sentence_id="s", words=["Абай", "ақын"], labels=["O"])


def test_gold_labels_length_is_checked():
    with pytest.raises(ValidationError):
        Sentence(sentence_id="s", words=["a", "b"], gold_labels=["O"])


def test_all_payloads_build():
    Annotation(batch_id="b", annotator="llm", sentences=[])
    NormalizedAnnotation(batch_id="b", annotator="llm", sentences=[], repairs_applied=0)
    EvaluationReport(batch_id="b", mode="summary")
    with pytest.raises(ValidationError):
        EvaluationReport(batch_id="b", mode="other")
