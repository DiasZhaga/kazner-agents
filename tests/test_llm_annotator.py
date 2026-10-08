import json

from kazner_agents.agents.llm_annotator import (
    FEWSHOT_PATH,
    LLMAnnotator,
    LLMAnswer,
    build_instructions,
    build_user_prompt,
    heuristic_entities,
    load_examples,
)
from kazner_agents.iob2 import iob2_to_spans
from kazner_agents.labels import ENTITY_TYPES
from kazner_agents.llm import FakeLLM, LLMClient
from kazner_agents.messages import DocumentBatch, Sentence
from kazner_agents.run_log import read_log

BATCH = DocumentBatch(
    batch_id="b001", doc_id="d", source_url="file:x.txt", licence="test",
    sentences=[
        Sentence(sentence_id="d-s0000", words=["Абай", "Құнанбайұлы", "1845", "жылы", "Семейде", "туған", "."]),
        Sentence(sentence_id="d-s0001", words=["Өсім", "12", "%", "болды", "."]),
    ],
)


def test_fewshot_examples_cover_all_25_types():
    examples = load_examples(FEWSHOT_PATH)
    covered = {t for e in examples for _, _, t in iob2_to_spans(e["labels"])}
    assert covered == set(ENTITY_TYPES)
    assert len(examples) <= 25


def test_prompt_has_lecture_structure_and_numbered_words():
    instructions = build_instructions(load_examples())
    for heading in ("## Role", "## Task", "## Entity types", "## Steps", "## Examples", "## Output format"):
        assert heading in instructions
    assert '"type": "NON_HUMAN"' in instructions
    user = build_user_prompt(BATCH)
    assert '<sentence id="d-s0000">0:Абай 1:Құнанбайұлы 2:1845' in user


def test_heuristic_fake_entities():
    words = BATCH.sentences[0].words
    # The first word is skipped (sentence start), so the surname alone looks like a GPE.
    assert heuristic_entities(words) == [
        {"start_word": 1, "end_word": 1, "type": "GPE", "text": "Құнанбайұлы"},
        {"start_word": 2, "end_word": 3, "type": "DATE", "text": "1845 жылы"},
        {"start_word": 4, "end_word": 4, "type": "GPE", "text": "Семейде"},
    ]


def test_answer_schema_is_accepted_as_strict_openai_schema():
    from openai.lib._pydantic import to_strict_json_schema

    schema = to_strict_json_schema(LLMAnswer)
    entity = schema["$defs"]["LLMEntity"]
    assert entity["additionalProperties"] is False
    assert set(entity["required"]) == {"start_word", "end_word", "type", "text"}
    assert len(entity["properties"]["type"]["enum"]) == 25


def test_annotator_turns_llm_answer_into_annotation(ctx):
    [annotation] = LLMAnnotator().handle(BATCH, ctx)
    assert annotation.annotator == "llm"
    first = annotation.sentences[0]
    assert [(s.start_word, s.end_word, s.type) for s in first.entities] == [
        (1, 1, "GPE"), (2, 3, "DATE"), (4, 4, "GPE"),
    ]
    kinds = [(r["kind"], r["name"]) for r in read_log(ctx.logger.path)]
    assert kinds == [("tool", "fewshot_store"), ("llm", "annotate_batch")]


def test_missing_sentence_and_negative_index_become_notes(ctx):
    answer = {"sentences": [{"sentence_id": "d-s0000", "entities": [
        {"start_word": -1, "end_word": 0, "type": "PERSON", "text": "Абай"}]}]}
    ctx.llm = LLMClient(FakeLLM(lambda r: answer), ctx.logger, max_calls=5, timeout_seconds=5)
    [annotation] = LLMAnnotator().handle(BATCH, ctx)
    assert annotation.sentences[0].entities == []
    assert "negative" in annotation.sentences[0].notes[0]
    assert annotation.sentences[1].notes == ["sentence missing from the LLM answer"]
