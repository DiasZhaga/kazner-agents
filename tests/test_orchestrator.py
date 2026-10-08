import dataclasses
import json
import time

from kazner_agents.agents.collector import Collector
from kazner_agents.agents.llm_annotator import LLMAnnotator, fake_annotation_responder
from kazner_agents.llm import FakeLLM
from kazner_agents.messages import CollectRequest
from kazner_agents.orchestrator import default_agents, run_pipeline
from kazner_agents.run_log import read_log
from kazner_agents.state import StateStore

from conftest import make_settings

TEXT = """Абай Құнанбайұлы 1845 жылы Шыңғыстауда туған. Ол Семейде оқыды.
Әкесі Құнанбай аға сұлтан болған. Абай орыс ақындарын аударды.
Өсім 12 % болды. Ақын 1904 жылы қайтыс болды. Оның өлеңдері кең тарады."""


def file_task(tmp_path, batch_size=3, max_sentences=20):
    path = tmp_path / "abai.txt"
    path.write_text(TEXT, encoding="utf-8")
    return CollectRequest(source="file", path=str(path), batch_size=batch_size, max_sentences=max_sentences)


def run(tmp_path, task=None, backend=None, agents=None, **limits):
    settings = dataclasses.replace(make_settings(tmp_path), **limits)
    backend = backend or FakeLLM(fake_annotation_responder)
    return run_pipeline(settings, task or file_task(tmp_path), backend, agents=agents, run_id="r1")


def test_end_to_end_run_with_fake_llm(tmp_path):
    summary = run(tmp_path)
    assert summary.status == "completed", summary.batch_errors
    assert summary.batches == {"b001": "done", "b002": "done", "b003": "done"}
    assert summary.steps == 1 + 4 * 3  # CollectRequest + 4 messages per batch
    assert summary.llm_calls == 3 and summary.input_tokens > 0

    iob2 = (summary.output_dir / "annotations.iob2").read_text(encoding="utf-8")
    assert "Семейде B-GPE" in iob2
    rows = [json.loads(l) for l in (summary.output_dir / "sentences.jsonl").read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 7 and rows[0]["source_url"] == "file:abai.txt"
    assert summary.reports[0].mode == "summary" and summary.reports[0].counts["sentences"] == 3

    records = read_log(summary.log_path)
    messages = [r for r in records if r["kind"] == "message"]
    assert [m["name"] for m in messages[:5]] == [
        "CollectRequest", "DocumentBatch", "Annotation", "NormalizedAnnotation", "EvaluationReport",
    ]
    assert {r["agent"] for r in records} == {
        "Orchestrator", "Collector", "LLMAnnotator", "BoundaryNormalizer", "Evaluator",
    }

    state = StateStore(tmp_path / "state" / "kna.sqlite3")
    assert state.get_run("r1")["status"] == "completed"
    assert [b["status"] for b in state.get_batches("r1")] == ["done"] * 3
    assert state.count_messages("r1") == 13
    state.close()


def test_step_limit_stops_the_run(tmp_path):
    summary = run(tmp_path, max_steps=5)
    assert summary.status == "stopped"
    assert summary.batches == {"b001": "done", "b002": "failed", "b003": "skipped"}
    assert "step limit" in summary.reason


def test_time_limit_stops_the_run(tmp_path):
    class SlowCollector(Collector):
        def handle(self, payload, ctx):
            time.sleep(0.3)
            return super().handle(payload, ctx)

    agents = default_agents()
    agents["Collector"] = SlowCollector()
    summary = run(tmp_path, agents=agents, max_seconds=0.2)
    assert summary.status == "stopped" and "time limit" in summary.reason
    assert summary.llm_calls == 0


def test_llm_call_cap_stops_the_run(tmp_path):
    summary = run(tmp_path, max_llm_calls=1)
    assert summary.status == "stopped" and "LLM call limit" in summary.reason
    assert summary.batches["b001"] == "done" and summary.batches["b003"] == "skipped"


def test_loop_protection_rejects_repeated_messages(tmp_path):
    class LoopingAnnotator(LLMAnnotator):
        def handle(self, payload, ctx):
            return [payload]  # sends the DocumentBatch back to itself: a loop

    agents = default_agents()
    agents["LLMAnnotator"] = LoopingAnnotator()
    summary = run(tmp_path, agents=agents)
    assert summary.batches["b001"] == "failed"
    assert "LoopDetected" in summary.batch_errors["b001"]
    rejected = [r for r in read_log(summary.log_path) if r["status"] == "rejected"]
    assert len(rejected) == 3  # one per batch: the 3rd identical delivery is refused


def test_failed_llm_batch_does_not_stop_the_run(tmp_path):
    backend = FakeLLM(fake_annotation_responder, fail_times=4)  # 1 attempt + 3 retries fail
    summary = run(tmp_path, backend=backend)
    assert summary.status == "completed_with_failures"
    assert summary.batches == {"b001": "failed", "b002": "done", "b003": "done"}
    assert "TransientError" in summary.batch_errors["b001"]
    assert summary.llm_calls == 6


def test_collector_failure_gives_a_readable_reason(tmp_path):
    task = CollectRequest(source="file", path=str(tmp_path / "missing.txt"))
    summary = run(tmp_path, task=task)
    assert summary.status == "failed"
    assert summary.reason.startswith("collecting text failed: cannot read file")


def test_repaired_sentences_go_to_the_review_list(tmp_path):
    def off_by_one(request):
        answer = fake_annotation_responder(request)
        for sentence in answer["sentences"]:
            for entity in sentence["entities"]:
                entity["start_word"] += 1  # a typical LLM counting error
                entity["end_word"] += 1
        return answer

    summary = run(tmp_path, backend=FakeLLM(off_by_one))
    assert summary.status == "completed"
    review = (summary.output_dir / "review.jsonl").read_text(encoding="utf-8").splitlines()
    first = json.loads(review[0])
    assert any("moved" in reason for reason in first["reasons"])
    assert "Семейде B-GPE" in (summary.output_dir / "annotations.iob2").read_text(encoding="utf-8")
