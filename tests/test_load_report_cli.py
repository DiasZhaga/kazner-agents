from kazner_agents.cli import main
from kazner_agents.config import DEFAULT_HOME
from kazner_agents.docs_gen import outdated_docs
from kazner_agents.load_report import compute_load, find_log, format_load_report, overloaded


def record(agent, kind):
    return {"agent": agent, "kind": kind}


def test_load_counts_llm_and_tool_calls_but_not_messages():
    records = [record("A", "llm"), record("A", "tool"), record("B", "tool"), record("A", "message")]
    loads = compute_load(records)
    assert [(l.agent, l.llm, l.tool, l.total) for l in loads] == [("A", 1, 1, 2), ("B", 0, 1, 1)]


def test_agent_over_40_percent_is_reported():
    balanced = [record(a, "tool") for a in "ABCDE"]
    assert overloaded(compute_load(balanced)) == []
    skewed = balanced + [record("A", "llm")] * 3  # A: 4 of 8 = 50%
    loads = compute_load(skewed)
    assert overloaded(loads) == ["A"]
    text = format_load_report("r1", loads)
    assert "FAIL" in text and "50.0%" in text


def test_exactly_40_percent_is_allowed():
    records = [record("A", "tool")] * 2 + [record("B", "tool")] * 2 + [record("C", "tool")]
    assert overloaded(compute_load(records)) == []


def test_find_log_latest(tmp_path):
    (tmp_path / "20261001-100000-aaaa.jsonl").write_text("", encoding="utf-8")
    (tmp_path / "20261008-090000-bbbb.jsonl").write_text("", encoding="utf-8")
    assert find_log(tmp_path, "latest").stem == "20261008-090000-bbbb"


def test_cli_run_and_load_report_offline(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KNA_HOME", str(tmp_path))
    text_file = tmp_path / "text.txt"
    text_file.write_text("Абай Семейде оқыды. Ол 1845 жылы туған. Абай ақын болды.", encoding="utf-8")
    code = main(["run", "--source", "file", "--path", str(text_file), "--batch-size", "2", "--llm", "fake"])
    out = capsys.readouterr().out
    assert code == 0, out
    assert ": completed" in out and "OK: no agent over the limit" in out
    assert main(["load-report", "latest"]) == 0


def test_cli_reports_bad_arguments(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("KNA_HOME", str(tmp_path))
    assert main(["run", "--source", "wikipedia"]) == 2
    assert "needs a title" in capsys.readouterr().err
    monkeypatch.setenv("OPENAI_API_KEY", "")
    assert main(["run", "--source", "wikipedia", "--title", "Абай", "--llm", "openai"]) == 2


def test_generated_docs_are_up_to_date():
    assert outdated_docs(DEFAULT_HOME / "docs") == [], "run `kna docs`"
