import pytest

from kazner_agents.agents.collector import Collector
from kazner_agents.errors import AgentError, TransientError
from kazner_agents.messages import CollectRequest
from kazner_agents.run_log import read_log
from kazner_agents.text_split import split_sentences, split_words
from kazner_agents.wikipedia import Article

ARTICLE_TEXT = """Абай Құнанбайұлы (1845–1904) — қазақ ақыны. Ол Семейде оқыды!
== Өмірбаяны ==

1845 ж. Шыңғыстауда туған. Әкесі Құнанбай аға сұлтан болған.
А. Байтұрсынұлы «Қазақтың бас ақыны» деп жазды. Өсім 12,4 % болды?! Келесі 2-інші сөйлем."""


def test_words_follow_kaznerd_conventions():
    assert split_words("«Атырау» 12,4 % бизнес-климатын 1-інші.") == [
        "«", "Атырау", "»", "12", ",", "4", "%", "бизнес-климатын", "1-інші", ".",
    ]


def test_sentences_handle_headings_abbreviations_initials_and_quotes():
    sentences = split_sentences(ARTICLE_TEXT)
    texts = [" ".join(s) for s in sentences]
    assert texts == [
        "Абай Құнанбайұлы ( 1845 – 1904 ) — қазақ ақыны .",
        "Ол Семейде оқыды !",
        "1845 ж . Шыңғыстауда туған .",
        "Әкесі Құнанбай аға сұлтан болған .",
        "А . Байтұрсынұлы « Қазақтың бас ақыны » деп жазды .",
        "Өсім 12 , 4 % болды ? !",
        "Келесі 2-інші сөйлем .",
    ]


def test_closing_quote_stays_with_its_sentence():
    assert split_sentences("Ол «Келемін.» Содан кетті.") == [
        ["Ол", "«", "Келемін", ".", "»"], ["Содан", "кетті", "."],
    ]


def fake_fetch(title, user_agent):
    return Article(title="Абай Құнанбайұлы", url="https://kk.wikipedia.org/wiki/Абай", text=ARTICLE_TEXT)


def test_collector_makes_batches_with_provenance(ctx):
    request = CollectRequest(source="wikipedia", title="Абай", max_sentences=5, batch_size=2)
    batches = Collector(fetch_article=fake_fetch).handle(request, ctx)
    assert [b.batch_id for b in batches] == ["b001", "b002", "b003"]
    assert [len(b.sentences) for b in batches] == [2, 2, 1]
    assert batches[0].licence == "CC BY-SA 4.0"
    assert batches[0].sentences[0].sentence_id == "kkwiki-абай-құнанбайұлы-s0000"
    names = [r["name"] for r in read_log(ctx.logger.path)]
    assert names == ["wikipedia_api", "sentence_splitter"]


def test_collector_retries_transient_wikipedia_errors(ctx):
    calls = []

    def flaky_fetch(title, user_agent):
        calls.append(title)
        if len(calls) == 1:
            raise TransientError("timeout")
        return fake_fetch(title, user_agent)

    Collector(fetch_article=flaky_fetch).handle(CollectRequest(source="wikipedia", title="Абай"), ctx)
    record = read_log(ctx.logger.path)[0]
    assert len(calls) == 2 and record["attempts"] == 2 and record["status"] == "ok"


def test_collector_reads_a_file(ctx, tmp_path):
    path = tmp_path / "мәтін.txt"
    path.write_text(ARTICLE_TEXT, encoding="utf-8")
    [batch] = Collector().handle(CollectRequest(source="file", path=str(path)), ctx)
    assert batch.source_url == "file:мәтін.txt"
    assert len(batch.sentences) == 7


def test_collector_errors_are_readable(ctx, tmp_path):
    with pytest.raises(AgentError, match="cannot read file"):
        Collector().handle(CollectRequest(source="file", path=str(tmp_path / "none.txt")), ctx)
    with pytest.raises(AgentError, match="Assignment 4"):
        Collector().handle(CollectRequest(source="kaznerd_test"), ctx)


class FakeResponse:
    def __init__(self, status_code, data=None):
        self.status_code = status_code
        self.data = data

    def json(self):
        return self.data


def test_fetch_article_parses_the_api_answer(monkeypatch):
    from kazner_agents import wikipedia

    seen = {}

    def fake_get(url, params, headers, timeout):
        seen.update(headers)
        page = {"title": "Абай Құнанбайұлы", "extract": "Абай — ақын."}
        return FakeResponse(200, {"query": {"pages": [page]}})

    monkeypatch.setattr(wikipedia.requests, "get", fake_get)
    article = wikipedia.fetch_article("Абай", user_agent="kna-test")
    assert article.url.startswith("https://kk.wikipedia.org/wiki/%D0%90")
    assert article.text == "Абай — ақын."
    assert seen["User-Agent"] == "kna-test"


def test_fetch_article_error_kinds(monkeypatch):
    from kazner_agents import wikipedia

    answers = iter([
        FakeResponse(503),
        FakeResponse(200, {"query": {"pages": [{"title": "Жоқ", "missing": True}]}}),
    ])
    monkeypatch.setattr(wikipedia.requests, "get", lambda *a, **k: next(answers))
    with pytest.raises(TransientError):
        wikipedia.fetch_article("Жоқ", "kna-test")
    with pytest.raises(AgentError, match="no article"):
        wikipedia.fetch_article("Жоқ", "kna-test")
