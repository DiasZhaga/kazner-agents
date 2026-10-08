from pathlib import Path

import pytest

from kazner_agents.config import DEFAULT_HOME, load_settings
from kazner_agents.labels import ENTITY_TYPES, IOB2_LABELS

KAZNERD_TEST = DEFAULT_HOME.parent / "ner-project" / "data" / "IOB2_test.txt"


def test_there_are_25_types_and_51_labels():
    assert len(ENTITY_TYPES) == 25
    assert len(IOB2_LABELS) == 51


@pytest.mark.skipif(not KAZNERD_TEST.exists(), reason="KazNERD data folder not available")
def test_label_set_matches_kaznerd_data():
    types = set()
    for line in KAZNERD_TEST.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[1] != "O":
            types.add(parts[1][2:])
    assert types == set(ENTITY_TYPES)


def test_settings_read_env_file_and_hide_key(tmp_path: Path, monkeypatch):
    for name in ("OPENAI_API_KEY", "KNA_MODEL", "KNA_MAX_STEPS"):
        monkeypatch.delenv(name, raising=False)
    (tmp_path / ".env").write_text("OPENAI_API_KEY=sk-secret\nKNA_MAX_STEPS=7\n", encoding="utf-8")
    settings = load_settings(tmp_path)
    assert settings.max_steps == 7
    assert settings.model == "gpt-6-luna"
    assert settings.openai_api_key == "sk-secret"
    assert "sk-secret" not in repr(settings)
    assert settings.logs_dir == tmp_path.resolve() / "logs"
