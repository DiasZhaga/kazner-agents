"""Deterministic offline stand-ins used by tests, demos and scenarios (no network, no model)."""

from __future__ import annotations

from kazner_agents.agents.llm_annotator import fake_annotation_responder
from kazner_agents.agents.pre_annotator import FakeTagger
from kazner_agents.agents.verifier import AdjudicationAnswer, fake_adjudication_responder
from kazner_agents.config import Settings
from kazner_agents.llm import LLMRequest


def fake_responder(request: LLMRequest) -> dict:
    """FakeLLM answers for both kinds of LLM requests: annotation and adjudication."""
    if request.schema is AdjudicationAnswer:
        return fake_adjudication_responder(request)
    return fake_annotation_responder(request)


def fake_tagger(settings: Settings) -> FakeTagger:
    return FakeTagger()
