# kazner-agents — multi-agent annotation system for Kazakh NER

## Purpose
Course project for "Artificial Intelligence and Machine Learning": a multi-agent AI system that
produces word-level named-entity annotations for Kazakh text. It supports the author's MSc thesis
on adapting language models to Kazakh (out-of-domain evaluation data; LLM-based NER as in WP3).

The project is graded in three stages. CURRENT STAGE: Assignment 2 (due tomorrow) — architecture,
agent specification, and a prototype in which at least two agents exchange structured messages.
Assignment 4 (week 10) will need the full system; design for it, but do not build it now.

All architecture decisions are in `DESIGN.md`. Treat it as the specification. If something in it
turns out to be wrong or impossible, say so and propose a change — do not silently deviate.

## Related folders (read-only for this project)
- `../ner-project` — pilot research code; KazNERD data in `../ner-project/data`.
- `../kazner` — the refactored NER toolkit (prediction contract and evaluator). Will be used by
  the Evaluator agent in Assignment 4; not required for Assignment 2.

## Hard rules
- API keys only in `.env` (gitignored). Provide `.env.example`. Never print or log a key.
- Every LLM call goes through ONE client module with: timeout, up to 3 retries with exponential
  backoff, token-usage logging, and a hard per-run cap on LLM calls from config.
- A deterministic `FakeLLM` backend must exist. All tests run with it — no network, no cost.
- Ask before running anything against the real API more than a few times. Report the token usage
  and estimated cost after each real run.
- Never commit datasets, model checkpoints, logs with personal data, or `.env`.
- Keep the code simple enough for the author to explain line by line at the exam (25% of the exam
  grade is "understanding of the code"). Plain Python over frameworks; no clever metaprogramming.
- Small Conventional Commits in English. Working directly on `main` is fine for this project, but
  every commit must leave the tests green.

## Stack
- Python 3.11, `pyproject.toml`, src layout, package `kazner_agents`, CLI `kna`.
- `pydantic` v2 for all messages (typed, validated, JSON-serialisable).
- `openai` Python SDK. Model name from `.env` (`KNA_MODEL`); choose an inexpensive current model
  after checking OpenAI's documentation, and use structured JSON output if the model supports it.
- `requests` for the Wikipedia API; `sqlite3` (stdlib) for task state; `pytest` for tests.
- No agent frameworks (LangGraph, CrewAI, AutoGen) — see the rationale in DESIGN.md §6.

## Logging
Every agent step and every tool/LLM call is appended to `logs/<run_id>.jsonl` with: timestamp,
run_id, step, agent, kind (`llm` | `tool` | `message`), name, status, duration_ms, and for LLM
calls the input/output tokens. `kna load-report <run_id>` prints each agent's share of LLM + tool
calls and fails if any agent exceeds 40%.

## Testing
- `pytest` must pass offline in under a minute.
- Cover: message schema validation, the boundary normaliser (span → IOB2, repair of invalid
  sequences), orchestrator step/time limits and loop protection, retry behaviour of the LLM client
  (with a fake that fails N times), and the load report.

## Environment
User is on Windows + PowerShell. Use `pathlib`; no assumptions about the current directory.

## Communication
Talk to the user in Russian. Code, comments, commit messages and repository docs in English.
