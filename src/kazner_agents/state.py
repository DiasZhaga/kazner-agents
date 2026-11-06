"""Task state in SQLite: runs, batch statuses and every delivered message.

One file, no server, and every write is committed at once, so the state survives a crash.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from kazner_agents.errors import AgentError
from kazner_agents.messages import CollectRequest, DocumentBatch, Envelope, Message

SCHEMA = """
CREATE TABLE IF NOT EXISTS runs (
    run_id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    task_json TEXT NOT NULL,
    status TEXT NOT NULL,
    reason TEXT,
    finished_at TEXT
);
CREATE TABLE IF NOT EXISTS batches (
    run_id TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    status TEXT NOT NULL,
    reason TEXT,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (run_id, batch_id)
);
CREATE TABLE IF NOT EXISTS messages (
    msg_id TEXT PRIMARY KEY,
    run_id TEXT NOT NULL,
    step INTEGER NOT NULL,
    batch_id TEXT,
    sender TEXT NOT NULL,
    recipient TEXT NOT NULL,
    type TEXT NOT NULL,
    envelope_json TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS review_queue (
    run_id TEXT NOT NULL,
    batch_id TEXT NOT NULL,
    sentence_id TEXT NOT NULL,
    start_word INTEGER NOT NULL,
    end_word INTEGER NOT NULL,
    words TEXT NOT NULL,
    mbert TEXT NOT NULL,
    llm TEXT NOT NULL,
    reason TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    created_at TEXT NOT NULL
);
"""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class StateStore:
    def __init__(self, path: Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(SCHEMA)

    def close(self) -> None:
        self.conn.close()

    def _write(self, sql: str, params: tuple) -> None:
        with self.conn:  # commits on success
            self.conn.execute(sql, params)

    # --- writes ----------------------------------------------------------------------------

    def create_run(self, run_id: str, task: CollectRequest) -> None:
        self._write(
            "INSERT INTO runs (run_id, created_at, task_json, status) VALUES (?, ?, ?, ?)",
            (run_id, _now(), task.model_dump_json(), "running"),
        )

    def set_batch_status(self, run_id: str, batch_id: str, status: str, reason: str | None) -> None:
        self._write(
            "INSERT OR REPLACE INTO batches (run_id, batch_id, status, reason, updated_at)"
            " VALUES (?, ?, ?, ?, ?)",
            (run_id, batch_id, status, reason, _now()),
        )

    def finish_run(self, run_id: str, status: str, reason: str | None) -> None:
        self._write(
            "UPDATE runs SET status = ?, reason = ?, finished_at = ? WHERE run_id = ?",
            (status, reason, _now(), run_id),
        )

    def save_message(self, envelope: Envelope, batch_id: str | None) -> None:
        self._write(
            "INSERT INTO messages (msg_id, run_id, step, batch_id, sender, recipient, type,"
            " envelope_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                envelope.msg_id, envelope.run_id, envelope.step, batch_id, envelope.sender,
                envelope.recipient, envelope.type, envelope.model_dump_json(),
            ),
        )

    def add_review_items(self, run_id: str, batch_id: str, items: list[dict]) -> int:
        """The human-review queue: regions the Verifier could not settle."""
        with self.conn:
            for item in items:
                self.conn.execute(
                    "INSERT INTO review_queue (run_id, batch_id, sentence_id, start_word, end_word,"
                    " words, mbert, llm, reason, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (
                        run_id, batch_id, item["sentence_id"], item["start_word"],
                        item["end_word"], item["words"], item["mbert"], item["llm"],
                        item["reason"], _now(),
                    ),
                )
        return len(items)

    # --- reads -----------------------------------------------------------------------------

    def get_run(self, run_id: str) -> dict | None:
        row = self.conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
        return dict(row) if row else None

    def get_batches(self, run_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM batches WHERE run_id = ? ORDER BY batch_id", (run_id,)
        ).fetchall()
        return [dict(row) for row in rows]

    def count_messages(self, run_id: str) -> int:
        row = self.conn.execute("SELECT COUNT(*) FROM messages WHERE run_id = ?", (run_id,))
        return row.fetchone()[0]

    def get_review_items(self, run_id: str) -> list[dict]:
        rows = self.conn.execute(
            "SELECT * FROM review_queue WHERE run_id = ? ORDER BY batch_id, sentence_id, start_word",
            (run_id,),
        ).fetchall()
        return [dict(row) for row in rows]

    def load_payloads(self, run_id: str, batch_id: str, type_name: str) -> list[Message]:
        """Every payload of one type delivered for a batch, in delivery order.

        A payload that went to two recipients (a DocumentBatch to both annotators) is stored
        once per delivery, so only the first copy of each message content is returned.
        """
        rows = self.conn.execute(
            "SELECT envelope_json FROM messages WHERE run_id = ? AND batch_id = ? AND type = ?"
            " ORDER BY step",
            (run_id, batch_id, type_name),
        ).fetchall()
        payloads: list[Message] = []
        for row in rows:
            payload = Envelope.model_validate_json(row["envelope_json"]).payload
            if payload not in payloads:
                payloads.append(payload)
        return payloads

    def load_document_batch(self, run_id: str, batch_id: str) -> DocumentBatch:
        """The DocumentBatch of a batch, as the Collector sent it (words and provenance)."""
        found = self.load_payloads(run_id, batch_id, "DocumentBatch")
        if not found:
            raise AgentError(f"no DocumentBatch stored for batch {batch_id}")
        return found[0]
