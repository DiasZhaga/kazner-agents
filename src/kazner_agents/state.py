"""Task state in SQLite: runs, batch statuses and every delivered message.

One file, no server, and every write is committed at once, so the state survives a crash.
"""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

from kazner_agents.errors import AgentError
from kazner_agents.messages import CollectRequest, DocumentBatch, Envelope

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

    def load_document_batch(self, run_id: str, batch_id: str) -> DocumentBatch:
        """The DocumentBatch of a batch, as the Collector sent it (words and provenance)."""
        row = self.conn.execute(
            "SELECT envelope_json FROM messages WHERE run_id = ? AND batch_id = ?"
            " AND type = 'DocumentBatch' ORDER BY step LIMIT 1",
            (run_id, batch_id),
        ).fetchone()
        if row is None:
            raise AgentError(f"no DocumentBatch stored for batch {batch_id}")
        return Envelope.model_validate_json(row["envelope_json"]).payload
