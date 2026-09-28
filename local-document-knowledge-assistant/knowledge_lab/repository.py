from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path


def now() -> str:
    return datetime.now(UTC).isoformat()


class Repository:
    """SQLite owns source documents, immutable index snapshots, runs and conversations."""

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path = path
        with self.connect() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS records (
                    kind TEXT NOT NULL, id TEXT NOT NULL, body TEXT NOT NULL,
                    PRIMARY KEY(kind, id));
                CREATE TABLE IF NOT EXISTS sources (id TEXT PRIMARY KEY, content BLOB NOT NULL);
                CREATE TABLE IF NOT EXISTS chunks (
                    index_id TEXT NOT NULL, id TEXT NOT NULL, ordinal INTEGER NOT NULL,
                    body TEXT NOT NULL, vector TEXT NOT NULL, PRIMARY KEY(index_id, id));
            """)

    @contextmanager
    def connect(self):
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        try:
            with db:
                yield db
        finally:
            db.close()

    def put(self, kind: str, record: dict):
        with self.connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO records VALUES (?, ?, ?)",
                (kind, record["id"], json.dumps(record)),
            )

    def get(self, kind: str, identifier: str) -> dict | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT body FROM records WHERE kind=? AND id=?", (kind, identifier)
            ).fetchone()
            return json.loads(row[0]) if row else None

    def save_chat(self, run: dict, conversation: dict):
        with self.connect() as db:
            db.executemany(
                "INSERT OR REPLACE INTO records VALUES (?, ?, ?)",
                [
                    ("runs", run["id"], json.dumps(run)),
                    ("conversations", conversation["id"], json.dumps(conversation)),
                ],
            )

    def list(self, kind: str) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT body FROM records WHERE kind=? ORDER BY rowid DESC", (kind,)
            ).fetchall()
            return [json.loads(row[0]) for row in rows]

    def delete(self, kind: str, identifier: str):
        with self.connect() as db:
            db.execute("DELETE FROM records WHERE kind=? AND id=?", (kind, identifier))

    def add_document(self, record: dict, content: bytes):
        with self.connect() as db:
            db.execute(
                "INSERT INTO records VALUES ('documents', ?, ?)", (record["id"], json.dumps(record))
            )
            db.execute("INSERT INTO sources VALUES (?, ?)", (record["id"], content))

    def source(self, identifier: str) -> bytes | None:
        with self.connect() as db:
            row = db.execute("SELECT content FROM sources WHERE id=?", (identifier,)).fetchone()
            return row[0] if row else None

    def remove_document(self, identifier: str):
        with self.connect() as db:
            db.execute("DELETE FROM sources WHERE id=?", (identifier,))
            db.execute("DELETE FROM records WHERE kind='documents' AND id=?", (identifier,))

    def save_chunks(self, index_id: str, chunks, vectors):
        with self.connect() as db:
            db.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?)",
                [
                    (index_id, c.id, i, c.model_dump_json(), json.dumps(v))
                    for i, (c, v) in enumerate(zip(chunks, vectors, strict=True))
                ],
            )

    def chunks(self, index_id: str) -> list[dict]:
        with self.connect() as db:
            rows = db.execute(
                "SELECT body, vector FROM chunks WHERE index_id=? ORDER BY ordinal", (index_id,)
            ).fetchall()
            return [{"chunk": json.loads(r[0]), "vector": json.loads(r[1])} for r in rows]

    def remove_chunks(self, index_id: str):
        with self.connect() as db:
            db.execute("DELETE FROM chunks WHERE index_id=?", (index_id,))

    def recover(self):
        for kind in ("indexes", "evaluations"):
            for record in self.list(kind):
                if record["status"] in {"queued", "building", "running"}:
                    if kind == "indexes" and record.get("stage") == "storing":
                        record["cleanup_pending"] = True
                    record.update(
                        status="failed", error="Server stopped during this job. Retry it."
                    )
                    self.put(kind, record)
