import json
import sqlite3
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from summarizer.models import HistoryItem, SavedSummary, SummaryRequest
from summarizer.provider import Generation
from summarizer.rendering import to_markdown


class RevisionConflict(Exception):
    pass


class Repository:
    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def connection(self):
        connection = sqlite3.connect(self.path, timeout=10)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA secure_delete=ON")
        try:
            with connection:
                yield connection
        finally:
            connection.close()

    def initialize(self):
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as connection:
            connection.execute("""CREATE TABLE IF NOT EXISTS summaries (
                id TEXT PRIMARY KEY, title TEXT NOT NULL, transcript TEXT NOT NULL,
                summary TEXT NOT NULL, metrics TEXT NOT NULL, draft_markdown TEXT NOT NULL,
                is_edited INTEGER NOT NULL DEFAULT 0, revision INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL, updated_at TEXT NOT NULL
            )""")

    @staticmethod
    def _record(row) -> SavedSummary:
        item = dict(row)
        item["summary"] = json.loads(item["summary"])
        item["metrics"] = json.loads(item["metrics"])
        return SavedSummary.model_validate(item)

    def create(self, request: SummaryRequest, generation: Generation) -> SavedSummary:
        now = datetime.now(UTC).isoformat()
        record_id = str(uuid4())
        with self.connection() as connection:
            connection.execute(
                "INSERT INTO summaries VALUES (?, ?, ?, ?, ?, ?, 0, 1, ?, ?)",
                (
                    record_id,
                    request.title,
                    request.transcript,
                    generation.summary.model_dump_json(),
                    generation.metrics.model_dump_json(),
                    to_markdown(generation.summary),
                    now,
                    now,
                ),
            )
        return self.get(record_id)

    def get(self, record_id: str) -> SavedSummary | None:
        with self.connection() as connection:
            row = connection.execute(
                "SELECT * FROM summaries WHERE id = ?", (record_id,)
            ).fetchone()
        return self._record(row) if row else None

    def list(self, limit: int, offset: int) -> list[HistoryItem]:
        with self.connection() as connection:
            rows = connection.execute(
                "SELECT id, title, created_at, updated_at, is_edited FROM summaries "
                "ORDER BY created_at DESC, id DESC LIMIT ? OFFSET ?",
                (limit, offset),
            ).fetchall()
        return [HistoryItem.model_validate(dict(row)) for row in rows]

    def update(self, record_id: str, markdown: str, revision: int) -> SavedSummary | None:
        with self.connection() as connection:
            cursor = connection.execute(
                "UPDATE summaries SET draft_markdown = ?, is_edited = 1, revision = revision + 1, "
                "updated_at = ? WHERE id = ? AND revision = ?",
                (markdown, datetime.now(UTC).isoformat(), record_id, revision),
            )
            if cursor.rowcount == 0:
                if connection.execute(
                    "SELECT 1 FROM summaries WHERE id = ?", (record_id,)
                ).fetchone():
                    raise RevisionConflict
                return None
        return self.get(record_id)

    def delete(self, record_id: str) -> bool:
        with self.connection() as connection:
            return (
                connection.execute("DELETE FROM summaries WHERE id = ?", (record_id,)).rowcount > 0
            )
