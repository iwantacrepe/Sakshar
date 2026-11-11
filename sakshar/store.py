"""SQLite persistence. Shared by request handlers and pipeline worker threads."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from pathlib import Path

from .models import ChatTurn, DocumentAnalysis

SCHEMA = """
CREATE TABLE IF NOT EXISTS documents (
    id TEXT PRIMARY KEY,
    filename TEXT NOT NULL,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    analysis TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS corrections (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    field TEXT NOT NULL,
    old_value TEXT,
    new_value TEXT,
    created_at REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS chat (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    doc_id TEXT NOT NULL,
    turn TEXT NOT NULL,
    created_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS chat_doc ON chat(doc_id);
"""


class Store:
    def __init__(self, path: Path | str):
        self.path = str(path)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=10)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            if self.path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
            self._conn.executescript(SCHEMA)
            self._conn.commit()

    # -- documents -------------------------------------------------------
    def save(self, doc: DocumentAnalysis) -> None:
        now = time.time()
        payload = doc.model_dump_json()
        with self._lock:
            self._conn.execute(
                "INSERT INTO documents(id, filename, created_at, updated_at, analysis) VALUES (?,?,?,?,?) "
                "ON CONFLICT(id) DO UPDATE SET analysis=excluded.analysis, updated_at=excluded.updated_at",
                (doc.id, doc.filename, now, now, payload),
            )
            self._conn.commit()

    def get(self, doc_id: str) -> DocumentAnalysis | None:
        with self._lock:
            row = self._conn.execute("SELECT analysis FROM documents WHERE id=?", (doc_id,)).fetchone()
        return DocumentAnalysis.model_validate_json(row["analysis"]) if row else None

    def list(self, limit: int = 20) -> list[DocumentAnalysis]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT analysis FROM documents ORDER BY created_at DESC LIMIT ?", (limit,)
            ).fetchall()
        return [DocumentAnalysis.model_validate_json(r["analysis"]) for r in rows]

    # -- corrections -----------------------------------------------------
    def log_correction(self, doc_id: str, field: str, old: str | None, new: str | None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO corrections(doc_id, field, old_value, new_value, created_at) VALUES (?,?,?,?,?)",
                (doc_id, field, old, new, time.time()),
            )
            self._conn.commit()

    # -- chat ------------------------------------------------------------
    def add_turn(self, doc_id: str, turn: ChatTurn) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO chat(doc_id, turn, created_at) VALUES (?,?,?)",
                (doc_id, turn.model_dump_json(), time.time()),
            )
            self._conn.commit()

    def turns(self, doc_id: str) -> list[ChatTurn]:
        with self._lock:
            rows = self._conn.execute("SELECT turn FROM chat WHERE doc_id=? ORDER BY id", (doc_id,)).fetchall()
        return [ChatTurn.model_validate(json.loads(r["turn"])) for r in rows]


_store: Store | None = None
_store_lock = threading.Lock()


def get_store() -> Store:
    global _store
    if _store is None:
        with _store_lock:
            if _store is None:
                from .config import get_settings

                _store = Store(get_settings().data_dir / "sakshar.db")
    return _store


def set_store(store: Store) -> None:
    """Used by tests to inject an in-memory store."""
    global _store
    _store = store
