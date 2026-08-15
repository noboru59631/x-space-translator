"""Small SQLite persistence layer for jobs, transcripts, and segments."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    """Thread-safe SQLite store using one short-lived connection per operation."""

    def __init__(self, path: Path):
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self.initialize()

    def connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.path, timeout=30)
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA foreign_keys = ON")
        return connection

    def initialize(self) -> None:
        with self._lock, self.connect() as db:
            db.executescript(
                """
                CREATE TABLE IF NOT EXISTS jobs (
                    id TEXT PRIMARY KEY,
                    source_type TEXT NOT NULL,
                    source_url TEXT DEFAULT '',
                    source_path TEXT DEFAULT '',
                    status TEXT NOT NULL,
                    stage TEXT NOT NULL,
                    progress INTEGER NOT NULL DEFAULT 0,
                    mode TEXT NOT NULL DEFAULT 'standard',
                    diarize INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    completed_at TEXT,
                    error TEXT,
                    cancel_requested INTEGER NOT NULL DEFAULT 0,
                    transcript_id INTEGER
                );
                CREATE TABLE IF NOT EXISTS transcripts (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    cache_key TEXT UNIQUE,
                    source_url TEXT DEFAULT '',
                    title TEXT DEFAULT '',
                    detected_language TEXT DEFAULT '',
                    language_probability REAL,
                    duration REAL NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL,
                    metadata_json TEXT DEFAULT '{}'
                );
                CREATE TABLE IF NOT EXISTS segments (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    transcript_id INTEGER NOT NULL REFERENCES transcripts(id) ON DELETE CASCADE,
                    speaker TEXT NOT NULL,
                    start REAL NOT NULL,
                    end REAL NOT NULL,
                    original TEXT NOT NULL,
                    translation_ja TEXT DEFAULT ''
                );
                CREATE INDEX IF NOT EXISTS idx_segments_transcript ON segments(transcript_id);
                """
            )

    def create_job(
        self,
        job_id: str,
        source_type: str,
        source_url: str = "",
        source_path: str = "",
        mode: str = "standard",
        diarize: bool = False,
    ) -> None:
        with self._lock, self.connect() as db:
            db.execute(
                "INSERT INTO jobs (id, source_type, source_url, source_path, status, stage, progress, mode, diarize, created_at) VALUES (?, ?, ?, ?, 'processing', 'queued', 0, ?, ?, ?)",
                (
                    job_id,
                    source_type,
                    source_url,
                    source_path,
                    mode,
                    int(diarize),
                    utc_now(),
                ),
            )

    def update_job(self, job_id: str, **values: Any) -> None:
        allowed = {
            "status",
            "stage",
            "progress",
            "completed_at",
            "error",
            "cancel_requested",
            "transcript_id",
            "source_path",
        }
        values = {key: value for key, value in values.items() if key in allowed}
        if not values:
            return
        assignments = ", ".join(f"{key} = ?" for key in values)
        with self._lock, self.connect() as db:
            db.execute(
                f"UPDATE jobs SET {assignments} WHERE id = ?",
                (*values.values(), job_id),
            )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute("SELECT * FROM jobs WHERE id = ?", (job_id,)).fetchone()
        return dict(row) if row else None

    def find_cached(self, cache_key: str) -> int | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT id FROM transcripts WHERE cache_key = ?", (cache_key,)
            ).fetchone()
        return int(row["id"]) if row else None

    def save_transcript(self, cache_key: str, result: dict[str, Any]) -> int:
        with self._lock, self.connect() as db:
            cursor = db.execute(
                "INSERT INTO transcripts (cache_key, source_url, title, detected_language, language_probability, duration, created_at, metadata_json) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    cache_key,
                    result.get("source_url", ""),
                    result.get("title", ""),
                    result.get("detected_language", ""),
                    result.get("language_probability"),
                    result.get("duration", 0),
                    utc_now(),
                    json.dumps(result.get("metadata", {}), ensure_ascii=False),
                ),
            )
            transcript_id = int(cursor.lastrowid)
            db.executemany(
                "INSERT INTO segments (transcript_id, speaker, start, end, original, translation_ja) VALUES (?, ?, ?, ?, ?, ?)",
                [
                    (
                        transcript_id,
                        s.get("speaker", "Speaker A"),
                        s["start"],
                        s["end"],
                        s["original"],
                        s.get("translation_ja", ""),
                    )
                    for s in result.get("segments", [])
                ],
            )
        return transcript_id

    def get_transcript(self, transcript_id: int) -> dict[str, Any] | None:
        with self.connect() as db:
            row = db.execute(
                "SELECT * FROM transcripts WHERE id = ?", (transcript_id,)
            ).fetchone()
            if not row:
                return None
            segments = [
                dict(item)
                for item in db.execute(
                    "SELECT speaker, start, end, original, translation_ja FROM segments WHERE transcript_id = ? ORDER BY start",
                    (transcript_id,),
                )
            ]
        result = dict(row)
        result.pop("cache_key", None)
        result.pop("metadata_json", None)
        result["segments"] = segments
        return result

    def update_translations(self, transcript_id: int, translations: list[str]) -> None:
        with self._lock, self.connect() as db:
            rows = db.execute(
                "SELECT id FROM segments WHERE transcript_id = ? ORDER BY start",
                (transcript_id,),
            ).fetchall()
            db.executemany(
                "UPDATE segments SET translation_ja = ? WHERE id = ?",
                [
                    (text, row["id"])
                    for row, text in zip(rows, translations, strict=False)
                ],
            )

    def rename_speakers(self, transcript_id: int, names: dict[str, str]) -> None:
        with self._lock, self.connect() as db:
            for old, new in names.items():
                db.execute(
                    "UPDATE segments SET speaker = ? WHERE transcript_id = ? AND speaker = ?",
                    (new, transcript_id, old),
                )

    def recover_interrupted_jobs(self) -> None:
        with self._lock, self.connect() as db:
            db.execute(
                "UPDATE jobs SET status = 'failed', stage = 'interrupted', error = 'アプリの再起動により処理が中断されました。' WHERE status = 'processing'"
            )
