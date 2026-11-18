"""Small, per-operation SQLite store for durable job state and result JSON."""
from __future__ import annotations

import os
import shutil
import sqlite3
import json
import threading
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from deep_shield.intelligence.schema import IntelligenceResult

ROOT = Path(__file__).resolve().parents[2]
_SCHEMA_LOCK = threading.Lock()
_SCHEMA_READY: set[str] = set()


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _database_path() -> Path:
    value = Path(os.getenv("DEEPSHIELD_DATABASE_PATH", "data/deep_shield.sqlite3")).expanduser()
    return value if value.is_absolute() else (ROOT / value).resolve()


class JobRecord(BaseModel):
    job_id: str
    status: Literal["queued", "running", "completed", "partial", "failed"]
    source_name: str
    created_at: str
    updated_at: str
    media_id: str | None = None
    result: IntelligenceResult | None = None
    error: str | None = None


class QueuedJob(BaseModel):
    job_id: str
    source_name: str
    upload_path: Path
    zones: list[dict]
    attempts: int


class AnalysisJobStore:
    def __init__(self, database_path: Path | None = None):
        self.path = database_path or _database_path()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 15000")
        schema_key = str(self.path.resolve())
        if schema_key not in _SCHEMA_READY:
            with _SCHEMA_LOCK:
                if schema_key not in _SCHEMA_READY:
                    conn.execute("BEGIN IMMEDIATE")
                    conn.execute("""CREATE TABLE IF NOT EXISTS analysis_jobs (
                        job_id TEXT PRIMARY KEY,
                        status TEXT NOT NULL,
                        source_name TEXT NOT NULL,
                        created_at TEXT NOT NULL,
                        updated_at TEXT NOT NULL,
                        media_id TEXT,
                        result_json TEXT,
                        error TEXT,
                        upload_path TEXT,
                        zones_json TEXT NOT NULL DEFAULT '[]',
                        attempts INTEGER NOT NULL DEFAULT 0,
                        claimed_by TEXT,
                        heartbeat_at TEXT
                    )""")
                    columns = {row[1] for row in conn.execute("PRAGMA table_info(analysis_jobs)")}
                    for name, definition in (
                        ("upload_path", "TEXT"),
                        ("zones_json", "TEXT NOT NULL DEFAULT '[]'"),
                        ("attempts", "INTEGER NOT NULL DEFAULT 0"),
                        ("claimed_by", "TEXT"),
                        ("heartbeat_at", "TEXT"),
                    ):
                        if name not in columns:
                            conn.execute(f"ALTER TABLE analysis_jobs ADD COLUMN {name} {definition}")
                    conn.execute("CREATE INDEX IF NOT EXISTS analysis_jobs_queue_idx ON analysis_jobs(status, created_at)")
                    conn.commit()
                    _SCHEMA_READY.add(schema_key)
        return conn

    @contextmanager
    def _connection(self):
        conn = self._connect()
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def create(self, job_id: str, source_name: str, upload_path: Path, zones: list[dict]) -> JobRecord:
        now = _timestamp()
        with self._connection() as conn:
            conn.execute(
                """INSERT INTO analysis_jobs
                   (job_id,status,source_name,created_at,updated_at,upload_path,zones_json)
                   VALUES(?,?,?,?,?,?,?)""",
                (job_id, "queued", source_name, now, now, str(upload_path), json.dumps(zones)),
            )
        return JobRecord(job_id=job_id, status="queued", source_name=source_name,
                         created_at=now, updated_at=now)

    def claim_next(self, worker_id: str, lease_seconds: int = 90, max_attempts: int = 3) -> QueuedJob | None:
        now = datetime.now(timezone.utc)
        stale_before = (now - timedelta(seconds=lease_seconds)).isoformat(timespec="seconds")
        now_text = now.isoformat(timespec="seconds")
        with self._connection() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                """SELECT * FROM analysis_jobs WHERE status='queued'
                   OR (status='running' AND (heartbeat_at IS NULL OR heartbeat_at < ?))
                   ORDER BY created_at, job_id LIMIT 1""",
                (stale_before,),
            ).fetchone()
            if row is None:
                return None
            if not row["upload_path"] or not Path(row["upload_path"]).is_file():
                conn.execute(
                    "UPDATE analysis_jobs SET status='failed', updated_at=?, error=?, claimed_by=NULL, heartbeat_at=NULL WHERE job_id=?",
                    (now_text, "Queued upload is missing from persistent storage.", row["job_id"]),
                )
                return None
            if row["attempts"] >= max_attempts:
                conn.execute(
                    "UPDATE analysis_jobs SET status='failed', updated_at=?, error=?, claimed_by=NULL, heartbeat_at=NULL WHERE job_id=?",
                    (now_text, "Worker stopped before completing this job after repeated attempts.", row["job_id"]),
                )
                return None
            attempts = row["attempts"] + 1
            conn.execute(
                """UPDATE analysis_jobs SET status='running', updated_at=?, attempts=?, claimed_by=?, heartbeat_at=?, error=NULL
                   WHERE job_id=?""",
                (now_text, attempts, worker_id, now_text, row["job_id"]),
            )
            return QueuedJob(job_id=row["job_id"], source_name=row["source_name"],
                             upload_path=Path(row["upload_path"]), zones=json.loads(row["zones_json"] or "[]"),
                             attempts=attempts)

    def heartbeat(self, job_id: str, worker_id: str) -> bool:
        now = _timestamp()
        with self._connection() as conn:
            changed = conn.execute(
                "UPDATE analysis_jobs SET heartbeat_at=?, updated_at=? WHERE job_id=? AND status='running' AND claimed_by=?",
                (now, now, job_id, worker_id),
            ).rowcount
        return changed == 1

    def finish(
        self,
        job_id: str,
        worker_id: str,
        *,
        status: Literal["completed", "partial", "failed"],
        media_id: str | None = None,
        result: IntelligenceResult | None = None,
        error: str | None = None,
    ) -> bool:
        now = _timestamp()
        with self._connection() as conn:
            changed = conn.execute(
                """UPDATE analysis_jobs SET status=?, updated_at=?, media_id=?, result_json=?, error=?,
                   claimed_by=NULL, heartbeat_at=NULL WHERE job_id=? AND status='running' AND claimed_by=?""",
                (status, now, media_id, result.model_dump_json() if result is not None else None,
                 error, job_id, worker_id),
            ).rowcount
        return changed == 1

    def update(
        self,
        job_id: str,
        *,
        status: Literal["running", "completed", "partial", "failed"],
        media_id: str | None = None,
        result: IntelligenceResult | None = None,
        error: str | None = None,
    ) -> None:
        with self._connection() as conn:
            conn.execute(
                """UPDATE analysis_jobs SET status=?, updated_at=?, media_id=?, result_json=?, error=?
                   WHERE job_id=?""",
                (status, _timestamp(), media_id,
                 result.model_dump_json() if result is not None else None, error, job_id),
            )

    def get(self, job_id: str) -> JobRecord | None:
        with self._connection() as conn:
            row = conn.execute("SELECT * FROM analysis_jobs WHERE job_id=?", (job_id,)).fetchone()
        if row is None:
            return None
        result = IntelligenceResult.model_validate_json(row["result_json"]) if row["result_json"] else None
        return JobRecord(
            job_id=row["job_id"], status=row["status"], source_name=row["source_name"],
            created_at=row["created_at"], updated_at=row["updated_at"],
            media_id=row["media_id"], result=result, error=row["error"],
        )

    def delete(self, job_id: str, evidence_root: Path | None = None) -> bool:
        """Delete a job record, any lingering source upload, and its derived evidence if no other job uses that media_id."""
        with self._connection() as conn:
            row = conn.execute("SELECT media_id, upload_path FROM analysis_jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                return False
            media_id = row["media_id"]
            upload_path = row["upload_path"]
            conn.execute("DELETE FROM analysis_jobs WHERE job_id=?", (job_id,))
            remaining = (
                conn.execute("SELECT COUNT(*) FROM analysis_jobs WHERE media_id=?", (media_id,)).fetchone()[0]
                if media_id else 0
            )
        if upload_path:
            Path(upload_path).unlink(missing_ok=True)
        if media_id and remaining == 0:
            self._delete_media_evidence(media_id, evidence_root)
        return True

    def purge_expired(self, max_age_hours: float, evidence_root: Path | None = None) -> int:
        """Purge completed/partial/failed jobs older than max_age_hours and their derived evidence."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=max_age_hours)).isoformat(timespec="seconds")
        with self._connection() as conn:
            rows = conn.execute(
                """SELECT job_id FROM analysis_jobs
                   WHERE status IN ('completed', 'partial', 'failed') AND updated_at <= ?""",
                (cutoff,),
            ).fetchall()
        purged = 0
        for row in rows:
            if self.delete(row["job_id"], evidence_root=evidence_root):
                purged += 1
        return purged

    @staticmethod
    def _delete_media_evidence(media_id: str, evidence_root: Path | None = None) -> None:
        if not media_id or "/" in media_id or "\\" in media_id or ".." in media_id:
            return
        base = evidence_root or (ROOT / "evidence_store")
        for mod in ("m01", "m02", "m03", "m04"):
            shutil.rmtree(base / mod / media_id, ignore_errors=True)
