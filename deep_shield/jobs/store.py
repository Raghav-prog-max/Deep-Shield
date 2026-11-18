"""Small, per-operation SQLite store for durable job state and result JSON."""
from __future__ import annotations

import os
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Literal

from pydantic import BaseModel

from deep_shield.intelligence.schema import IntelligenceResult

ROOT = Path(__file__).resolve().parents[2]


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


class AnalysisJobStore:
    def __init__(self, database_path: Path | None = None):
        self.path = database_path or _database_path()

    def _connect(self) -> sqlite3.Connection:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path, timeout=15)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA busy_timeout = 15000")
        conn.execute("""CREATE TABLE IF NOT EXISTS analysis_jobs (
            job_id TEXT PRIMARY KEY,
            status TEXT NOT NULL,
            source_name TEXT NOT NULL,
            created_at TEXT NOT NULL,
            updated_at TEXT NOT NULL,
            media_id TEXT,
            result_json TEXT,
            error TEXT
        )""")
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

    def create(self, job_id: str, source_name: str) -> JobRecord:
        now = _timestamp()
        with self._connection() as conn:
            conn.execute(
                "INSERT INTO analysis_jobs(job_id,status,source_name,created_at,updated_at) VALUES(?,?,?,?,?)",
                (job_id, "queued", source_name, now, now),
            )
        return JobRecord(job_id=job_id, status="queued", source_name=source_name,
                         created_at=now, updated_at=now)

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
