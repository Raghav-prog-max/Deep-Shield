import json
import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator


SCHEMA = """
PRAGMA foreign_keys = ON;
CREATE TABLE IF NOT EXISTS media_assets (
    media_id TEXT PRIMARY KEY,
    original_name TEXT NOT NULL,
    media_type TEXT NOT NULL CHECK (media_type IN ('image', 'video')),
    storage_name TEXT NOT NULL UNIQUE,
    sha256 TEXT NOT NULL,
    size_bytes INTEGER NOT NULL CHECK (size_bytes > 0),
    created_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS analysis_jobs (
    job_id TEXT PRIMARY KEY,
    media_id TEXT NOT NULL REFERENCES media_assets(media_id),
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'completed', 'failed')),
    requested_engines TEXT NOT NULL,
    progress INTEGER NOT NULL DEFAULT 0 CHECK (progress BETWEEN 0 AND 100),
    error TEXT,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS engine_runs (
    job_id TEXT NOT NULL REFERENCES analysis_jobs(job_id) ON DELETE CASCADE,
    engine TEXT NOT NULL,
    status TEXT NOT NULL CHECK (status IN ('queued', 'running', 'completed', 'failed', 'unavailable')),
    error_code TEXT,
    message TEXT,
    PRIMARY KEY (job_id, engine)
);
CREATE TABLE IF NOT EXISTS findings (
    finding_id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL REFERENCES analysis_jobs(job_id) ON DELETE CASCADE,
    engine TEXT NOT NULL,
    media_hash TEXT NOT NULL,
    timestamp REAL,
    frame_number INTEGER,
    track_id TEXT,
    label TEXT NOT NULL,
    score REAL,
    score_type TEXT,
    evidence_path TEXT,
    explanation TEXT NOT NULL,
    model_version TEXT,
    limitations TEXT NOT NULL DEFAULT '[]'
);
CREATE INDEX IF NOT EXISTS idx_jobs_media ON analysis_jobs(media_id, created_at);
CREATE INDEX IF NOT EXISTS idx_findings_job ON findings(job_id, engine);
"""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Database:
    def __init__(self, path: Path):
        self.path = path

    def initialize(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.path, timeout=30)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA foreign_keys = ON")
        try:
            yield conn
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()

    def add_media(self, media: dict[str, Any]) -> None:
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO media_assets
                   (media_id, original_name, media_type, storage_name, sha256, size_bytes, created_at)
                   VALUES (:media_id, :original_name, :media_type, :storage_name, :sha256, :size_bytes, :created_at)""",
                media,
            )

    def get_media(self, media_id: str) -> sqlite3.Row | None:
        with self.connect() as conn:
            return conn.execute("SELECT * FROM media_assets WHERE media_id = ?", (media_id,)).fetchone()

    def add_job(self, job_id: str, media_id: str, engines: list[str]) -> None:
        now = utc_now()
        with self.connect() as conn:
            conn.execute(
                """INSERT INTO analysis_jobs
                   (job_id, media_id, status, requested_engines, progress, created_at, updated_at)
                   VALUES (?, ?, 'queued', ?, 0, ?, ?)""",
                (job_id, media_id, json.dumps(engines), now, now),
            )
            conn.executemany(
                "INSERT INTO engine_runs (job_id, engine, status) VALUES (?, ?, 'queued')",
                [(job_id, engine) for engine in engines],
            )

    def update_job(self, job_id: str, status: str, progress: int, error: str | None = None) -> None:
        with self.connect() as conn:
            conn.execute(
                "UPDATE analysis_jobs SET status=?, progress=?, error=?, updated_at=? WHERE job_id=?",
                (status, progress, error, utc_now(), job_id),
            )

    def update_engine_run(
        self, job_id: str, engine: str, status: str, error_code: str | None = None, message: str | None = None
    ) -> None:
        with self.connect() as conn:
            conn.execute(
                """UPDATE engine_runs SET status=?, error_code=?, message=?
                   WHERE job_id=? AND engine=?""",
                (status, error_code, message, job_id, engine),
            )

    def add_findings(self, findings: list[dict[str, Any]]) -> None:
        if not findings:
            return
        with self.connect() as conn:
            conn.executemany(
                """INSERT INTO findings
                   (finding_id, job_id, engine, media_hash, timestamp, frame_number, track_id,
                    label, score, score_type, evidence_path, explanation, model_version, limitations)
                   VALUES (:finding_id, :job_id, :engine, :media_hash, :timestamp, :frame_number,
                    :track_id, :label, :score, :score_type, :evidence_path, :explanation,
                    :model_version, :limitations)""",
                [
                    {**finding, "limitations": json.dumps(finding.get("limitations", []))}
                    for finding in findings
                ],
            )

    def get_job(self, job_id: str) -> dict[str, Any] | None:
        with self.connect() as conn:
            row = conn.execute("SELECT * FROM analysis_jobs WHERE job_id=?", (job_id,)).fetchone()
            if row is None:
                return None
            result = dict(row)
            result["requested_engines"] = json.loads(result["requested_engines"])
            result["engine_runs"] = [
                dict(item) for item in conn.execute(
                    "SELECT engine, status, error_code, message FROM engine_runs WHERE job_id=? ORDER BY engine",
                    (job_id,),
                ).fetchall()
            ]
            return result

    def list_findings(self, job_id: str) -> list[dict[str, Any]]:
        with self.connect() as conn:
            rows = conn.execute(
                "SELECT * FROM findings WHERE job_id=? ORDER BY engine, timestamp, frame_number",
                (job_id,),
            ).fetchall()
        results = []
        for row in rows:
            item = dict(row)
            item["limitations"] = json.loads(item["limitations"])
            results.append(item)
        return results
