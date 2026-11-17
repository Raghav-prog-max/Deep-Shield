import logging
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from fastapi import HTTPException

from .database import Database
from .engines.base import EngineContext, REGISTRY
from .schemas import EngineName, FindingOut

logger = logging.getLogger(__name__)


class JobService:
    def __init__(self, database: Database, media_root: Path):
        self.database = database
        self.media_root = media_root
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="deep-shield-job")

    def create(self, media_id: str, engines: list[EngineName]) -> dict:
        with self.database.connect() as conn:
            media = conn.execute("SELECT * FROM media_assets WHERE media_id=?", (media_id,)).fetchone()
        if media is None:
            raise HTTPException(status_code=404, detail="Media not found")
        job_id = str(uuid.uuid4())
        unique_engines = list(dict.fromkeys(engines))
        self.database.add_job(job_id, media_id, unique_engines)
        self.executor.submit(self._run, job_id, dict(media))
        return self.database.get_job(job_id)

    def _run(self, job_id: str, media: dict) -> None:
        try:
            self.database.update_job(job_id, "running", 0)
            job = self.database.get_job(job_id)
            if job is None:
                return
            engines: list[str] = job["requested_engines"]
            failures = []
            completed = 0
            context = EngineContext(
                media_id=media["media_id"],
                media_hash=media["sha256"],
                media_type=media["media_type"],
                path=self.media_root / media["storage_name"],
            )
            for index, engine_name in enumerate(engines):
                engine = REGISTRY.get(engine_name)
                if engine is None:
                    message = "This analysis engine is not implemented or configured in this checkout. No finding was generated."
                    self.database.update_engine_run(job_id, engine_name, "unavailable", "engine_unavailable", message)
                    failures.append(f"{engine_name}: unavailable")
                else:
                    self.database.update_engine_run(job_id, engine_name, "running")
                    try:
                        raw_findings = engine.analyze(context)
                        normalized = []
                        for item in raw_findings:
                            payload = {
                                **item,
                                "finding_id": str(uuid.uuid4()),
                                "job_id": job_id,
                                "engine": engine_name,
                                "media_hash": media["sha256"],
                            }
                            normalized.append(FindingOut.model_validate(payload).model_dump())
                        self.database.add_findings(normalized)
                        self.database.update_engine_run(job_id, engine_name, "completed")
                        completed += 1
                    except Exception:
                        logger.exception("Engine %s failed for job %s", engine_name, job_id)
                        message = "Engine failed. See the server log for diagnostics."
                        self.database.update_engine_run(job_id, engine_name, "failed", "engine_failed", message)
                        failures.append(f"{engine_name}: failed")
                self.database.update_job(job_id, "running", round((index + 1) * 100 / len(engines)))

            if failures and not completed:
                status = "failed"
                error = "No requested engine completed: " + "; ".join(failures)
            else:
                status = "completed"
                error = "; ".join(failures) if failures else None
            self.database.update_job(job_id, status, 100, error)
        except Exception:
            logger.exception("Job dispatcher failed for job %s", job_id)
            self.database.update_job(job_id, "failed", 100, "Job dispatcher failed; see server log for diagnostics.")

    def shutdown(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=False)
