"""Durable SQLite-backed analysis worker. Run with ``python -m deep_shield.jobs.worker``."""
from __future__ import annotations

import logging
import os
import socket
import threading
import time
import uuid
from pathlib import Path

from deep_shield.api.routes_intelligence import _analyze_path
from deep_shield.engines.threat.config import parse_zones
from deep_shield.jobs.store import AnalysisJobStore

logging.basicConfig(level=os.getenv("DEEPSHIELD_LOG_LEVEL", "INFO"))
logger = logging.getLogger("deep_shield.worker")


def _keep_lease(store: AnalysisJobStore, job_id: str, worker_id: str, every_seconds: int, stop: threading.Event) -> None:
    while not stop.wait(every_seconds):
        if not store.heartbeat(job_id, worker_id):
            logger.warning("Lost lease for job %s", job_id)
            return


def run() -> None:
    poll_seconds = max(0.2, float(os.getenv("DEEPSHIELD_WORKER_POLL_SECONDS", "2")))
    lease_seconds = max(30, int(os.getenv("DEEPSHIELD_WORKER_LEASE_SECONDS", "90")))
    max_attempts = max(1, int(os.getenv("DEEPSHIELD_WORKER_MAX_ATTEMPTS", "3")))
    heartbeat_seconds = max(5, lease_seconds // 3)
    retention_hours = float(os.getenv("DEEPSHIELD_RETENTION_HOURS", "168"))
    worker_id = f"{socket.gethostname()}:{os.getpid()}:{uuid.uuid4().hex[:8]}"
    store = AnalysisJobStore()
    last_purge = 0.0
    logger.info("Worker %s polling durable queue (retention_hours=%s)", worker_id, retention_hours)

    while True:
        if retention_hours > 0 and (time.monotonic() - last_purge) >= 3600:
            purged = store.purge_expired(retention_hours)
            if purged:
                logger.info("Purged %s expired job(s) older than %s hours", purged, retention_hours)
            last_purge = time.monotonic()
        job = store.claim_next(worker_id, lease_seconds=lease_seconds, max_attempts=max_attempts)
        if job is None:
            time.sleep(poll_seconds)
            continue

        logger.info("Running job %s (attempt %s)", job.job_id, job.attempts)
        stop_heartbeat = threading.Event()
        heartbeat = threading.Thread(
            target=_keep_lease,
            args=(store, job.job_id, worker_id, heartbeat_seconds, stop_heartbeat),
            name=f"lease-{job.job_id[:8]}",
            daemon=True,
        )
        heartbeat.start()
        completed_while_owned = False
        try:
            result = _analyze_path(job.upload_path, job.source_name, parse_zones(job.zones))
            failures = sum(item.status == "failed" for item in result.module_results)
            status = "failed" if failures == len(result.module_results) else (
                "partial" if failures else "completed")
            completed_while_owned = store.finish(
                job.job_id, worker_id, status=status, media_id=result.media_id, result=result)
            logger.info("Finished job %s with status %s (lease held: %s)", job.job_id, status,
                        completed_while_owned)
        except Exception as exc:  # persist unexpected failures for polling clients
            logger.exception("Job %s failed", job.job_id)
            completed_while_owned = store.finish(
                job.job_id, worker_id, status="failed", error=f"{type(exc).__name__}: {exc}")
        finally:
            stop_heartbeat.set()
            heartbeat.join(timeout=1)
            if completed_while_owned:
                job.upload_path.unlink(missing_ok=True)


if __name__ == "__main__":
    run()
