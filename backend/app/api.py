from fastapi import APIRouter, File, HTTPException, UploadFile
from starlette.concurrency import run_in_threadpool

from .database import Database
from .engines.base import REGISTRY
from .jobs import JobService
from .media import MediaService
from .schemas import FindingOut, JobCreate, JobOut, MediaOut


def build_router(database: Database, media_service: MediaService, jobs: JobService) -> APIRouter:
    router = APIRouter()

    @router.get("/health")
    def health() -> dict:
        return {
            "status": "ok",
            "service": "deep-shield-api",
            "version": "0.1.0",
            "available_engines": sorted(REGISTRY),
        }

    @router.post("/v1/media", response_model=MediaOut, status_code=201)
    async def upload_media(file: UploadFile = File(...)) -> dict:
        return await run_in_threadpool(media_service.accept, file)

    @router.post("/v1/jobs", response_model=JobOut, status_code=202)
    def create_job(payload: JobCreate) -> dict:
        job = jobs.create(payload.media_id, payload.requested_engines)
        return job

    @router.get("/v1/jobs/{job_id}", response_model=JobOut)
    def get_job(job_id: str) -> dict:
        job = database.get_job(job_id)
        if job is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return job

    @router.get("/v1/jobs/{job_id}/findings", response_model=list[FindingOut])
    def get_findings(job_id: str) -> list[dict]:
        if database.get_job(job_id) is None:
            raise HTTPException(status_code=404, detail="Job not found")
        return database.list_findings(job_id)

    return router
