"""API for linking independent engine outputs for reviewer navigation."""
from pathlib import Path
import os
import re
import tempfile
import uuid
from io import BytesIO

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse

from deep_shield.api.routes_deepfake import get_engine as get_m02_engine
from deep_shield.api.routes_forensics import get_engine as get_m04_engine
from deep_shield.api.routes_threat import get_engine as get_m01_engine
from deep_shield.api.routes_video_events import get_engine as get_m03_engine
from deep_shield.engines.deepfake.media import IMAGE_EXT, VIDEO_EXT, MediaError, media_kind
from deep_shield.intelligence import IntelligenceResult, LinkRequest, link_module_results
from deep_shield.engines.threat.config import parse_zones
from deep_shield.jobs import AnalysisJobStore, JobRecord
from deep_shield.reporting import render_analysis_report

router = APIRouter(prefix="/api/v1/intelligence", tags=["Cross-module Intelligence"])
_JOB_ID = re.compile(r"^[a-f0-9]{32}$")


def _analyze_path(path: Path, filename: str, zones) -> IntelligenceResult:
    kind = media_kind(path)
    results = [
        get_m01_engine().analyze(path, zones=zones, source_name=filename),
        get_m02_engine().analyze(path, source_name=filename),
    ]
    if kind == "video":
        results.append(get_m03_engine().analyze(path, zones=zones, source_name=filename))
    results.append(get_m04_engine().analyze(path, source_name=filename))
    return link_module_results(results)


@router.post("/link", response_model=IntelligenceResult)
def link_results(request: LinkRequest):
    """Link submitted ModuleResults by same-media time and region overlap."""
    try:
        return link_module_results(request.module_results)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc


@router.post("/analyze", response_model=IntelligenceResult)
def analyze_all(
    file: UploadFile = File(...),
    zones: str = Form("", description="Optional normalized polygons used by M01 and M03."),
):
    """Run applicable engines sequentially on one uploaded source file."""
    filename = Path((file.filename or "upload").replace("\\", "/")).name
    extension = Path(filename).suffix.lower()
    if extension not in IMAGE_EXT | VIDEO_EXT:
        raise HTTPException(415, f"Unsupported file type '{extension}'")
    try:
        parsed_zones = parse_zones(zones or None)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc

    limit = int(os.getenv("DEEPSHIELD_MAX_UPLOAD_MB", "200")) * 1024 * 1024
    written = 0
    try:
        with tempfile.TemporaryDirectory(prefix="deep_shield_analysis_") as temporary:
            destination = Path(temporary) / f"source{extension}"
            with destination.open("wb") as output:
                while chunk := file.file.read(1 << 20):
                    written += len(chunk)
                    if written > limit:
                        raise HTTPException(413, f"File exceeds {limit // (1024 * 1024)} MB limit")
                    output.write(chunk)
            if not written:
                raise HTTPException(400, "Empty file")
            try:
                return _analyze_path(destination, filename, parsed_zones)
            except (MediaError, OSError, ValueError) as exc:
                raise HTTPException(422, f"Unreadable media: {exc}") from exc
    finally:
        file.file.close()


@router.post("/jobs", response_model=JobRecord, status_code=202)
def create_job(
    file: UploadFile = File(...),
    zones: str = Form("", description="Optional normalized polygons used by M01 and M03."),
):
    """Queue analysis and return a job ID for status/result polling."""
    filename = Path((file.filename or "upload").replace("\\", "/")).name
    extension = Path(filename).suffix.lower()
    if extension not in IMAGE_EXT | VIDEO_EXT:
        raise HTTPException(415, f"Unsupported file type '{extension}'")
    try:
        parsed_zones = parse_zones(zones or None)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    limit = int(os.getenv("DEEPSHIELD_MAX_UPLOAD_MB", "200")) * 1024 * 1024
    job_id = uuid.uuid4().hex
    destination = Path(os.getenv("DEEPSHIELD_JOB_UPLOAD_DIR", "evidence_store/jobs/uploads"))
    if not destination.is_absolute():
        destination = Path(__file__).resolve().parents[2] / destination
    destination.mkdir(parents=True, exist_ok=True)
    source_path = destination / f"{job_id}{extension}"
    written = 0
    try:
        with source_path.open("wb") as output:
            while chunk := file.file.read(1 << 20):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(413, f"File exceeds {limit // (1024 * 1024)} MB limit")
                output.write(chunk)
        if not written:
            raise HTTPException(400, "Empty file")
        job = AnalysisJobStore().create(
            job_id, filename, source_path,
            [zone.model_dump() for zone in parsed_zones],
        )
        return job
    except HTTPException:
        source_path.unlink(missing_ok=True)
        raise
    except Exception:
        source_path.unlink(missing_ok=True)
        raise
    finally:
        file.file.close()


@router.get("/jobs/{job_id}", response_model=JobRecord)
def get_job(job_id: str):
    if not _JOB_ID.fullmatch(job_id):
        raise HTTPException(400, "Invalid job ID")
    job = AnalysisJobStore().get(job_id)
    if job is None:
        raise HTTPException(404, "Analysis job not found")
    return job


@router.get("/jobs/{job_id}/result", response_model=IntelligenceResult)
def get_job_result(job_id: str):
    if not _JOB_ID.fullmatch(job_id):
        raise HTTPException(400, "Invalid job ID")
    job = AnalysisJobStore().get(job_id)
    if job is None:
        raise HTTPException(404, "Analysis job not found")
    if job.result is None:
        raise HTTPException(409, f"Job is {job.status}; result is not available yet")
    return job.result


@router.get("/jobs/{job_id}/report.pdf")
def get_job_report(job_id: str):
    if not _JOB_ID.fullmatch(job_id):
        raise HTTPException(400, "Invalid job ID")
    job = AnalysisJobStore().get(job_id)
    if job is None:
        raise HTTPException(404, "Analysis job not found")
    if job.result is None:
        raise HTTPException(409, f"Job is {job.status}; report is not available yet")
    content = render_analysis_report(job.result, job.job_id)
    return StreamingResponse(
        BytesIO(content),
        media_type="application/pdf",
        headers={"Content-Disposition": f'attachment; filename="deep-shield-{job.job_id}.pdf"'},
    )


@router.delete("/jobs/{job_id}")
def delete_job(job_id: str):
    """Retention control: delete a job record and its derived evidence."""
    if not _JOB_ID.fullmatch(job_id):
        raise HTTPException(400, "Invalid job ID")
    if not AnalysisJobStore().delete(job_id):
        raise HTTPException(404, "Analysis job not found")
    return {"deleted": job_id}


@router.post("/retention/purge")
def purge_expired_jobs(max_age_hours: float = 168.0):
    """Retention control: purge completed/failed jobs and derived evidence older than max_age_hours."""
    if max_age_hours < 0:
        raise HTTPException(422, "max_age_hours must be non-negative")
    purged = AnalysisJobStore().purge_expired(max_age_hours)
    return {"purged_jobs": purged, "max_age_hours": max_age_hours}


@router.get("/models")
def list_loaded_models_and_parameters():
    """Return all loaded weights, revisions, parameter counts, and calibrated thresholds across M01-M04."""
    return {
        "m01_threat_intelligence": get_m01_engine().health(),
        "m02_deepfake_image_integrity": get_m02_engine().health(),
        "m03_video_events": get_m03_engine().health(),
        "m04_visual_forensics": get_m04_engine().health(),
    }
