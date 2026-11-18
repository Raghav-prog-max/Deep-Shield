"""FastAPI router for Module 02. Mount into the main DEEP SHIELD API:

    from deep_shield.api.routes_deepfake import router as m02_router
    app.include_router(m02_router)
"""
from __future__ import annotations

import re
import shutil
import uuid
from functools import lru_cache
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from deep_shield.common.schema import ModuleResult
from deep_shield.engines.deepfake import ALL_TASKS, DeepfakeImageIntegrityEngine, EngineConfig
from deep_shield.engines.deepfake.config import DEFAULT_TASKS
from deep_shield.engines.deepfake.media import IMAGE_EXT, VIDEO_EXT

router = APIRouter(prefix="/api/v1/m02", tags=["M02 Deepfake & Image Integrity"])

_SAFE = re.compile(r"^[A-Za-z0-9_.\-]+$")


@lru_cache(maxsize=1)
def get_engine() -> DeepfakeImageIntegrityEngine:
    return DeepfakeImageIntegrityEngine(EngineConfig())


@router.get("/health")
def health():
    return get_engine().health()


@router.post("/analyze", response_model=ModuleResult)
def analyze(
    file: UploadFile = File(...),
    tasks: str = Form("", description=("Comma-separated. Empty = defaults (images: " + ",".join(DEFAULT_TASKS)
                                        + "; videos: face_manipulation). Options: face_manipulation, "
                                        "frequency_analysis, image_manipulation, synthetic_image (experimental).")),
    assume_face_crop: bool = Form(False, description="Image only: treat the whole image as one face crop"),
):
    """Synchronous analysis (runs in FastAPI's threadpool). For long videos call
    ``engine.analyze`` from the background worker (RQ/Celery) with the same arguments."""
    eng = get_engine()
    cfg = eng.cfg
    name = Path(file.filename or "upload").name
    ext = Path(name).suffix.lower()
    if ext not in IMAGE_EXT | VIDEO_EXT:
        raise HTTPException(415, f"Unsupported file type '{ext}'")
    task_list = [t.strip() for t in tasks.split(",") if t.strip()] or None
    bad = [t for t in (task_list or []) if t not in ALL_TASKS]
    if bad:
        raise HTTPException(422, f"Invalid tasks {bad}; valid: {list(ALL_TASKS)}")

    up_dir = cfg.evidence_dir / "_uploads"
    up_dir.mkdir(parents=True, exist_ok=True)
    dest = up_dir / f"{uuid.uuid4().hex}{ext}"
    limit = cfg.max_upload_mb * 1024 * 1024
    written = 0
    try:
        with open(dest, "wb") as out:
            while chunk := file.file.read(1 << 20):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(413, f"File exceeds {cfg.max_upload_mb} MB limit")
                out.write(chunk)
        if written == 0:
            raise HTTPException(400, "Empty file")
        result = eng.analyze(dest, tasks=task_list, assume_face_crop=assume_face_crop, source_name=name)
    finally:
        if not cfg.keep_uploads:
            dest.unlink(missing_ok=True)
    if result.status == "failed":
        raise HTTPException(422, detail={"error": result.error, "media_id": result.media_id})
    return result


@router.get("/evidence/{media_id}/{filename}")
def evidence(media_id: str, filename: str):
    if not (_SAFE.match(media_id) and _SAFE.match(filename)) or media_id.startswith("_"):
        raise HTTPException(400, "Invalid path")
    root = get_engine().cfg.evidence_dir.resolve()
    p = (root / media_id / filename).resolve()
    if root not in p.parents or not p.is_file():
        raise HTTPException(404, "Not found")
    return FileResponse(p)


@router.delete("/evidence/{media_id}")
def delete_evidence(media_id: str):
    """Retention control: remove all derived evidence for one media item."""
    if not _SAFE.match(media_id) or media_id.startswith("_"):
        raise HTTPException(400, "Invalid id")
    root = get_engine().cfg.evidence_dir.resolve()
    d = (root / media_id).resolve()
    if root not in d.parents or not d.is_dir():
        raise HTTPException(404, "Not found")
    shutil.rmtree(d)
    return {"deleted": media_id}
