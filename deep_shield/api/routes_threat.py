"""FastAPI router for Module 01 — Threat and Anomaly Intelligence."""
from __future__ import annotations

import re
import shutil
import uuid
from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

from deep_shield.common.schema import ModuleResult
from deep_shield.engines.deepfake.media import IMAGE_EXT, VIDEO_EXT
from deep_shield.engines.threat import ThreatConfig, ThreatIntelligenceEngine

router = APIRouter(prefix="/api/v1/m01", tags=["M01 Threat Intelligence"])
_SAFE = re.compile(r"^[A-Za-z0-9_.-]+$")


@lru_cache(maxsize=1)
def get_engine() -> ThreatIntelligenceEngine:
    return ThreatIntelligenceEngine(ThreatConfig())


@router.get("/health")
def health():
    return get_engine().health()


@router.post("/analyze", response_model=ModuleResult)
def analyze(
    file: UploadFile = File(...),
    zones: str = Form("", description="Optional JSON array of normalized polygon zones."),
):
    engine = get_engine()
    cfg = engine.cfg
    filename = Path((file.filename or "upload").replace("\\", "/")).name
    extension = Path(filename).suffix.lower()
    if extension not in IMAGE_EXT | VIDEO_EXT:
        raise HTTPException(415, f"Unsupported file type '{extension}'")

    upload_dir = cfg.evidence_dir / "_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    destination = upload_dir / f"{uuid.uuid4().hex}{extension}"
    written = 0
    limit = cfg.max_upload_mb * 1024 * 1024
    try:
        with destination.open("wb") as output:
            while chunk := file.file.read(1 << 20):
                written += len(chunk)
                if written > limit:
                    raise HTTPException(413, f"File exceeds {cfg.max_upload_mb} MB limit")
                output.write(chunk)
        if not written:
            raise HTTPException(400, "Empty file")
        result = engine.analyze(destination, zones=zones or None, source_name=filename)
    except ValueError as exc:
        raise HTTPException(422, str(exc)) from exc
    finally:
        if not cfg.keep_uploads:
            destination.unlink(missing_ok=True)
        file.file.close()
    if result.status == "failed":
        raise HTTPException(422, detail={"error": result.error, "media_id": result.media_id})
    return result


@router.get("/evidence/{media_id}/{filename}")
def evidence(media_id: str, filename: str):
    if not (_SAFE.fullmatch(media_id) and _SAFE.fullmatch(filename)) or media_id.startswith("_"):
        raise HTTPException(400, "Invalid evidence path")
    root = get_engine().cfg.evidence_dir.resolve()
    path = (root / media_id / filename).resolve()
    if root not in path.parents or not path.is_file():
        raise HTTPException(404, "Evidence not found")
    return FileResponse(path)


@router.delete("/evidence/{media_id}")
def delete_evidence(media_id: str):
    if not _SAFE.fullmatch(media_id) or media_id.startswith("_"):
        raise HTTPException(400, "Invalid media id")
    root = get_engine().cfg.evidence_dir.resolve()
    directory = (root / media_id).resolve()
    if root not in directory.parents or not directory.is_dir():
        raise HTTPException(404, "Evidence not found")
    shutil.rmtree(directory)
    return {"deleted": media_id}
