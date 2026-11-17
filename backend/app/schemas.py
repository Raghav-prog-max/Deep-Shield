from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


EngineName = Literal["engine1", "engine2", "engine3", "engine4"]
MediaType = Literal["image", "video"]
JobStatus = Literal["queued", "running", "completed", "failed"]
EngineRunStatus = Literal["queued", "running", "completed", "failed", "unavailable"]


class MediaOut(BaseModel):
    media_id: str
    original_name: str
    media_type: MediaType
    sha256: str
    size_bytes: int
    created_at: datetime
    status: Literal["accepted"] = "accepted"


class JobCreate(BaseModel):
    media_id: str
    requested_engines: list[EngineName] = Field(min_length=1, max_length=4)


class EngineRunOut(BaseModel):
    engine: EngineName
    status: EngineRunStatus
    error_code: str | None = None
    message: str | None = None


class JobOut(BaseModel):
    job_id: str
    media_id: str
    status: JobStatus
    progress: int = Field(ge=0, le=100)
    requested_engines: list[EngineName]
    engine_runs: list[EngineRunOut] = Field(default_factory=list)
    error: str | None = None
    created_at: datetime
    updated_at: datetime


class FindingOut(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    finding_id: str
    job_id: str
    engine: EngineName
    media_hash: str
    timestamp: float | None = None
    frame_number: int | None = None
    track_id: str | None = None
    label: str
    score: float | None = None
    score_type: str | None = None
    evidence_path: str | None = None
    explanation: str
    model_version: str | None = None
    limitations: list[str] = Field(default_factory=list)
