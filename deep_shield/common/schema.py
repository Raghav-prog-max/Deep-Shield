"""DEEP SHIELD — common finding schema shared by all four engines.

Every engine returns a ``ModuleResult`` holding a list of ``Finding`` objects.
The cross-module intelligence layer links findings using ``media_id``,
``frame_index`` / ``timestamp_s``, ``track_id`` and ``region`` — so every engine
must fill those fields whenever they apply.

Score semantics are explicit (``score_type``) so that detector confidence,
classifier output, anomaly scores and rule scores are never merged into one
uncalibrated "threat probability".
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, ConfigDict, Field


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class ScoreType(str, Enum):
    CLASSIFIER_PROBABILITY = "classifier_probability"      # softmax output of one model, uncalibrated
    AGGREGATE_MEAN_CLASSIFIER = "aggregate_mean_classifier_probability"  # mean over frames of a track/video
    DETECTOR_CONFIDENCE = "detector_confidence"
    ANOMALY_SCORE = "anomaly_score"
    RULE_SCORE = "rule_score"
    FORENSIC_INDICATOR = "forensic_indicator"


class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


class BBox(BaseModel):
    x: int
    y: int
    w: int
    h: int


class ModelInfo(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    model_id: str
    revision: Optional[str] = None
    task: str
    license: Optional[str] = None
    source_url: Optional[str] = None
    backend: str = "transformers"


class EvidenceRef(BaseModel):
    kind: str                     # face_crop | annotated_frame | annotated_image | score_timeline
    path: str                     # relative to the engine's evidence root
    description: Optional[str] = None


class TimePoint(BaseModel):
    t: float
    frame_index: int
    score: float


class Segment(BaseModel):
    start_s: float
    end_s: float
    peak_score: float


class TemporalSummary(BaseModel):
    n_observations: int
    mean: float
    median: float
    max: float
    std: float
    fraction_above_high: float
    segments_above_high: List[Segment] = Field(default_factory=list)
    series: List[TimePoint] = Field(default_factory=list)


class Finding(BaseModel):
    finding_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    module: str
    task: str
    level: str                    # frame | image | track | video
    media_id: str
    media_type: MediaType
    classification: str           # task-specific: e.g. face_manipulation_indicated | inconclusive | ...
    score: Optional[float] = None  # None for visual-only (exploratory) outputs
    score_type: ScoreType
    thresholds: Dict[str, float] = Field(default_factory=dict)
    predicted_label: Optional[str] = None   # raw top label from the model
    raw_scores: Dict[str, float] = Field(default_factory=dict)
    frame_index: Optional[int] = None
    timestamp_s: Optional[float] = None
    time_start_s: Optional[float] = None
    time_end_s: Optional[float] = None
    track_id: Optional[str] = None
    region: Optional[BBox] = None
    detector_confidence: Optional[float] = None
    temporal: Optional[TemporalSummary] = None
    explanation: Optional[Dict[str, Any]] = None   # method-specific details (heatmap, spectrum, maps)
    model: ModelInfo
    evidence: List[EvidenceRef] = Field(default_factory=list)
    limitations: List[str] = Field(default_factory=list)
    created_at: str = Field(default_factory=_now)


class TaskSummary(BaseModel):
    task: str
    classification: str
    score: Optional[float] = None
    score_type: Optional[ScoreType] = None
    basis: str
    model: Optional[ModelInfo] = None


class ModuleResult(BaseModel):
    module: str
    engine_version: str
    media_id: str
    media_sha256: str
    media_type: MediaType
    source_name: str
    status: str                   # completed | partial | failed
    tasks_requested: List[str]
    task_summaries: List[TaskSummary] = Field(default_factory=list)
    findings: List[Finding] = Field(default_factory=list)
    media_info: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)
    error: Optional[str] = None
    disclaimer: str = (
        "Advisory indicators for human review. Model scores are uncalibrated and are "
        "not proof of authenticity, manipulation, identity or intent."
    )
    processing_ms: int = 0
    created_at: str = Field(default_factory=_now)
