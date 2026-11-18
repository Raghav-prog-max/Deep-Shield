"""Configuration for Module 01 and shared detector settings."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

ROOT = Path(__file__).resolve().parents[3]


def _rooted_path(value: str, default: str) -> Path:
    path = Path(value or default).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


class Zone(BaseModel):
    """A polygon in normalized image coordinates, independent of resolution."""

    model_config = ConfigDict(extra="forbid")
    zone_id: str = Field(min_length=1, max_length=64, pattern=r"^[A-Za-z0-9_.-]+$")
    polygon: list[tuple[float, float]] = Field(min_length=3, max_length=64)

    @field_validator("polygon")
    @classmethod
    def validate_polygon(cls, polygon: list[tuple[float, float]]) -> list[tuple[float, float]]:
        if any(not (0.0 <= x <= 1.0 and 0.0 <= y <= 1.0) for x, y in polygon):
            raise ValueError("zone polygon coordinates must be normalized to the inclusive range 0..1")
        return polygon


def parse_zones(value: str | list[dict[str, Any]] | None) -> list[Zone]:
    """Parse request zones or the optional DEEPSHIELD_M01_ZONES JSON setting."""
    if value is None or value == "":
        value = os.getenv("DEEPSHIELD_M01_ZONES", "")
    if isinstance(value, str):
        if not value.strip():
            return []
        value = json.loads(value)
    if not isinstance(value, list) or len(value) > 32:
        raise ValueError("zones must be a JSON array containing at most 32 polygons")
    zones = [Zone.model_validate(item) for item in value]
    ids = [zone.zone_id for zone in zones]
    if len(set(ids)) != len(ids):
        raise ValueError("zone_id values must be unique")
    return zones


@dataclass(frozen=True)
class ThreatConfig:
    model_path: Path = field(default_factory=lambda: _rooted_path(
        os.getenv("DEEPSHIELD_M01_MODEL", ""), "models/yolo26n.pt"))
    evidence_dir: Path = field(default_factory=lambda: _rooted_path(
        os.getenv("DEEPSHIELD_M01_EVIDENCE_DIR", ""), "evidence_store/m01"))
    device: str = field(default_factory=lambda: os.getenv("DEEPSHIELD_M01_DEVICE", "cpu"))
    confidence: float = field(default_factory=lambda: float(os.getenv("DEEPSHIELD_M01_CONFIDENCE", "0.25")))
    image_size: int = field(default_factory=lambda: int(os.getenv("DEEPSHIELD_M01_IMAGE_SIZE", "416")))
    sample_fps: float = field(default_factory=lambda: float(os.getenv("DEEPSHIELD_M01_SAMPLE_FPS", "1")))
    max_duration_s: float = field(default_factory=lambda: float(os.getenv("DEEPSHIELD_M01_MAX_DURATION", "60")))
    dwell_seconds: float = field(default_factory=lambda: float(os.getenv("DEEPSHIELD_M01_DWELL_SECONDS", "30")))
    unattended_seconds: float = field(default_factory=lambda: float(os.getenv("DEEPSHIELD_M01_UNATTENDED_SECONDS", "60")))
    unattended_stationary_body_lengths: float = field(default_factory=lambda: float(
        os.getenv("DEEPSHIELD_M01_UNATTENDED_STATIONARY_BODY_LENGTHS", "0.75")))
    max_upload_mb: int = field(default_factory=lambda: int(os.getenv("DEEPSHIELD_MAX_UPLOAD_MB", "200")))
    keep_uploads: bool = field(default_factory=lambda: os.getenv("DEEPSHIELD_KEEP_UPLOADS", "0") == "1")

    def __post_init__(self) -> None:
        if not 0 < self.confidence <= 1:
            raise ValueError("DEEPSHIELD_M01_CONFIDENCE must be within (0, 1]")
        if self.image_size < 32:
            raise ValueError("DEEPSHIELD_M01_IMAGE_SIZE must be at least 32")
        if self.sample_fps <= 0 or self.max_duration_s <= 0 or self.max_upload_mb <= 0:
            raise ValueError("Engine 1 sample rate, duration and upload limit must be positive")
        if self.dwell_seconds <= 0 or self.unattended_seconds <= 0:
            raise ValueError("Engine 1 dwell and unattended durations must be positive")
        if self.unattended_stationary_body_lengths <= 0:
            raise ValueError("Unattended stationary distance must be positive")
