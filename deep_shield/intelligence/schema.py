"""API schemas for cross-module evidence navigation."""
from __future__ import annotations

from pydantic import BaseModel, Field

from deep_shield.common.schema import ModuleResult


class LinkRequest(BaseModel):
    """Results from distinct engines analyzing the same source media."""

    module_results: list[ModuleResult] = Field(min_length=2, max_length=4)


class FindingLink(BaseModel):
    finding_ids: list[str] = Field(min_length=2, max_length=2)
    modules: list[str] = Field(min_length=2, max_length=2)
    relationship: str
    time_overlap_s: float | None = None
    region_overlap_ratio: float | None = Field(default=None, ge=0.0, le=1.0)
    explanation: str


class FindingGroup(BaseModel):
    group_id: str
    finding_ids: list[str]
    modules: list[str]
    link_indices: list[int]


class IntelligenceResult(BaseModel):
    media_id: str
    media_sha256: str
    module_results: list[ModuleResult]
    links: list[FindingLink] = Field(default_factory=list)
    groups: list[FindingGroup] = Field(default_factory=list)
    unlinked_finding_ids: list[str] = Field(default_factory=list)
    disclaimer: str = (
        "Links are for evidence navigation only. They do not combine engine scores or establish "
        "authenticity, identity, intent, causation, or wrongdoing."
    )
