"""Deterministic, non-scoring links between findings from separate modules."""
from __future__ import annotations

import hashlib

from deep_shield.common.schema import BBox, Finding, ModuleResult

from .schema import FindingGroup, FindingLink, IntelligenceResult

_TIME_TOLERANCE_S = 1.0
_MIN_IOU = 0.10
_MIN_CONTAINMENT = 0.50
_LINK_EXPLANATION = (
    "Findings overlap in source time and/or image region. This is a navigation association, "
    "not a calibrated agreement or causal conclusion. Track identifiers are module-local."
)


def _interval(finding: Finding) -> tuple[float, float] | None:
    start = finding.time_start_s
    end = finding.time_end_s
    if start is None:
        start = finding.timestamp_s
    if end is None:
        end = finding.timestamp_s
    if start is None or end is None:
        return None
    return (min(start, end), max(start, end))


def _time_relation(left: Finding, right: Finding) -> tuple[bool, float | None]:
    a, b = _interval(left), _interval(right)
    if a is None or b is None:
        return False, None
    overlap = min(a[1], b[1]) - max(a[0], b[0])
    related = overlap >= 0 or overlap >= -_TIME_TOLERANCE_S
    return related, round(max(0.0, overlap), 3) if related else None


def _region_relation(left: BBox | None, right: BBox | None) -> tuple[bool, float | None]:
    if left is None or right is None:
        return False, None
    left_area = max(0, left.w) * max(0, left.h)
    right_area = max(0, right.w) * max(0, right.h)
    if not left_area or not right_area:
        return False, None
    ix1, iy1 = max(left.x, right.x), max(left.y, right.y)
    ix2 = min(left.x + left.w, right.x + right.w)
    iy2 = min(left.y + left.h, right.y + right.h)
    intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
    if intersection == 0:
        return False, 0.0
    iou = intersection / (left_area + right_area - intersection)
    containment = intersection / min(left_area, right_area)
    related = iou >= _MIN_IOU or containment >= _MIN_CONTAINMENT
    return related, round(max(iou, containment), 4) if related else round(iou, 4)


def _make_link(left: Finding, right: Finding) -> FindingLink | None:
    time_related, overlap_s = _time_relation(left, right)
    region_related, overlap_ratio = _region_relation(left.region, right.region)
    if not time_related and not region_related:
        return None
    relationship = (
        "spatial_temporal_overlap" if time_related and region_related
        else "region_overlap" if region_related
        else "temporal_cooccurrence"
    )
    return FindingLink(
        finding_ids=[left.finding_id, right.finding_id],
        modules=[left.module, right.module],
        relationship=relationship,
        time_overlap_s=overlap_s,
        region_overlap_ratio=overlap_ratio,
        explanation=_LINK_EXPLANATION,
    )


def _groups(links: list[FindingLink], findings: list[Finding]) -> list[FindingGroup]:
    parent = {item.finding_id: item.finding_id for item in findings}

    def root(value: str) -> str:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = parent[value]
        return value

    for link in links:
        a, b = (root(value) for value in link.finding_ids)
        if a != b:
            parent[max(a, b)] = min(a, b)

    members: dict[str, set[str]] = {}
    for item in findings:
        key = root(item.finding_id)
        members.setdefault(key, set()).add(item.finding_id)
    groups = []
    for ids in members.values():
        if len(ids) < 2:
            continue
        ordered_ids = sorted(ids)
        indices = [i for i, link in enumerate(links) if set(link.finding_ids) <= ids]
        modules = sorted({module for link in (links[i] for i in indices) for module in link.modules})
        digest = hashlib.sha256("|".join(ordered_ids).encode("utf-8")).hexdigest()[:12]
        groups.append(FindingGroup(
            group_id=f"link-{digest}", finding_ids=ordered_ids, modules=modules, link_indices=indices,
        ))
    return sorted(groups, key=lambda group: group.group_id)


def link_module_results(module_results: list[ModuleResult]) -> IntelligenceResult:
    if len(module_results) < 2 or len(module_results) > 4:
        raise ValueError("Provide results from two to four modules.")
    modules = [result.module for result in module_results]
    if len(set(modules)) != len(modules):
        raise ValueError("Each module may appear only once in a link request.")
    media_ids = {result.media_id for result in module_results}
    hashes = {result.media_sha256 for result in module_results}
    if len(media_ids) != 1 or len(hashes) != 1:
        raise ValueError("All module results must refer to the same media_id and SHA-256.")

    findings = [finding for result in module_results for finding in result.findings]
    if len(findings) > 5000:
        raise ValueError("A link request may contain at most 5,000 findings.")
    links = []
    for index, left in enumerate(findings):
        for right in findings[index + 1:]:
            if left.module == right.module:
                continue
            link = _make_link(left, right)
            if link is not None:
                links.append(link)
    linked_ids = {finding_id for link in links for finding_id in link.finding_ids}
    return IntelligenceResult(
        media_id=next(iter(media_ids)),
        media_sha256=next(iter(hashes)),
        module_results=module_results,
        links=links,
        groups=_groups(links, findings),
        unlinked_finding_ids=sorted(item.finding_id for item in findings if item.finding_id not in linked_ids),
    )
