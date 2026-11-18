"""Deterministic, explainable rules shared by Modules 01 and 03."""
from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Iterable

import cv2
import numpy as np

from .config import Zone
from .detector import Detection


@dataclass(frozen=True)
class FrameSnapshot:
    frame_index: int
    timestamp_s: float
    width: int
    height: int
    detections: list[Detection]


@dataclass(frozen=True)
class RuleEvent:
    event_type: str
    start: Detection
    end: Detection
    track_id: str | None
    zone_id: str | None
    details: dict


def _inside(point: tuple[float, float], zone: Zone, width: int, height: int) -> bool:
    polygon = np.asarray([(x * width, y * height) for x, y in zone.polygon], dtype=np.float32)
    return cv2.pointPolygonTest(polygon, point, False) >= 0


def _person_tracks(detections: Iterable[Detection]) -> dict[str, list[Detection]]:
    tracks: dict[str, list[Detection]] = {}
    for item in detections:
        if item.label == "person" and item.track_id is not None:
            tracks.setdefault(item.track_id, []).append(item)
    for sequence in tracks.values():
        sequence.sort(key=lambda item: item.timestamp_s if item.timestamp_s is not None else -1)
    return tracks


def _bag_tracks(detections: Iterable[Detection]) -> dict[str, list[Detection]]:
    tracks: dict[str, list[Detection]] = {}
    for item in detections:
        if item.label in {"backpack", "handbag", "suitcase"} and item.track_id is not None:
            tracks.setdefault(item.track_id, []).append(item)
    for sequence in tracks.values():
        sequence.sort(key=lambda item: item.timestamp_s if item.timestamp_s is not None else -1)
    return tracks


def zone_transition_events(
    detections: Iterable[Detection],
    zones: list[Zone],
    width: int,
    height: int,
    *,
    include_exits: bool,
    max_gap_s: float,
) -> list[RuleEvent]:
    events: list[RuleEvent] = []
    for track_id, sequence in _person_tracks(detections).items():
        for zone in zones:
            previous: Detection | None = None
            previous_inside = False
            for current in sequence:
                is_inside = _inside(current.footpoint, zone, width, height)
                gap = None if previous is None else (current.timestamp_s or 0) - (previous.timestamp_s or 0)
                if previous is not None and gap is not None and gap > max_gap_s:
                    previous = None
                    previous_inside = False
                if previous is not None and not previous_inside and is_inside:
                    events.append(RuleEvent(
                        "zone_entry", current, current, track_id, zone.zone_id,
                        {"zone_id": zone.zone_id, "rule": "person footpoint crossed outside-to-inside"},
                    ))
                elif previous is not None and previous_inside and not is_inside and include_exits:
                    events.append(RuleEvent(
                        "zone_exit", previous, current, track_id, zone.zone_id,
                        {"zone_id": zone.zone_id, "rule": "person footpoint crossed inside-to-outside"},
                    ))
                previous, previous_inside = current, is_inside
    return events


def loitering_events(
    detections: Iterable[Detection],
    zones: list[Zone],
    width: int,
    height: int,
    *,
    dwell_seconds: float,
    sample_interval_s: float,
) -> list[RuleEvent]:
    max_gap = max(2.5 * sample_interval_s, 2.0)
    events: list[RuleEvent] = []
    for track_id, sequence in _person_tracks(detections).items():
        for zone in zones:
            segment: list[Detection] = []
            for item in sequence:
                inside = _inside(item.footpoint, zone, width, height)
                gap = (item.timestamp_s - segment[-1].timestamp_s) if segment and item.timestamp_s is not None and segment[-1].timestamp_s is not None else 0
                if segment and (not inside or gap > max_gap):
                    events.extend(_loiter_segment(segment, track_id, zone.zone_id, dwell_seconds))
                    segment = []
                if inside:
                    segment.append(item)
            if segment:
                events.extend(_loiter_segment(segment, track_id, zone.zone_id, dwell_seconds))
    return events


def _loiter_segment(sequence: list[Detection], track_id: str, zone_id: str, dwell_seconds: float) -> list[RuleEvent]:
    if len(sequence) < 2:
        return []
    elapsed = (sequence[-1].timestamp_s or 0) - (sequence[0].timestamp_s or 0)
    if elapsed < dwell_seconds:
        return []
    return [RuleEvent(
        "loitering_candidate", sequence[0], sequence[-1], track_id, zone_id,
        {"zone_id": zone_id, "observed_dwell_seconds": round(elapsed, 2), "threshold_seconds": dwell_seconds},
    )]


def crowding_events(
    snapshots: list[FrameSnapshot],
    zones: list[Zone],
    *,
    min_people: int,
    min_duration_s: float,
    merge_gap_s: float,
) -> list[RuleEvent]:
    if not snapshots:
        return []
    first = snapshots[0]
    scan_zones = zones or [Zone(zone_id="full_frame", polygon=[(0, 0), (1, 0), (1, 1), (0, 1)])]
    events: list[RuleEvent] = []
    for zone in scan_zones:
        active: list[tuple[Detection, int]] = []
        for snapshot in snapshots:
            in_zone = [
                d for d in snapshot.detections
                if d.label == "person" and _inside(d.footpoint, zone, snapshot.width, snapshot.height)
            ]
            unique: dict[str, Detection] = {}
            for d in in_zone:
                unique[d.track_id or f"frame-{snapshot.frame_index}-{len(unique)}"] = d
            if len(unique) >= min_people:
                active.append((max(unique.values(), key=lambda d: d.confidence), len(unique)))
            elif active:
                events.extend(_crowd_segment(active, zone.zone_id, min_people, min_duration_s))
                active = []
        if active:
            events.extend(_crowd_segment(active, zone.zone_id, min_people, min_duration_s))

    # Merge short gaps in the same zone; entries/exits are deliberately not merged.
    return merge_intervals(events, merge_gap_s)


def _crowd_segment(
    segment: list[tuple[Detection, int]], zone_id: str, min_people: int, min_duration_s: float
) -> list[RuleEvent]:
    start, end = segment[0][0], segment[-1][0]
    elapsed = (end.timestamp_s or 0) - (start.timestamp_s or 0)
    if elapsed < min_duration_s:
        return []
    peak = max(count for _, count in segment)
    return [RuleEvent(
        "crowding_candidate", start, end, None, zone_id,
        {"zone_id": zone_id, "peak_person_count": peak, "minimum_people": min_people,
          "observed_duration_seconds": round(elapsed, 2), "threshold_seconds": min_duration_s},
    )]


def running_events(
    detections: Iterable[Detection],
    *,
    threshold_body_lengths_per_second: float,
    sample_interval_s: float,
    min_consecutive_intervals: int = 2,
) -> list[RuleEvent]:
    events: list[RuleEvent] = []
    max_gap = max(2.5 * sample_interval_s, 2.0)
    for track_id, sequence in _person_tracks(detections).items():
        speeds: list[tuple[Detection, Detection, float]] = []
        for previous, current in zip(sequence, sequence[1:]):
            if previous.timestamp_s is None or current.timestamp_s is None:
                continue
            elapsed = current.timestamp_s - previous.timestamp_s
            if elapsed <= 0 or elapsed > max_gap:
                continue
            px, py = previous.center
            cx, cy = current.center
            body_height = max(1.0, (previous.box[3] + current.box[3]) / 2)
            speed = float(np.hypot(cx - px, cy - py) / elapsed / body_height)
            speeds.append((previous, current, speed))
        active: list[tuple[Detection, Detection, float]] = []
        for item in speeds:
            if item[2] >= threshold_body_lengths_per_second:
                active.append(item)
            else:
                events.extend(_running_segment(active, track_id, threshold_body_lengths_per_second, min_consecutive_intervals))
                active = []
        events.extend(_running_segment(active, track_id, threshold_body_lengths_per_second, min_consecutive_intervals))
    return merge_intervals(events, sample_interval_s * 1.5)


def _running_segment(
    segment: list[tuple[Detection, Detection, float]], track_id: str,
    threshold: float, min_intervals: int,
) -> list[RuleEvent]:
    if len(segment) < min_intervals:
        return []
    start, end = segment[0][0], segment[-1][1]
    return [RuleEvent(
        "running_candidate", start, end, track_id, None,
        {"peak_body_lengths_per_second": round(max(x[2] for x in segment), 3),
          "threshold_body_lengths_per_second": threshold, "intervals_above_threshold": len(segment)},
    )]


def unattended_object_events(
    snapshots: list[FrameSnapshot],
    *,
    unattended_seconds: float,
    stationary_body_lengths: float,
    sample_interval_s: float,
) -> list[RuleEvent]:
    detections = [item for snapshot in snapshots for item in snapshot.detections]
    people = [item for item in detections if item.label == "person"]
    typical_person_height = float(np.median([p.box[3] for p in people])) if people else 48.0
    max_displacement = max(18.0, typical_person_height * stationary_body_lengths)
    max_gap = max(2.5 * sample_interval_s, 2.0)
    events: list[RuleEvent] = []

    for track_id, sequence in _bag_tracks(detections).items():
        segment: list[Detection] = []
        for item in sequence:
            near_person = any(_person_near_bag(person, item) for person in people
                              if person.frame_index == item.frame_index)
            gap = (item.timestamp_s - segment[-1].timestamp_s) if segment and item.timestamp_s is not None and segment[-1].timestamp_s is not None else 0
            if segment and (near_person or gap > max_gap):
                events.extend(_unattended_segment(segment, track_id, unattended_seconds, max_displacement))
                segment = []
            if not near_person:
                segment.append(item)
        if segment:
            events.extend(_unattended_segment(segment, track_id, unattended_seconds, max_displacement))
    return merge_intervals(events, sample_interval_s * 1.5)


def _person_near_bag(person: Detection, bag: Detection) -> bool:
    px, py = person.box[0], person.box[1]
    pw, ph = person.box[2], person.box[3]
    bx, by = bag.center
    margin = max(12.0, 0.18 * ph)
    return px - margin <= bx <= px + pw + margin and py - margin <= by <= py + ph + margin


def _unattended_segment(
    sequence: list[Detection], track_id: str, threshold_s: float, max_displacement: float
) -> list[RuleEvent]:
    if len(sequence) < 2:
        return []
    start, end = sequence[0], sequence[-1]
    elapsed = (end.timestamp_s or 0) - (start.timestamp_s or 0)
    if elapsed < threshold_s:
        return []
    points = np.asarray([item.center for item in sequence], dtype=np.float32)
    anchor = np.median(points, axis=0)
    max_observed_displacement = float(np.linalg.norm(points - anchor, axis=1).max())
    if max_observed_displacement > max_displacement:
        return []
    return [RuleEvent(
        "unattended_object_candidate", start, end, track_id, None,
        {"object_label": start.label, "observed_seconds_without_nearby_person": round(elapsed, 2),
          "threshold_seconds": threshold_s, "max_displacement_px": round(max_observed_displacement, 1),
          "stationary_threshold_px": round(max_displacement, 1)},
    )]


def merge_intervals(events: list[RuleEvent], gap_s: float) -> list[RuleEvent]:
    mergeable = {"loitering_candidate", "crowding_candidate", "running_candidate", "unattended_object_candidate"}
    ordered = sorted(events, key=lambda e: (e.event_type, e.track_id or "", e.zone_id or "", e.start.timestamp_s or 0))
    merged: list[RuleEvent] = []
    for event in ordered:
        if merged:
            previous = merged[-1]
            same = (event.event_type == previous.event_type and event.track_id == previous.track_id
                    and event.zone_id == previous.zone_id and event.event_type in mergeable)
            gap = (event.start.timestamp_s or 0) - (previous.end.timestamp_s or 0)
            if same and 0 <= gap <= gap_s:
                details = dict(previous.details)
                details.update(event.details)
                if "peak_body_lengths_per_second" in previous.details:
                    details["peak_body_lengths_per_second"] = max(
                        previous.details["peak_body_lengths_per_second"], event.details["peak_body_lengths_per_second"])
                if "peak_person_count" in previous.details:
                    details["peak_person_count"] = max(previous.details["peak_person_count"], event.details["peak_person_count"])
                merged[-1] = replace(previous, end=event.end, details=details)
                continue
        merged.append(event)
    return merged
