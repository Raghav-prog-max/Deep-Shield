"""Module 01: people/bag detection, tracking, zones and alert candidates."""
from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Iterable

from deep_shield.common.schema import (
    BBox, Finding, MediaType, ModelInfo, ModuleResult, ScoreType, TaskSummary,
)
from deep_shield.engines.deepfake.evidence import EvidenceStore
from deep_shield.engines.deepfake.media import MediaError, load_image, media_kind, probe_video, sha256_file

from .config import ThreatConfig, Zone, parse_zones
from .detector import Detection, TrackedFrame, detector_for
from .rules import (
    FrameSnapshot, RuleEvent, loitering_events, unattended_object_events,
    zone_transition_events,
)

log = logging.getLogger("deep_shield.m01")

MODULE_NAME = "M01_threat_intelligence"
ENGINE_VERSION = "m01-0.1.0"
TASK_DETECTION = "object_detection"
TASK_ENTRY = "zone_entry"
TASK_LOITER = "loitering"
TASK_UNATTENDED = "unattended_object"
ALL_TASKS = (TASK_DETECTION, TASK_ENTRY, TASK_LOITER, TASK_UNATTENDED)

LIMITATIONS = [
    "COCO-pretrained object detections are not fine-tuned or evaluated on campus footage.",
    "Detector confidence is uncalibrated and is not a probability of danger or wrongdoing.",
    "Zone, dwell and unattended-object events are rule-based candidates; thresholds require local validation.",
    "The system does not identify people or infer identity, intent, criminality or sensitive attributes.",
]

_COLORS = {"person": (70, 190, 90), "backpack": (40, 165, 245), "handbag": (220, 150, 45), "suitcase": (210, 75, 190)}


def _bbox(detection: Detection) -> BBox:
    x, y, w, h = detection.box
    return BBox(x=x, y=y, w=w, h=h)


def _model(task: str) -> ModelInfo:
    return ModelInfo(model_id="rule-based-v1", task=task, backend="python-rules")


class ThreatIntelligenceEngine:
    def __init__(self, cfg: ThreatConfig | None = None):
        self.cfg = cfg or ThreatConfig()
        self.detector = detector_for(self.cfg)

    def health(self) -> dict:
        return {
            "module": MODULE_NAME,
            "engine_version": ENGINE_VERSION,
            "tasks": list(ALL_TASKS),
            "detector": self.detector.health(),
            "thresholds": {
                "confidence": self.cfg.confidence,
                "image_size": self.cfg.image_size,
                "video_sample_fps": self.cfg.sample_fps,
                "max_video_duration_s": self.cfg.max_duration_s,
                "loitering_seconds": self.cfg.dwell_seconds,
                "unattended_seconds": self.cfg.unattended_seconds,
            },
            "zones_configured": len(parse_zones(None)),
        }

    def analyze(
        self,
        path: str | Path,
        *,
        zones: list[Zone] | list[dict] | None = None,
        source_name: str | None = None,
    ) -> ModuleResult:
        started = time.perf_counter()
        path = Path(path)
        zones = parse_zones(zones) if zones is not None else parse_zones(None)
        if not path.is_file():
            raise FileNotFoundError(path)
        kind = media_kind(path)
        digest = sha256_file(path)
        media_type = MediaType(kind)
        result = ModuleResult(
            module=MODULE_NAME,
            engine_version=ENGINE_VERSION,
            media_id=digest[:16],
            media_sha256=digest,
            media_type=media_type,
            source_name=source_name or path.name,
            status="completed",
            tasks_requested=list(ALL_TASKS),
        )
        evidence = EvidenceStore(self.cfg.evidence_dir, result.media_id)
        detections: list[Detection] = []
        snapshots: list[FrameSnapshot] = []
        evidence_by_frame = {}
        width = height = 0
        try:
            if kind == "image":
                image, _ = load_image(path)
                height, width = image.shape[:2]
                found = self.detector.predict_image(image)
                detections.extend(found)
                ref = self._save_frame(evidence, image, found, None, None)
                if ref is not None:
                    evidence_by_frame[None] = ref
                for item in found:
                    result.findings.append(self._detection_finding(item, ref, result.media_id, media_type, level="image"))
                result.media_info = {"width": width, "height": height, "objects_detected": len(found)}
            else:
                meta = probe_video(path, self.cfg.sample_fps)
                width, height = meta.width, meta.height
                sample_count = 0
                for frame in self.detector.track_video(
                    path, sample_fps=self.cfg.sample_fps, max_duration_s=self.cfg.max_duration_s,
                ):
                    sample_count += 1
                    detections.extend(frame.detections)
                    ref = self._save_frame(evidence, frame.image_bgr, frame.detections,
                                           frame.frame_index, frame.timestamp_s)
                    if ref is not None:
                        evidence_by_frame[frame.frame_index] = ref
                    snapshots.append(FrameSnapshot(
                        frame_index=frame.frame_index,
                        timestamp_s=frame.timestamp_s,
                        width=frame.image_bgr.shape[1],
                        height=frame.image_bgr.shape[0],
                        detections=frame.detections,
                    ))
                    for item in frame.detections:
                        result.findings.append(self._detection_finding(
                            item, ref, result.media_id, media_type, level="frame"))
                truncated = bool(meta.duration_s and meta.duration_s > self.cfg.max_duration_s)
                result.media_info = {
                    **meta.as_dict(), "frames_analysed": sample_count,
                    "truncated_at_duration_s": self.cfg.max_duration_s if truncated else None,
                    "objects_detected": len(detections),
                }
                if truncated:
                    result.warnings.append(
                        f"Analysis stopped at the configured {self.cfg.max_duration_s:g}-second limit.")

                if zones:
                    interval = 1.0 / self.cfg.sample_fps
                    transitions = zone_transition_events(
                        detections, zones, width, height, include_exits=False,
                        max_gap_s=max(2.5 * interval, 2.0),
                    )
                    dwell = loitering_events(
                        detections, zones, width, height,
                        dwell_seconds=self.cfg.dwell_seconds, sample_interval_s=interval,
                    )
                    self._append_events(result, transitions, evidence_by_frame, media_type)
                    self._append_events(result, dwell, evidence_by_frame, media_type)
                else:
                    result.warnings.append("No zones were configured; zone-entry and loitering rules were skipped.")

                unattended = unattended_object_events(
                    snapshots,
                    unattended_seconds=self.cfg.unattended_seconds,
                    stationary_body_lengths=self.cfg.unattended_stationary_body_lengths,
                    sample_interval_s=1.0 / self.cfg.sample_fps,
                )
                self._append_events(result, unattended, evidence_by_frame, media_type)

            self._summaries(result, detections, zones, kind)
        except (MediaError, OSError, RuntimeError, ValueError) as exc:
            log.exception("M01 analysis failed for %s", result.media_id)
            result.status = "failed"
            result.error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001
            log.exception("Unexpected M01 analysis failure for %s", result.media_id)
            result.status = "failed"
            result.error = f"{type(exc).__name__}: {exc}"
        result.processing_ms = int((time.perf_counter() - started) * 1000)
        return result

    def _save_frame(self, store: EvidenceStore, image, detections: list[Detection],
                    frame_index: int | None, timestamp_s: float | None):
        if not detections:
            return None
        suffix = "image" if frame_index is None else f"frame_{frame_index:06d}"
        boxes = [
            (item.box, f"{item.label} {item.confidence:.2f}" + (f" · T{item.track_id}" if item.track_id else ""),
             _COLORS.get(item.label, (80, 180, 220)))
            for item in detections
        ]
        header = "M01 detector · advisory" if timestamp_s is None else f"M01 detector · {timestamp_s:.2f}s · advisory"
        return store.save_annotated(
            image, boxes, f"{suffix}.jpg", "annotated_frame" if frame_index is not None else "annotated_image",
            "Person and bag detector outputs; candidate observations for human review.", header=header,
        )

    def _detection_finding(self, item: Detection, ref, media_id: str, media_type: MediaType, *, level: str) -> Finding:
        return Finding(
            module=MODULE_NAME,
            task=TASK_DETECTION,
            level=level,
            media_id=media_id,
            media_type=media_type,
            classification=f"{item.label}_detected",
            score=round(item.confidence, 6),
            score_type=ScoreType.DETECTOR_CONFIDENCE,
            predicted_label=item.label,
            raw_scores={"detector_confidence": round(item.confidence, 6)},
            frame_index=item.frame_index,
            timestamp_s=item.timestamp_s,
            track_id=item.track_id,
            region=_bbox(item),
            detector_confidence=round(item.confidence, 6),
            explanation={"label": item.label, "bbox_xywh": list(item.box), "track_id": item.track_id},
            model=self.detector.model_info,
            evidence=[ref] if ref is not None else [],
            limitations=LIMITATIONS,
        )

    def _append_events(self, result: ModuleResult, events: Iterable[RuleEvent], evidence_by_frame: dict,
                       media_type: MediaType) -> None:
        for event in events:
            refs = []
            for frame_index in (event.start.frame_index, event.end.frame_index):
                ref = evidence_by_frame.get(frame_index)
                if ref is not None and all(existing.path != ref.path for existing in refs):
                    refs.append(ref)
            representative = event.end
            result.findings.append(Finding(
                module=MODULE_NAME,
                task=TASK_ENTRY if event.event_type == "zone_entry" else (
                    TASK_LOITER if event.event_type == "loitering_candidate" else TASK_UNATTENDED),
                level="track" if event.track_id else "video",
                media_id=result.media_id,
                media_type=media_type,
                classification=event.event_type,
                score=None,
                score_type=ScoreType.RULE_SCORE,
                frame_index=representative.frame_index,
                timestamp_s=representative.timestamp_s,
                time_start_s=event.start.timestamp_s,
                time_end_s=event.end.timestamp_s,
                track_id=event.track_id,
                region=_bbox(representative),
                explanation={**event.details, "review_note": "Rule candidate only; verify against source video."},
                model=_model(event.event_type),
                evidence=refs,
                limitations=LIMITATIONS,
            ))

    def _summaries(self, result: ModuleResult, detections: list[Detection], zones: list[Zone], kind: str) -> None:
        object_count = len(detections)
        result.task_summaries.append(TaskSummary(
            task=TASK_DETECTION,
            classification="detections_found" if object_count else "no_target_detections",
            basis=f"{object_count} person or bag detection(s); output confidence is uncalibrated.",
            score_type=ScoreType.DETECTOR_CONFIDENCE,
            model=self.detector.model_info,
        ))
        for task, label in ((TASK_ENTRY, "zone_entry"), (TASK_LOITER, "loitering_candidate"),
                            (TASK_UNATTENDED, "unattended_object_candidate")):
            count = sum(1 for item in result.findings if item.classification == label)
            if task in (TASK_ENTRY, TASK_LOITER) and kind == "video" and not zones:
                classification, basis = "zones_not_configured", "Add normalized polygon zones to enable this rule."
            elif kind == "image":
                classification, basis = "not_applicable_image", "This rule requires timestamped video tracks."
            else:
                classification = "candidates_found" if count else "no_candidates"
                basis = f"{count} rule candidate(s); thresholds are configurable and require local validation."
            result.task_summaries.append(TaskSummary(
                task=task,
                classification=classification,
                basis=basis,
                score_type=ScoreType.RULE_SCORE,
                model=_model(task),
            ))
