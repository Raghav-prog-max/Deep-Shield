"""Module 03: timestamped zone, dwell, crowding and rapid-movement events."""
from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from deep_shield.common.schema import (
    BBox, Finding, MediaType, ModelInfo, ModuleResult, ScoreType, TaskSummary,
)
from deep_shield.engines.deepfake.evidence import EvidenceStore
from deep_shield.engines.deepfake.media import MediaError, media_kind, probe_video, sha256_file
from deep_shield.engines.threat.config import ROOT, ThreatConfig, Zone, parse_zones
from deep_shield.engines.threat.detector import Detection, detector_for
from deep_shield.engines.threat.rules import (
    FrameSnapshot, RuleEvent, crowding_events, loitering_events,
    merge_intervals, running_events, zone_transition_events,
)
from .gru import TrackGRUObservation, load_trained_gru, score_tracks_gru

log = logging.getLogger("deep_shield.m03")

MODULE_NAME = "M03_video_events"
ENGINE_VERSION = "m03-0.1.0"
TASK_ENTRY_EXIT = "entry_exit"
TASK_LOITER = "loitering"
TASK_CROWD = "crowding"
TASK_RUNNING = "running"
TASK_GRU = "temporal_gru"
ALL_TASKS = (TASK_ENTRY_EXIT, TASK_LOITER, TASK_CROWD, TASK_RUNNING, TASK_GRU)

LIMITATIONS = [
    "Events are heuristic candidates based on detector boxes and configured rules, not ground truth.",
    "Crowding and running thresholds require camera-specific calibration and validation.",
    "Temporal GRU sequence scores are advisory kinematic indicators and remain separate from rule triggers.",
    "No identity, intent, criminality or sensitive-attribute inference is performed.",
]


@dataclass(frozen=True)
class VideoEventConfig:
    model_path: Path = field(default_factory=lambda: ThreatConfig().model_path)
    evidence_dir: Path = field(default_factory=lambda: _rooted(
        os.getenv("DEEPSHIELD_M03_EVIDENCE_DIR", "evidence_store/m03")))
    device: str = field(default_factory=lambda: os.getenv("DEEPSHIELD_M01_DEVICE", "cpu"))
    confidence: float = field(default_factory=lambda: float(os.getenv("DEEPSHIELD_M01_CONFIDENCE", "0.25")))
    image_size: int = field(default_factory=lambda: int(os.getenv("DEEPSHIELD_M01_IMAGE_SIZE", "416")))
    sample_fps: float = field(default_factory=lambda: float(os.getenv("DEEPSHIELD_M03_SAMPLE_FPS", "2")))
    max_duration_s: float = field(default_factory=lambda: float(os.getenv("DEEPSHIELD_M03_MAX_DURATION", "60")))
    dwell_seconds: float = field(default_factory=lambda: float(os.getenv("DEEPSHIELD_M03_DWELL_SECONDS", "30")))
    crowd_min_people: int = field(default_factory=lambda: int(os.getenv("DEEPSHIELD_M03_CROWD_MIN_PEOPLE", "4")))
    crowd_seconds: float = field(default_factory=lambda: float(os.getenv("DEEPSHIELD_M03_CROWD_SECONDS", "5")))
    running_body_lengths_per_second: float = field(default_factory=lambda: float(
        os.getenv("DEEPSHIELD_M03_RUNNING_BODY_LENGTHS_PER_SECOND", "2.5")))
    merge_gap_seconds: float = field(default_factory=lambda: float(os.getenv("DEEPSHIELD_M03_MERGE_GAP_SECONDS", "3")))
    max_upload_mb: int = field(default_factory=lambda: int(os.getenv("DEEPSHIELD_MAX_UPLOAD_MB", "200")))
    keep_uploads: bool = field(default_factory=lambda: os.getenv("DEEPSHIELD_KEEP_UPLOADS", "0") == "1")
    gru_enabled: bool = field(default_factory=lambda: os.getenv("DEEPSHIELD_M03_GRU", "1") == "1")

    def __post_init__(self) -> None:
        if not 0 < self.confidence <= 1 or self.image_size < 32:
            raise ValueError("M03 detector confidence or image size is invalid")
        if self.sample_fps < 2 or self.sample_fps > 5:
            raise ValueError("DEEPSHIELD_M03_SAMPLE_FPS must be between 2 and 5")
        if self.max_duration_s <= 0 or self.max_upload_mb <= 0 or self.dwell_seconds <= 0:
            raise ValueError("M03 duration, upload limit and dwell threshold must be positive")
        if self.crowd_min_people < 2 or self.crowd_seconds <= 0 or self.merge_gap_seconds < 0:
            raise ValueError("M03 crowd and merge thresholds are invalid")
        if self.running_body_lengths_per_second <= 0:
            raise ValueError("M03 running speed threshold must be positive")


def _rooted(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else (ROOT / path).resolve()


def _bbox(detection: Detection) -> BBox:
    x, y, w, h = detection.box
    return BBox(x=x, y=y, w=w, h=h)


def _rule_model(task: str) -> ModelInfo:
    return ModelInfo(model_id="rule-based-v1", task=task, backend="python-rules")


class VideoEventEngine:
    def __init__(self, cfg: VideoEventConfig | None = None):
        self.cfg = cfg or VideoEventConfig()
        threat_cfg = ThreatConfig(
            model_path=self.cfg.model_path,
            evidence_dir=self.cfg.evidence_dir,
            device=self.cfg.device,
            confidence=self.cfg.confidence,
            image_size=self.cfg.image_size,
        )
        self.detector = detector_for(threat_cfg)

    def health(self) -> dict:
        return {
            "module": MODULE_NAME,
            "engine_version": ENGINE_VERSION,
            "tasks": list(ALL_TASKS),
            "detector": self.detector.health(),
            "sampling": {"fps": self.cfg.sample_fps, "max_duration_s": self.cfg.max_duration_s},
            "rules": {
                "loitering_seconds": self.cfg.dwell_seconds,
                "crowding_people": self.cfg.crowd_min_people,
                "crowding_seconds": self.cfg.crowd_seconds,
                "running_body_lengths_per_second": self.cfg.running_body_lengths_per_second,
                "event_merge_gap_seconds": self.cfg.merge_gap_seconds,
            },
            "gru_enabled": self.cfg.gru_enabled,
            "gru_model": (
                {k: v for k, v in load_trained_gru().items() if k not in ("model", "mean", "std")}
                if self.cfg.gru_enabled else None
            ),
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
        if media_kind(path) != "video":
            raise MediaError("Module 03 accepts video input only")

        digest = sha256_file(path)
        result = ModuleResult(
            module=MODULE_NAME,
            engine_version=ENGINE_VERSION,
            media_id=digest[:16],
            media_sha256=digest,
            media_type=MediaType.VIDEO,
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
            meta = probe_video(path, self.cfg.sample_fps)
            width, height = meta.width, meta.height
            count = 0
            for frame in self.detector.track_video(
                path, sample_fps=self.cfg.sample_fps, max_duration_s=self.cfg.max_duration_s,
            ):
                count += 1
                detections.extend(frame.detections)
                boxes = [
                    (item.box, f"{item.label} {item.confidence:.2f}" + (f" · T{item.track_id}" if item.track_id else ""),
                     (70, 190, 90) if item.label == "person" else (40, 165, 245))
                    for item in frame.detections
                ]
                ref = None
                if boxes:
                    ref = evidence.save_annotated(
                        frame.image_bgr, boxes, f"frame_{frame.frame_index:06d}.jpg", "annotated_frame",
                        "M01 person/bag detections used to derive M03 event candidates.",
                        header=f"M03 event analysis · {frame.timestamp_s:.2f}s · advisory",
                    )
                    evidence_by_frame[frame.frame_index] = ref
                snapshots.append(FrameSnapshot(
                    frame_index=frame.frame_index,
                    timestamp_s=frame.timestamp_s,
                    width=frame.image_bgr.shape[1],
                    height=frame.image_bgr.shape[0],
                    detections=frame.detections,
                ))

            truncated = bool(meta.duration_s and meta.duration_s > self.cfg.max_duration_s)
            result.media_info = {
                **meta.as_dict(), "frames_analysed": count,
                "truncated_at_duration_s": self.cfg.max_duration_s if truncated else None,
                "tracked_persons": len({d.track_id for d in detections if d.label == "person" and d.track_id}),
            }
            if truncated:
                result.warnings.append(f"Analysis stopped at the configured {self.cfg.max_duration_s:g}-second limit.")
            events: list[RuleEvent] = []
            interval = 1.0 / self.cfg.sample_fps
            if zones:
                events.extend(zone_transition_events(
                    detections, zones, width, height, include_exits=True,
                    max_gap_s=max(2.5 * interval, 2.0),
                ))
                events.extend(loitering_events(
                    detections, zones, width, height,
                    dwell_seconds=self.cfg.dwell_seconds, sample_interval_s=interval,
                ))
            else:
                result.warnings.append("No zones were configured; entry, exit and loitering rules were skipped.")
            events.extend(crowding_events(
                snapshots, zones, min_people=self.cfg.crowd_min_people,
                min_duration_s=self.cfg.crowd_seconds, merge_gap_s=self.cfg.merge_gap_seconds,
            ))
            events.extend(running_events(
                detections,
                threshold_body_lengths_per_second=self.cfg.running_body_lengths_per_second,
                sample_interval_s=interval,
            ))
            events = merge_intervals(events, self.cfg.merge_gap_seconds)
            self._append_events(result, events, evidence_by_frame)
            gru_obs = score_tracks_gru(detections, width, height) if self.cfg.gru_enabled else []
            self._append_gru(result, gru_obs, evidence_by_frame)
            self._summaries(result, events, zones, gru_obs, self.cfg.gru_enabled)
        except (MediaError, OSError, RuntimeError, ValueError) as exc:
            log.exception("M03 analysis failed for %s", result.media_id)
            result.status = "failed"
            result.error = f"{type(exc).__name__}: {exc}"
        except Exception as exc:  # noqa: BLE001
            log.exception("Unexpected M03 analysis failure for %s", result.media_id)
            result.status = "failed"
            result.error = f"{type(exc).__name__}: {exc}"
        result.processing_ms = int((time.perf_counter() - started) * 1000)
        return result

    def _append_events(self, result: ModuleResult, events: Iterable[RuleEvent], evidence_by_frame: dict) -> None:
        task_by_event = {
            "zone_entry": TASK_ENTRY_EXIT,
            "zone_exit": TASK_ENTRY_EXIT,
            "loitering_candidate": TASK_LOITER,
            "crowding_candidate": TASK_CROWD,
            "running_candidate": TASK_RUNNING,
        }
        for event in events:
            refs = []
            for frame_index in (event.start.frame_index, event.end.frame_index):
                ref = evidence_by_frame.get(frame_index)
                if ref is not None and all(existing.path != ref.path for existing in refs):
                    refs.append(ref)
            representative = event.end
            result.findings.append(Finding(
                module=MODULE_NAME,
                task=task_by_event[event.event_type],
                level="track" if event.track_id else "video",
                media_id=result.media_id,
                media_type=MediaType.VIDEO,
                classification=event.event_type,
                score=None,
                score_type=ScoreType.RULE_SCORE,
                frame_index=representative.frame_index,
                timestamp_s=representative.timestamp_s,
                time_start_s=event.start.timestamp_s,
                time_end_s=event.end.timestamp_s,
                track_id=event.track_id,
                region=_bbox(representative),
                explanation={**event.details, "review_note": "Event candidate only; verify the evidence frames."},
                model=_rule_model(event.event_type),
                evidence=refs,
                limitations=LIMITATIONS,
            ))

    @staticmethod
    def _append_gru(result: ModuleResult, gru_obs: list[TrackGRUObservation], evidence_by_frame: dict) -> None:
        for obs in gru_obs:
            gru_model = ModelInfo(model_id=obs.model_id, revision=obs.revision, task=TASK_GRU, backend="pytorch-gru")
            ref = evidence_by_frame.get(obs.representative.frame_index)
            result.findings.append(Finding(
                module=MODULE_NAME,
                task=TASK_GRU,
                level="track",
                media_id=result.media_id,
                media_type=MediaType.VIDEO,
                classification="kinematic_anomaly_candidate" if obs.score >= obs.threshold else "normal_kinematic_trajectory",
                score=obs.score,
                score_type=ScoreType.ANOMALY_SCORE,
                thresholds={"high": obs.threshold},
                frame_index=obs.representative.frame_index,
                timestamp_s=obs.representative.timestamp_s,
                time_start_s=obs.start_s,
                time_end_s=obs.end_s,
                track_id=obs.track_id,
                region=_bbox(obs.representative),
                explanation={
                    "n_steps": obs.n_steps,
                    "peak_speed_body_lengths_per_s": obs.peak_speed_bls,
                    "calibrated_threshold": obs.threshold,
                    "method": "2-layer PyTorch Temporal GRU (models/m03_temporal_gru.pt) over normalized track kinematics",
                },
                model=gru_model,
                evidence=[ref] if ref is not None else [],
                limitations=LIMITATIONS,
            ))

    @staticmethod
    def _summaries(
        result: ModuleResult,
        events: list[RuleEvent],
        zones: list[Zone],
        gru_obs: list[TrackGRUObservation] | None = None,
        gru_enabled: bool = True,
    ) -> None:
        event_by_task = {
            TASK_ENTRY_EXIT: {"zone_entry", "zone_exit"},
            TASK_LOITER: {"loitering_candidate"},
            TASK_CROWD: {"crowding_candidate"},
            TASK_RUNNING: {"running_candidate"},
        }
        for task, kinds in event_by_task.items():
            count = sum(1 for event in events if event.event_type in kinds)
            if task in (TASK_ENTRY_EXIT, TASK_LOITER) and not zones:
                classification, basis = "zones_not_configured", "Provide normalized polygon zones to enable this task."
            else:
                classification = "events_found" if count else "no_event_candidates"
                basis = f"{count} event candidate(s); no identity or intent is inferred."
            result.task_summaries.append(TaskSummary(
                task=task,
                classification=classification,
                basis=basis,
                score_type=ScoreType.RULE_SCORE,
                model=_rule_model(task),
            ))
        if gru_enabled:
            obs_list = gru_obs or []
            bundle = load_trained_gru()
            top_score = max((o.score for o in obs_list), default=None)
            thr = float(bundle["threshold"])
            result.task_summaries.append(TaskSummary(
                task=TASK_GRU,
                classification=(
                    "kinematic_anomaly_candidate" if (top_score is not None and top_score >= thr)
                    else "normal_kinematic_trajectory" if top_score is not None
                    else "no_tracks_scored"
                ),
                score=top_score,
                score_type=ScoreType.ANOMALY_SCORE,
                basis=f"PyTorch Temporal GRU ({bundle['model_id']}, {bundle['revision']}) evaluated {len(obs_list)} person track(s)",
                model=ModelInfo(model_id=bundle["model_id"], revision=bundle["revision"], task=TASK_GRU, backend="pytorch-gru"),
            ))
