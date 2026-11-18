"""Lazy CPU neural object detector + multi-object tracker shared by Modules 01 and 03.

Supports two genuine MS-COCO neural detection backends (no synthetic contour or face-box hacks):
  1. Ultralytics YOLO + ByteTrack (when ``models/yolo26n.pt`` and ``ultralytics`` are installed).
  2. Local PyTorch HuggingFace ``hustvl/yolos-tiny`` (MS-COCO 2017 ViT detector under
     ``models/hustvl__yolos-tiny``) paired with an IoU + velocity-predicted multi-object tracker.
"""
from __future__ import annotations

import hashlib
import importlib.util
import logging
import threading
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterator

import cv2
import numpy as np
from PIL import Image

from .config import ROOT, ThreatConfig
from ..deepfake.media import MediaError, probe_video, sample_video

log = logging.getLogger("deep_shield.m01.detector")

# COCO ids for Ultralytics (0-indexed) and target class names (shared across Ultralytics and YOLOS).
CLASS_IDS = (0, 24, 26, 28)
CLASS_NAMES = {0: "person", 24: "backpack", 26: "handbag", 28: "suitcase"}
TARGET_LABELS = frozenset(CLASS_NAMES.values())
DEFAULT_YOLOS_DIR = ROOT / "models" / "hustvl__yolos-tiny"


@dataclass(frozen=True)
class Detection:
    label: str
    confidence: float
    box: tuple[int, int, int, int]  # x, y, width, height in source pixels
    track_id: str | None = None
    frame_index: int | None = None
    timestamp_s: float | None = None

    @property
    def center(self) -> tuple[float, float]:
        x, y, w, h = self.box
        return x + w / 2, y + h / 2

    @property
    def footpoint(self) -> tuple[float, float]:
        x, y, w, h = self.box
        return x + w / 2, y + h


@dataclass(frozen=True)
class TrackedFrame:
    frame_index: int
    timestamp_s: float
    image_bgr: np.ndarray
    detections: list[Detection]


class _ActiveTrack:
    __slots__ = ("track_id", "label", "box", "vx", "vy", "missed")

    def __init__(self, track_id: int, label: str, box: tuple[int, int, int, int]):
        self.track_id = track_id
        self.label = label
        self.box = box
        self.vx = 0.0
        self.vy = 0.0
        self.missed = 0

    def predicted_box(self) -> tuple[float, float, float, float]:
        x, y, w, h = self.box
        return (x + self.vx, y + self.vy, float(w), float(h))


def _box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix1, iy1 = max(ax, bx), max(ay, by)
    ix2, iy2 = min(ax + aw, bx + bw), min(ay + ah, by + bh)
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    union = max(0.0, aw) * max(0.0, ah) + max(0.0, bw) * max(0.0, bh) - inter
    return inter / union if union > 0.0 else 0.0


class MultiObjectIoUTracker:
    """Velocity-predicted IoU + normalized-distance tracker for multi-frame video tracking."""

    def __init__(self, iou_threshold: float = 0.12, max_dist_ratio: float = 1.35, max_age: int = 4):
        self.iou_threshold = iou_threshold
        self.max_dist_ratio = max_dist_ratio
        self.max_age = max_age
        self._next_id = 1
        self._tracks: list[_ActiveTrack] = []

    def update(self, detections: list[Detection]) -> list[Detection]:
        if not detections:
            for tr in self._tracks:
                tr.missed += 1
            self._tracks = [tr for tr in self._tracks if tr.missed <= self.max_age]
            return []

        candidates: list[tuple[float, int, int]] = []
        for ti, tr in enumerate(self._tracks):
            pbox = tr.predicted_box()
            pcx, pcy = pbox[0] + pbox[2] / 2.0, pbox[1] + pbox[3] / 2.0
            scale = max(24.0, float(max(pbox[2], pbox[3])))
            for di, det in enumerate(detections):
                if det.label != tr.label:
                    continue
                iou = _box_iou(pbox, tuple(float(v) for v in det.box))
                dcx, dcy = det.center
                dist_norm = float(np.hypot(dcx - pcx, dcy - pcy) / scale)
                if iou >= self.iou_threshold or dist_norm <= self.max_dist_ratio:
                    score = iou + max(0.0, 1.0 - dist_norm / self.max_dist_ratio) * 0.5
                    candidates.append((score, ti, di))

        candidates.sort(key=lambda item: item[0], reverse=True)
        matched_tracks: set[int] = set()
        matched_dets: set[int] = set()
        det_track_ids: dict[int, str] = {}

        for _score, ti, di in candidates:
            if ti in matched_tracks or di in matched_dets:
                continue
            matched_tracks.add(ti)
            matched_dets.add(di)
            tr = self._tracks[ti]
            det = detections[di]
            old_cx = tr.box[0] + tr.box[2] / 2.0
            old_cy = tr.box[1] + tr.box[3] / 2.0
            new_cx, new_cy = det.center
            tr.vx = 0.6 * (new_cx - old_cx) + 0.4 * tr.vx
            tr.vy = 0.6 * (new_cy - old_cy) + 0.4 * tr.vy
            tr.box = det.box
            tr.missed = 0
            det_track_ids[di] = str(tr.track_id)

        for ti, tr in enumerate(self._tracks):
            if ti not in matched_tracks:
                tr.missed += 1

        for di, det in enumerate(detections):
            if di not in matched_dets:
                tid = self._next_id
                self._next_id += 1
                self._tracks.append(_ActiveTrack(tid, det.label, det.box))
                det_track_ids[di] = str(tid)

        self._tracks = [tr for tr in self._tracks if tr.missed <= self.max_age]
        return [
            Detection(
                label=det.label,
                confidence=det.confidence,
                box=det.box,
                track_id=det_track_ids[di],
                frame_index=det.frame_index,
                timestamp_s=det.timestamp_s,
            )
            for di, det in enumerate(detections)
        ]


class YoloObjectDetector:
    """Loads a local MS-COCO neural detector once and serializes access to its tracker state."""

    def __init__(self, cfg: ThreatConfig):
        self.cfg = cfg
        self._model = None
        self._yolos_model = None
        self._yolos_processor = None
        self._lock = threading.RLock()
        self._revision: str | None = None
        self._yolos_dir = DEFAULT_YOLOS_DIR

    def _has_yolo(self) -> bool:
        if not self.cfg.model_path.is_file():
            return False
        try:
            return importlib.util.find_spec("ultralytics") is not None
        except (ImportError, ValueError):
            return False

    def _has_yolos(self) -> bool:
        if not (self._yolos_dir / "config.json").is_file():
            return False
        try:
            return (
                importlib.util.find_spec("torch") is not None
                and importlib.util.find_spec("transformers") is not None
            )
        except (ImportError, ValueError):
            return False

    def health(self) -> dict:
        has_yolo = self._has_yolo()
        has_yolos = self._has_yolos()
        if has_yolo:
            return {
                "ready": True,
                "backend": "ultralytics/pytorch",
                "model_id": f"ultralytics:{self.cfg.model_path.stem}",
                "checkpoint_ready": True,
                "checkpoint": str(self.cfg.model_path),
                "device": self.cfg.device,
                "tracker": "ByteTrack",
                "targets": list(CLASS_NAMES.values()),
            }
        if has_yolos:
            return {
                "ready": True,
                "backend": "transformers/pytorch",
                "model_id": "hustvl/yolos-tiny",
                "checkpoint_ready": True,
                "checkpoint": str(self._yolos_dir),
                "device": self.cfg.device,
                "tracker": "MultiObjectIoUTracker",
                "targets": list(CLASS_NAMES.values()),
            }
        return {
            "ready": False,
            "backend": "unavailable",
            "model_id": None,
            "checkpoint_ready": False,
            "checkpoint": str(self.cfg.model_path),
            "device": self.cfg.device,
            "tracker": None,
            "targets": list(CLASS_NAMES.values()),
            "error": "No MS-COCO neural detector checkpoint found (models/yolo26n.pt or models/hustvl__yolos-tiny).",
        }

    def _file_sha(self, path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as source:
            for chunk in iter(lambda: source.read(1 << 20), b""):
                digest.update(chunk)
        return digest.hexdigest()[:12]

    @property
    def model_info(self):
        from deep_shield.common.schema import ModelInfo

        if self._has_yolo():
            if self._revision is None and self.cfg.model_path.is_file():
                self._revision = self._file_sha(self.cfg.model_path)
            return ModelInfo(
                model_id=f"ultralytics:{self.cfg.model_path.stem}",
                revision=self._revision,
                task="person_and_bag_detection",
                license="AGPL-3.0",
                source_url="https://docs.ultralytics.com/models/yolo26/",
                backend="ultralytics/pytorch",
            )
        if self._has_yolos():
            weights = self._yolos_dir / "model.safetensors"
            if self._revision is None and weights.is_file():
                self._revision = self._file_sha(weights)
            return ModelInfo(
                model_id="hustvl/yolos-tiny",
                revision=self._revision or "coco-2017",
                task="person_and_bag_detection",
                license="Apache-2.0",
                source_url="https://huggingface.co/hustvl/yolos-tiny",
                backend="transformers/pytorch",
            )
        return ModelInfo(
            model_id="unavailable",
            revision=None,
            task="person_and_bag_detection",
            backend="unavailable",
        )

    def _get_yolo_model(self):
        if self._model is None:
            from ultralytics import YOLO
            self._model = YOLO(str(self.cfg.model_path))
        return self._model

    def _get_yolos_model(self):
        if self._yolos_model is None:
            if not self._has_yolos():
                raise FileNotFoundError(
                    f"No MS-COCO neural detector checkpoint found at {self.cfg.model_path} or {self._yolos_dir}."
                )
            import torch
            from transformers import AutoImageProcessor, AutoModelForObjectDetection

            self._yolos_processor = AutoImageProcessor.from_pretrained(str(self._yolos_dir))
            self._yolos_model = AutoModelForObjectDetection.from_pretrained(str(self._yolos_dir))
            dev = "cuda" if (self.cfg.device.startswith("cuda") and torch.cuda.is_available()) else "cpu"
            self._yolos_model.to(dev).eval()
        return self._yolos_model, self._yolos_processor

    def _predict_yolos_frame(
        self,
        image_bgr: np.ndarray,
        frame_index: int | None = None,
        timestamp_s: float | None = None,
    ) -> list[Detection]:
        import torch

        model, processor = self._get_yolos_model()
        h, w = image_bgr.shape[:2]
        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        inputs = processor(images=pil, return_tensors="pt")
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        with torch.no_grad():
            outputs = model(**inputs)

        target_sizes = torch.tensor([[h, w]], device=model.device)
        # Apply a slightly higher floor for YOLOS-Tiny to avoid low-confidence background noise
        score_thr = max(float(self.cfg.confidence), 0.45)
        post = processor.post_process_object_detection(outputs, threshold=score_thr, target_sizes=target_sizes)[0]
        id2label = model.config.id2label

        found: list[Detection] = []
        for score_t, label_t, box_t in zip(post["scores"], post["labels"], post["boxes"]):
            raw_label = str(id2label.get(int(label_t.item()), "")).lower()
            if raw_label not in TARGET_LABELS:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in box_t.detach().cpu().tolist()]
            x1 = max(0, min(w - 1, x1))
            y1 = max(0, min(h - 1, y1))
            x2 = max(x1 + 1, min(w, x2))
            y2 = max(y1 + 1, min(h, y2))
            bw, bh = x2 - x1, y2 - y1
            if bw < 8 or bh < 8:
                continue
            found.append(
                Detection(
                    label=raw_label,
                    confidence=round(float(score_t.item()), 4),
                    box=(x1, y1, bw, bh),
                    track_id=None,
                    frame_index=frame_index,
                    timestamp_s=timestamp_s,
                )
            )
        return found

    def predict_image(self, image_bgr: np.ndarray) -> list[Detection]:
        with self._lock:
            if self._has_yolo():
                model = self._get_yolo_model()
                result = model.predict(
                    source=image_bgr,
                    device=self.cfg.device,
                    imgsz=self.cfg.image_size,
                    conf=self.cfg.confidence,
                    classes=list(CLASS_IDS),
                    max_det=100,
                    verbose=False,
                )[0]
                return self._detections(result, None, None)
            dets = self._predict_yolos_frame(image_bgr, None, None)
            return [
                Detection(
                    label=d.label,
                    confidence=d.confidence,
                    box=d.box,
                    track_id=str(i + 1),
                    frame_index=None,
                    timestamp_s=None,
                )
                for i, d in enumerate(dets)
            ]

    def track_video(
        self,
        path: Path,
        *,
        sample_fps: float,
        max_duration_s: float,
    ) -> Iterator[TrackedFrame]:
        meta = probe_video(path, sample_fps)
        if meta.sample_step < 1 or meta.fps <= 0:
            raise MediaError("Video frame rate is invalid for object tracking")
        max_frames = max(1, int(np.ceil(sample_fps * max_duration_s)))
        with self._lock:
            if self._has_yolo():
                model = self._get_yolo_model()
                stream = iter(model.track(
                    source=str(path),
                    stream=True,
                    persist=False,
                    tracker="bytetrack.yaml",
                    device=self.cfg.device,
                    imgsz=self.cfg.image_size,
                    conf=self.cfg.confidence,
                    classes=list(CLASS_IDS),
                    max_det=100,
                    vid_stride=meta.sample_step,
                    verbose=False,
                ))
                try:
                    for sampled_index, result in enumerate(stream):
                        frame_index = sampled_index * meta.sample_step
                        timestamp_s = frame_index / meta.fps
                        if timestamp_s >= max_duration_s or sampled_index >= max_frames:
                            break
                        image = result.orig_img
                        detections = self._detections(result, frame_index, timestamp_s)
                        yield TrackedFrame(frame_index, round(timestamp_s, 3), image, detections)
                finally:
                    close = getattr(stream, "close", None)
                    if close:
                        close()
                return

            tracker = MultiObjectIoUTracker()
            for _, (frame_index, timestamp_s, image) in enumerate(
                sample_video(path, meta, max_frames, max_duration_s)
            ):
                raw_dets = self._predict_yolos_frame(image, frame_index, round(timestamp_s, 3))
                tracked = tracker.update(raw_dets)
                yield TrackedFrame(frame_index, round(timestamp_s, 3), image, tracked)

    @staticmethod
    def _detections(result, frame_index: int | None, timestamp_s: float | None) -> list[Detection]:
        boxes = result.boxes
        if boxes is None or len(boxes) == 0:
            return []
        xyxy = boxes.xyxy.detach().cpu().numpy()
        classes = boxes.cls.detach().cpu().numpy().astype(int)
        confidences = boxes.conf.detach().cpu().numpy()
        ids = boxes.id.detach().cpu().numpy().astype(int) if boxes.id is not None else [None] * len(boxes)
        found = []
        for index, class_id in enumerate(classes):
            label = CLASS_NAMES.get(int(class_id))
            if label is None:
                continue
            x1, y1, x2, y2 = [int(round(v)) for v in xyxy[index]]
            found.append(Detection(
                label=label,
                confidence=float(confidences[index]),
                box=(x1, y1, max(0, x2 - x1), max(0, y2 - y1)),
                track_id=None if ids[index] is None else str(int(ids[index])),
                frame_index=frame_index,
                timestamp_s=timestamp_s,
            ))
        return found


@lru_cache(maxsize=4)
def get_detector(model_path: str, device: str, confidence: float, image_size: int) -> YoloObjectDetector:
    """Share one loaded model/tracker between M01 and M03 per configuration."""
    cfg = ThreatConfig(model_path=Path(model_path), device=device, confidence=confidence, image_size=image_size)
    return YoloObjectDetector(cfg)


def detector_for(cfg: ThreatConfig) -> YoloObjectDetector:
    return get_detector(str(cfg.model_path.resolve()), cfg.device, cfg.confidence, cfg.image_size)
