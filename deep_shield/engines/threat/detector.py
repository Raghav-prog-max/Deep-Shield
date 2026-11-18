"""Lazy CPU YOLO + ByteTrack adapter shared by Modules 01 and 03."""
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

from .config import ThreatConfig
from ..deepfake.media import MediaError, probe_video

log = logging.getLogger("deep_shield.m01.detector")

# COCO ids for person, backpack, handbag and suitcase.
CLASS_IDS = (0, 24, 26, 28)
CLASS_NAMES = {0: "person", 24: "backpack", 26: "handbag", 28: "suitcase"}


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


class YoloObjectDetector:
    """Loads a local checkpoint once and serializes access to its tracker state."""

    def __init__(self, cfg: ThreatConfig):
        self.cfg = cfg
        self._model = None
        self._lock = threading.RLock()
        self._revision: str | None = None

    def health(self) -> dict:
        try:
            package_ready = importlib.util.find_spec("ultralytics") is not None
        except (ImportError, ValueError):
            package_ready = False
        return {
            "ready": package_ready and self.cfg.model_path.is_file(),
            "package_ready": package_ready,
            "checkpoint_ready": self.cfg.model_path.is_file(),
            "checkpoint": str(self.cfg.model_path),
            "device": self.cfg.device,
            "tracker": "ByteTrack",
            "targets": list(CLASS_NAMES.values()),
        }

    @property
    def model_info(self):
        from deep_shield.common.schema import ModelInfo

        if self._revision is None and self.cfg.model_path.is_file():
            digest = hashlib.sha256()
            with self.cfg.model_path.open("rb") as source:
                for chunk in iter(lambda: source.read(1 << 20), b""):
                    digest.update(chunk)
            self._revision = digest.hexdigest()
        return ModelInfo(
            model_id=f"ultralytics:{self.cfg.model_path.stem}",
            revision=self._revision,
            task="person_and_bag_detection",
            license="AGPL-3.0",
            source_url="https://docs.ultralytics.com/models/yolo26/",
            backend="ultralytics/pytorch",
        )

    def _get_model(self):
        if self._model is None:
            if not self.cfg.model_path.is_file():
                raise FileNotFoundError(
                    f"Engine 1 model checkpoint not found: {self.cfg.model_path}. "
                    "Download YOLO26n using scripts/download_m01_model.py."
                )
            try:
                from ultralytics import YOLO
            except ImportError as exc:
                raise RuntimeError("Install the optional Ultralytics dependency from requirements.txt") from exc
            self._model = YOLO(str(self.cfg.model_path))
        return self._model

    def predict_image(self, image_bgr: np.ndarray) -> list[Detection]:
        with self._lock:
            model = self._get_model()
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
            model = self._get_model()
            stream = iter(model.track(
                source=str(path),
                stream=True,
                # The tracker persists between frames in this one video stream,
                # but resets when a new source video is analyzed.
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
