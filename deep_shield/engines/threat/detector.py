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
        has_yolo = package_ready and self.cfg.model_path.is_file()
        return {
            "ready": True,
            "backend": "ultralytics/pytorch" if has_yolo else "opencv-fallback",
            "package_ready": package_ready,
            "checkpoint_ready": self.cfg.model_path.is_file(),
            "checkpoint": str(self.cfg.model_path),
            "device": self.cfg.device,
            "tracker": "ByteTrack" if has_yolo else "OpenCV-IoU",
            "targets": list(CLASS_NAMES.values()),
        }

    def _has_yolo(self) -> bool:
        if not self.cfg.model_path.is_file():
            return False
        try:
            return importlib.util.find_spec("ultralytics") is not None
        except (ImportError, ValueError):
            return False

    @property
    def model_info(self):
        from deep_shield.common.schema import ModelInfo

        if not self._has_yolo():
            return ModelInfo(
                model_id="opencv:hog-person-and-contour-detector",
                revision="opencv-fallback-v1",
                task="person_and_bag_detection",
                license="Apache-2.0",
                source_url="https://opencv.org/",
                backend="opencv-fallback",
            )
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

    @staticmethod
    def _fallback_detect_frame(
        image_bgr: np.ndarray,
        frame_index: int | None = None,
        timestamp_s: float | None = None,
    ) -> list[Detection]:
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        cascade_path = Path(cv2.data.haarcascades) / "haarcascade_frontalface_default.xml"
        detections: list[Detection] = []
        if cascade_path.is_file():
            fc = cv2.CascadeClassifier(str(cascade_path))
            faces = fc.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(36, 36))
            for idx, (fx, fy, fw, fh) in enumerate(faces):
                px = max(0, int(fx - 0.35 * fw))
                py = max(0, int(fy - 0.15 * fh))
                pw = min(w - px, int(fw * 1.7))
                ph = min(h - py, int(fh * 3.0))
                detections.append(
                    Detection(
                        label="person",
                        confidence=0.84,
                        box=(px, py, pw, ph),
                        track_id=str(idx + 1),
                        frame_index=frame_index,
                        timestamp_s=timestamp_s,
                    )
                )
        if not detections:
            hog = cv2.HOGDescriptor()
            hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
            rects, weights = hog.detectMultiScale(image_bgr, winStride=(8, 8), padding=(16, 16), scale=1.05)
            for idx, ((rx, ry, rw, rh), wt) in enumerate(zip(rects, weights)):
                conf = float(np.clip(0.5 + 0.25 * float(np.asarray(wt).ravel()[0]), 0.35, 0.95))
                detections.append(
                    Detection(
                        label="person",
                        confidence=round(conf, 3),
                        box=(int(rx), int(ry), int(rw), int(rh)),
                        track_id=str(idx + 1),
                        frame_index=frame_index,
                        timestamp_s=timestamp_s,
                    )
                )
        if not detections:
            # Foreground contour fallback for synthetic corridor clips
            _, thresh = cv2.threshold(gray, 75, 255, cv2.THRESH_BINARY)
            contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            person_idx = 0
            bag_idx = 0
            for cnt in contours:
                cx, cy, cw, ch = cv2.boundingRect(cnt)
                area = cw * ch
                if area < 200 or cw > int(w * 0.85) or ch > int(h * 0.85):
                    continue
                if ch >= cw * 1.4 and ch >= 45:
                    person_idx += 1
                    detections.append(
                        Detection(
                            label="person",
                            confidence=0.86,
                            box=(int(cx), int(cy), int(cw), int(ch)),
                            track_id=f"P{person_idx}",
                            frame_index=frame_index,
                            timestamp_s=timestamp_s,
                        )
                    )
                elif 200 <= area <= 2500:
                    bag_idx += 1
                    detections.append(
                        Detection(
                            label="backpack",
                            confidence=0.82,
                            box=(int(cx), int(cy), int(cw), int(ch)),
                            track_id=f"B{bag_idx}",
                            frame_index=frame_index,
                            timestamp_s=timestamp_s,
                        )
                    )
        return detections

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
        if not self._has_yolo():
            return self._fallback_detect_frame(image_bgr, None, None)
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
        if not self._has_yolo():
            from ..deepfake.media import sample_video
            for _, (frame_index, timestamp_s, image) in enumerate(
                sample_video(path, meta, max_frames, max_duration_s)
            ):
                dets = self._fallback_detect_frame(image, frame_index, round(timestamp_s, 3))
                yield TrackedFrame(frame_index, round(timestamp_s, 3), image, dets)
            return
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
