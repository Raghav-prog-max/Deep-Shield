"""Face localisation for the face-manipulation task.

Preference order:
  1. OpenCV YuNet (ONNX, Apache-2.0) — accurate, fast on CPU. Fetched by scripts/download_models.py.
  2. OpenCV Haar cascade — ships with opencv-python 4.x, zero download.
  3. None — face task reports "no face detector available".

Faces are localised only so the classifier can see a face crop. No identity,
recognition or attribute inference is performed.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

from .config import EngineConfig

YUNET_FILE = "face_detection_yunet_2023mar.onnx"
HAAR_FILE = "haarcascade_frontalface_default.xml"


@dataclass
class FaceBox:
    x: int
    y: int
    w: int
    h: int
    confidence: Optional[float] = None   # None for Haar (no calibrated score)

    def as_tuple(self) -> Tuple[int, int, int, int]:
        return self.x, self.y, self.w, self.h


class FaceDetector:
    def __init__(self, cfg: EngineConfig):
        self.cfg = cfg
        self.backend = "none"
        self._yunet = None
        self._haar = None

        yunet_path = cfg.yunet_path or (cfg.models_dir / YUNET_FILE)
        if Path(yunet_path).exists() and hasattr(cv2, "FaceDetectorYN"):
            try:
                self._yunet = cv2.FaceDetectorYN.create(
                    str(yunet_path), "", (320, 320), cfg.face_score_threshold, 0.3, 5000
                )
                self.backend = "yunet"
                return
            except cv2.error:
                self._yunet = None

        candidates = [cfg.haar_path, cfg.models_dir / HAAR_FILE]
        if hasattr(cv2, "data") and getattr(cv2.data, "haarcascades", None):
            candidates.append(Path(cv2.data.haarcascades) / HAAR_FILE)
        for c in candidates:
            if c and Path(c).exists():
                clf = cv2.CascadeClassifier(str(c))
                if not clf.empty():
                    self._haar = clf
                    self.backend = "haar"
                    return

    @property
    def available(self) -> bool:
        return self.backend != "none"

    # (target longest side in px, border padding as a fraction of the longest side).
    # Close-up portraits (face filling a 1024 px frame, e.g. GAN faces or selfies) are missed by
    # YuNet/Haar at full resolution, so later attempts shrink the image and/or add a border.
    _ATTEMPTS = ((640, 0.0), (320, 0.0), (480, 0.35))

    def _detect_raw(self, img: np.ndarray, min_size: int) -> List[FaceBox]:
        h, w = img.shape[:2]
        out: List[FaceBox] = []
        if self._yunet is not None:
            self._yunet.setInputSize((w, h))
            _, faces = self._yunet.detect(img)
            if faces is not None:
                for f in faces:
                    out.append(FaceBox(int(f[0]), int(f[1]), int(f[2]), int(f[3]), float(f[-1])))
        elif self._haar is not None:
            gray = cv2.equalizeHist(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY))
            for (x, y, fw, fh) in self._haar.detectMultiScale(
                    gray, scaleFactor=1.1, minNeighbors=5, minSize=(min_size, min_size)):
                out.append(FaceBox(int(x), int(y), int(fw), int(fh), None))
        return out

    def detect(self, bgr: np.ndarray) -> List[FaceBox]:
        if not self.available:
            return []
        h, w = bgr.shape[:2]
        boxes: List[FaceBox] = []
        for target, pad_frac in self._ATTEMPTS:
            pad = int(round(pad_frac * max(h, w)))
            img = cv2.copyMakeBorder(bgr, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(0, 0, 0)) if pad else bgr
            s = min(1.0, target / float(max(img.shape[:2])))
            small = cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1.0 else img
            raw = self._detect_raw(small, max(20, int(self.cfg.face_min_size * s)))
            if raw:
                boxes = [FaceBox(int(round(b.x / s)) - pad, int(round(b.y / s)) - pad,
                                 int(round(b.w / s)), int(round(b.h / s)), b.confidence) for b in raw]
                break

        boxes = [_clip(b, w, h) for b in boxes]
        boxes = [b for b in boxes if b.w >= self.cfg.face_min_size and b.h >= self.cfg.face_min_size]
        boxes.sort(key=lambda b: b.w * b.h, reverse=True)
        return boxes[: self.cfg.max_faces_per_frame]


def _clip(b: FaceBox, w: int, h: int) -> FaceBox:
    x0, y0 = max(0, b.x), max(0, b.y)
    x1, y1 = min(w, b.x + b.w), min(h, b.y + b.h)
    return FaceBox(x0, y0, max(0, x1 - x0), max(0, y1 - y0), b.confidence)


def crop_box(shape, box: FaceBox, margin: float) -> Tuple[int, int, int, int]:
    """(x, y, w, h) of the square context crop around a face, clipped to the image."""
    h, w = shape[:2]
    cx, cy = box.x + box.w / 2.0, box.y + box.h / 2.0
    side = max(box.w, box.h) * (1.0 + 2.0 * margin)
    x0, y0 = int(max(0, cx - side / 2)), int(max(0, cy - side / 2))
    x1, y1 = int(min(w, cx + side / 2)), int(min(h, cy + side / 2))
    return x0, y0, x1 - x0, y1 - y0


def crop_face(bgr: np.ndarray, box: FaceBox, margin: float) -> Image.Image:
    """Square crop around the face with context margin (face classifiers are trained on such crops)."""
    x0, y0, cw, ch = crop_box(bgr.shape, box, margin)
    crop = bgr[y0:y0 + ch, x0:x0 + cw]
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
