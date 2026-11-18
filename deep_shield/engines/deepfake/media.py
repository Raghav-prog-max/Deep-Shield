"""Media loading, hashing and timestamped video frame sampling."""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Tuple

import cv2
import numpy as np
from PIL import Image, ImageOps

IMAGE_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
VIDEO_EXT = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"}


class MediaError(ValueError):
    pass


def media_kind(path: Path) -> str:
    ext = Path(path).suffix.lower()
    if ext in IMAGE_EXT:
        return "image"
    if ext in VIDEO_EXT:
        return "video"
    raise MediaError(f"Unsupported file type '{ext}'. Images: {sorted(IMAGE_EXT)}; videos: {sorted(VIDEO_EXT)}")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def load_image(path: Path) -> Tuple[np.ndarray, Image.Image]:
    """Return (BGR ndarray, RGB PIL) with EXIF orientation applied."""
    try:
        with Image.open(path) as im:
            im = ImageOps.exif_transpose(im).convert("RGB")
            pil = im.copy()
    except Exception as exc:  # noqa: BLE001
        raise MediaError(f"Cannot decode image: {exc}") from exc
    bgr = cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)
    return bgr, pil


@dataclass
class VideoMeta:
    fps: float
    frame_count: int
    duration_s: float
    width: int
    height: int
    sample_step: int
    fps_assumed: bool

    def as_dict(self) -> Dict:
        return self.__dict__.copy()


def probe_video(path: Path, sample_fps: float) -> VideoMeta:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise MediaError("Cannot open video (codec unsupported or file corrupt)")
    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    assumed = False
    if not fps or fps != fps or fps > 240:
        fps, assumed = 25.0, True
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    cap.release()
    step = max(1, int(round(fps / max(sample_fps, 1e-6))))
    return VideoMeta(fps, n, n / fps if n else 0.0, w, h, step, assumed)


def sample_video(path: Path, meta: VideoMeta, max_frames: int, max_duration_s: float
                 ) -> Iterator[Tuple[int, float, np.ndarray]]:
    """Yield (frame_index, timestamp_s, bgr) for every ``meta.sample_step``-th frame."""
    cap = cv2.VideoCapture(str(path))
    idx, emitted = 0, 0
    try:
        while emitted < max_frames:
            ok = cap.grab()
            if not ok:
                break
            t = idx / meta.fps
            if t > max_duration_s:
                break
            if idx % meta.sample_step == 0:
                ok, frame = cap.retrieve()
                if ok and frame is not None:
                    yield idx, round(t, 3), frame
                    emitted += 1
            idx += 1
    finally:
        cap.release()


def read_frames(path: Path, indices: Iterable[int]) -> Dict[int, np.ndarray]:
    """Re-read specific frames (used for evidence images, keeps memory low during analysis)."""
    wanted = sorted(set(indices))
    out: Dict[int, np.ndarray] = {}
    if not wanted:
        return out
    cap = cv2.VideoCapture(str(path))
    idx, last = 0, wanted[-1]
    want = set(wanted)
    try:
        while idx <= last:
            ok = cap.grab()
            if not ok:
                break
            if idx in want:
                ok, frame = cap.retrieve()
                if ok:
                    out[idx] = frame
            idx += 1
    finally:
        cap.release()
    return out
