"""Evidence assets: face crops, annotated frames, score timeline. Paths are stored relative to the evidence root."""
from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import cv2
import numpy as np
from PIL import Image

from deep_shield.common.schema import EvidenceRef

# BGR colours by verdict band
_RED, _AMBER, _GREEN = (40, 40, 220), (0, 170, 255), (80, 170, 60)


def band_colour(score: float, low: float, high: float) -> Tuple[int, int, int]:
    return _RED if score >= high else (_GREEN if score <= low else _AMBER)


class EvidenceStore:
    def __init__(self, root: Path, media_id: str):
        self.root = Path(root)
        self.media_id = media_id
        self.dir = self.root / media_id
        self.dir.mkdir(parents=True, exist_ok=True)

    def _rel(self, name: str) -> str:
        return f"{self.media_id}/{name}"

    def save_crop(self, img: Image.Image, name: str, description: str) -> EvidenceRef:
        im = img.copy()
        im.thumbnail((320, 320))
        im.save(self.dir / name, quality=90)
        return EvidenceRef(kind="face_crop", path=self._rel(name), description=description)

    def save_bgr(self, bgr: np.ndarray, name: str, kind: str, description: str) -> EvidenceRef:
        cv2.imwrite(str(self.dir / name), bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return EvidenceRef(kind=kind, path=self._rel(name), description=description)

    def path_for(self, name: str) -> Path:
        return self.dir / name

    def ref(self, name: str, kind: str, description: str) -> EvidenceRef:
        return EvidenceRef(kind=kind, path=self._rel(name), description=description)

    def save_annotated(self, bgr: np.ndarray, boxes: Sequence[Tuple[Tuple[int, int, int, int], str, Tuple[int, int, int]]],
                       name: str, kind: str, description: str, header: Optional[str] = None) -> EvidenceRef:
        img = bgr.copy()
        h, w = img.shape[:2]
        scale = 1280.0 / max(h, w) if max(h, w) > 1280 else 1.0
        if scale != 1.0:
            img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
        th = max(1, int(round(2 * max(img.shape[:2]) / 900)))
        fs = 0.45 * max(1.0, max(img.shape[:2]) / 900)
        top = 0
        if header:
            top = int(26 * fs / 0.45)
            cv2.rectangle(img, (0, 0), (img.shape[1], top), (30, 30, 30), -1)
            cv2.putText(img, header, (8, int(18 * fs / 0.45)), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
        for (x, y, bw, bh), label, colour in boxes:
            x, y, bw, bh = [int(round(v * scale)) for v in (x, y, bw, bh)]
            cv2.rectangle(img, (x, y), (x + bw, y + bh), colour, th)
            (tw, tht), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)
            ty = y - 4 if y - 4 - tht - 6 >= top else y + tht + 8  # label inside box if no room above
            cv2.rectangle(img, (x, ty - tht - 6), (x + tw + 6, ty + 2), colour, -1)
            cv2.putText(img, label, (x + 3, ty - 3), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imwrite(str(self.dir / name), img, [cv2.IMWRITE_JPEG_QUALITY, 88])
        return EvidenceRef(kind=kind, path=self._rel(name), description=description)

    def save_timeline(self, series: List[Tuple[str, List[float], List[float]]], low: float, high: float,
                      duration_s: float, name: str = "face_score_timeline.png") -> Optional[EvidenceRef]:
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
        except ImportError:
            return None
        fig, ax = plt.subplots(figsize=(9, 3), dpi=120)
        ax.axhspan(high, 1.0, color="#f4cccc", alpha=0.5, lw=0)
        ax.axhspan(low, high, color="#fde9c8", alpha=0.5, lw=0)
        for label, ts, ss in series:
            ax.plot(ts, ss, marker="o", ms=3, lw=1.5, label=label)
        ax.set_ylim(0, 1)
        ax.set_xlim(0, max(duration_s, max((t[-1] for _, t, _ in series if t), default=1.0)))
        ax.set_xlabel("time (s)")
        ax.set_ylabel("face-manipulation score")
        ax.set_title("Module 02 — per-track face-manipulation score over time (uncalibrated)", fontsize=9)
        if series:
            ax.legend(fontsize=7, loc="upper right")
        ax.grid(alpha=0.3)
        fig.tight_layout()
        fig.savefig(self.dir / name)
        plt.close(fig)
        return EvidenceRef(kind="score_timeline", path=self._rel(name),
                           description="Per-track face-manipulation score vs time; shaded bands = thresholds")
