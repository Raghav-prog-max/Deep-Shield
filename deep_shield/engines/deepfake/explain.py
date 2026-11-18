"""Suspicious-region map for the face classifier (occlusion sensitivity).

Model-agnostic: grey patches are slid over the face crop and the face-manipulation score is
re-computed. Regions whose occlusion lowers the score most are the regions the classifier relied on
for its "fake" output. This explains the *classifier*; it is not a pixel-level manipulation mask.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
from PIL import Image

SIDE = 224  # classifier input size; crops are resized once so all occlusions share one geometry


def occlusion_map(clf, crop: Image.Image, grid: int = 5, patch_frac: float = 0.34,
                  min_drop: float = 0.15) -> Tuple[np.ndarray, Dict]:
    """Return (heat[SIDE,SIDE] in 0..1, info). heat = share of the score removed by hiding each pixel."""
    im = crop.convert("RGB").resize((SIDE, SIDE), Image.BILINEAR)
    arr = np.asarray(im).copy()
    fill = arr.reshape(-1, 3).mean(axis=0).astype(np.uint8)
    p = max(8, int(round(SIDE * patch_frac)))
    pos = np.linspace(0, SIDE - p, grid).round().astype(int)
    boxes: List[Tuple[int, int, int]] = []
    masked: List[Image.Image] = []
    for y in pos:
        for x in pos:
            a = arr.copy()
            a[y:y + p, x:x + p] = fill
            masked.append(Image.fromarray(a))
            boxes.append((int(x), int(y), p))
    preds = clf.classify([im] + masked)
    base = preds[0].score
    drops = np.array([base - q.score for q in preds[1:]], dtype=np.float32)

    heat = np.zeros((SIDE, SIDE), np.float32)
    cnt = np.zeros((SIDE, SIDE), np.float32)
    for (x, y, s), d in zip(boxes, drops):
        heat[y:y + s, x:x + s] += d
        cnt[y:y + s, x:x + s] += 1
    heat = heat / np.maximum(cnt, 1)
    pos_heat = np.clip(heat, 0, None)
    peak = float(pos_heat.max())
    informative = peak >= min_drop
    # colour scale is absolute with a floor: a 0.05 drop is drawn faint, >= min_drop is drawn full.
    # This prevents tiny, noisy differences from being rendered as a confident "hot spot".
    norm = np.clip(pos_heat / max(peak, min_drop), 0, 1)

    k = int(np.argmax(drops))
    x, y, s = boxes[k]
    info = {
        "method": "occlusion_sensitivity",
        "grid": grid,
        "patch_fraction": patch_frac,
        "base_score": round(float(base), 4),
        "max_score_drop": round(float(drops.max()), 4),
        "mean_score_drop": round(float(drops.mean()), 4),
        # top region in normalised crop coordinates (0..1)
        "top_region_norm": [round(x / SIDE, 4), round(y / SIDE, 4), round(s / SIDE, 4), round(s / SIDE, 4)],
        "informative": bool(informative),
        "min_informative_drop": min_drop,
        "map_scale": (f"absolute, full colour at a score drop >= {min_drop}") ,
        "interpretation": ("a localised region drives the score" if informative else
                           "evidence spread across the face: no single region changes the score by "
                           f">= {min_drop} (typical of fully synthetic faces)"),
        "note": ("Regions whose occlusion most reduced the face-manipulation score; explains the "
                 "classifier's decision, not a ground-truth manipulation mask."),
    }
    return norm, info


def norm_box_to_image(norm_box, crop_box: Tuple[int, int, int, int]) -> Tuple[int, int, int, int]:
    nx, ny, nw, nh = norm_box
    cx, cy, cw, ch = crop_box
    return int(cx + nx * cw), int(cy + ny * ch), int(nw * cw), int(nh * ch)


def overlay(crop: Image.Image, heat: np.ndarray, top_norm: Optional[List[float]] = None,
            label: Optional[str] = None) -> np.ndarray:
    """BGR overlay of the heatmap on the crop (crop shown at up to 448 px)."""
    im = crop.convert("RGB")
    scale = min(1.0, 448.0 / max(im.size)) if max(im.size) > 448 else max(1.0, 224.0 / max(im.size))
    im = im.resize((max(1, int(im.width * scale)), max(1, int(im.height * scale))), Image.BILINEAR)
    bgr = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGB2BGR)
    h, w = bgr.shape[:2]
    hm = cv2.resize((heat * 255).astype(np.uint8), (w, h), interpolation=cv2.INTER_CUBIC)
    col = cv2.applyColorMap(hm, cv2.COLORMAP_JET)
    alpha = (hm.astype(np.float32) / 255.0 * 0.6)[..., None]
    out = (bgr * (1 - alpha) + col * alpha).astype(np.uint8)
    if top_norm:
        x, y, bw, bh = top_norm
        cv2.rectangle(out, (int(x * w), int(y * h)), (int((x + bw) * w), int((y + bh) * h)), (255, 255, 255), 2)
    if label:
        fs = 0.42
        while cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, fs, 1)[0][0] > w - 10 and fs > 0.28:
            fs -= 0.02
        cv2.rectangle(out, (0, 0), (w, 20), (30, 30, 30), -1)
        cv2.putText(out, label, (5, 14), cv2.FONT_HERSHEY_SIMPLEX, fs, (255, 255, 255), 1, cv2.LINE_AA)
    return out
