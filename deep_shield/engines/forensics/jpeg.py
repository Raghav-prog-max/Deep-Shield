"""JPEG-specific forensic analyses.

* Quantisation-table quality estimate (from the file's own tables, libjpeg scaling).
* JPEG ghosts (Farid, IEEE TIFS 2009): re-compress at a range of qualities; a region that was
  previously saved at a lower quality shows a local minimum ("ghost") at that quality.
* 8x8 block-grid alignment: JPEG blocking artefacts sit on a grid at offset (0, 0); a shifted grid
  suggests the image was cropped or re-positioned after an earlier JPEG save.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image

# libjpeg standard luminance table (natural row-major order)
_STD_LUMA = np.array([
    16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
    14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113, 92,
    49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=np.float64)
_ZIGZAG = np.array([
    0, 1, 8, 16, 9, 2, 3, 10, 17, 24, 32, 25, 18, 11, 4, 5, 12, 19, 26, 33, 40, 48, 41, 34, 27, 20, 13, 6,
    7, 14, 21, 28, 35, 42, 49, 56, 57, 50, 43, 36, 29, 22, 15, 23, 30, 37, 44, 51, 58, 59, 52, 45, 38, 31,
    39, 46, 53, 60, 61, 54, 47, 55, 62, 63])


def _scaled(q: int) -> np.ndarray:
    s = 5000.0 / q if q < 50 else 200.0 - 2.0 * q
    return np.clip(np.floor((_STD_LUMA * s + 50.0) / 100.0), 1, 255)


def quality_from_tables(path: Path) -> Dict:
    out: Dict = {"is_jpeg": False}
    try:
        with Image.open(path) as im:
            if (im.format or "").upper() not in ("JPEG", "MPO"):
                return out
            out["is_jpeg"] = True
            qt = getattr(im, "quantization", None) or {}
    except Exception as exc:  # noqa: BLE001
        out["error"] = str(exc)
        return out
    if 0 not in qt:
        return out
    t = np.array(qt[0], dtype=np.float64)
    if t.size != 64:
        return out
    best = None
    for order, tab in (("natural", t), ("zigzag", np.empty(64))):
        if order == "zigzag":
            tab[_ZIGZAG] = t               # undo zig-zag ordering if Pillow returned it that way
        for q in range(1, 101):
            err = float(np.abs(_scaled(q) - tab).mean())
            if best is None or err < best[0]:
                best = (err, q, order)
    err, q, order = best
    out.update({"estimated_quality": q, "table_match_error": round(err, 2),
                "standard_libjpeg_table": err < 0.5, "tables": len(qt)})
    return out


def _jpeg(img: np.ndarray, q: int) -> np.ndarray:
    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
    return cv2.imdecode(enc, cv2.IMREAD_COLOR)


def ghosts(bgr: np.ndarray, qualities: List[int] = None, block: int = 16, max_side: int = 1024) -> Dict:
    qualities = qualities or list(range(50, 100, 5))
    s = min(1.0, max_side / float(max(bgr.shape[:2])))
    img = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else bgr
    f = img.astype(np.float32)
    diffs = []
    for q in qualities:
        d = ((f - _jpeg(img, q).astype(np.float32)) ** 2).mean(axis=2)
        diffs.append(cv2.blur(d, (block, block)))
    D = np.stack(diffs)
    mn, mx = D.min(0), D.max(0)
    N = (D - mn) / np.maximum(mx - mn, 1e-6)              # per-pixel normalised over qualities
    return {"qualities": qualities, "_maps": N, "_img": img, "_scale": s}


def render_ghosts(g: Dict, show: List[int] = None) -> np.ndarray:
    show = show or [60, 70, 80, 90]
    tiles = []
    for q in show:
        if q not in g["qualities"]:
            continue
        m = g["_maps"][g["qualities"].index(q)]
        t = cv2.applyColorMap((255 * (1 - m)).astype(np.uint8), cv2.COLORMAP_INFERNO)  # bright = ghost
        cv2.rectangle(t, (0, 0), (t.shape[1], 22), (30, 30, 30), -1)
        cv2.putText(t, f"q{q}: bright = earlier save at ~q{q}", (6, 16),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1, cv2.LINE_AA)
        tiles.append(t)
    if not tiles:
        return g["_img"]
    while len(tiles) % 2 and len(tiles) > 1:
        tiles = tiles[:-1]
    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)] if len(tiles) > 1 else [tiles[0]]
    panel = np.vstack(rows)
    if panel.shape[1] > 2000:
        f = 2000.0 / panel.shape[1]
        panel = cv2.resize(panel, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return panel


def grid_alignment(bgr: np.ndarray) -> Dict:
    """Strength and offset of the 8x8 blocking grid from second differences along rows/columns."""
    g = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY).astype(np.float32)
    dx = np.abs(np.diff(g, 2, axis=1)).mean(axis=0)
    dy = np.abs(np.diff(g, 2, axis=0)).mean(axis=1)

    def period8(v):
        n = (len(v) // 8) * 8
        if n < 16:
            return 0, 1.0
        e = v[:n].reshape(-1, 8).mean(axis=0)
        k = int(np.argmax(e))
        return (k + 1) % 8, float(e.max() / (np.median(e) + 1e-6))   # +1: second diff is centred
    ox, sx = period8(dx)
    oy, sy = period8(dy)
    strength = round(min(sx, sy), 3)
    return {"grid_offset": [ox, oy], "grid_strength": strength,
            "blocking_visible": strength > 1.1,
            "aligned": (ox, oy) == (0, 0)}


def observations(qinfo: Dict, grid: Dict) -> List[str]:
    obs: List[str] = []
    if not qinfo.get("is_jpeg"):
        obs.append("Not a JPEG file: quantisation-table and ghost analyses are less informative.")
    else:
        q = qinfo.get("estimated_quality")
        if q is not None:
            std = "standard libjpeg tables" if qinfo.get("standard_libjpeg_table") else "non-standard (camera/app-specific) tables"
            obs.append(f"Last JPEG save at estimated quality ~{q} ({std}).")
    if grid["blocking_visible"] and not grid["aligned"]:
        obs.append(f"8x8 JPEG block grid is shifted by {grid['grid_offset']} px: consistent with cropping or "
                   f"re-positioning after an earlier JPEG save (not proof of manipulation).")
    elif grid["blocking_visible"]:
        obs.append("8x8 JPEG block grid aligned at (0, 0).")
    return obs
