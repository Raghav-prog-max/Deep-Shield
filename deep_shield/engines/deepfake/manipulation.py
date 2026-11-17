"""Exploratory image-integrity (manipulation) analysis — visual maps, no verdict.

  * Error Level Analysis (ELA): re-compress as JPEG and show per-pixel compression error.
  * Local noise-level map: robust high-pass noise estimate per block, shown as a ratio to the image
    median (red = noisier than the rest, blue = smoother, e.g. pasted vs. blurred/retouched areas).
  * Metadata review: EXIF camera / software / time fields.

Why no automatic verdict or flagged regions: in our checks, automatic block-outlier flagging both
missed real pastes and flagged untouched textured photos. Following the proposal (section 10), these
outputs are therefore labelled exploratory visualizations for human review.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Tuple

import cv2
import numpy as np
from PIL import Image

EDIT_TOOLS = ("photoshop", "gimp", "lightroom", "snapseed", "picsart", "canva", "affinity",
              "pixelmator", "facetune", "meitu", "faceapp", "remini", "midjourney", "stable diffusion",
              "dall", "firefly")
_K = np.array([[1, -2, 1], [-2, 4, -2], [1, -2, 1]], np.float32)  # Immerkaer noise-estimation kernel


def _noise_blocks(gray: np.ndarray, b: int) -> np.ndarray:
    hp = np.abs(cv2.filter2D(gray.astype(np.float32), -1, _K))
    h, w = hp.shape
    gh, gw = h // b, w // b
    blk = hp[: gh * b, : gw * b].reshape(gh, b, gw, b).transpose(0, 2, 1, 3).reshape(gh, gw, -1)
    q = np.percentile(blk, 75, axis=2, keepdims=True)           # trimmed mean: drop edge pixels
    m = blk <= q
    return ((blk * m).sum(axis=2) / np.maximum(m.sum(axis=2), 1)).astype(np.float32)


def analyse(bgr: np.ndarray, jpeg_quality: int = 90, block: int = 16, max_side: int = 1536) -> Dict:
    h0, w0 = bgr.shape[:2]
    s = min(1.0, max_side / float(max(h0, w0)))
    img = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else bgr

    ok, enc = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, jpeg_quality])
    rec = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    ela = cv2.absdiff(img, rec).astype(np.float32).max(axis=2)

    noise = _noise_blocks(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), block)
    med = float(np.median(noise)) + 1e-3
    log_ratio = np.log2((noise + 1e-3) / med)
    return {
        "method": "ela+local_noise_level_map (exploratory visualization)",
        "jpeg_quality_for_ela": jpeg_quality,
        "block_px": int(round(block / s)),
        "analysis_scale": round(s, 4),
        "ela_mean": round(float(ela.mean()), 3),
        "ela_p99": round(float(np.percentile(ela, 99)), 3),
        "noise_median": round(med, 3),
        "noise_ratio_p05_p95": [round(float(2 ** np.percentile(log_ratio, 5)), 2),
                                round(float(2 ** np.percentile(log_ratio, 95)), 2)],
        "share_blocks_noise_2x_off": round(float((np.abs(log_ratio) >= 1).mean()), 4),
        "_ela": ela, "_logr": log_ratio, "_img": img, "_block": block,
    }


def public(res: Dict) -> Dict:
    return {k: v for k, v in res.items() if not k.startswith("_")}


def _caption(img: np.ndarray, text: str) -> np.ndarray:
    out = img.copy()
    cv2.rectangle(out, (0, 0), (out.shape[1], 24), (30, 30, 30), -1)
    cv2.putText(out, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return out


def render(res: Dict) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(ELA map, noise-ratio overlay, side-by-side panel) as BGR images."""
    img, ela, logr, b = res["_img"], res["_ela"], res["_logr"], res["_block"]
    ela_vis = cv2.applyColorMap(
        np.clip(ela * (255.0 / max(1.0, float(np.percentile(ela, 99.5)))), 0, 255).astype(np.uint8),
        cv2.COLORMAP_INFERNO)

    h, w = img.shape[:2]
    gh, gw = logr.shape
    lr_full = np.zeros((h, w), np.float32)
    lr_full[: gh * b, : gw * b] = cv2.resize(logr.astype(np.float32), (gw * b, gh * b),
                                             interpolation=cv2.INTER_NEAREST)
    # colour only blocks >= 2x away from the image median; full colour at >= 8x
    pos = np.clip((lr_full - 1.0) / 2.0, 0, 1)   # noisier -> red
    neg = np.clip((-lr_full - 1.0) / 2.0, 0, 1)  # smoother -> blue
    col = np.zeros_like(img, dtype=np.float32)
    col[..., 2] = 255 * pos
    col[..., 0] = 255 * neg
    alpha = (np.maximum(pos, neg) * 0.65)[..., None]
    noise_vis = (img * (1 - alpha) + col * alpha).astype(np.uint8)

    panel = np.hstack([_caption(img, "input"), _caption(ela_vis, "ELA (JPEG re-compression error)"),
                       _caption(noise_vis, "noise level vs median: red >=2x, blue <=0.5x")])
    if panel.shape[1] > 2400:
        f = 2400.0 / panel.shape[1]
        panel = cv2.resize(panel, None, fx=f, fy=f, interpolation=cv2.INTER_AREA)
    return ela_vis, noise_vis, panel


def read_metadata(path: Path) -> Dict:
    out: Dict = {"format": None, "has_exif": False, "notes": []}
    try:
        with Image.open(path) as im:
            out["format"] = im.format
            ex = im.getexif()
            if ex:
                out["has_exif"] = True
                sub = ex.get_ifd(0x8769) if hasattr(ex, "get_ifd") else {}
                fields = {"make": ex.get(271), "model": ex.get(272), "software": ex.get(305),
                          "datetime": ex.get(306), "datetime_original": sub.get(36867) if sub else None}
                out.update({k: (str(v).strip("\x00 ") if v else None) for k, v in fields.items()})
                sw = (out.get("software") or "").lower()
                if any(t in sw for t in EDIT_TOOLS):
                    out["notes"].append(f"Software tag names an editing/generation tool ({out['software']}).")
                if out.get("datetime") and out.get("datetime_original") and out["datetime"] != out["datetime_original"]:
                    out["notes"].append("Modification time differs from original capture time.")
            else:
                out["notes"].append("No EXIF metadata (common after messaging/social-media upload; not evidence).")
    except Exception as exc:  # noqa: BLE001
        out["notes"].append(f"Metadata could not be read: {exc}")
    if out["format"] and out["format"].upper() not in ("JPEG", "MPO"):
        out["notes"].append(f"Source is {out['format']}, not JPEG: ELA is less informative for lossless formats.")
    return out
