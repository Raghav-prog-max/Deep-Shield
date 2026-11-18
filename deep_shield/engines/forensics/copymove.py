"""Copy-move (cloning) detection by keypoint self-matching.

Method (Amerini et al., IEEE TIFS 2011, simplified): SIFT keypoints are matched against the
other keypoints of the SAME image with a generalised 2-nearest-neighbour ratio test; spatially
distant matches are clustered by RANSAC-estimated similarity transforms. A cluster with enough
consistent inliers is reported as a candidate (source -> target) cloned region pair.

Known limitation: repeated real-world structures (tiles, windows, flags, text) can also produce
consistent matches; results are candidates for human review.
"""
from __future__ import annotations

from typing import Dict, List

import cv2
import numpy as np


def _detector(contrast: float = 0.02, nfeatures: int = 5000):
    if hasattr(cv2, "SIFT_create"):
        return cv2.SIFT_create(nfeatures=nfeatures, contrastThreshold=contrast), cv2.NORM_L2, "sift"
    return cv2.ORB_create(nfeatures=6000), cv2.NORM_HAMMING, "orb"


def detect(bgr: np.ndarray, max_side: int = 1200, ratio: float = 0.6, min_dist: float = 40.0,
           min_inliers: int = 10, max_clusters: int = 3, contrast: float = 0.01, nfeatures: int = 10000,
           min_side: int = 0) -> Dict:
    h0, w0 = bgr.shape[:2]
    s = min(1.0, max_side / float(max(h0, w0)))
    if min_side and max(h0, w0) < min_side:          # enlarge small images so small clones get keypoints
        s = min_side / float(max(h0, w0))
    img = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA if s < 1 else cv2.INTER_CUBIC) if s != 1 else bgr
    min_dist = min_dist * max(1.0, s)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    det, norm, name = _detector(contrast, nfeatures)
    kps, des = det.detectAndCompute(gray, None)
    out = {"method": f"{name} self-matching (g2NN ratio {ratio}) + RANSAC similarity clustering",
           "keypoints": int(len(kps) if kps is not None else 0), "matches": 0, "clusters": [],
           "params": {"ratio": ratio, "min_dist_px": min_dist, "min_inliers": min_inliers},
           "_scale": s}
    if des is None or len(kps) < min_inliers * 2:
        return out
    k = min(6, len(kps))
    knn = cv2.BFMatcher(norm).knnMatch(des, des, k=k)
    pts = np.float32([kp.pt for kp in kps])
    pairs = set()
    for row in knn:
        if len(row) < 3:
            continue
        q = row[0].queryIdx
        cands = [m for m in row if m.trainIdx != q]
        for i in range(len(cands) - 1):
            if cands[i].distance < ratio * max(cands[i + 1].distance, 1e-6):
                t = cands[i].trainIdx
                if np.linalg.norm(pts[q] - pts[t]) > min_dist:
                    a, b = (q, t) if (pts[q][0], pts[q][1]) <= (pts[t][0], pts[t][1]) else (t, q)
                    pairs.add((a, b))
            else:
                break
    out["matches"] = len(pairs)
    if len(pairs) < min_inliers:
        return out
    P = np.array(sorted(pairs))
    A, B = pts[P[:, 0]], pts[P[:, 1]]
    rem = np.arange(len(P))
    while len(rem) >= min_inliers and len(out["clusters"]) < max_clusters:
        M, inl = cv2.estimateAffinePartial2D(A[rem], B[rem], method=cv2.RANSAC,
                                             ransacReprojThreshold=3.0, maxIters=4000, confidence=0.995)
        if M is None or inl is None:
            break
        inl = inl.ravel().astype(bool)
        if inl.sum() < min_inliers:
            break
        src, dst = A[rem][inl], B[rem][inl]
        scale = float(np.hypot(M[0, 0], M[1, 0]))
        shift = float(np.linalg.norm(dst.mean(0) - src.mean(0)))
        # spatial support: inliers must cover an area, not a handful of coincident points
        spread = float(min(np.ptp(src[:, 0]) + np.ptp(src[:, 1]), np.ptp(dst[:, 0]) + np.ptp(dst[:, 1])))
        if shift > min_dist and spread > 2 * min_dist and 0.5 < scale < 2.0:
            def box(p):
                x0, y0 = p.min(0); x1, y1 = p.max(0)
                return [int(x0 / s), int(y0 / s), int((x1 - x0) / s) + 1, int((y1 - y0) / s) + 1]
            out["clusters"].append({
                "source_bbox": box(src), "target_bbox": box(dst), "inliers": int(inl.sum()),
                "shift_px": round(shift / s, 1), "scale": round(scale, 3),
                "rotation_deg": round(float(np.degrees(np.arctan2(M[1, 0], M[0, 0]))), 1),
                "_src": src, "_dst": dst})
        rem = rem[~inl]
    return out


def render(bgr: np.ndarray, res: Dict) -> np.ndarray:
    s = res["_scale"]
    img = cv2.resize(bgr, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s < 1 else bgr.copy()
    for i, c in enumerate(res["clusters"]):
        for p, q in zip(c["_src"][:200], c["_dst"][:200]):
            cv2.line(img, tuple(int(v) for v in p), tuple(int(v) for v in q), (0, 220, 255), 1, cv2.LINE_AA)
        for key, col in (("source_bbox", (80, 200, 60)), ("target_bbox", (40, 40, 230))):
            x, y, w, h = [int(v * s) for v in c[key]]
            cv2.rectangle(img, (x, y), (x + w, y + h), col, 2)
        x, y = int(c["target_bbox"][0] * s), int(c["target_bbox"][1] * s)
        cv2.putText(img, f"C{i + 1}: {c['inliers']} matches", (x, max(14, y - 5)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    label = (f"copy-move candidates: {len(res['clusters'])}  (green=source, red=target)"
             if res["clusters"] else "no copy-move candidates")
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (30, 30, 30), -1)
    cv2.putText(img, label, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
    return img


def public(res: Dict) -> Dict:
    return {**{k: v for k, v in res.items() if not k.startswith("_") and k != "clusters"},
            "clusters": [{k: v for k, v in c.items() if not k.startswith("_")} for c in res["clusters"]]}
