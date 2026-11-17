"""Temporal reasoning for video: face tracks across sampled frames and per-track score aggregation."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from deep_shield.common.schema import Segment, TemporalSummary, TimePoint


@dataclass
class FaceObs:
    sample_idx: int
    frame_index: int
    timestamp_s: float
    box: Tuple[int, int, int, int]
    det_conf: Optional[float]
    score: float = 0.0
    raw: Dict[str, float] = field(default_factory=dict)
    predicted_label: str = ""


@dataclass
class FaceTrack:
    track_id: str
    obs: List[FaceObs] = field(default_factory=list)


def iou(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int]) -> float:
    ax0, ay0, aw, ah = a
    bx0, by0, bw, bh = b
    ix = max(0, min(ax0 + aw, bx0 + bw) - max(ax0, bx0))
    iy = max(0, min(ay0 + ah, by0 + bh) - max(ay0, by0))
    inter = ix * iy
    union = aw * ah + bw * bh - inter
    return inter / union if union > 0 else 0.0


def _link_score(a: Tuple[int, int, int, int], b: Tuple[int, int, int, int], iou_thr: float,
                center_dist: float) -> float:
    """IoU, or (if the box jumped between sparse samples) a just-passing score when the centres
    are within ``center_dist`` x face size and the sizes are similar."""
    v = iou(a, b)
    if v >= iou_thr or center_dist <= 0:
        return v
    ax, ay = a[0] + a[2] / 2, a[1] + a[3] / 2
    bx, by = b[0] + b[2] / 2, b[1] + b[3] / 2
    size = max(a[2], a[3], b[2], b[3])
    ratio = min(a[2], b[2]) / max(a[2], b[2], 1)
    if ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5 <= center_dist * size and ratio >= 0.6:
        return iou_thr
    return v


def build_tracks(per_sample: List[List[FaceObs]], iou_thr: float, max_gap: int,
                 prefix: str = "m02-face-T", center_dist: float = 0.0) -> List[FaceTrack]:
    """Greedy association across sampled frames by IoU with a centre-distance fallback
    (sufficient for short clips with one or a few people)."""
    tracks: List[FaceTrack] = []
    for obs_list in per_sample:
        if not obs_list:
            continue
        cur = obs_list[0].sample_idx
        active = [t for t in tracks if cur - t.obs[-1].sample_idx <= max_gap + 1]
        pairs = sorted(
            ((_link_score(t.obs[-1].box, o.box, iou_thr, center_dist), ti, oi) for ti, t in enumerate(active) for oi, o in enumerate(obs_list)),
            reverse=True,
        )
        used_t, used_o = set(), set()
        for v, ti, oi in pairs:
            if v < iou_thr:
                break
            if ti in used_t or oi in used_o:
                continue
            active[ti].obs.append(obs_list[oi])
            used_t.add(ti)
            used_o.add(oi)
        for oi, o in enumerate(obs_list):
            if oi not in used_o:
                tracks.append(FaceTrack(f"{prefix}{len(tracks) + 1}", [o]))
    return tracks


def summarise(points: List[Tuple[float, int, float]], thr_high: float, sample_period_s: float,
              max_series: int = 200) -> TemporalSummary:
    """points = [(t, frame_index, score)] in time order."""
    scores = np.array([p[2] for p in points], dtype=np.float64)
    segs: List[Segment] = []
    start = None
    peak = 0.0
    prev_t = None
    for t, _, s in points:
        if s >= thr_high:
            if start is None or (prev_t is not None and t - prev_t > 2.5 * sample_period_s):
                if start is not None:
                    segs.append(Segment(start_s=start, end_s=prev_t, peak_score=round(peak, 4)))
                start, peak = t, s
            peak = max(peak, s)
            prev_t = t
        else:
            if start is not None:
                segs.append(Segment(start_s=start, end_s=prev_t, peak_score=round(peak, 4)))
                start, prev_t = None, None
    if start is not None:
        segs.append(Segment(start_s=start, end_s=prev_t, peak_score=round(peak, 4)))

    stride = max(1, len(points) // max_series)
    series = [TimePoint(t=p[0], frame_index=p[1], score=round(p[2], 4)) for p in points[::stride]]
    return TemporalSummary(
        n_observations=len(points),
        mean=round(float(scores.mean()), 4),
        median=round(float(np.median(scores)), 4),
        max=round(float(scores.max()), 4),
        std=round(float(scores.std()), 4),
        fraction_above_high=round(float((scores >= thr_high).mean()), 4),
        segments_above_high=segs,
        series=series,
    )
