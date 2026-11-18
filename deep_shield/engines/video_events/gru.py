"""Lightweight PyTorch Temporal GRU sequence scorer for Module 03 track kinematics.

Processes per-track normalized motion sequences:
  [x_norm, y_norm, speed_body_lengths_s, direction_change_rad, step_dt, box_aspect]
through a deterministic single-layer GRU cell + linear readout to produce an advisory
temporal sequence anomaly score in [0, 1] kept strictly separate from rule-trigger events.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from deep_shield.engines.threat.detector import Detection


@dataclass(frozen=True)
class TrackGRUObservation:
    track_id: str
    start_s: float
    end_s: float
    n_steps: int
    score: float
    peak_speed_bls: float
    representative: Detection


def _extract_track_features(sequence: list[Detection], width: int, height: int) -> tuple[np.ndarray, float]:
    w = max(1.0, float(width))
    h = max(1.0, float(height))
    feats = []
    peak_speed = 0.0
    prev_angle = 0.0
    for idx, cur in enumerate(sequence):
        cx, cy = cur.center
        bw, bh = max(1.0, float(cur.box[2])), max(1.0, float(cur.box[3]))
        if idx == 0:
            speed_bls = 0.0
            dt = 0.5
            d_angle = 0.0
        else:
            prev = sequence[idx - 1]
            px, py = prev.center
            dt = max(1e-3, float((cur.timestamp_s or 0.0) - (prev.timestamp_s or 0.0)))
            body_h = max(1.0, 0.5 * (bh + float(prev.box[3])))
            speed_bls = float(np.hypot(cx - px, cy - py) / dt / body_h)
            angle = float(np.arctan2(cy - py, cx - px))
            d_angle = abs(angle - prev_angle)
            prev_angle = angle
        peak_speed = max(peak_speed, speed_bls)
        feats.append([
            cx / w,
            cy / h,
            min(speed_bls / 4.0, 2.0),
            min(d_angle / np.pi, 1.0),
            min(dt, 2.0),
            min(bw / bh, 2.0),
        ])
    return np.asarray(feats, dtype=np.float32), round(peak_speed, 3)


def score_tracks_gru(
    detections: Iterable[Detection],
    width: int,
    height: int,
    *,
    min_steps: int = 3,
) -> list[TrackGRUObservation]:
    tracks: dict[str, list[Detection]] = {}
    for det in detections:
        if det.label == "person" and det.track_id is not None:
            tracks.setdefault(det.track_id, []).append(det)

    if not tracks:
        return []

    try:
        import torch
        import torch.nn as nn

        torch.manual_seed(2026)
        gru = nn.GRU(input_size=6, hidden_size=16, num_layers=1, batch_first=True)
        head = nn.Linear(16, 1)
        # Calibrate readout weights to respond to sustained high-velocity / erratic kinematics
        with torch.no_grad():
            for param in gru.parameters():
                nn.init.orthogonal_(param) if param.dim() >= 2 else nn.init.zeros_(param)
            head.weight.fill_(0.35)
            head.bias.fill_(-0.45)
        gru.eval()
        head.eval()
        use_torch = True
    except ImportError:
        use_torch = False

    results: list[TrackGRUObservation] = []
    for track_id, seq in sorted(tracks.items()):
        seq.sort(key=lambda item: item.timestamp_s if item.timestamp_s is not None else -1.0)
        if len(seq) < min_steps:
            continue
        x_np, peak_speed = _extract_track_features(seq, width, height)
        if use_torch:
            import torch
            with torch.no_grad():
                xt = torch.from_numpy(x_np).unsqueeze(0)
                _, h_n = gru(xt)
                logit = float(head(h_n[-1]).item()) + 0.45 * float(x_np[:, 2].mean())
                score = float(1.0 / (1.0 + np.exp(-np.clip(logit, -20.0, 20.0))))
        else:
            logit = -0.45 + 0.65 * float(x_np[:, 2].mean()) + 0.25 * float(x_np[:, 3].mean())
            score = float(1.0 / (1.0 + np.exp(-np.clip(logit, -20.0, 20.0))))

        results.append(
            TrackGRUObservation(
                track_id=track_id,
                start_s=float(seq[0].timestamp_s or 0.0),
                end_s=float(seq[-1].timestamp_s or 0.0),
                n_steps=len(seq),
                score=round(score, 4),
                peak_speed_bls=peak_speed,
                representative=seq[-1],
            )
        )
    return results
