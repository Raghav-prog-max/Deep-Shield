"""PyTorch Temporal GRU sequence model for Module 03 track kinematics.

Processes per-track normalized motion sequences:
  [x_norm, y_norm, speed_body_lengths_s_scaled, direction_change_rad_scaled, step_dt, box_aspect]
Loads fine-tuned weights and feature normalization parameters from ``models/m03_temporal_gru.pt``
(trained via ``scripts/train_m03_gru.py``) to produce an advisory temporal sequence anomaly score
in [0, 1] kept strictly separate from rule-trigger events.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn

from deep_shield.engines.threat.config import ROOT
from deep_shield.engines.threat.detector import Detection

DEFAULT_GRU_PATH = ROOT / "models" / "m03_temporal_gru.pt"


class TemporalKinematicGRU(nn.Module):
    def __init__(self, input_size: int = 6, hidden_size: int = 24, num_layers: int = 2, dropout: float = 0.15):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor, lengths: torch.Tensor | None = None) -> torch.Tensor:
        out, _ = self.gru(x)
        if lengths is not None:
            idx = (lengths - 1).clamp(min=0)
            last = out[torch.arange(out.size(0), device=out.device), idx]
            mean_pool = torch.stack([out[i, : lengths[i]].mean(dim=0) for i in range(out.size(0))])
            pooled = 0.5 * (last + mean_pool)
        else:
            pooled = 0.5 * (out[:, -1, :] + out.mean(dim=1))
        return self.head(pooled).squeeze(-1)


@dataclass(frozen=True)
class TrackGRUObservation:
    track_id: str
    start_s: float
    end_s: float
    n_steps: int
    score: float
    threshold: float
    peak_speed_bls: float
    representative: Detection
    model_id: str
    revision: str


@lru_cache(maxsize=2)
def load_trained_gru(checkpoint_path: str = str(DEFAULT_GRU_PATH)) -> dict:
    path = Path(checkpoint_path)
    if path.is_file():
        payload = torch.load(path, map_location="cpu", weights_only=True)
        arch = payload.get("architecture", {"input_size": 6, "hidden_size": 24, "num_layers": 2, "dropout": 0.15})
        model = TemporalKinematicGRU(**arch)
        model.load_state_dict(payload["state_dict"])
        model.eval()
        sha = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        return {
            "model": model,
            "mean": np.asarray(payload["mean"], dtype=np.float32),
            "std": np.asarray(payload["std"], dtype=np.float32),
            "threshold": float(payload.get("threshold", 0.55)),
            "model_id": payload.get("model_id", "temporal-gru-kinematic-v2"),
            "revision": f"trained:{sha}",
            "n_parameters": int(payload.get("n_parameters", sum(p.numel() for p in model.parameters()))),
            "test_auc": payload.get("test_auc"),
            "loaded_from": str(path),
        }

    # Deterministic fallback if checkpoint has not been built yet
    torch.manual_seed(2026)
    model = TemporalKinematicGRU()
    model.eval()
    return {
        "model": model,
        "mean": np.zeros(6, dtype=np.float32),
        "std": np.ones(6, dtype=np.float32),
        "threshold": 0.65,
        "model_id": "temporal-gru-kinematic-v1",
        "revision": "untrained-init",
        "n_parameters": sum(p.numel() for p in model.parameters()),
        "test_auc": None,
        "loaded_from": "memory-init",
    }


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
            if d_angle > np.pi:
                d_angle = 2 * np.pi - d_angle
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

    bundle = load_trained_gru()
    model: TemporalKinematicGRU = bundle["model"]
    mean: np.ndarray = bundle["mean"]
    std: np.ndarray = bundle["std"]
    threshold: float = bundle["threshold"]

    results: list[TrackGRUObservation] = []
    for track_id, seq in sorted(tracks.items()):
        seq.sort(key=lambda item: item.timestamp_s if item.timestamp_s is not None else -1.0)
        if len(seq) < min_steps:
            continue
        x_np, peak_speed = _extract_track_features(seq, width, height)
        x_norm = (x_np - mean) / std
        with torch.no_grad():
            xt = torch.from_numpy(x_norm).unsqueeze(0)
            logit = float(model(xt).item())
            score = float(1.0 / (1.0 + np.exp(-np.clip(logit, -20.0, 20.0))))

        results.append(
            TrackGRUObservation(
                track_id=track_id,
                start_s=float(seq[0].timestamp_s or 0.0),
                end_s=float(seq[-1].timestamp_s or 0.0),
                n_steps=len(seq),
                score=round(score, 4),
                threshold=threshold,
                peak_speed_bls=peak_speed,
                representative=seq[-1],
                model_id=bundle["model_id"],
                revision=bundle["revision"],
            )
        )
    return results
