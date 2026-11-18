"""Train, calibrate, and evaluate the Module 03 PyTorch Temporal GRU on REAL pedestrian tracks.

Pipeline:
  1. Runs the real MS-COCO neural object detector (hustvl/yolos-tiny) + MultiObjectIoUTracker
     across the OpenCV real-world outdoor pedestrian surveillance clip (data/raw_videos/vtest.avi,
     79.5s, 768x576, 10 FPS) and caches extracted tracks to data/real_pedestrian_tracks_vtest.json.
  2. Splits real pedestrian tracks by DISJOINT track IDs into train (60%), validation (20%),
     and held-out test (20%) splits so no trajectory leaks across splits.
  3. Builds normal kinematic sequences directly from real pedestrian walking windows + minor
     camera-angle/scale augmentations, and builds realistic anomalous kinematic sequences
     (sustained rapid running, abrupt burst acceleration, erratic heading reversals) by
     kinematically warping real track windows with realistic bounding-box detector noise.
  4. Trains TemporalKinematicGRU (6-D normalized features, 2-layer GRU + LayerNorm MLP head),
     calibrates the anomaly threshold on the validation tracks (<=5% FPR on normal tracks),
     and evaluates once on the held-out test tracks.
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deep_shield.engines.deepfake.metrics import binary_metrics, roc_auc  # noqa: E402
from deep_shield.engines.threat.config import ThreatConfig  # noqa: E402
from deep_shield.engines.threat.detector import Detection, YoloObjectDetector  # noqa: E402
from deep_shield.engines.video_events.gru import TemporalKinematicGRU, _extract_track_features  # noqa: E402

VTEST_PATH = ROOT / "data" / "raw_videos" / "vtest.avi"
TRACK_CACHE = ROOT / "data" / "real_pedestrian_tracks_vtest.json"


def extract_or_load_real_tracks() -> tuple[int, int, dict[str, list[dict]]]:
    if TRACK_CACHE.is_file():
        payload = json.loads(TRACK_CACHE.read_text())
        if len(payload.get("tracks", {})) >= 10:
            print(f"Loaded {len(payload['tracks'])} real pedestrian tracks from {TRACK_CACHE}")
            return int(payload["width"]), int(payload["height"]), payload["tracks"]

    if not VTEST_PATH.is_file():
        raise FileNotFoundError(
            f"Real pedestrian surveillance clip not found at {VTEST_PATH}. "
            "Download OpenCV samples/data/vtest.avi first."
        )

    print(f"Extracting REAL pedestrian tracks from {VTEST_PATH} using hustvl/yolos-tiny + MultiObjectIoUTracker...")
    det = YoloObjectDetector(ThreatConfig())
    tracks: dict[str, list[dict]] = {}
    width = height = 0
    for fr in det.track_video(VTEST_PATH, sample_fps=2.0, max_duration_s=80.0):
        height, width = fr.image_bgr.shape[:2]
        for d in fr.detections:
            if d.label == "person" and d.track_id is not None:
                tracks.setdefault(d.track_id, []).append({
                    "frame_index": d.frame_index,
                    "timestamp_s": d.timestamp_s,
                    "box": list(d.box),
                    "confidence": d.confidence,
                })

    # Keep tracks with at least 6 observations (>= 3 seconds of real pedestrian motion)
    valid = {tid: obs for tid, obs in tracks.items() if len(obs) >= 6}
    TRACK_CACHE.parent.mkdir(parents=True, exist_ok=True)
    TRACK_CACHE.write_text(json.dumps({
        "source_video": "OpenCV samples/data/vtest.avi (MIT/Apache real outdoor pedestrian surveillance)",
        "detector": det.model_info.model_dump(),
        "width": width,
        "height": height,
        "n_tracks": len(valid),
        "tracks": valid,
    }, indent=2))
    print(f"Extracted and cached {len(valid)} real pedestrian tracks (>=6 steps) -> {TRACK_CACHE}")
    return width, height, valid


def _obs_to_detections(tid: str, obs: list[dict]) -> list[Detection]:
    return [
        Detection(
            label="person",
            confidence=float(o["confidence"]),
            box=tuple(int(v) for v in o["box"]),
            track_id=tid,
            frame_index=int(o["frame_index"]),
            timestamp_s=float(o["timestamp_s"]),
        )
        for o in obs
    ]


def _build_windows_from_tracks(
    track_ids: list[str],
    tracks: dict[str, list[dict]],
    width: int,
    height: int,
    seed: int,
    n_aug_per_window: int = 4,
) -> tuple[list[np.ndarray], np.ndarray]:
    """Create normal (y=0) and kinematically warped anomalous (y=1) windows from real tracks."""
    rng = np.random.default_rng(seed)
    seqs: list[np.ndarray] = []
    labels: list[int] = []

    for tid in track_ids:
        dets = _obs_to_detections(tid, tracks[tid])
        base_feat, _ = _extract_track_features(dets, width, height)
        T = len(base_feat)
        if T < 6:
            continue

        # Sliding windows along each real pedestrian trajectory
        win_len = min(T, 14)
        starts = list(range(0, max(1, T - win_len + 1), 3))
        for st in starts:
            w = base_feat[st : st + win_len].copy()
            if len(w) < 6:
                continue

            # 1. Genuine unperturbed real walking window (y = 0)
            seqs.append(w)
            labels.append(0)

            for _ in range(n_aug_per_window):
                # Normal walking variation (y = 0): slight camera shift, brisk walk up to ~1.9 BL/s, minor turn
                wn = w.copy()
                wn[:, 0] = np.clip(wn[:, 0] + rng.normal(0.0, 0.05), 0.02, 0.98)
                wn[:, 1] = np.clip(wn[:, 1] + rng.normal(0.0, 0.05), 0.02, 0.98)
                speed_scale = float(rng.uniform(0.75, 1.45))
                wn[:, 2] = np.clip(wn[:, 2] * speed_scale + rng.normal(0.0, 0.04, len(wn)), 0.0, 0.55)
                wn[:, 3] = np.clip(wn[:, 3] + abs(rng.normal(0.0, 0.04, len(wn))), 0.0, 0.32)
                wn[:, 5] = np.clip(wn[:, 5] + rng.normal(0.0, 0.03, len(wn)), 0.2, 1.8)
                seqs.append(wn.astype(np.float32))
                labels.append(0)

                # Realistic kinematic anomaly derived from the real track (y = 1)
                wa = w.copy()
                mode = int(rng.integers(0, 3))
                if mode == 0:
                    # Sustained rapid running / sprinting (> 2.2..4.4 BL/s, scaled /4.0 -> 0.55..1.10)
                    sprint_v = float(rng.uniform(0.56, 1.15))
                    wa[:, 2] = np.clip(
                        np.maximum(wa[:, 2] * float(rng.uniform(2.5, 4.2)), sprint_v)
                        + rng.normal(0.0, 0.08, len(wa)),
                        0.38, 1.8,
                    )
                    wa[:, 3] = np.clip(wa[:, 3] + abs(rng.normal(0.0, 0.08, len(wa))), 0.0, 0.85)
                elif mode == 1:
                    # Erratic pacing / abrupt heading reversals while moving moderately fast
                    wa[:, 2] = np.clip(
                        np.maximum(wa[:, 2] * 1.8, float(rng.uniform(0.32, 0.72)))
                        + rng.normal(0.0, 0.06, len(wa)),
                        0.22, 1.4,
                    )
                    rev = np.abs(rng.normal(0.55, 0.20, len(wa)))
                    wa[:, 3] = np.clip(rev, 0.25, 1.0)
                else:
                    # Calm real walk followed by sudden mid-track burst sprint + sharp turn
                    mid = len(wa) // 2
                    wa[mid:, 2] = np.clip(
                        float(rng.uniform(0.62, 1.10)) + rng.normal(0.0, 0.09, len(wa) - mid),
                        0.45, 1.8,
                    )
                    wa[mid:, 3] = np.clip(
                        wa[mid:, 3] + np.abs(rng.normal(0.35, 0.15, len(wa) - mid)),
                        0.08, 1.0,
                    )
                seqs.append(wa.astype(np.float32))
                labels.append(1)

    return seqs, np.asarray(labels, dtype=np.int64)


def pad_batch(seqs: list[np.ndarray], mean: np.ndarray, std: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded = np.zeros((len(seqs), max_len, 6), dtype=np.float32)
    for i, s in enumerate(seqs):
        padded[i, : len(s)] = (s - mean) / std
    return torch.from_numpy(padded), lengths


def main() -> int:
    torch.manual_seed(2026)
    width, height, tracks = extract_or_load_real_tracks()

    # Disjoint track-ID split: 60% train, 20% validation, 20% held-out test
    tids = sorted(tracks.keys(), key=lambda x: int(x) if x.isdigit() else x)
    rng = np.random.default_rng(2026)
    perm_tids = [tids[i] for i in rng.permutation(len(tids))]
    n_tr = max(4, int(round(0.60 * len(perm_tids))))
    n_va = max(2, int(round(0.20 * len(perm_tids))))
    train_tids = perm_tids[:n_tr]
    val_tids = perm_tids[n_tr : n_tr + n_va]
    test_tids = perm_tids[n_tr + n_va :]
    if not test_tids:
        test_tids = val_tids[-2:]
        val_tids = val_tids[:-2] or train_tids[-2:]

    print(f"Disjoint real pedestrian track split: train={len(train_tids)} val={len(val_tids)} test={len(test_tids)}")
    train_seqs, y_train = _build_windows_from_tracks(train_tids, tracks, width, height, seed=101, n_aug_per_window=4)
    val_seqs, y_val = _build_windows_from_tracks(val_tids, tracks, width, height, seed=202, n_aug_per_window=4)
    test_seqs, y_test = _build_windows_from_tracks(test_tids, tracks, width, height, seed=303, n_aug_per_window=4)
    print(f"Windows: train={len(train_seqs)} (pos={int(y_train.sum())}), val={len(val_seqs)} (pos={int(y_val.sum())}), test={len(test_seqs)} (pos={int(y_test.sum())})")

    all_train_steps = np.concatenate(train_seqs, axis=0)
    mean = all_train_steps.mean(axis=0).astype(np.float32)
    std = (all_train_steps.std(axis=0) + 1e-5).astype(np.float32)

    model = TemporalKinematicGRU(input_size=6, hidden_size=24, num_layers=2, dropout=0.2)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=2.5e-3, weight_decay=5e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=40)
    # Class-balanced BCE loss
    pos_weight = torch.tensor([(y_train == 0).sum() / max(1, (y_train == 1).sum())], dtype=torch.float32)
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auc = -1.0
    best_state = None
    x_val, len_val = pad_batch(val_seqs, mean, std)

    for epoch in range(1, 41):
        model.train()
        perm = np.random.default_rng(epoch).permutation(len(train_seqs))
        for start in range(0, len(train_seqs), 64):
            idx = perm[start : start + 64]
            xb, lb = pad_batch([train_seqs[i] for i in idx], mean, std)
            yb = torch.tensor(y_train[idx], dtype=torch.float32)
            loss = loss_fn(model(xb, lb), yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(x_val, len_val)).numpy()
        val_auc = roc_auc(y_val.tolist(), val_probs.tolist())
        if val_auc >= best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        val_probs = torch.sigmoid(model(x_val, len_val)).numpy()
        x_test, len_test = pad_batch(test_seqs, mean, std)
        test_probs = torch.sigmoid(model(x_test, len_test)).numpy()

    # Calibrate threshold on validation split (<=5% FPR on real normal tracks)
    threshold = round(float(np.quantile(val_probs[y_val == 0], 0.95)), 4)
    threshold = float(np.clip(threshold, 0.35, 0.85))

    test_auc = round(roc_auc(y_test.tolist(), test_probs.tolist()), 4)
    test_metrics = binary_metrics(y_test.tolist(), test_probs.tolist(), threshold)

    # Also evaluate genuine unperturbed real walking tracks in the held-out test split (false-alarm check)
    raw_normal_test = [
        _extract_track_features(_obs_to_detections(tid, tracks[tid]), width, height)[0]
        for tid in test_tids
        if len(tracks[tid]) >= 6
    ]
    raw_scores = []
    if raw_normal_test:
        x_raw, len_raw = pad_batch(raw_normal_test, mean, std)
        with torch.no_grad():
            raw_scores = torch.sigmoid(model(x_raw, len_raw)).numpy().tolist()
    raw_fp = sum(1 for s in raw_scores if s >= threshold)

    out_path = ROOT / "models" / "m03_temporal_gru.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": "temporal-gru-kinematic-v3-realtracks",
        "architecture": {"input_size": 6, "hidden_size": 24, "num_layers": 2, "dropout": 0.2},
        "n_parameters": n_params,
        "state_dict": best_state,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "threshold": threshold,
        "val_auc": round(best_val_auc, 4),
        "test_auc": test_auc,
        "test_metrics_at_threshold": test_metrics,
        "real_video_source": "OpenCV samples/data/vtest.avi (hustvl/yolos-tiny + MultiObjectIoUTracker)",
        "real_normal_test_tracks": len(raw_scores),
        "real_normal_test_false_alarms": raw_fp,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
    }
    torch.save(payload, out_path)
    sha = hashlib.sha256(out_path.read_bytes()).hexdigest()[:12]

    rep_dir = ROOT / "reports" / "m03_gru_training"
    rep_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "model_id": payload["model_id"],
        "checkpoint": "models/m03_temporal_gru.pt",
        "sha": sha,
        "n_parameters": n_params,
        "real_video_source": payload["real_video_source"],
        "disjoint_tracks": {"train": len(train_tids), "val": len(val_tids), "test": len(test_tids)},
        "n_train_windows": len(train_seqs),
        "n_val_windows": len(val_seqs),
        "n_test_windows": len(test_seqs),
        "calibrated_threshold": threshold,
        "val_roc_auc": round(best_val_auc, 4),
        "test_roc_auc": test_auc,
        "test_metrics": test_metrics,
        "real_unperturbed_normal_test_tracks": {
            "n_tracks": len(raw_scores),
            "false_alarms": raw_fp,
            "mean_score": round(float(np.mean(raw_scores)), 4) if raw_scores else 0.0,
        },
        "trained_at": payload["trained_at"],
    }
    (rep_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"Trained M03 Temporal GRU ({n_params} params) on REAL pedestrian tracks -> {out_path} (sha {sha})")
    print(f"Validation ROC-AUC: {best_val_auc:.4f} | Held-out Test ROC-AUC: {test_auc:.4f} | Threshold: {threshold}")
    print(f"Test Acc: {test_metrics['accuracy']:.4f} | Precision: {test_metrics['precision']:.4f} | Recall: {test_metrics['recall']:.4f} | F1: {test_metrics['f1']:.4f}")
    print(f"Unperturbed real normal held-out tracks: {raw_fp}/{len(raw_scores)} false alarms (mean score {report['real_unperturbed_normal_test_tracks']['mean_score']})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
