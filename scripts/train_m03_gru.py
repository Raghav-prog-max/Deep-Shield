"""Train, fine-tune, and evaluate the Module 03 PyTorch Temporal GRU on track kinematic sequences.

Saves trained weights, feature normalization statistics, calibrated threshold, and validation/test
metrics to models/m03_temporal_gru.pt and reports/m03_gru_training/report.json.
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
from deep_shield.engines.video_events.gru import TemporalKinematicGRU  # noqa: E402


def generate_kinematic_dataset(n_samples: int, seed: int) -> tuple[list[np.ndarray], np.ndarray]:
    """Generate realistic 6-D track kinematic sequences:
    [x_norm, y_norm, speed_bls_scaled, d_angle_scaled, dt, aspect_ratio]
    Label 0: normal corridor walk / brief pause (speed 0.4..1.8 BL/s, smooth heading)
    Label 1: rapid sprinting (>2.5 BL/s), erratic pacing/zig-zag, or sudden burst acceleration
    """
    rng = np.random.default_rng(seed)
    sequences: list[np.ndarray] = []
    labels: list[int] = []

    for i in range(n_samples):
        y = i % 2
        T = int(rng.integers(8, 28))
        dt = 0.5
        x, y_pos = float(rng.uniform(0.1, 0.9)), float(rng.uniform(0.15, 0.85))
        heading = float(rng.uniform(-np.pi, np.pi))
        aspect = float(rng.uniform(0.32, 0.48))
        seq = []

        mode = int(rng.integers(0, 3))
        for t in range(T):
            if y == 0:
                speed_bls = float(np.clip(rng.normal(1.05, 0.32), 0.1, 2.1))
                d_angle = float(np.clip(abs(rng.normal(0.0, 0.18)), 0.0, 0.7))
            else:
                if mode == 0:  # sustained sprint
                    speed_bls = float(np.clip(rng.normal(3.1, 0.45), 2.3, 4.8))
                    d_angle = float(np.clip(abs(rng.normal(0.0, 0.25)), 0.0, 1.0))
                elif mode == 1:  # erratic zig-zag / rapid direction reversals
                    speed_bls = float(np.clip(rng.normal(2.2, 0.5), 1.4, 3.8))
                    d_angle = float(np.clip(abs(rng.normal(1.6, 0.45)), 0.8, np.pi))
                else:  # calm walk then sudden burst sprint
                    if t < T // 2:
                        speed_bls = float(np.clip(rng.normal(0.9, 0.25), 0.2, 1.5))
                        d_angle = float(np.clip(abs(rng.normal(0.0, 0.15)), 0.0, 0.5))
                    else:
                        speed_bls = float(np.clip(rng.normal(3.35, 0.4), 2.5, 4.9))
                        d_angle = float(np.clip(abs(rng.normal(0.9, 0.4)), 0.2, 2.5))

            heading += d_angle * (1 if rng.random() > 0.5 else -1)
            x = float(np.clip(x + 0.03 * speed_bls * np.cos(heading), 0.02, 0.98))
            y_pos = float(np.clip(y_pos + 0.03 * speed_bls * np.sin(heading), 0.02, 0.98))
            seq.append([
                x,
                y_pos,
                min(speed_bls / 4.0, 2.0),
                min(d_angle / np.pi, 1.0),
                dt,
                aspect + float(rng.normal(0, 0.02)),
            ])
        sequences.append(np.asarray(seq, dtype=np.float32))
        labels.append(y)

    return sequences, np.asarray(labels, dtype=np.int64)


def pad_batch(seqs: list[np.ndarray], mean: np.ndarray, std: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = torch.tensor([len(s) for s in seqs], dtype=torch.long)
    max_len = int(lengths.max().item())
    padded = np.zeros((len(seqs), max_len, 6), dtype=np.float32)
    for i, s in enumerate(seqs):
        normed = (s - mean) / std
        padded[i, : len(s)] = normed
    return torch.from_numpy(padded), lengths


def main() -> int:
    torch.manual_seed(2026)
    train_seqs, y_train = generate_kinematic_dataset(960, seed=101)
    val_seqs, y_val = generate_kinematic_dataset(240, seed=202)
    test_seqs, y_test = generate_kinematic_dataset(300, seed=303)

    all_train_steps = np.concatenate(train_seqs, axis=0)
    mean = all_train_steps.mean(axis=0).astype(np.float32)
    std = (all_train_steps.std(axis=0) + 1e-5).astype(np.float32)

    model = TemporalKinematicGRU(input_size=6, hidden_size=24, num_layers=2, dropout=0.15)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3, weight_decay=1e-3)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=35)
    loss_fn = nn.BCEWithLogitsLoss()

    best_val_auc = -1.0
    best_state = None

    x_val, len_val = pad_batch(val_seqs, mean, std)
    for epoch in range(1, 36):
        model.train()
        perm = np.random.default_rng(epoch).permutation(len(train_seqs))
        for start in range(0, len(train_seqs), 64):
            idx = perm[start : start + 64]
            xb, lb = pad_batch([train_seqs[i] for i in idx], mean, std)
            yb = torch.tensor(y_train[idx], dtype=torch.float32)
            logits = model(xb, lb)
            loss = loss_fn(logits, yb)
            opt.zero_grad()
            loss.backward()
            opt.step()
        sched.step()

        model.eval()
        with torch.no_grad():
            val_probs = torch.sigmoid(model(x_val, len_val)).numpy()
        val_auc = roc_auc(y_val.tolist(), val_probs.tolist())
        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()

    with torch.no_grad():
        val_probs = torch.sigmoid(model(x_val, len_val)).numpy()
        x_test, len_test = pad_batch(test_seqs, mean, std)
        test_probs = torch.sigmoid(model(x_test, len_test)).numpy()

    # Calibrate threshold on validation split (<=5% FPR on normal trajectories)
    threshold = round(float(np.quantile(val_probs[y_val == 0], 0.95)), 4)
    threshold = float(np.clip(threshold, 0.35, 0.80))

    test_auc = round(roc_auc(y_test.tolist(), test_probs.tolist()), 4)
    test_metrics = binary_metrics(y_test.tolist(), test_probs.tolist(), threshold)

    out_path = ROOT / "models" / "m03_temporal_gru.pt"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_id": "temporal-gru-kinematic-v2",
        "architecture": {"input_size": 6, "hidden_size": 24, "num_layers": 2, "dropout": 0.15},
        "n_parameters": n_params,
        "state_dict": best_state,
        "mean": mean.tolist(),
        "std": std.tolist(),
        "threshold": threshold,
        "val_auc": round(best_val_auc, 4),
        "test_auc": test_auc,
        "test_metrics_at_threshold": test_metrics,
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
        "n_train": len(train_seqs),
        "n_val": len(val_seqs),
        "n_test": len(test_seqs),
        "calibrated_threshold": threshold,
        "val_roc_auc": round(best_val_auc, 4),
        "test_roc_auc": test_auc,
        "test_metrics": test_metrics,
        "trained_at": payload["trained_at"],
    }
    (rep_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(f"Trained M03 Temporal GRU ({n_params} params) -> {out_path} (sha {sha})")
    print(f"Validation ROC-AUC: {best_val_auc:.4f} | Held-out Test ROC-AUC: {test_auc:.4f} | Threshold: {threshold}")
    print(f"Test Acc: {test_metrics['accuracy']:.4f} | Precision: {test_metrics['precision']:.4f} | Recall: {test_metrics['recall']:.4f} | F1: {test_metrics['f1']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
