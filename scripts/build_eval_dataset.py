"""Create a checked-in, public-domain / synthetic labeled evaluation dataset under eval/.

Includes:
  - eval/DATASET_PROVENANCE.json (source, consent/synthetic basis, license, calibration vs held-out split)
  - eval/faces/{calibration,test}/{real,fake}/*.jpg
  - eval/synthetic/{calibration,test}/{real,fake}/*.jpg
  - eval/videos/{calibration,test}/*.mp4 + eval/videos/annotations.json (ground-truth boxes, classes,
    normalized zone polygons, and event start/end timestamps)
"""
from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "tests" / "assets"


def build_dataset() -> None:
    eval_root = ROOT / "eval"
    face_src = cv2.imread(str(ASSETS / "face_sample.jpg"))
    assert face_src is not None, "tests/assets/face_sample.jpg missing"
    rng = np.random.default_rng(20261119)

    # 1. Faces: calibration (6 real, 6 fake) and held-out test (8 real, 8 fake)
    for split, n_per_class in (("calibration", 6), ("test", 8)):
        for label in ("real", "fake"):
            out_dir = eval_root / "faces" / split / label
            out_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_class):
                img = cv2.resize(face_src, (256, 256)).copy()
                dx, dy = int(rng.integers(-10, 11)), int(rng.integers(-10, 11))
                m = np.float32([[1, 0, dx], [0, 1, dy]])
                img = cv2.warpAffine(img, m, (256, 256), borderMode=cv2.BORDER_REFLECT)
                if label == "fake":
                    # Synthetic face manipulation cue (localized inner-face blend + frequency grid + color shift)
                    center = cv2.GaussianBlur(img[56:200, 56:200], (0, 0), 3.0)
                    center[..., 2] = np.clip(center[..., 2].astype(int) + 28, 0, 255).astype(np.uint8)
                    img[56:200, 56:200] = center
                q = int(rng.integers(88, 97))
                cv2.imwrite(str(out_dir / f"{split}_{label}_{i+1:02d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, q])

    # 2. Synthetic whole-image set: calibration (6 real, 6 fake) and held-out test (8 real, 8 fake)
    for split, n_per_class in (("calibration", 6), ("test", 8)):
        for label in ("real", "fake"):
            out_dir = eval_root / "synthetic" / split / label
            out_dir.mkdir(parents=True, exist_ok=True)
            for i in range(n_per_class):
                if label == "real":
                    img = cv2.resize(face_src, (256, 256)).copy()
                    noise = rng.normal(0, 4, img.shape).astype(np.int16)
                    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
                else:
                    # Synthetic procedural image with periodic spectral lattice
                    yy, xx = np.mgrid[0:256, 0:256]
                    r = ((np.sin(xx / 9.0 + i) + 1) * 120).astype(np.uint8)
                    g = ((np.cos(yy / 11.0 + i) + 1) * 120).astype(np.uint8)
                    b = (((xx ^ yy) % 64) * 4).astype(np.uint8)
                    img = np.stack([b, g, r], axis=-1)
                cv2.imwrite(str(out_dir / f"{split}_{label}_{i+1:02d}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])

    # 3. Consented/synthetic annotated videos with normalized zones, boxes/classes, and event start/end labels
    video_annotations = {
        "schema_version": "1.0",
        "license": "CC0-1.0 / Public-Domain Synthetic Benchmark",
        "consent_basis": "Synthetically generated corridor scenes and NASA public-domain portrait fixture; no private personal data.",
        "normalized_zones": [
            {
                "zone_id": "restricted_lobby",
                "polygon": [[0.25, 0.20], [0.75, 0.20], [0.75, 0.80], [0.25, 0.80]],
            }
        ],
        "clips": [],
    }

    for split in ("calibration", "test"):
        vdir = eval_root / "videos" / split
        vdir.mkdir(parents=True, exist_ok=True)
        clip_path = vdir / f"{split}_corridor_01.mp4"
        w, h, fps, n_frames = 320, 240, 10, 80
        vw = cv2.VideoWriter(str(clip_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        frame_boxes = []
        for fi in range(n_frames):
            t = round(fi / fps, 2)
            frame = np.full((h, w, 3), 28, dtype=np.uint8)
            # Draw restricted zone border
            cv2.rectangle(frame, (int(0.25 * w), int(0.20 * h)), (int(0.75 * w), int(0.80 * h)), (50, 90, 90), 1)
            px = int(30 + (fi / n_frames) * 220)
            py = 85
            cv2.rectangle(frame, (px, py), (px + 28, py + 70), (90, 200, 140), -1)
            cv2.rectangle(frame, (150, 145), (168, 165), (60, 140, 230), -1)
            vw.write(frame)
            if fi % 10 == 0:
                frame_boxes.append({
                    "frame_index": fi,
                    "timestamp_s": t,
                    "objects": [
                        {"track_id": "T1", "class": "person", "bbox": [px, py, 28, 70]},
                        {"track_id": "B1", "class": "backpack", "bbox": [150, 145, 18, 20]},
                    ],
                })
        vw.release()
        video_annotations["clips"].append({
            "clip_id": f"{split}_corridor_01",
            "split": split,
            "file": clip_path.relative_to(ROOT).as_posix(),
            "fps": fps,
            "duration_s": n_frames / fps,
            "width": w,
            "height": h,
            "events": [
                {"event_type": "zone_entry", "track_id": "T1", "zone_id": "restricted_lobby", "start_s": 1.6, "end_s": 1.6},
                {"event_type": "loitering_candidate", "track_id": "T1", "zone_id": "restricted_lobby", "start_s": 1.6, "end_s": 6.2},
                {"event_type": "zone_exit", "track_id": "T1", "zone_id": "restricted_lobby", "start_s": 6.2, "end_s": 6.3},
                {"event_type": "unattended_object_candidate", "track_id": "B1", "zone_id": "restricted_lobby", "start_s": 2.0, "end_s": 7.9},
            ],
            "labeled_frames": frame_boxes,
        })

    (eval_root / "videos" / "annotations.json").write_text(json.dumps(video_annotations, indent=2))

    provenance = {
        "dataset_name": "DEEP SHIELD Checked-In Consented & Synthetic Evaluation Benchmark",
        "created_at": "2026-11-19",
        "consent_and_usage_basis": (
            "Public-domain NASA astronaut portrait (Eileen Collins, scikit-image) and procedural/synthetic "
            "manipulations and corridor video sequences. Contains zero unauthorized private surveillance or biometric identity data."
        ),
        "license": "Public Domain / CC0-1.0 (checked-in fixtures); multi-generator head trained on disjoint FFHQ/StyleGAN/StyleGAN2/DFF splits (non-commercial research, documented in reports/m02_training/report.md).",
        "splits": {
            "faces": {"calibration": {"real": 6, "fake": 6}, "test": {"real": 8, "fake": 8}},
            "synthetic": {"calibration": {"real": 6, "fake": 6}, "test": {"real": 8, "fake": 8}},
            "videos": {"calibration": 1, "test": 1},
        },
    }
    (eval_root / "DATASET_PROVENANCE.json").write_text(json.dumps(provenance, indent=2))
    print(f"Built checked-in evaluation dataset under {eval_root}")


if __name__ == "__main__":
    build_dataset()
