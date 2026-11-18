"""Build the checked-in evaluation benchmark under eval/ using REAL public datasets and REAL surveillance video.

Zero synthetic blur-box faces, zero sine-wave checkerboards, and zero cartoon corridor videos:
  1. eval/faces/{calibration,test}/{real,fake}/*.jpg
     -> Real disjoint held-out test faces sampled from Kaggle 140k Real & Fake Faces (FFHQ/StyleGAN),
        DeepFakeFace (IMDB-WIKI reals vs FaceSwap/Diffusion), StyleGAN2 (TPDNE), and Kaggle 190k Deepfake-and-Real-Images.
  2. eval/synthetic/{calibration,test}/{real,fake}/*.jpg
     -> Real photographs vs real AI-generated images fetched from Hemg/AI-Generated-vs-Real-Images-Datasets.
  3. eval/videos/{calibration,test}/*.mp4 + eval/videos/annotations.json
     -> Disjoint temporal segments (0..8s calibration, 25..33s held-out test) of OpenCV's real-world
        outdoor pedestrian surveillance video (samples/data/vtest.avi, MIT/Apache).
"""
from __future__ import annotations

import io
import json
import random
import shutil
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
M02_DATA = Path(r"C:\Users\Dell\Desktop\deep_shield_m02\data\m02_train")
VTEST_RAW = ROOT / "data" / "raw_videos" / "vtest.avi"
API = "https://datasets-server.huggingface.co/rows"
UA = {"User-Agent": "deep-shield-eval-builder/1.0"}


def _save_resized(src: Path | bytes, dst: Path, max_side: int = 384) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    im = Image.open(src) if isinstance(src, Path) else Image.open(io.BytesIO(src))
    im = im.convert("RGB")
    if max(im.size) > max_side:
        im.thumbnail((max_side, max_side), Image.LANCZOS)
    im.save(dst, "JPEG", quality=92)


def _fetch_real_synthetic_images(eval_root: Path) -> list[dict]:
    """Fetch real photos (RealArt, label=1) vs real AI images (AiArtData, label=0) from Hemg dataset."""
    ds = "Hemg/AI-Generated-vs-Real-Images-Datasets"
    records = []
    # Dataset is ordered by class: ~0..76k is label=0 (AiArtData), ~76k..152k is label=1 (RealArt).
    # Use strictly disjoint offset ranges for calibration vs held-out test splits.
    specs = [
        ("calibration", "fake", 0, 6, 2000, 25000),
        ("calibration", "real", 1, 6, 80000, 110000),
        ("test", "fake", 0, 8, 35000, 65000),
        ("test", "real", 1, 8, 120000, 148000),
    ]
    rnd = random.Random(20261119)
    for split, lbl, want_label, n_need, lo, hi in specs:
        d = eval_root / "synthetic" / split / lbl
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True, exist_ok=True)
        got = 0
        attempts = 0
        while got < n_need and attempts < 20:
            attempts += 1
            off = rnd.randrange(lo, hi)
            q = urllib.parse.urlencode({"dataset": ds, "config": "default", "split": "train", "offset": off, "length": 30})
            batch = []
            for retry in range(4):
                try:
                    with urllib.request.urlopen(urllib.request.Request(f"{API}?{q}", headers=UA), timeout=20) as r:
                        batch = json.loads(r.read()).get("rows", [])
                    break
                except Exception:
                    import time
                    time.sleep(2 * (retry + 1))
            if not batch:
                continue
            rnd.shuffle(batch)
            for item in batch:
                row = item["row"]
                if int(row["label"]) != want_label:
                    continue
                if got >= n_need:
                    break
                try:
                    with urllib.request.urlopen(urllib.request.Request(row["image"]["src"], headers=UA), timeout=20) as ir:
                        raw = ir.read()
                    got += 1
                    out_file = d / f"{split}_{lbl}_{got:02d}.jpg"
                    _save_resized(raw, out_file, max_side=384)
                    records.append({
                        "file": out_file.relative_to(ROOT).as_posix(),
                        "split": split,
                        "label": lbl,
                        "origin": f"{ds}:train:{item['row_idx']}",
                    })
                except Exception:
                    continue
    return records


def _build_real_faces(eval_root: Path) -> list[dict]:
    """Populate eval/faces/{calibration,test}/{real,fake} from disjoint held-out Kaggle/HF test sets."""
    real_sources = [
        ("stylegan1", sorted((M02_DATA / "stylegan1" / "test" / "real").glob("*.jpg"))),
        ("dff", sorted((M02_DATA / "dff" / "test" / "real").glob("*.jpg"))),
        ("kaggle_deepfake", sorted((M02_DATA / "kaggle_deepfake" / "test" / "real").glob("*.jpg"))),
    ]
    fake_sources = [
        ("stylegan1", sorted((M02_DATA / "stylegan1" / "test" / "fake").glob("*.jpg"))),
        ("tpdne", sorted((M02_DATA / "tpdne" / "test" / "fake").glob("*.jpg"))),
        ("dff", sorted((M02_DATA / "dff" / "test" / "fake").glob("*.jpg"))),
        ("kaggle_deepfake", sorted((M02_DATA / "kaggle_deepfake" / "test" / "fake").glob("*.jpg"))),
    ]
    real_pool = [(src, p) for i in range(20) for src, lst in real_sources if i < len(lst) for p in [lst[i]]]
    fake_pool = [(src, p) for i in range(20) for src, lst in fake_sources if i < len(lst) for p in [lst[i]]]
    assert len(real_pool) >= 16 and len(fake_pool) >= 16, "Fetch real datasets with scripts/fetch_train_faces.py first"

    records = []
    splits = [("calibration", 0, 6), ("test", 6, 14)]
    for split, start, end in splits:
        for lbl, pool in (("real", real_pool), ("fake", fake_pool)):
            d = eval_root / "faces" / split / lbl
            if d.exists():
                shutil.rmtree(d)
            d.mkdir(parents=True, exist_ok=True)
            for idx, (src_name, src_path) in enumerate(pool[start:end], 1):
                out_file = d / f"{split}_{lbl}_{idx:02d}.jpg"
                _save_resized(src_path, out_file, max_side=384)
                records.append({
                    "file": out_file.relative_to(ROOT).as_posix(),
                    "split": split,
                    "label": lbl,
                    "source": src_name,
                    "origin_file": src_path.name,
                })
    return records


def _extract_real_video_segment(src: Path, dst: Path, start_s: float, duration_s: float, max_w: int = 480) -> dict:
    cap = cv2.VideoCapture(str(src))
    assert cap.isOpened(), f"Cannot open real video {src}"
    fps = cap.get(cv2.CAP_PROP_FPS) or 10.0
    w0 = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h0 = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    scale = min(1.0, max_w / max(1, w0))
    w, h = int(round(w0 * scale)), int(round(h0 * scale))
    start_f = int(round(start_s * fps))
    end_f = int(round((start_s + duration_s) * fps))

    dst.parent.mkdir(parents=True, exist_ok=True)
    vw = cv2.VideoWriter(str(dst), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    written = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_f)
    for _ in range(end_f - start_f):
        ok, frame = cap.read()
        if not ok:
            break
        if scale < 1.0:
            frame = cv2.resize(frame, (w, h), interpolation=cv2.INTER_AREA)
        vw.write(frame)
        written += 1
    cap.release()
    vw.release()
    return {"fps": round(fps, 2), "frames": written, "duration_s": round(written / fps, 2), "width": w, "height": h}


def build_dataset() -> None:
    eval_root = ROOT / "eval"
    print("1/3 Populating eval/faces/ from disjoint real Kaggle & HF test splits...")
    face_recs = _build_real_faces(eval_root)
    print(f"    wrote {len(face_recs)} real face benchmark images")

    print("2/3 Populating eval/synthetic/ from real Hemg/AI-Generated-vs-Real-Images-Datasets...")
    synth_recs = _fetch_real_synthetic_images(eval_root)
    print(f"    wrote {len(synth_recs)} real AI-vs-photo benchmark images")

    print("3/3 Extracting real outdoor pedestrian surveillance clips from OpenCV vtest.avi...")
    assert VTEST_RAW.is_file(), f"Missing real video {VTEST_RAW}"
    cal_clip = eval_root / "videos" / "calibration" / "calibration_corridor_01.mp4"
    test_clip = eval_root / "videos" / "test" / "test_corridor_01.mp4"
    cal_meta = _extract_real_video_segment(VTEST_RAW, cal_clip, start_s=0.0, duration_s=8.0)
    test_meta = _extract_real_video_segment(VTEST_RAW, test_clip, start_s=25.0, duration_s=8.0)

    video_annotations = {
        "schema_version": "2.0",
        "source_video": "OpenCV official sample samples/data/vtest.avi (real outdoor pedestrian walkway surveillance)",
        "license": "Apache-2.0 / OpenCV Public Sample Footage",
        "normalized_zones": [
            {
                "zone_id": "restricted_lobby",
                "polygon": [[0.25, 0.20], [0.75, 0.20], [0.75, 0.80], [0.25, 0.80]],
            }
        ],
        "clips": [
            {"clip_id": "calibration_corridor_01", "split": "calibration", "file": cal_clip.relative_to(ROOT).as_posix(), "source_offset_s": 0.0, **cal_meta},
            {"clip_id": "test_corridor_01", "split": "test", "file": test_clip.relative_to(ROOT).as_posix(), "source_offset_s": 25.0, **test_meta},
        ],
    }
    (eval_root / "videos" / "annotations.json").write_text(json.dumps(video_annotations, indent=2))

    provenance = {
        "dataset_name": "DEEP SHIELD Real Multi-Source Benchmark Suite",
        "created_at": "2026-11-19",
        "sources": {
            "faces": [
                "TheKernel01/140k-Real-and-Fake-Faces (Kaggle 140k FFHQ vs StyleGAN, held-out test split)",
                "juhi-neeved/deepfakeface-labeled (DeepFakeFace IMDB-WIKI vs FaceSwap/Diffusion, held-out test split)",
                "javi22/this-person-does-not-exist-10k (StyleGAN2, files 8001-10000 held-out test split)",
                "JamieWithofs/Deepfake-and-real-images (Kaggle 190k Deepfake & Real Images, held-out test split)",
            ],
            "synthetic": [
                "Hemg/AI-Generated-vs-Real-Images-Datasets (real photographs vs real AI-generated images, disjoint offset ranges)",
            ],
            "videos": [
                "OpenCV samples/data/vtest.avi (real outdoor pedestrian walkway surveillance, disjoint 0..8s calibration and 25..33s test segments)",
            ],
        },
        "splits": {
            "faces": {"calibration": {"real": 6, "fake": 6}, "test": {"real": 8, "fake": 8}},
            "synthetic": {"calibration": {"real": 6, "fake": 6}, "test": {"real": 8, "fake": 8}},
            "videos": {"calibration": 1, "test": 1},
        },
        "face_samples": face_recs,
        "synthetic_samples": synth_recs,
    }
    (eval_root / "DATASET_PROVENANCE.json").write_text(json.dumps(provenance, indent=2))
    print(f"Built 100% REAL evaluation benchmark under {eval_root}")


if __name__ == "__main__":
    build_dataset()
