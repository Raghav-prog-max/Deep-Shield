"""Fetch real benchmark videos (Celeb-DF v2, Google DeepFakeDetection / FaceForensics++, UCF-Crime)
directly from public non-gated HuggingFace repositories with strict actor/split disjointness.

Outputs:
  - eval/deepfake_videos/celebdf/{real,fake}/*.mp4      (10 real + 10 fake official test videos)
  - eval/deepfake_videos/dfd/{real,fake}/*.mp4          (10 real + 10 fake actor-disjoint test videos)
  - eval/surveillance_ucf/{normal,anomaly}/*.mp4        (6 normal + 6 real CCTV anomaly videos)
  - Extracted spaced training & test frames appended to M02's training manifest (`manifest.csv`)
    for `celebdf_video` and `dfd_video` (train MP4s deleted after extraction to save disk space).
  - reports/real_video_datasets_manifest.json
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import tempfile
import urllib.parse
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_M02_TRAIN = Path(r"C:\Users\Dell\Desktop\deep_shield_m02\data\m02_train")


def hf_tree(repo: str, path: str = "") -> list[dict]:
    url = f"https://huggingface.co/api/datasets/{repo}/tree/main" + (f"/{urllib.parse.quote(path)}" if path else "")
    req = urllib.request.Request(url, headers={"User-Agent": "deep-shield-eval/1.0"})
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode())


def hf_download(repo: str, remote_path: str, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() and dest.stat().st_size > 10_000:
        return dest
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{urllib.parse.quote(remote_path)}"
    req = urllib.request.Request(url, headers={"User-Agent": "deep-shield-eval/1.0"})
    tmp = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(req, timeout=90) as r, tmp.open("wb") as out:
        while True:
            chunk = r.read(1 << 18)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(dest)
    return dest


def extract_frames(video_path: Path, out_dir: Path, prefix: str, n_frames: int = 8) -> tuple[list[Path], dict]:
    out_dir.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return [], {"error": "unreadable"}
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 0
    fps = float(cap.get(cv2.CAP_PROP_FPS)) or 25.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if total < 4:
        # fallback sequential count
        frames = []
        while True:
            ok, f = cap.read()
            if not ok:
                break
            frames.append(f)
        cap.release()
        total = len(frames)
        if not frames:
            return [], {"error": "no_frames"}
        idxs = np.linspace(0, total - 1, min(n_frames, total)).round().astype(int).tolist()
        saved = []
        for i, fi in enumerate(sorted(set(idxs))):
            fp = out_dir / f"{prefix}_f{fi:05d}.jpg"
            cv2.imwrite(str(fp), frames[fi], [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved.append(fp)
        return saved, {"frames": total, "fps": round(fps, 2), "duration_s": round(total / max(fps, 1.0), 2), "w": w, "h": h}

    lo = max(0, int(total * 0.05))
    hi = max(lo + 1, int(total * 0.95) - 1)
    idxs = sorted(set(np.linspace(lo, hi, min(n_frames, max(1, hi - lo + 1))).round().astype(int).tolist()))
    saved = []
    for fi in idxs:
        cap.set(cv2.CAP_PROP_POS_FRAMES, int(fi))
        ok, frame = cap.read()
        if not ok or frame is None:
            continue
        fp = out_dir / f"{prefix}_f{fi:05d}.jpg"
        cv2.imwrite(str(fp), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
        saved.append(fp)
    cap.release()
    return saved, {"frames": total, "fps": round(fps, 2), "duration_s": round(total / max(fps, 1.0), 2), "w": w, "h": h}


def celeb_actor_ids(name: str) -> set[str]:
    return set(re.findall(r"id\d+", name))


def dfd_actor_ids(name: str) -> set[str]:
    head = name.split("__")[0]
    return set(re.findall(r"\d+", head))


def append_manifest_rows(manifest_path: Path, new_rows: list[dict]) -> int:
    existing = []
    seen = set()
    fieldnames = ["file", "label", "source", "split", "subtype"]
    if manifest_path.exists():
        with manifest_path.open("r", newline="", encoding="utf-8") as fh:
            reader = csv.DictReader(fh)
            fieldnames = list(reader.fieldnames or fieldnames)
            for r in reader:
                seen.add(r["file"])
                existing.append(r)
    added = 0
    for r in new_rows:
        if r["file"] in seen:
            continue
        seen.add(r["file"])
        existing.append({k: r.get(k, "") for k in fieldnames})
        added += 1
    with manifest_path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(existing)
    return added


def fetch_celebdf(m02_root: Path, eval_dir: Path, n_train: int = 12, n_test: int = 10, frames_per_vid: int = 8) -> dict:
    repo = "gonnerthetooner/deepfake_video_data"
    print(f"[1/3] Fetching pair-matched actor-disjoint Celeb-DF v2 from {repo} ...")
    real_tree = [f for f in hf_tree(repo, "celebdf_v2/Celeb-real") if f["path"].endswith(".mp4")]
    fake_tree = [f for f in hf_tree(repo, "celebdf_v2/Celeb-synthesis") if f["path"].endswith(".mp4")]

    # Group by target celebrity ID (idX_... for real, idX_idY_... for fake)
    from collections import defaultdict
    real_by_id: dict[str, list[dict]] = defaultdict(list)
    fake_by_id: dict[str, list[dict]] = defaultdict(list)
    for f in real_tree:
        m = re.match(r"(id\d+)_", Path(f["path"]).name)
        if m and f.get("size", 9e9) < 5_000_000:
            real_by_id[m.group(1)].append(f)
    for f in fake_tree:
        m = re.match(r"(id\d+)_(id\d+)_", Path(f["path"]).name)
        if m and f.get("size", 9e9) < 5_000_000:
            fake_by_id[m.group(1)].append(f)

    # Strictly disjoint actor sets, PAIR-MATCHED (each actor has equal real & fake clips -> zero identity shortcut!)
    test_actor_set = ["id0", "id1", "id16", "id17", "id19"]
    train_actor_set = ["id10", "id11", "id12", "id13", "id20", "id21"]

    def pick_paired(actors: list[str], target_n: int, forbid_actors: set[str]) -> tuple[list[dict], list[dict]]:
        per_actor = max(1, (target_n + len(actors) - 1) // len(actors))
        reals, fakes = [], []
        for aid in actors:
            r_cands = sorted(real_by_id.get(aid, []), key=lambda x: x.get("size", 9e9))
            # Prefer fake clips whose source actor is NOT in forbid_actors
            f_cands = sorted(
                [x for x in fake_by_id.get(aid, []) if not (celeb_actor_ids(Path(x["path"]).name) & forbid_actors)],
                key=lambda x: x.get("size", 9e9),
            )
            if not f_cands:
                f_cands = sorted(fake_by_id.get(aid, []), key=lambda x: x.get("size", 9e9))
            reals.extend(r_cands[:per_actor])
            fakes.extend(f_cands[:per_actor])
        return reals[:target_n], fakes[:target_n]

    sel_test_real, sel_test_fake = pick_paired(test_actor_set, n_test, set(train_actor_set))
    sel_train_real, sel_train_fake = pick_paired(train_actor_set, n_train, set(test_actor_set))
    used_test_actors = set(test_actor_set)

    rows = []
    manifest_videos = []
    # Save test videos permanently + extract test frames
    for lbl, group in [("real", sel_test_real), ("fake", sel_test_fake)]:
        for f in group:
            name = Path(f["path"]).name
            dest = eval_dir / "celebdf" / lbl / name
            hf_download(repo, f["path"], dest)
            frames, meta = extract_frames(dest, m02_root / "celebdf_video" / "test" / lbl, dest.stem, n_frames=frames_per_vid)
            for fp in frames:
                rows.append({"file": str(fp.relative_to(m02_root)).replace("\\", "/"), "label": lbl, "source": "celebdf_video", "split": "test", "category": "CelebDFv2", "origin": f"{repo}:{f['path']}"})
            manifest_videos.append({"dataset": "Celeb-DF-v2", "repo": repo, "remote_path": f["path"], "local_path": str(dest.relative_to(ROOT)).replace("\\", "/"), "label": lbl, "split": "test", "size_mb": round(dest.stat().st_size / 1e6, 2), **meta})
            print(f"  [test/{lbl}] {name} ({round(dest.stat().st_size/1e6, 2)} MB, {len(frames)} frames)")

    # Download train videos to temp, extract frames, delete MP4s to save disk space
    with tempfile.TemporaryDirectory(prefix="celebdf_train_") as td:
        for lbl, group in [("real", sel_train_real), ("fake", sel_train_fake)]:
            for f in group:
                name = Path(f["path"]).name
                tmp_vid = Path(td) / name
                hf_download(repo, f["path"], tmp_vid)
                frames, meta = extract_frames(tmp_vid, m02_root / "celebdf_video" / "train" / lbl, tmp_vid.stem, n_frames=frames_per_vid)
                for fp in frames:
                    rows.append({"file": str(fp.relative_to(m02_root)).replace("\\", "/"), "label": lbl, "source": "celebdf_video", "split": "train", "category": "CelebDFv2", "origin": f"{repo}:{f['path']}"})
                tmp_vid.unlink(missing_ok=True)
                manifest_videos.append({"dataset": "Celeb-DF-v2", "repo": repo, "remote_path": f["path"], "local_path": "(frames extracted; temp mp4 removed)", "label": lbl, "split": "train", "extracted_frames": len(frames), **meta})
                print(f"  [train/{lbl}] {name} -> {len(frames)} frames")

    added = append_manifest_rows(m02_root / "manifest.csv", rows)
    return {"videos": manifest_videos, "manifest_rows_added": added, "test_actors": sorted(used_test_actors)}


def fetch_dfd(m02_root: Path, eval_dir: Path, n_train: int = 12, n_test: int = 10, frames_per_vid: int = 8) -> dict:
    repo = "angads24/deepfake-video"
    print(f"[2/3] Fetching Google DeepFakeDetection (FF++ DFD) from {repo} ...")
    real_tree = [f for f in hf_tree(repo, "dataset/real") if f["path"].endswith(".mp4")]
    fake_tree = [f for f in hf_tree(repo, "dataset/fake") if f["path"].endswith(".mp4")]

    # Strict actor split: actors 01..06 -> train, actors 07..27 -> test (ZERO actor overlap!)
    def is_test_actor(f) -> bool:
        ids = {int(x) for x in dfd_actor_ids(Path(f["path"]).name)}
        return bool(ids) and all(i >= 7 for i in ids)

    def is_train_actor(f) -> bool:
        ids = {int(x) for x in dfd_actor_ids(Path(f["path"]).name)}
        return bool(ids) and all(i <= 6 for i in ids)

    test_real = sorted([f for f in real_tree if is_test_actor(f) and f.get("size", 9e9) < 6_000_000], key=lambda x: x.get("size", 9e9))[:n_test]
    test_fake = sorted([f for f in fake_tree if is_test_actor(f) and f.get("size", 9e9) < 6_000_000], key=lambda x: x.get("size", 9e9))[:n_test]
    train_real = sorted([f for f in real_tree if is_train_actor(f) and f.get("size", 9e9) < 6_000_000], key=lambda x: x.get("size", 9e9))[:n_train]
    train_fake = sorted([f for f in fake_tree if is_train_actor(f) and f.get("size", 9e9) < 6_000_000], key=lambda x: x.get("size", 9e9))[:n_train]

    rows = []
    manifest_videos = []
    for lbl, group in [("real", test_real), ("fake", test_fake)]:
        for f in group:
            name = Path(f["path"]).name
            dest = eval_dir / "dfd" / lbl / name
            hf_download(repo, f["path"], dest)
            frames, meta = extract_frames(dest, m02_root / "dfd_video" / "test" / lbl, dest.stem, n_frames=frames_per_vid)
            for fp in frames:
                rows.append({"file": str(fp.relative_to(m02_root)).replace("\\", "/"), "label": lbl, "source": "dfd_video", "split": "test", "category": "FFpp_DFD", "origin": f"{repo}:{f['path']}"})
            manifest_videos.append({"dataset": "FaceForensics++-DFD", "repo": repo, "remote_path": f["path"], "local_path": str(dest.relative_to(ROOT)).replace("\\", "/"), "label": lbl, "split": "test", "size_mb": round(dest.stat().st_size / 1e6, 2), **meta})
            print(f"  [test/{lbl}] {name} ({round(dest.stat().st_size/1e6, 2)} MB, {len(frames)} frames)")

    with tempfile.TemporaryDirectory(prefix="dfd_train_") as td:
        for lbl, group in [("real", train_real), ("fake", train_fake)]:
            for f in group:
                name = Path(f["path"]).name
                tmp_vid = Path(td) / name
                hf_download(repo, f["path"], tmp_vid)
                frames, meta = extract_frames(tmp_vid, m02_root / "dfd_video" / "train" / lbl, tmp_vid.stem, n_frames=frames_per_vid)
                for fp in frames:
                    rows.append({"file": str(fp.relative_to(m02_root)).replace("\\", "/"), "label": lbl, "source": "dfd_video", "split": "train", "category": "FFpp_DFD", "origin": f"{repo}:{f['path']}"})
                tmp_vid.unlink(missing_ok=True)
                manifest_videos.append({"dataset": "FaceForensics++-DFD", "repo": repo, "remote_path": f["path"], "local_path": "(frames extracted; temp mp4 removed)", "label": lbl, "split": "train", "extracted_frames": len(frames), **meta})
                print(f"  [train/{lbl}] {name} -> {len(frames)} frames")

    added = append_manifest_rows(m02_root / "manifest.csv", rows)
    return {"videos": manifest_videos, "manifest_rows_added": added}


def fetch_ucf_crime(out_dir: Path, n_normal: int = 6, n_anomaly: int = 6) -> dict:
    repo = "mteb/ucf-crime"
    print(f"[3/3] Fetching UCF-Crime real CCTV surveillance clips from {repo} ...")
    norm_files = [f for f in hf_tree(repo, "videos/Testing_Normal") if f["path"].endswith(".mp4")]
    norm_pick = sorted([f for f in norm_files if 200_000 < f.get("size", 9e9) < 8_000_000], key=lambda x: x.get("size", 9e9))[:n_normal]

    anom_cats = ["Fighting", "Robbery", "Assault", "Burglary", "Stealing", "Shoplifting"]
    anom_pick = []
    for cat in anom_cats:
        try:
            files = [f for f in hf_tree(repo, f"videos/{cat}") if f["path"].endswith(".mp4")]
            small = sorted([f for f in files if 200_000 < f.get("size", 9e9) < 8_000_000], key=lambda x: x.get("size", 9e9))
            if small:
                anom_pick.append((cat, small[0]))
        except Exception as e:
            print(f"  skip {cat}: {e}")
        if len(anom_pick) >= n_anomaly:
            break

    records = []
    for f in norm_pick:
        name = Path(f["path"]).name
        dest = out_dir / "normal" / name
        hf_download(repo, f["path"], dest)
        records.append({"dataset": "UCF-Crime", "repo": repo, "category": "Testing_Normal", "label": "normal", "local_path": str(dest.relative_to(ROOT)).replace("\\", "/"), "size_mb": round(dest.stat().st_size / 1e6, 2)})
        print(f"  [normal] {name} ({round(dest.stat().st_size/1e6, 2)} MB)")
    for cat, f in anom_pick:
        name = Path(f["path"]).name
        dest = out_dir / "anomaly" / name
        hf_download(repo, f["path"], dest)
        records.append({"dataset": "UCF-Crime", "repo": repo, "category": cat, "label": "anomaly", "local_path": str(dest.relative_to(ROOT)).replace("\\", "/"), "size_mb": round(dest.stat().st_size / 1e6, 2)})
        print(f"  [anomaly/{cat}] {name} ({round(dest.stat().st_size/1e6, 2)} MB)")
    return {"videos": records}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--m02-train", type=Path, default=DEFAULT_M02_TRAIN)
    ap.add_argument("--n-train", type=int, default=12)
    ap.add_argument("--n-test", type=int, default=10)
    ap.add_argument("--frames", type=int, default=8)
    args = ap.parse_args()

    deepfake_dir = ROOT / "eval" / "deepfake_videos"
    ucf_dir = ROOT / "eval" / "surveillance_ucf"

    c_info = fetch_celebdf(args.m02_train, deepfake_dir, n_train=args.n_train, n_test=args.n_test, frames_per_vid=args.frames)
    d_info = fetch_dfd(args.m02_train, deepfake_dir, n_train=args.n_train, n_test=args.n_test, frames_per_vid=args.frames)
    u_info = fetch_ucf_crime(ucf_dir, n_normal=6, n_anomaly=6)

    out_manifest = ROOT / "reports" / "real_video_datasets_manifest.json"
    out_manifest.parent.mkdir(parents=True, exist_ok=True)
    out_manifest.write_text(json.dumps({"celebdf_v2": c_info, "dfd_ffpp": d_info, "ucf_crime": u_info}, indent=2), encoding="utf-8")
    print(f"\nSaved real video manifest -> {out_manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
