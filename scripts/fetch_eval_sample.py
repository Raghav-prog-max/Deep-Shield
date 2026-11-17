"""Fetch a small labelled real-vs-AI sample from a public Hugging Face dataset into eval/synthetic/{real,fake}.

    python scripts/fetch_eval_sample.py            # 25 per class (~15 MB)
    python scripts/fetch_eval_sample.py --n 50

Source: Hemg/AI-Generated-vs-Real-Images-Datasets (AiArtData = AI-generated, RealArt = real).
Web-collected images with no stated licence: use for internal sanity checks only; do not redistribute,
and do not report these numbers as a formal benchmark. For the report, add your own consented photos
and a documented dataset (e.g. FaceForensics++ under its access agreement).
"""
from __future__ import annotations

import argparse
import random
import shutil
import tempfile
from pathlib import Path

REPO = "Hemg/AI-Generated-vs-Real-Images-Datasets"
CLASSES = {"fake": "AiArtData/", "real": "RealArt/"}
EXT = (".jpg", ".jpeg", ".png", ".webp")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25, help="images per class")
    ap.add_argument("--out", default="eval/synthetic")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()

    from huggingface_hub import HfApi, hf_hub_download

    files = HfApi().list_repo_files(REPO, repo_type="dataset")
    rnd = random.Random(a.seed)
    with tempfile.TemporaryDirectory() as tmp:
        for label, prefix in CLASSES.items():
            pool = [f for f in files if f.startswith(prefix) and f.lower().endswith(EXT)]
            pick = rnd.sample(pool, min(a.n, len(pool)))
            dest = Path(a.out) / label
            dest.mkdir(parents=True, exist_ok=True)
            print(f"{label}: {len(pick)} of {len(pool)} images -> {dest}")
            for i, f in enumerate(pick, 1):
                p = hf_hub_download(REPO, f, repo_type="dataset", local_dir=tmp)
                shutil.copy(p, dest / f"{label}_{i:03d}{Path(f).suffix.lower()}")
    print("done")


if __name__ == "__main__":
    main()
