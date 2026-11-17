"""Fetch a small labelled real-vs-fake FACE sample into eval/faces/{real,fake} (about 5 MB).

    python scripts/fetch_face_sample.py           # 25 per class from the test split
    python scripts/fetch_face_sample.py --n 50

Source: TheKernel01/140k-Real-and-Fake-Faces (real = FFHQ photos, fake = StyleGAN faces), fetched
row-by-row through the Hugging Face Dataset Viewer API so the 4 GB parquet files are NOT downloaded.
FFHQ is non-commercial research data: internal evaluation only, do not redistribute the images.
"""
from __future__ import annotations

import argparse
import json
import random
import time
import urllib.parse
import urllib.request
from pathlib import Path

DATASET = "TheKernel01/140k-Real-and-Fake-Faces"
API = "https://datasets-server.huggingface.co/rows"
LABELS = {0: "real", 1: "fake"}
UA = {"User-Agent": "deep-shield-m02-eval/0.1"}


def get(url: str, tries: int = 5) -> bytes:
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=60) as r:
                return r.read()
        except Exception as exc:  # noqa: BLE001 — retry on 429 / transient network errors
            if k == tries - 1:
                raise
            wait = 3 * (k + 1)
            print(f"  retry in {wait}s ({exc})")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def rows(split: str, offset: int, length: int = 100) -> dict:
    q = urllib.parse.urlencode({"dataset": DATASET, "config": "default", "split": split,
                                "offset": offset, "length": length})
    return json.loads(get(f"{API}?{q}"))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=25, help="faces per class")
    ap.add_argument("--split", default="test")
    ap.add_argument("--out", default="eval/faces")
    ap.add_argument("--seed", type=int, default=11)
    a = ap.parse_args()

    rnd = random.Random(a.seed)
    out = {lbl: Path(a.out) / name for lbl, name in LABELS.items()}
    for d in out.values():
        d.mkdir(parents=True, exist_ok=True)
    got = {0: 0, 1: 0}

    first = rows(a.split, 0, 1)
    total = int(first.get("num_rows_total", 20000))
    attempts = 0
    while min(got.values()) < a.n and attempts < 40:
        attempts += 1
        off = rnd.randrange(0, max(1, total - 100))
        batch = rows(a.split, off, 100).get("rows", [])
        rnd.shuffle(batch)
        for item in batch:
            r = item["row"]
            lbl = int(r["label"])
            if got[lbl] >= a.n:
                continue
            src = r["image"]["src"]
            ext = ".png" if ".png" in src.split("?")[0].lower() else ".jpg"
            got[lbl] += 1
            (out[lbl] / f"{LABELS[lbl]}_{got[lbl]:03d}{ext}").write_bytes(get(src))
            if min(got.values()) >= a.n:
                break
        print(f"  real {got[0]}/{a.n}   fake {got[1]}/{a.n}")
    print(f"done -> {Path(a.out).resolve()}")


if __name__ == "__main__":
    main()
