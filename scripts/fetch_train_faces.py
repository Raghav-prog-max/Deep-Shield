"""Fetch a labelled real-vs-fake FACE set for training and testing the Module 02 face head.

    python scripts/fetch_train_faces.py                 # defaults below, ~3,500 images, ~250 MB on disk
    python scripts/fetch_train_faces.py --scale 0.5     # half size (faster, fewer images)

Three sources, three generator families. Train and test images come from DISJOINT splits:

| source    | real images        | fake images                                   | train pool from        | test from            |
|-----------|--------------------|-----------------------------------------------|------------------------|----------------------|
| stylegan1 | FFHQ photos        | StyleGAN (1M Fake Faces)                      | 140k `train` split     | 140k `test` split    |
| tpdne     | (uses stylegan1's) | StyleGAN2, thispersondoesnotexist.com, 1024px | files 1-6000           | files 8001-10000     |
| dff       | IMDB-WIKI photos   | face swap / diffusion (DeepFakeFace-derived)  | `train` split          | `test` split         |

Images are fetched one by one (Dataset Viewer API / file URLs), so the multi-GB parquet files are NOT
downloaded. TPDNE images are resized from 1024 to 512 px to save disk (faces are classified at 224 px).

Licences: FFHQ is non-commercial research data; check each dataset card before any redistribution.
Use for internal training/evaluation only; do not commit the images to git (data/ is git-ignored).
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import random
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

API = "https://datasets-server.huggingface.co/rows"
UA = {"User-Agent": "deep-shield-m02-train/0.1"}
DS_140K = "TheKernel01/140k-Real-and-Fake-Faces"           # label: 0 real, 1 fake
DS_DFF = "juhi-neeved/deepfakeface-labeled"                 # binary_label: 0 real, 1 fake; class_name
DS_TPDNE = "javi22/this-person-does-not-exist-10k"          # files "face (N).jpg", N = 1..10000
TPDNE_URL = "https://huggingface.co/datasets/" + DS_TPDNE + "/resolve/main/{name}"
FIELDS = ["file", "source", "split", "label", "category", "origin"]


def get(url: str, tries: int = 6) -> bytes:
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=90) as r:
                return r.read()
        except Exception as exc:  # noqa: BLE001 - retry on 429 / transient network errors
            if k == tries - 1:
                raise
            wait = 3 * (k + 1)
            print(f"    retry in {wait}s ({str(exc)[:80]})")
            time.sleep(wait)
    raise RuntimeError("unreachable")


def rows(dataset: str, split: str, offset: int, length: int = 100) -> dict:
    q = urllib.parse.urlencode({"dataset": dataset, "config": "default", "split": split,
                                "offset": offset, "length": length})
    return json.loads(get(f"{API}?{q}"))


def save_image(data: bytes, path: Path, max_side: int = 0) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if max_side:
        im = Image.open(io.BytesIO(data)).convert("RGB")
        if max(im.size) > max_side:
            im.thumbnail((max_side, max_side), Image.LANCZOS)
        im.save(path, quality=95)
    else:
        path.write_bytes(data)


def existing(root: Path, source: str, split: str, label: str) -> int:
    d = root / source / split / label
    return len(list(d.glob("*.jpg"))) if d.exists() else 0


def fetch_viewer(root, dataset, source, split, ds_split, need, label_key, cat_key, rnd, pool, log):
    """Sample rows at random offsets until `need` = {"real": n, "fake": n} is met."""
    got = {lbl: existing(root, source, split, lbl) for lbl in need}
    if all(got[l] >= need[l] for l in need):
        print(f"  {source}/{split}: already complete {got}")
        return
    total = int(rows(dataset, ds_split, 0, 1).get("num_rows_total", 10000))
    seen, attempts = set(), 0
    while any(got[l] < need[l] for l in need) and attempts < 400:
        attempts += 1
        off = rnd.randrange(0, max(1, total - 100))
        batch = rows(dataset, ds_split, off, 100).get("rows", [])
        rnd.shuffle(batch)
        jobs = []
        for item in batch:
            r, idx = item["row"], item["row_idx"]
            if idx in seen:
                continue
            seen.add(idx)
            lbl = "fake" if int(r[label_key]) == 1 else "real"
            if lbl not in need or got[lbl] >= need[lbl]:
                continue
            got[lbl] += 1
            cat = str(r.get(cat_key, lbl)) if cat_key else ("StyleGAN" if lbl == "fake" else "FFHQ")
            path = root / source / split / lbl / f"{source}_{split}_{lbl}_{idx:06d}.jpg"
            jobs.append((r["image"]["src"], path, lbl, cat, f"{dataset}:{ds_split}:{idx}"))
        results = list(pool.map(lambda j: (j, _try_save(j[0], j[1])), jobs))
        for (src, path, lbl, cat, origin), ok in results:
            if ok:
                log.writerow({"file": path.relative_to(root).as_posix(), "source": source, "split": split,
                              "label": lbl, "category": cat, "origin": origin})
            else:
                got[lbl] -= 1
        print(f"  {source}/{split}: real {got.get('real', 0)}/{need.get('real', 0)}  "
              f"fake {got.get('fake', 0)}/{need.get('fake', 0)}")


def _try_save(url: str, path: Path, max_side: int = 0) -> bool:
    try:
        save_image(get(url), path, max_side)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"    skip {path.name}: {str(exc)[:80]}")
        return False


def fetch_tpdne(root, split, n, lo, hi, rnd, pool, log):
    have = existing(root, "tpdne", split, "fake")
    if have >= n:
        print(f"  tpdne/{split}: already complete ({have})")
        return
    picks = rnd.sample(range(lo, hi + 1), n)[have:]
    jobs = []
    for i in picks:
        name = f"face ({i}).jpg"
        url = TPDNE_URL.format(name=urllib.parse.quote(name))
        jobs.append((url, root / "tpdne" / split / "fake" / f"tpdne_{split}_fake_{i:05d}.jpg", i))
    done = 0
    for (url, path, i), ok in zip(jobs, pool.map(lambda j: _try_save(j[0], j[1], 512), jobs)):
        if ok:
            done += 1
            log.writerow({"file": path.relative_to(root).as_posix(), "source": "tpdne", "split": split,
                          "label": "fake", "category": "StyleGAN2 (thispersondoesnotexist)",
                          "origin": f"{DS_TPDNE}:face ({i}).jpg"})
        if done and done % 50 == 0:
            print(f"  tpdne/{split}: {have + done}/{n}")
    print(f"  tpdne/{split}: {have + done}/{n}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/m02_train")
    ap.add_argument("--scale", type=float, default=1.0, help="multiply all counts (0.5 = half)")
    ap.add_argument("--train", type=int, default=500, help="train-pool images per class per source")
    ap.add_argument("--test", type=int, default=200, help="test images per class per source")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--sources", default="stylegan1,tpdne,dff")
    a = ap.parse_args()

    ntr, nte = max(10, int(a.train * a.scale)), max(10, int(a.test * a.scale))
    root = Path(a.out)
    root.mkdir(parents=True, exist_ok=True)
    mf = root / "manifest.csv"
    new = not mf.exists()
    rnd = random.Random(a.seed)
    srcs = set(a.sources.split(","))
    t0 = time.time()
    with open(mf, "a", newline="") as fh, ThreadPoolExecutor(a.workers) as pool:
        log = csv.DictWriter(fh, fieldnames=FIELDS)
        if new:
            log.writeheader()
        if "stylegan1" in srcs:
            print("[stylegan1] 140k Real and Fake Faces (FFHQ vs StyleGAN)")
            fetch_viewer(root, DS_140K, "stylegan1", "train", "train", {"real": ntr, "fake": ntr},
                         "label", None, rnd, pool, log); fh.flush()
            fetch_viewer(root, DS_140K, "stylegan1", "test", "test", {"real": nte, "fake": nte},
                         "label", None, rnd, pool, log); fh.flush()
        if "tpdne" in srcs:
            print("[tpdne] thispersondoesnotexist (StyleGAN2) - fakes only, paired with FFHQ reals")
            fetch_tpdne(root, "train", ntr, 1, 6000, rnd, pool, log); fh.flush()
            fetch_tpdne(root, "test", nte, 8001, 10000, rnd, pool, log); fh.flush()
        if "dff" in srcs:
            print("[dff] DeepFakeFace-derived set (IMDB-WIKI real vs swap/diffusion fakes)")
            fetch_viewer(root, DS_DFF, "dff", "train", "train", {"real": ntr, "fake": ntr},
                         "binary_label", "class_name", rnd, pool, log); fh.flush()
            fetch_viewer(root, DS_DFF, "dff", "test", "test", {"real": nte, "fake": nte},
                         "binary_label", "class_name", rnd, pool, log); fh.flush()
    print(f"done in {time.time() - t0:.0f}s -> {root.resolve()}  (manifest: {mf.name})")


if __name__ == "__main__":
    main()
