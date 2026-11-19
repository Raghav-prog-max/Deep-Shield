"""Evaluate one Module 02 task on a labelled, held-out image folder.

Folder layout (names are case-insensitive):
    DATA/real/...   negatives  (authentic faces / real photographs)
    DATA/fake/...   positives  (manipulated faces / AI-generated images)

    python scripts/evaluate_m02.py --data eval/faces --task face_manipulation --assume-face-crop
    python scripts/evaluate_m02.py --data eval/synthetic --task synthetic_image

Writes eval_out/<task>_<timestamp>/{metrics.json, scores.csv}. Report the dataset source,
licence, split and class balance alongside these numbers (proposal section 10).
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deep_shield.engines.deepfake import ALL_TASKS, DeepfakeImageIntegrityEngine, EngineConfig  # noqa: E402
from deep_shield.engines.deepfake.config import SCORED_TASKS, SPECS  # noqa: E402
from deep_shield.engines.deepfake.media import IMAGE_EXT  # noqa: E402
from deep_shield.engines.deepfake.metrics import binary_metrics, roc_auc, sweep, three_band  # noqa: E402

LABEL_DIRS = {"real": 0, "authentic": 0, "fake": 1, "manipulated": 1, "synthetic": 1}


def collect(root: Path):
    items = []
    for d in root.iterdir():
        if d.is_dir() and d.name.lower() in LABEL_DIRS:
            y = LABEL_DIRS[d.name.lower()]
            items += [(p, y) for p in d.rglob("*") if p.suffix.lower() in IMAGE_EXT]
    return items


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--task", choices=SCORED_TASKS, required=True)
    ap.add_argument("--assume-face-crop", action="store_true", help="images are pre-cropped faces")
    ap.add_argument("--limit", type=int, default=0, help="max images per class (random, seeded)")
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--out", default="eval_out")
    a = ap.parse_args()

    items = collect(Path(a.data))
    if not items:
        print("No images found under real/ and fake/ subfolders.")
        return 1
    if a.limit:
        rnd = random.Random(a.seed)
        by = {0: [i for i in items if i[1] == 0], 1: [i for i in items if i[1] == 1]}
        items = [x for k in by for x in rnd.sample(by[k], min(a.limit, len(by[k])))]

    cfg = EngineConfig()
    cfg.evidence_dir = Path(a.out) / "_evidence"
    cfg.explain_faces = False      # heatmaps cost ~25 extra passes per face and do not change the score
    cfg.evidence_plots = False     # spectrum charts are not needed for metrics
    eng = DeepfakeImageIntegrityEngine(cfg)
    rows, y, s, skipped = [], [], [], 0
    thresholds = {}
    t0 = time.perf_counter()
    for n, (p, label) in enumerate(items, 1):
        r = eng.analyze(p, tasks=[a.task], assume_face_crop=a.assume_face_crop)
        fs = [f for f in r.findings if f.task == a.task]
        if r.status == "failed" or not fs:
            skipped += 1
            rows.append({"file": str(p), "label": label, "score": "", "note": r.error or "no face / no finding"})
            continue
        finding = max(fs, key=lambda f: f.score)
        score = finding.score
        thresholds = finding.thresholds
        y.append(label); s.append(score)
        rows.append({"file": str(p), "label": label, "score": round(score, 5), "note": ""})
        if n % 50 == 0:
            print(f"  {n}/{len(items)}")
    secs = time.perf_counter() - t0

    spec = SPECS.get(a.task)
    out = Path(a.out) / f"{a.task}_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    metrics = {
        "task": a.task, "model_id": spec.model_id if spec else "classical", "backend": cfg.backend, "data": str(a.data),
        "n_images": len(items), "n_scored": len(s), "n_skipped": skipped,
        "assume_face_crop": a.assume_face_crop, "seconds": round(secs, 1),
        "ms_per_image": round(1000 * secs / max(1, len(items)), 1),
        "roc_auc": roc_auc(y, s) if s else None,
        "at_0.5": binary_metrics(y, s, 0.5) if s else None,
        "three_band": three_band(y, s, thresholds["low"], thresholds["high"]) if (s and spec) else None,
        "sweep": sweep(y, s) if s else None,
    }
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2))
    with open(out / "scores.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["file", "label", "score", "note"])
        w.writeheader(); w.writerows(rows)

    m = metrics["at_0.5"] or {}
    print(f"\n{a.task} · {spec.model_id if spec else 'classical spectral indicator'}")
    print(f"scored {len(s)}/{len(items)} (skipped {skipped})  ROC-AUC {metrics['roc_auc']}")
    if m:
        print(f"@0.5  acc {m['accuracy']:.3f}  P {m['precision']:.3f}  R {m['recall']:.3f}  F1 {m['f1']:.3f}  "
              f"(tp {m['tp']} fp {m['fp']} tn {m['tn']} fn {m['fn']})")
        tb = metrics["three_band"]
        if tb:
            print(f"three-band [{tb['low']}, {tb['high']}]: coverage {tb['coverage']:.2%}, inconclusive {tb['inconclusive']}")
    print(f"-> {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
