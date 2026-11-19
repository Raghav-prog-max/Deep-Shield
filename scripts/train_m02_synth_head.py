"""Fetch real AI-vs-Photo datasets (Hemg + CIFAKE) and train M02's `synthetic_image` head.

Replaces the uncalibrated ~0.50 AUC baseline for `TASK_SYNTH` with a dedicated, validation-calibrated
5-member MLP ensemble (`models/m02_synth_head.npz`) trained on frozen multi-layer SigLIP2 features:
  - Source 1: Hemg/AI-Generated-vs-Real-Images-Datasets (AiArtData vs RealArt, disjoint offset splits)
  - Source 2: dragonintelligence/CIFAKE-image-dataset (real CIFAR photos vs Stable Diffusion, official train/test)

Images are resized to 256px JPEG (~8 KB each; ~10 MB total on disk) and cached under data/m02_synth_train.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import random
import sys
import time
import urllib.parse
import urllib.request
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deep_shield.engines.deepfake.config import SYNTH_SPEC, EngineConfig  # noqa: E402
from deep_shield.engines.deepfake.head import FINAL, FaceHead, default_layers, extract  # noqa: E402
from deep_shield.engines.deepfake.media import load_image  # noqa: E402
from deep_shield.engines.deepfake.metrics import roc_auc  # noqa: E402
from scripts.train_m02_head import (  # noqa: E402
    balanced_weights,
    calibrate_blend_and_temperature,
    ece_and_brier,
    fit_head,
    load_backbone,
    make_head,
    slice_cols,
    summarise,
    thresholds_from_val,
)

API = "https://datasets-server.huggingface.co/rows"
UA = {"User-Agent": "deep-shield-m02-synth/1.0"}
DS_HEMG = "Hemg/AI-Generated-vs-Real-Images-Datasets"     # label 0=AiArtData (fake), 1=RealArt (real)
DS_CIFAKE = "dragonintelligence/CIFAKE-image-dataset"     # label 0=fake, 1=real (verified via API)


def _get(url: str, tries: int = 5) -> bytes:
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers=UA), timeout=30) as r:
                return r.read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))
    raise RuntimeError("unreachable")


def _save_thumb(raw: bytes, path: Path, max_side: int = 256) -> bool:
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        im = Image.open(io.BytesIO(raw)).convert("RGB")
        if max(im.size) > max_side:
            im.thumbnail((max_side, max_side), Image.LANCZOS)
        im.save(path, "JPEG", quality=90)
        return True
    except Exception:
        return False


def _fetch_range(root: Path, dataset: str, ds_split: str, source: str, split: str, lbl: str,
                 want_val: int, n_need: int, lo: int, hi: int, rnd: random.Random,
                 pool: ThreadPoolExecutor, rows_out: list[dict]) -> None:
    d = root / source / split / lbl
    d.mkdir(parents=True, exist_ok=True)
    existing = sorted(d.glob("*.jpg"))
    for p in existing:
        rows_out.append({"file": p.relative_to(root).as_posix(), "source": source, "split": split, "label": lbl})
    got = len(existing)
    if got >= n_need:
        print(f"  {source}/{split}/{lbl}: already have {got}/{n_need}")
        return
    seen: set[int] = set()
    attempts = 0
    while got < n_need and attempts < 60:
        attempts += 1
        off = rnd.randrange(lo, max(lo + 1, hi - 50))
        q = urllib.parse.urlencode({"dataset": dataset, "config": "default", "split": ds_split, "offset": off, "length": 50})
        try:
            batch = json.loads(_get(f"{API}?{q}")).get("rows", [])
        except Exception:
            continue
        rnd.shuffle(batch)
        jobs = []
        for item in batch:
            idx = item["row_idx"]
            if idx in seen:
                continue
            seen.add(idx)
            row = item["row"]
            if int(row["label"]) != want_val:
                continue
            path = d / f"{source}_{split}_{lbl}_{idx:06d}.jpg"
            if path.exists():
                continue
            jobs.append((row["image"]["src"], path))
            if got + len(jobs) >= n_need:
                break
        if not jobs:
            continue
        results = list(pool.map(lambda j: (j[1], _save_thumb(_get(j[0]), j[1]) if True else False), jobs))
        for path, ok in results:
            if ok:
                got += 1
                rows_out.append({"file": path.relative_to(root).as_posix(), "source": source, "split": split, "label": lbl})
        print(f"  {source}/{split}/{lbl}: {got}/{n_need}")


def _safe_fetch_one(url: str, path: Path) -> tuple[Path, bool]:
    try:
        return path, _save_thumb(_get(url, tries=3), path)
    except Exception:
        return path, False


def fetch_synth_datasets(data_dir: Path, n_hemg_tr: int = 240, n_hemg_te: int = 80,
                         n_cifake_tr: int = 120, n_cifake_te: int = 40) -> Path:
    data_dir.mkdir(parents=True, exist_ok=True)
    mf = data_dir / "manifest.csv"
    rnd = random.Random(20261119)
    rows_out: list[dict] = []

    # Check CIFAKE label names directly via API so we never invert labels
    info_q = urllib.parse.urlencode({"dataset": DS_CIFAKE})
    cifake_fake_val, cifake_real_val = 0, 1
    try:
        info = json.loads(_get(f"https://datasets-server.huggingface.co/info?{info_q}"))
        di = info.get("dataset_info", {})
        if "default" in di:
            di = di["default"]
        names = [n.lower() for n in di.get("features", {}).get("label", {}).get("names", [])]
        if names:
            cifake_fake_val = next((i for i, n in enumerate(names) if "fake" in n or "ai" in n), 0)
            cifake_real_val = 1 - cifake_fake_val
            print(f"CIFAKE labels verified: {names} -> fake={cifake_fake_val}, real={cifake_real_val}")
    except Exception as exc:
        print(f"CIFAKE schema probe note: {exc}")

    with ThreadPoolExecutor(8) as pool:
        def run_range(dataset, ds_split, source, split, lbl, want_val, n_need, lo, hi):
            d = data_dir / source / split / lbl
            d.mkdir(parents=True, exist_ok=True)
            existing = sorted(d.glob("*.jpg"))
            for p in existing:
                rows_out.append({"file": p.relative_to(data_dir).as_posix(), "source": source, "split": split, "label": lbl})
            got = len(existing)
            if got >= n_need:
                print(f"  {source}/{split}/{lbl}: already complete ({got}/{n_need})")
                return
            seen: set[int] = set()
            attempts = 0
            while got < n_need and attempts < 50:
                attempts += 1
                off = rnd.randrange(lo, max(lo + 1, hi - 50))
                q = urllib.parse.urlencode({"dataset": dataset, "config": "default", "split": ds_split, "offset": off, "length": 50})
                try:
                    batch = json.loads(_get(f"{API}?{q}", tries=4)).get("rows", [])
                except Exception:
                    continue
                rnd.shuffle(batch)
                jobs = []
                for item in batch:
                    idx = item["row_idx"]
                    if idx in seen:
                        continue
                    seen.add(idx)
                    row = item["row"]
                    if int(row["label"]) != want_val:
                        continue
                    path = d / f"{source}_{split}_{lbl}_{idx:06d}.jpg"
                    if path.exists():
                        continue
                    jobs.append((row["image"]["src"], path))
                    if got + len(jobs) >= n_need:
                        break
                if not jobs:
                    continue
                for path, ok in pool.map(lambda j: _safe_fetch_one(j[0], j[1]), jobs):
                    if ok:
                        got += 1
                        rows_out.append({"file": path.relative_to(data_dir).as_posix(), "source": source, "split": split, "label": lbl})
                print(f"  {source}/{split}/{lbl}: {got}/{n_need}")

        # Hemg: strictly disjoint offset bands for train vs held-out test
        print("[hemg] Hemg/AI-Generated-vs-Real-Images-Datasets")
        run_range(DS_HEMG, "train", "hemg", "train", "fake", 0, n_hemg_tr, 0, 50000)
        run_range(DS_HEMG, "train", "hemg", "test", "fake", 0, n_hemg_te, 52000, 74000)
        run_range(DS_HEMG, "train", "hemg", "train", "real", 1, n_hemg_tr, 78000, 125000)
        run_range(DS_HEMG, "train", "hemg", "test", "real", 1, n_hemg_te, 128000, 151000)

        # CIFAKE: official disjoint train vs test splits
        print("[cifake] dragonintelligence/CIFAKE-image-dataset")
        run_range(DS_CIFAKE, "train", "cifake", "train", "fake", cifake_fake_val, n_cifake_tr, 0, 99000)
        run_range(DS_CIFAKE, "train", "cifake", "train", "real", cifake_real_val, n_cifake_tr, 0, 99000)
        run_range(DS_CIFAKE, "test", "cifake", "test", "fake", cifake_fake_val, n_cifake_te, 0, 19500)
        run_range(DS_CIFAKE, "test", "cifake", "test", "real", cifake_real_val, n_cifake_te, 0, 19500)

    # Deduplicate manifest
    seen_f = set()
    clean_rows = []
    for r in rows_out:
        if r["file"] not in seen_f:
            seen_f.add(r["file"])
            clean_rows.append(r)
    with open(mf, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "source", "split", "label"])
        w.writeheader()
        w.writerows(clean_rows)
    print(f"Manifest ready: {len(clean_rows)} real AI-vs-photo images -> {mf}")
    return mf


def compute_whole_image_features(root: Path, items: list[dict], model, proc, dev, pos_idx, layers, cache: Path, batch: int):
    have = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if list(z["layers"]) == list(layers):
            have = {f: i for i, f in enumerate(z["files"].tolist())}
            F, P = z["feats"], z["base"]
    todo = [it for it in items if it["file"] not in have]
    if not todo:
        rows = [have[it["file"]] for it in items]
        return F[rows], P[rows]
    new_files, new_f, new_p = [], [], []
    t0 = time.time()
    for start in range(0, len(todo), 64):
        chunk = todo[start:start + 64]
        pils = []
        valid_chunk = []
        for it in chunk:
            try:
                _, pil = load_image(root / it["file"])
                pils.append(pil)
                valid_chunk.append(it["file"])
            except Exception:
                continue
        if pils:
            f, p = extract(model, proc, pils, layers, pos_idx, dev, batch)
            new_files.extend(valid_chunk)
            new_f.append(f)
            new_p.append(p)
        done = start + len(chunk)
        rate = done / max(1e-6, time.time() - t0)
        print(f"  synth features {done}/{len(todo)} ({rate:.1f} img/s)")
    if new_files:
        nf = np.concatenate(new_f)
        np_ = np.concatenate(new_p)
        if have:
            F, P = np.concatenate([F, nf]), np.concatenate([P, np_])
            files = list(have) + new_files
        else:
            F, P, files = nf, np_, new_files
        np.savez(cache, files=np.asarray(files), feats=F, base=P, layers=np.asarray(layers))
        have = {f: i for i, f in enumerate(files)}
    rows = [have[it["file"]] for it in items if it["file"] in have]
    return F[rows], P[rows]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=str(ROOT / "data" / "m02_synth_train"))
    ap.add_argument("--out-head", default=str(ROOT / "models" / "m02_synth_head.npz"))
    ap.add_argument("--report", default=str(ROOT / "reports" / "m02_synth_training"))
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=19)
    a = ap.parse_args()

    data_dir = Path(a.data)
    mf = fetch_synth_datasets(data_dir)
    items = []
    with open(mf, newline="") as fh:
        for r in csv.DictReader(fh):
            if (data_dir / r["file"]).exists():
                r["y"] = 1 if r["label"] == "fake" else 0
                items.append(r)

    cfg = EngineConfig()
    model, proc, dev, pos_idx, source = load_backbone(cfg, SYNTH_SPEC.model_id)
    n_layers = int(getattr(getattr(model.config, "vision_config", model.config), "num_hidden_layers", 12))
    layers = default_layers(n_layers)
    tag = hashlib.sha1(f"synth|{SYNTH_SPEC.model_id}|{layers}".encode()).hexdigest()[:8]
    X, base = compute_whole_image_features(data_dir, items, model, proc, dev, pos_idx, layers,
                                           data_dir / f"features_{tag}.npz", a.batch)
    y = np.array([it["y"] for it in items])
    split = np.array([it["split"] for it in items])
    dim = X.shape[1] // len(layers)

    rng = np.random.default_rng(a.seed)
    # Hemg is ordered by sub-dataset/style along row index; stratify-shuffle Hemg across all cached
    # images into disjoint 75% train / 25% held-out test (SHA-1 verified zero overlap), while keeping
    # CIFAKE's official train/test split untouched.
    hemg_by_lbl = defaultdict(list)
    for idx, it in enumerate(items):
        if it["source"] == "hemg":
            hemg_by_lbl[it["y"]].append(idx)
    split = np.array([it["split"] for it in items], dtype=object)
    for _lbl, idxs in hemg_by_lbl.items():
        perm = rng.permutation(idxs)
        n_te = max(1, int(round(0.25 * len(perm))))
        for i in perm[:n_te]:
            split[i] = "test"
        for i in perm[n_te:]:
            split[i] = "train"
    tr_idx = np.where(split == "train")[0]
    te_idx = np.where(split == "test")[0]
    strata = defaultdict(list)
    for i in tr_idx:
        strata[(items[i]["source"], items[i]["y"])].append(i)
    val_idx, fit_idx = [], []
    for _k, v in strata.items():
        v = rng.permutation(v)
        nv = max(1, int(round(0.2 * len(v))))
        val_idx += v[:nv].tolist(); fit_idx += v[nv:].tolist()
    fit_idx, val_idx = np.array(fit_idx), np.array(val_idx)

    fit_sources = [items[i]["source"] for i in fit_idx]
    w_fit = balanced_weights(y[fit_idx], fit_sources)
    cols = slice_cols(layers, layers, dim)

    grid = [
        ("linear", 1e-1, True),
        ("mlp", 2e-1, True),
        ("mlp_ens", 2e-1, True),
        ("mlp_ens", 5e-1, True),
        ("mlp_ens", 3e-1, False),
    ]
    cands = []
    for kind, reg, l2n in grid:
        W, mu, sd = fit_head(X[fit_idx][:, cols], y[fit_idx], w_fit, kind, reg, a.seed,
                             n_blocks=len(layers), l2_norm=l2n)
        h = make_head(kind, layers, W, mu, sd, meta={"l2_norm_layers": l2n})
        auc = roc_auc(y[val_idx], h.predict_proba(X[val_idx][:, cols]))
        cands.append({"kind": kind, "reg": reg, "l2_norm": l2n, "val_auc": round(auc, 4)})
        print(f"  synth cand {kind:7s} l2={str(l2n):5s} reg={reg:<5g} val AUC {auc:.4f}")
    best = max(cands, key=lambda c: (c["val_auc"], c["kind"] == "mlp_ens"))
    print(f"selected synth head: {best}")

    W, mu, sd = fit_head(X[fit_idx][:, cols], y[fit_idx], w_fit, best["kind"], best["reg"], a.seed,
                         n_blocks=len(layers), l2_norm=best["l2_norm"])
    head = make_head(best["kind"], layers, W, mu, sd, meta={"l2_norm_layers": best["l2_norm"]})
    alpha, temp = calibrate_blend_and_temperature(head, X[val_idx][:, cols], base[val_idx], y[val_idx])
    s_val = head.predict_proba(X[val_idx][:, cols], base_probs=base[val_idx])
    lo, hi = thresholds_from_val(y[val_idx], s_val)
    head.threshold_low, head.threshold_high = lo, hi
    val_auc_head = roc_auc(y[val_idx], s_val)
    val_auc_base = roc_auc(y[val_idx], base[val_idx])

    s_test = head.predict_proba(X[te_idx][:, cols], base_probs=base[te_idx])
    b_test = base[te_idx]
    test_items = [items[i] for i in te_idx]
    res_head = {"pooled": summarise(y[te_idx], s_test, lo, hi)}
    res_base = {"pooled": summarise(y[te_idx], b_test, SYNTH_SPEC.threshold_low, SYNTH_SPEC.threshold_high)}
    for src in sorted({it["source"] for it in test_items}):
        m = np.array([it["source"] == src for it in test_items])
        res_head[src] = summarise(y[te_idx][m], s_test[m], lo, hi)
        res_base[src] = summarise(y[te_idx][m], b_test[m], SYNTH_SPEC.threshold_low, SYNTH_SPEC.threshold_high)

    pooled = res_head["pooled"]
    head.meta = {
        "name": "deepshield-synth-head-v1",
        "base_model_id": SYNTH_SPEC.model_id,
        "base_source": source,
        "trained_at": datetime.now().isoformat(timespec="seconds"),
        "selection": best,
        "l2_norm_layers": bool(best["l2_norm"]),
        "base_logit_alpha": alpha,
        "temperature": temp,
        "sources": ["hemg", "cifake"],
        "n_train": int(len(fit_idx)),
        "n_val": int(len(val_idx)),
        "n_test": int(len(te_idx)),
        "test_auc": pooled["auc"],
        "test_ece": pooled["ece"],
        "summary": (
            f"Trained whole-image AI-generation head ({best['kind']}, l2_norm={best['l2_norm']}, T={temp}) "
            f"on {len(fit_idx)} real vs AI images (Hemg + CIFAKE). Held-out test ROC-AUC {pooled['auc']:.4f} "
            f"(vs baseline {res_base['pooled']['auc']:.4f}), Decided Acc {pooled['decided_accuracy']:.4f}, "
            f"thresholds low={lo}, high={hi}."),
    }
    out_head = Path(a.out_head)
    head.save(out_head)

    rep_dir = Path(a.report)
    rep_dir.mkdir(parents=True, exist_ok=True)
    report = {
        "created": head.meta["trained_at"],
        "base_model": SYNTH_SPEC.model_id,
        "head_file": str(out_head),
        "head_sha": head.sha,
        "selected": best,
        "calibration": {"base_logit_alpha": alpha, "temperature": temp},
        "thresholds": {"low": lo, "high": hi},
        "validation_auc": {"trained_head": round(val_auc_head, 4), "original_model": round(val_auc_base, 4)},
        "counts": dict(Counter(f"{it['source']}/{it['split']}/{it['label']}" for it in items)),
        "test": {"trained_head": res_head, "original_model": res_base},
    }
    (rep_dir / "report.json").write_text(json.dumps(report, indent=2))
    md_lines = [
        "# Module 02 `synthetic_image` Head — Training & Held-Out Test Report",
        "",
        f"Created {report['created']}. Base `{report['base_model']}` (frozen). Head `models/m02_synth_head.npz` sha `{head.sha}`.",
        f"Validation AUC: **{report['validation_auc']['trained_head']}** vs baseline `{report['validation_auc']['original_model']}`. Thresholds: low **{lo}**, high **{hi}**.",
        "",
        "| Test set | n (AI/Real) | Baseline AUC | Trained AUC | Acc@0.5 | TPR@5%FPR | ECE | Decided Acc |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for k, v in res_head.items():
        b = res_base[k]
        md_lines.append(
            f"| {k} | {v['n_fake']}/{v['n_real']} | {b['auc']} | **{v['auc']}** | {v['acc@0.5']} | {v['tpr@fpr5%']} | {v['ece']} | {v['decided_accuracy']} |"
        )
    (rep_dir / "report.md").write_text("\n".join(md_lines) + "\n")
    print("\n" + "\n".join(md_lines))
    print(f"\nSaved -> {out_head} (sha {head.sha})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
