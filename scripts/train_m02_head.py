"""Train and test the Module 02 face-manipulation head on DEEP SHIELD's multi-generator face set.

    python scripts/fetch_train_faces.py           # 1. data (once)
    python scripts/train_m02_head.py              # 2. features -> train -> test -> report -> head file

What it does (CPU-friendly; backbone frozen):
  1. Detects and crops faces EXACTLY as the engine does (same detector, same margin).
  2. Extracts frozen SigLIP features from three encoder depths (cached in data/m02_train/features_*.npz).
  3. Splits the train pool 80/20 (stratified by source and label) into train / validation.
  4. Trains candidate heads (linear logistic regression and a 1-hidden-layer MLP, several regularisation
     strengths, final-layer vs multi-layer features); picks the best by validation ROC-AUC.
  5. Sets the three-band thresholds on validation (<=5% of real faces above HIGH, <=5% of fakes below LOW).
  6. Tests ONCE on the held-out test split, against the original model on the same faces.
  7. Leave-one-generator-out: retrains without each source and tests on it (generalisation to unseen fakes).
  8. Saves models/m02_face_head.npz (loaded automatically by the engine) and reports/m02_training/.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from deep_shield.engines.deepfake.classifiers import load_image_processor, local_model_dir, resolve_device  # noqa: E402
from deep_shield.engines.deepfake.config import FACE_SPEC, EngineConfig  # noqa: E402
from deep_shield.engines.deepfake.faces import FaceDetector, crop_face  # noqa: E402
from deep_shield.engines.deepfake.head import FINAL, FaceHead, default_layers, extract  # noqa: E402
from deep_shield.engines.deepfake.media import load_image  # noqa: E402
from deep_shield.engines.deepfake.metrics import binary_metrics, roc_auc, three_band  # noqa: E402

PAIR_REALS = {"tpdne": "stylegan1"}   # fake-only sources are tested against this source's real faces


# --------------------------------------------------------------------------- data
def read_manifest(root: Path):
    seen, items = set(), []
    with open(root / "manifest.csv", newline="") as fh:
        for r in csv.DictReader(fh):
            if r["file"] in seen or not (root / r["file"]).exists():
                continue
            seen.add(r["file"])
            r["y"] = 1 if r["label"] == "fake" else 0
            items.append(r)
    return items


def drop_test_duplicates(root: Path, items):
    """Remove any test image whose bytes also appear in the train pool (leakage guard)."""
    h = {it["file"]: hashlib.sha1((root / it["file"]).read_bytes()).hexdigest() for it in items}
    train_hashes = {h[it["file"]] for it in items if it["split"] == "train"}
    keep = [it for it in items if not (it["split"] == "test" and h[it["file"]] in train_hashes)]
    return keep, len(items) - len(keep)


# --------------------------------------------------------------------------- features
def load_backbone(cfg: EngineConfig, model_id: str):
    import torch
    from transformers import AutoModelForImageClassification

    local = local_model_dir(cfg, model_id)
    source = str(local) if (local / "config.json").exists() else model_id
    model = AutoModelForImageClassification.from_pretrained(source)
    proc = load_image_processor(source)
    dev = resolve_device(cfg.device)
    dev = "cpu" if dev == -1 else (f"cuda:{dev}" if isinstance(dev, int) else dev)
    model.to(dev).eval()
    torch.set_grad_enabled(False)
    pos = {l.lower() for l in FACE_SPEC.positive_labels}
    pos_idx = [int(i) for i, l in model.config.id2label.items() if l.lower() in pos]
    if not pos_idx:
        raise SystemExit(f"positive label {FACE_SPEC.positive_labels} not in {model.config.id2label}")
    return model, proc, dev, pos_idx, source


def compute_features(root, items, cfg, model, proc, dev, pos_idx, layers, cache: Path, batch: int):
    have = {}
    if cache.exists():
        z = np.load(cache, allow_pickle=False)
        if list(z["layers"]) == list(layers):
            have = {f: i for i, f in enumerate(z["files"].tolist())}
            F, P, S = z["feats"], z["base"], z["status"]
    todo = [it for it in items if it["file"] not in have]
    det = FaceDetector(cfg)
    if not det.available:
        raise SystemExit("No face detector available (YuNet/Haar) - cannot crop faces like the engine.")
    new_files, new_f, new_p, new_s = [], [], [], []
    t0 = time.time()
    for start in range(0, len(todo), 64):
        chunk = todo[start:start + 64]
        crops, local = [], []
        for j, it in enumerate(chunk):
            new_files.append(it["file"])
            try:
                bgr, _ = load_image(root / it["file"])
                boxes = det.detect(bgr)
            except Exception as exc:  # noqa: BLE001 - unreadable image -> skip like "no face"
                print(f"    unreadable {it['file']}: {exc}")
                boxes = []
            if not boxes:
                new_s.append(0)
                continue
            b = max(boxes, key=lambda b: b.w * b.h)
            crops.append(crop_face(bgr, b, cfg.face_margin))
            local.append(j)
            new_s.append(1)
        feats = np.zeros((len(chunk), 0), dtype=np.float32)
        base = np.full(len(chunk), np.nan, dtype=np.float32)
        if crops:
            f, p = extract(model, proc, crops, layers, pos_idx, dev, batch)
            feats = np.zeros((len(chunk), f.shape[1]), dtype=np.float32)
            feats[local] = f
            base[local] = p
        new_f.append(feats); new_p.append(base)
        done = start + len(chunk)
        rate = done / max(1e-6, time.time() - t0)
        print(f"  features {done}/{len(todo)}  ({rate:.1f} img/s, ~{(len(todo) - done) / max(rate, 1e-6) / 60:.1f} min left)")
    if new_files:
        dims = [a.shape[1] for a in new_f if a.shape[1]]
        D = dims[0] if dims else (F.shape[1] if have else 0)
        nf = np.concatenate([a if a.shape[1] else np.zeros((len(a), D), np.float32) for a in new_f])
        np_ = np.concatenate(new_p)
        ns = np.asarray(new_s, dtype=np.int8)
        if have:
            F, P, S = np.concatenate([F, nf]), np.concatenate([P, np_]), np.concatenate([S, ns])
            files = list(have) + new_files
        else:
            F, P, S, files = nf, np_, ns, new_files
        np.savez(cache, files=np.asarray(files), feats=F, base=P, status=S, layers=np.asarray(layers))
        have = {f: i for i, f in enumerate(files)}
    rows = [have[it["file"]] for it in items]
    return F[rows], P[rows], S[rows].astype(bool)


# --------------------------------------------------------------------------- heads
def fit_head(X, y, w_sample, kind: str, reg: float, seed: int = 0):
    """Returns (weights list, standardisation mean, std). Class-balanced, L2-regularised."""
    import torch

    torch.manual_seed(seed)
    mean = X.mean(0)
    std = X.std(0) + 1e-6
    Xt = torch.tensor((X - mean) / std, dtype=torch.float32)
    yt = torch.tensor(y, dtype=torch.float32)
    wt = torch.tensor(w_sample, dtype=torch.float32)
    D = X.shape[1]
    with torch.enable_grad():
        if kind == "linear":
            W = torch.zeros(D, 1, requires_grad=True); b = torch.zeros(1, requires_grad=True)
            params = [W, b]
            opt = torch.optim.LBFGS(params, lr=1.0, max_iter=300, line_search_fn="strong_wolfe")

            def closure():
                opt.zero_grad()
                z = (Xt @ W).squeeze(1) + b
                loss = (torch.nn.functional.binary_cross_entropy_with_logits(z, yt, reduction="none") * wt).sum() / wt.sum()
                loss = loss + reg * (W ** 2).sum()
                loss.backward()
                return loss
            opt.step(closure)
            return [W.detach().numpy(), b.detach().numpy()], mean, std
        H = 128
        net = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(D, H), torch.nn.ReLU(),
                                  torch.nn.Dropout(0.35), torch.nn.Linear(H, 1))
        opt = torch.optim.AdamW(net.parameters(), lr=8e-4, weight_decay=reg)
        n = len(Xt)
        g = torch.Generator().manual_seed(seed)
        for _ in range(70):
            net.train()
            perm = torch.randperm(n, generator=g)
            for i in range(0, n, 128):
                k = perm[i:i + 128]
                z = net(Xt[k]).squeeze(1)
                loss = (torch.nn.functional.binary_cross_entropy_with_logits(z, yt[k], reduction="none") * wt[k]).sum() / wt[k].sum()
                opt.zero_grad(); loss.backward(); opt.step()
        net.eval()
        l1, l2 = net[1], net[4]
        return [l1.weight.detach().numpy().T, l1.bias.detach().numpy(),
                l2.weight.detach().numpy().T, l2.bias.detach().numpy()], mean, std


def balanced_weights(y, sources=None):
    """Balance class totals 50/50 and equalize contribution across sources within each class."""
    if sources is None:
        c = Counter(y.tolist())
        return np.array([len(y) / (2.0 * c[v]) for v in y], dtype=np.float32)
    by_cls = defaultdict(Counter)
    for yi, si in zip(y.tolist(), sources):
        by_cls[yi][si] += 1
    w = np.zeros(len(y), dtype=np.float32)
    for i, (yi, si) in enumerate(zip(y.tolist(), sources)):
        n_src = max(1, len(by_cls[yi]))
        w[i] = 0.5 / (n_src * by_cls[yi][si])
    return w * len(y)


def calibrate_temperature(head: FaceHead, X_val: np.ndarray, y_val: np.ndarray) -> float:
    """Guo et al. (ICML 2017) validation-split temperature scaling folded into the last linear layer."""
    x = (X_val - head.mean) / head.std
    if head.kind == "linear":
        z = (x @ head.weights[0] + head.weights[1]).ravel()
    else:
        h = np.maximum(0.0, x @ head.weights[0] + head.weights[1])
        z = (h @ head.weights[2] + head.weights[3]).ravel()
    best_t, best_nll = 1.0, float("inf")
    for t in np.geomspace(0.5, 8.0, 60):
        p = 1.0 / (1.0 + np.exp(-np.clip(z / t, -40.0, 40.0)))
        p = np.clip(p, 1e-6, 1.0 - 1e-6)
        nll = float(-np.mean(y_val * np.log(p) + (1.0 - y_val) * np.log(1.0 - p)))
        if nll < best_nll:
            best_nll, best_t = nll, float(t)
    if head.kind == "linear":
        head.weights = [head.weights[0] / best_t, head.weights[1] / best_t]
    else:
        head.weights = [head.weights[0], head.weights[1], head.weights[2] / best_t, head.weights[3] / best_t]
    return round(best_t, 4)


def make_head(kind, layers, weights, mean, std, lo=0.35, hi=0.65, meta=None) -> FaceHead:
    return FaceHead(kind=kind, layers=list(layers), mean=mean, std=std, weights=weights,
                    threshold_low=lo, threshold_high=hi, meta=meta or {})


def slice_cols(layers_all, layers_sel, dim):
    cols = []
    for i, l in enumerate(layers_all):
        if l in layers_sel:
            cols.extend(range(i * dim, (i + 1) * dim))
    return np.asarray(cols)


def thresholds_from_val(y, s, fpr=0.05, fnr=0.05):
    hi = float(np.quantile(s[y == 0], 1 - fpr))   # above HIGH: at most `fpr` of real faces
    lo = float(np.quantile(s[y == 1], fnr))       # below LOW: at most `fnr` of fakes
    if lo >= hi or hi - lo < 0.10:                # separable (or nearly): narrow band around the gap
        mid = (lo + hi) / 2
        lo, hi = mid - 0.05, mid + 0.05
    return round(float(np.clip(lo, 0.02, 0.96)), 3), round(float(np.clip(hi, lo + 0.02, 0.98)), 3)


def tpr_at_fpr(y, s, fpr=0.05):
    t = np.quantile(s[y == 0], 1 - fpr)
    return float((s[y == 1] > t).mean())


def summarise(y, s, lo, hi):
    if len(set(y.tolist())) < 2:
        return None
    m = binary_metrics(y, s, 0.5)
    tb = three_band(y, s, lo, hi)
    return {"n": int(len(y)), "n_fake": int(y.sum()), "n_real": int((1 - y).sum()),
            "auc": round(roc_auc(y, s), 4), "acc@0.5": round(m["accuracy"], 4),
            "precision@0.5": round(m["precision"], 4), "recall@0.5": round(m["recall"], 4),
            "f1@0.5": round(m["f1"], 4), "tpr@fpr5%": round(tpr_at_fpr(y, s), 4),
            "band_low": lo, "band_high": hi, "coverage": round(tb["coverage"], 4),
            "decided_accuracy": round(tb.get("decided_accuracy", float("nan")), 4)}


def eval_groups(items, y, s, lo, hi):
    """Per source (fake-only sources paired with their real partner), per DFF category, and pooled."""
    out = {"pooled": summarise(y, s, lo, hi)}
    srcs = sorted({it["source"] for it in items})
    for src in srcs:
        real_src = PAIR_REALS.get(src, src)
        m = np.array([(it["source"] == src and it["y"] == 1) or (it["source"] == real_src and it["y"] == 0)
                      for it in items])
        if m.any():
            out[src] = summarise(y[m], s[m], lo, hi)
    for cat in sorted({it["category"] for it in items if it["source"] == "dff" and it["y"] == 1}):
        m = np.array([(it["source"] == "dff" and (it["y"] == 0 or it["category"] == cat)) for it in items])
        out[f"dff:{cat}"] = summarise(y[m], s[m], lo, hi)
    return {k: v for k, v in out.items() if v}


# --------------------------------------------------------------------------- main
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/m02_train")
    ap.add_argument("--out-head", default="models/m02_face_head.npz")
    ap.add_argument("--report", default="reports/m02_training")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--quick", action="store_true", help="fewer candidate heads (smoke test)")
    a = ap.parse_args()

    root = Path(a.data)
    if not (root / "manifest.csv").exists():
        print(f"No {root}/manifest.csv - run scripts/fetch_train_faces.py first.")
        return 1
    cfg = EngineConfig()
    rng = np.random.default_rng(a.seed)
    items = read_manifest(root)
    items, n_dup = drop_test_duplicates(root, items)
    print(f"{len(items)} images in manifest ({n_dup} test duplicates of train images removed)")
    print("by source/split/label:", dict(Counter((i["source"], i["split"], i["label"]) for i in items)))

    model, proc, dev, pos_idx, source = load_backbone(cfg, FACE_SPEC.model_id)
    n_layers = int(getattr(getattr(model.config, "vision_config", model.config), "num_hidden_layers", 12))
    layers = default_layers(n_layers)
    print(f"backbone {FACE_SPEC.model_id} on {dev}; feature layers {layers} (frozen)")
    tag = hashlib.sha1(f"{FACE_SPEC.model_id}|{layers}|{cfg.face_margin}".encode()).hexdigest()[:8]
    X, base, ok = compute_features(root, items, cfg, model, proc, dev, pos_idx, layers,
                                   root / f"features_{tag}.npz", a.batch)
    no_face = Counter(it["source"] + "/" + it["split"] for it, k in zip(items, ok) if not k)
    items = [it for it, k in zip(items, ok) if k]
    X, base = X[ok], base[ok]
    y = np.array([it["y"] for it in items])
    split = np.array([it["split"] for it in items])
    dim = X.shape[1] // len(layers)
    print(f"faces used: {len(items)}   no face detected (skipped, as the engine would): {dict(no_face)}")

    # ---- train / validation split of the train pool (stratified by source+label)
    tr_idx = np.where(split == "train")[0]
    te_idx = np.where(split == "test")[0]
    strata = defaultdict(list)
    for i in tr_idx:
        strata[(items[i]["source"], items[i]["y"])].append(i)
    val_idx, fit_idx = [], []
    for k, v in strata.items():
        v = rng.permutation(v)
        nv = max(1, int(round(0.2 * len(v))))
        val_idx += v[:nv].tolist(); fit_idx += v[nv:].tolist()
    fit_idx, val_idx = np.array(fit_idx), np.array(val_idx)

    # ---- candidate heads, chosen on validation AUC
    feature_sets = {"final": [FINAL], "multi": layers}
    grid = [("linear", r) for r in ([1e-3, 1e-2] if a.quick else [1e-3, 1e-2, 1e-1, 0.5, 2.0])]
    grid += [("mlp", r) for r in ([1e-2] if a.quick else [1e-2, 5e-2, 2e-1, 5e-1])]
    cands = []
    fit_sources = [items[i]["source"] for i in fit_idx]
    w_fit = balanced_weights(y[fit_idx], fit_sources)
    for fs_name, fs in feature_sets.items():
        cols = slice_cols(layers, fs, dim)
        for kind, reg in grid:
            W, mu, sd = fit_head(X[fit_idx][:, cols], y[fit_idx], w_fit, kind, reg, a.seed)
            h = make_head(kind, fs, W, mu, sd)
            s_val = h.predict_proba(X[val_idx][:, cols])
            auc = roc_auc(y[val_idx], s_val)
            cands.append({"features": fs_name, "kind": kind, "reg": reg, "val_auc": round(auc, 4)})
            print(f"  candidate {fs_name:5s} {kind:6s} reg={reg:<7g} val AUC {auc:.4f}")
    best = max(cands, key=lambda c: (c["val_auc"], c["kind"] == "linear"))  # tie -> simpler head
    print(f"selected: {best}")

    # ---- final head: fit on the fit split, temperature-calibrate & set thresholds on validation
    fs = feature_sets[best["features"]]
    cols = slice_cols(layers, fs, dim)
    W, mu, sd = fit_head(X[fit_idx][:, cols], y[fit_idx], w_fit, best["kind"], best["reg"], a.seed)
    head = make_head(best["kind"], fs, W, mu, sd)
    temp = calibrate_temperature(head, X[val_idx][:, cols], y[val_idx])
    s_val = head.predict_proba(X[val_idx][:, cols])
    lo, hi = thresholds_from_val(y[val_idx], s_val)
    head.threshold_low, head.threshold_high = lo, hi
    print(f"temperature calibration on validation: T={temp:.4f} -> thresholds low={lo}, high={hi}")
    val_auc_head, val_auc_base = roc_auc(y[val_idx], s_val), roc_auc(y[val_idx], base[val_idx])
    activate = val_auc_head >= val_auc_base   # deployment gate decided on validation, never on test
    print(f"validation AUC: trained head {val_auc_head:.4f} vs original model {val_auc_base:.4f}"
          f" -> {'ACTIVATE head' if activate else 'keep ORIGINAL model (head saved as .rejected)'}")

    # ---- held-out test: trained head vs original model, same faces
    test_items = [items[i] for i in te_idx]
    s_test = head.predict_proba(X[te_idx][:, cols])
    b_test = base[te_idx]
    res_head = eval_groups(test_items, y[te_idx], s_test, lo, hi)
    res_base = eval_groups(test_items, y[te_idx], b_test, FACE_SPEC.threshold_low, FACE_SPEC.threshold_high)

    # ---- leave-one-generator-out (unseen generator at test time)
    loso = {}
    for held in sorted({it["source"] for it in items}):
        tr = np.array([i for i in tr_idx if items[i]["source"] != held])
        if len(set(y[tr].tolist())) < 2:
            continue
        tr_sources = [items[i]["source"] for i in tr]
        Wl, mul, sdl = fit_head(X[tr][:, cols], y[tr], balanced_weights(y[tr], tr_sources), best["kind"], best["reg"], a.seed)
        hl = make_head(best["kind"], fs, Wl, mul, sdl)
        real_src = PAIR_REALS.get(held, held)
        m = np.array([i for i in te_idx if (items[i]["source"] == held and items[i]["y"] == 1)
                      or (items[i]["source"] == real_src and items[i]["y"] == 0)])
        r = summarise(y[m], hl.predict_proba(X[m][:, cols]), lo, hi)
        rb = summarise(y[m], base[m], FACE_SPEC.threshold_low, FACE_SPEC.threshold_high)
        if r:
            loso[held] = {"head_trained_without_it": r, "original_model": rb}

    # ---- save head + report
    counts = dict(Counter(f"{it['source']}/{it['split']}/{it['label']}" for it in items))
    pooled = res_head["pooled"]
    per_src = ", ".join(f"{k} {v['auc']:.3f}" for k, v in res_head.items() if ":" not in k and k != "pooled")
    loso_txt = ", ".join(f"{k} {v['head_trained_without_it']['auc']:.3f}" for k, v in loso.items())
    src_list = sorted({it["source"] for it in items})
    head.meta = {
        "name": "deepshield-face-head-v2", "base_model_id": FACE_SPEC.model_id, "base_source": source,
        "trained_at": datetime.now().isoformat(timespec="seconds"), "selection": best,
        "temperature": temp, "sources": src_list,
        "n_train": int(len(fit_idx)), "n_val": int(len(val_idx)), "n_test": int(len(te_idx)),
        "test_auc": pooled["auc"], "summary": (
            f"Classification head retrained by DEEP SHIELD on {len(fit_idx)} face crops across {len(src_list)} "
            f"real datasets ({', '.join(src_list)}); SigLIP2 backbone frozen, temperature-calibrated (T={temp}). "
            f"Held-out test ROC-AUC {pooled['auc']:.3f} (per source: {per_src}). "
            f"Unseen-generator (leave-one-out) ROC-AUC: {loso_txt}. "
            f"Validation-calibrated thresholds: <=5% real above {hi}, <=5% fake below {lo}."),
    }
    out_head = Path(a.out_head)
    if not activate:
        out_head = out_head.with_suffix(".rejected.npz")
        Path(a.out_head).unlink(missing_ok=True)
    head.save(out_head)
    rep = Path(a.report)
    rep.mkdir(parents=True, exist_ok=True)
    report = {"created": head.meta["trained_at"], "base_model": FACE_SPEC.model_id, "head_file": str(out_head),
              "activated": bool(activate),
              "validation_auc": {"trained_head": round(val_auc_head, 4), "original_model": round(val_auc_base, 4)},
              "head_sha": head.sha, "feature_layers": layers, "selected": best, "candidates": cands,
              "thresholds": {"low": lo, "high": hi}, "counts_used": counts, "no_face_skipped": dict(no_face),
              "test_duplicates_removed": n_dup, "test": {"trained_head": res_head, "original_model": res_base},
              "leave_one_generator_out": loso}
    (rep / "report.json").write_text(json.dumps(report, indent=2))
    with open(rep / "test_scores.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["file", "source", "category", "label", "original_score", "trained_head_score"])
        for it, b_, s_ in zip(test_items, b_test, s_test):
            w.writerow([it["file"], it["source"], it["category"], it["label"], round(float(b_), 5), round(float(s_), 5)])
    (rep / "report.md").write_text(render_md(report))
    print("\n" + render_md(report))
    print(f"\nhead -> {out_head}  (sha {head.sha})  {'ACTIVE: the engine will use it' if activate else 'NOT active'}"
          f"\nreport -> {rep / 'report.md'}")
    return 0


def render_md(r) -> str:
    L = [f"# Module 02 face head - training and test report", "",
         f"Created {r['created']}. Base model `{r['base_model']}` (frozen). Head `{r['head_file']}` sha `{r['head_sha']}`.",
         f"Selected head: **{r['selected']['kind']}** on **{r['selected']['features']}** features "
         f"(reg {r['selected']['reg']}, validation AUC {r['selected']['val_auc']}).",
         f"Three-band thresholds (set on validation): low **{r['thresholds']['low']}**, high **{r['thresholds']['high']}**.",
         f"Validation AUC: trained head {r['validation_auc']['trained_head']} vs original {r['validation_auc']['original_model']} "
         f"-> head **{'ACTIVATED' if r['activated'] else 'NOT activated (original model kept)'}**.", "",
         "## Held-out test: original model vs trained head (same faces)", "",
         "| Test set | n (fake/real) | Original AUC | Trained AUC | Original acc@0.5 | Trained acc@0.5 | Original TPR@5%FPR | Trained TPR@5%FPR | Trained coverage | Trained decided acc |",
         "|---|---|---|---|---|---|---|---|---|---|"]
    th, ob = r["test"]["trained_head"], r["test"]["original_model"]
    for k in th:
        t, o = th[k], ob.get(k) or {}
        L.append(f"| {k} | {t['n_fake']}/{t['n_real']} | {o.get('auc', '-')} | **{t['auc']}** | {o.get('acc@0.5', '-')} | "
                 f"{t['acc@0.5']} | {o.get('tpr@fpr5%', '-')} | {t['tpr@fpr5%']} | {t['coverage']} | {t['decided_accuracy']} |")
    L += ["", "## Unseen generator (leave-one-generator-out)", "",
          "Head retrained WITHOUT the named source, then tested on it. This is the honest estimate for a new generator.", "",
          "| Held-out source | n (fake/real) | Original AUC | Head-without-it AUC | Head-without-it TPR@5%FPR |", "|---|---|---|---|---|"]
    for k, v in r["leave_one_generator_out"].items():
        h, o = v["head_trained_without_it"], v["original_model"] or {}
        L.append(f"| {k} | {h['n_fake']}/{h['n_real']} | {o.get('auc', '-')} | {h['auc']} | {h['tpr@fpr5%']} |")
    L += ["", "## Data", "", "| source/split/label | faces used |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in sorted(r["counts_used"].items())]
    L += ["", f"No face detected (skipped, as the engine would): {r['no_face_skipped'] or 'none'}. "
              f"Test images identical to a train image removed: {r['test_duplicates_removed']}.",
          "", "Notes: tpdne has fake images only; it is tested against the stylegan1 (FFHQ) real faces. "
              "dff category names are as labelled by the dataset uploader. Small test sets: treat differences "
              "below ~0.02 AUC as noise. Not a formal benchmark."]
    return "\n".join(L)


if __name__ == "__main__":
    raise SystemExit(main())
