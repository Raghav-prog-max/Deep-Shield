"""Tests for the trained Module 02 face head (training script + engine integration).

Uses a TINY randomly initialised SigLIP classifier written to a temp models dir, so nothing is
downloaded and the real model / real head on disk are never touched.
"""
from __future__ import annotations

import csv
import importlib.util
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

from deep_shield.engines.deepfake import classifiers
from deep_shield.engines.deepfake.config import FACE_SPEC, TASK_FACE, EngineConfig
from deep_shield.engines.deepfake.head import FINAL, FaceHead, default_layers

ROOT = Path(__file__).resolve().parents[1]
ASSETS = Path(__file__).parent / "assets"
needs_torch = pytest.mark.skipif(
    importlib.util.find_spec("torch") is None or importlib.util.find_spec("transformers") is None,
    reason="torch/transformers not installed")


def _random_head(dim, layers, kind="linear", base=FACE_SPEC.model_id, seed=0):
    rng = np.random.default_rng(seed)
    D = dim * len(layers)
    if kind == "linear":
        w = [rng.normal(size=(D, 1)).astype(np.float32) * 0.1, np.zeros(1, np.float32)]
    else:
        w = [rng.normal(size=(D, 8)).astype(np.float32) * 0.1, np.zeros(8, np.float32),
             rng.normal(size=(8, 1)).astype(np.float32), np.zeros(1, np.float32)]
    return FaceHead(kind=kind, layers=list(layers), mean=np.zeros(D, np.float32), std=np.ones(D, np.float32),
                    weights=w, threshold_low=0.3, threshold_high=0.7,
                    meta={"name": "test-head", "base_model_id": base, "summary": "test head summary"})


def test_head_roundtrip(tmp_path):
    for kind in ("linear", "mlp"):
        h = _random_head(16, [2, 3, FINAL], kind)
        x = np.random.default_rng(1).normal(size=(5, 48)).astype(np.float32)
        p = tmp_path / f"{kind}.npz"
        h.save(p)
        g = FaceHead.load(p)
        assert g.kind == kind and g.layers == [2, 3, FINAL] and len(g.sha) == 12
        assert np.allclose(h.predict_proba(x), g.predict_proba(x), atol=1e-6)
        assert ((g.predict_proba(x) > 0) & (g.predict_proba(x) < 1)).all()
    assert default_layers(12) == [6, 9, FINAL]


@pytest.fixture()
def tiny_models(tmp_path, monkeypatch):
    """Tiny SiglipForImageClassification saved where the engine looks for the face model."""
    from transformers import SiglipConfig, SiglipForImageClassification, SiglipImageProcessor

    cfg = SiglipConfig(
        vision_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4,
                           image_size=32, patch_size=8),
        text_config=dict(hidden_size=32, intermediate_size=64, num_hidden_layers=1, num_attention_heads=4,
                         vocab_size=100, max_position_embeddings=16))
    cfg.id2label, cfg.label2id = {0: "Fake", 1: "Real"}, {"Fake": 0, "Real": 1}
    import torch
    torch.manual_seed(0)
    model = SiglipForImageClassification(cfg)
    mdir = tmp_path / "models"
    target = mdir / FACE_SPEC.model_id.replace("/", "__")
    model.save_pretrained(target)
    SiglipImageProcessor(size={"height": 32, "width": 32}).save_pretrained(target)
    monkeypatch.setenv("DEEPSHIELD_MODELS_DIR", str(mdir))
    from deep_shield.engines.deepfake.faces import YUNET_FILE
    if not os.getenv("DEEPSHIELD_YUNET_PATH") and (ROOT / "models" / YUNET_FILE).exists():
        monkeypatch.setenv("DEEPSHIELD_YUNET_PATH", str(ROOT / "models" / YUNET_FILE))  # keep the real detector
    classifiers._CACHE.clear(); classifiers._BACKENDS.clear()
    yield mdir
    classifiers._CACHE.clear(); classifiers._BACKENDS.clear()


def _cfg(mdir):
    c = EngineConfig()
    c.backend, c.device, c.models_dir = "hf", "cpu", mdir
    return c


@needs_torch
def test_engine_uses_trained_head(tiny_models, tmp_path):
    from PIL import Image
    head = _random_head(32, default_layers(4))
    head.save(tiny_models / "m02_face_head.npz")
    clf = classifiers.get_classifier(TASK_FACE, _cfg(tiny_models))
    assert clf.head is not None and clf.backend.name == "transformers+trained_head"
    assert clf.thresholds() == {"low": 0.3, "high": 0.7}
    assert "test-head" in clf.model_info.model_id and "+head:" in clf.model_info.revision
    assert clf.spec.limitations[0] == "test head summary"
    im = Image.open(ASSETS / "face_sample.jpg")
    pred = clf.classify([im, im.rotate(10)])
    feats, _ = clf.backend.base.features([im, im.rotate(10)], head.layers, clf.backend.pos_idx)
    want = head.predict_proba(feats)
    assert np.allclose([p.score for p in pred], np.round(want, 4), atol=1e-4)
    assert set(pred[0].raw) == {"Fake", "Real"}


@needs_torch
def test_head_ignored_for_other_base_model_and_disabled(tiny_models, monkeypatch):
    _random_head(32, default_layers(4), base="someone/else").save(tiny_models / "m02_face_head.npz")
    clf = classifiers.get_classifier(TASK_FACE, _cfg(tiny_models))
    assert clf.head is None and "ignored" in (clf.head_note or "")
    assert clf.backend.name == "transformers"
    classifiers._CACHE.clear(); classifiers._BACKENDS.clear()
    _random_head(32, default_layers(4)).save(tiny_models / "m02_face_head.npz")
    monkeypatch.setenv("DEEPSHIELD_M02_FACE_HEAD", "none")
    assert classifiers.get_classifier(TASK_FACE, _cfg(tiny_models)).head is None


@needs_torch
def test_training_script_end_to_end(tiny_models, tmp_path, monkeypatch):
    """Fetch-style folder -> train_m02_head.py -> head file + report -> engine picks the head up."""
    from deep_shield.engines.deepfake.faces import FaceDetector
    if not FaceDetector(_cfg(tiny_models)).available:
        pytest.skip("no face detector available")
    base = cv2.imread(str(ASSETS / "face_sample.jpg"))
    data = tmp_path / "data"
    rows = []
    rng = np.random.default_rng(0)
    for source in ("stylegan1", "dff", "tpdne"):
        for split, n in (("train", 10), ("test", 6)):
            for label in ("real", "fake"):
                if source == "tpdne" and label == "real":
                    continue
                for i in range(n):
                    img = base.copy()
                    if label == "fake":   # a consistent, learnable artefact
                        img = cv2.GaussianBlur(img, (0, 0), 2.5)
                        img[..., 2] = np.clip(img[..., 2].astype(int) + 40, 0, 255).astype(np.uint8)
                    m = np.float32([[1, 0, rng.integers(-6, 7)], [0, 1, rng.integers(-6, 7)]])
                    img = cv2.warpAffine(img, m, (img.shape[1], img.shape[0]), borderMode=cv2.BORDER_REFLECT)
                    img = np.clip(img.astype(int) + rng.integers(-12, 13), 0, 255).astype(np.uint8)
                    rel = f"{source}/{split}/{label}/{source}_{split}_{label}_{i:03d}.jpg"
                    (data / rel).parent.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(data / rel), img)
                    rows.append({"file": rel, "source": source, "split": split, "label": label,
                                 "category": "FS" if (source == "dff" and label == "fake") else label,
                                 "origin": "synthetic-test"})
    with open(data / "manifest.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["file", "source", "split", "label", "category", "origin"])
        w.writeheader(); w.writerows(rows)

    spec = importlib.util.spec_from_file_location("train_m02_head", ROOT / "scripts" / "train_m02_head.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    head_path, rep = tiny_models / "m02_face_head.npz", tmp_path / "rep"
    monkeypatch.setattr(sys, "argv", ["train", "--data", str(data), "--out-head", str(head_path),
                                      "--report", str(rep), "--quick"])
    assert mod.main() == 0
    import json
    r = json.loads((rep / "report.json").read_text())
    assert r["test"]["trained_head"]["pooled"]["auc"] >= 0.9          # learnable artefact is learnt
    assert r["activated"] and r["validation_auc"]["trained_head"] >= r["validation_auc"]["original_model"]
    assert set(r["leave_one_generator_out"]) >= {"stylegan1", "dff", "tpdne"}
    assert "dff:FS" in r["test"]["trained_head"]
    assert (rep / "report.md").read_text().startswith("# Module 02 face head")
    # features are cached: a second run must not recompute them
    cache = list(data.glob("features_*.npz"))
    assert len(cache) == 1
    mtime = cache[0].stat().st_mtime
    assert mod.main() == 0 and cache[0].stat().st_mtime == mtime
    # the engine now loads the trained head
    classifiers._CACHE.clear(); classifiers._BACKENDS.clear()
    clf = classifiers.get_classifier(TASK_FACE, _cfg(tiny_models))
    assert clf.head is not None and clf.head.meta["n_test"] > 0


def test_fetcher_offline(tmp_path, monkeypatch):
    """fetch_train_faces.py with the network replaced: layout, labels, splits, manifest, resume."""
    import io
    import json as _json
    from PIL import Image
    spec = importlib.util.spec_from_file_location("fetch_train_faces", ROOT / "scripts" / "fetch_train_faces.py")
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    buf = io.BytesIO(); Image.new("RGB", (1024, 1024), (120, 90, 60)).save(buf, "JPEG")
    jpeg = buf.getvalue()
    calls = {"rows": 0, "img": 0}

    def fake_get(url, tries=6):
        if url.startswith(mod.API):
            calls["rows"] += 1
            from urllib.parse import parse_qs, urlparse
            q = parse_qs(urlparse(url).query)
            off, n = int(q["offset"][0]), int(q["length"][0])
            ds = q["dataset"][0]
            key = "label" if ds == mod.DS_140K else "binary_label"
            rows = [{"row_idx": off + i, "row": {key: (off + i) % 2, "class_name": "FS" if (off + i) % 2 else "real",
                                                 "image": {"src": f"https://cdn/{ds}/{off + i}.jpg"}}}
                    for i in range(n)]
            return _json.dumps({"rows": rows, "num_rows_total": 5000}).encode()
        calls["img"] += 1
        return jpeg

    monkeypatch.setattr(mod, "get", fake_get)
    out = tmp_path / "d"
    monkeypatch.setattr(sys, "argv", ["f", "--out", str(out), "--train", "12", "--test", "10", "--workers", "2"])
    mod.main()
    with open(out / "manifest.csv") as fh:
        rows = list(csv.DictReader(fh))
    from collections import Counter
    c = Counter((r["source"], r["split"], r["label"]) for r in rows)
    assert c[("stylegan1", "train", "real")] == 12 and c[("stylegan1", "test", "fake")] == 10
    assert c[("tpdne", "train", "fake")] == 12 and c[("tpdne", "test", "fake")] == 10
    assert ("tpdne", "train", "real") not in c and c[("dff", "test", "real")] == 10
    assert all((out / r["file"]).exists() for r in rows)
    tp = [r for r in rows if r["source"] == "tpdne"]
    idx = {r["split"]: {int(r["origin"].split("(")[1].split(")")[0]) for r in tp if r["split"] == s}
           for s in ("train", "test") for r in tp if r["split"] == s}
    assert max(idx["train"]) <= 6000 < 8000 < min(idx["test"])            # disjoint file ranges
    assert max(Image.open(out / tp[0]["file"]).size) == 512                # resized to save disk
    assert {r["category"] for r in rows if r["source"] == "dff" and r["label"] == "fake"} == {"FS"}
    n_img = calls["img"]
    mod.main()                                                             # resume: nothing re-downloaded
    assert calls["img"] == n_img
