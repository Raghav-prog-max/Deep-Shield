"""Task-specific classifiers for Module 02.

Two tasks are kept strictly separate (different models, labels, thresholds,
limitations):
  * face_manipulation — applied to face crops
  * synthetic_image   — applied to the whole image / video keyframe

Backends:
  * ``hf``   — Hugging Face transformers image-classification pipeline (real models)
  * ``stub`` — deterministic pseudo-scores for tests / UI wiring ONLY; every
               result it produces is labelled STUB so it cannot be mistaken for a model.
"""
from __future__ import annotations

import json
import os
import threading
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

from deep_shield.common.schema import ModelInfo

from .config import SPECS, TASK_FACE, ClassifierSpec, EngineConfig
from .head import FaceHead, extract, load_head


@dataclass
class Prediction:
    score: float                      # sum of positive-label probabilities
    raw: Dict[str, float]             # label -> probability
    predicted_label: str              # argmax label
    classification: str               # task verdict from three-band thresholds


def local_model_dir(cfg: EngineConfig, model_id: str) -> Path:
    return cfg.models_dir / model_id.replace("/", "__")


def _manifest_revision(cfg: EngineConfig, model_id: str) -> Optional[str]:
    mf = cfg.models_dir / "MANIFEST.json"
    if mf.exists():
        try:
            data = json.loads(mf.read_text())
            return data.get("models", {}).get(model_id, {}).get("revision")
        except (ValueError, OSError):
            return None
    return None


def resolve_device(device: str):
    if device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return 0
        if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
            return "mps"
    except ImportError:
        pass
    return -1


def load_image_processor(source: str):
    """AutoImageProcessor, with a fallback for checkpoints whose preprocessor_config.json names a
    legacy class (e.g. ``ViTFeatureExtractor``) that transformers 5.x no longer registers."""
    import transformers
    from transformers import AutoImageProcessor

    try:
        return AutoImageProcessor.from_pretrained(source)
    except (ValueError, KeyError) as first_error:
        p = Path(source) / "preprocessor_config.json"
        if not p.exists():
            from huggingface_hub import hf_hub_download
            p = Path(hf_hub_download(source, "preprocessor_config.json"))
        legacy = json.loads(p.read_text()).get("image_processor_type") or \
            json.loads(p.read_text()).get("feature_extractor_type") or ""
        modern = legacy.replace("FeatureExtractor", "ImageProcessor")
        for name in (modern, modern + "Fast", modern.removesuffix("Fast")):
            cls = getattr(transformers, name, None) if name else None
            if cls is not None:
                return cls.from_pretrained(source)
        raise first_error


class HFBackend:
    name = "transformers"

    def __init__(self, spec: ClassifierSpec, cfg: EngineConfig):
        from transformers import pipeline  # imported lazily: heavy

        local = local_model_dir(cfg, spec.model_id)
        source = str(local) if (local / "config.json").exists() else spec.model_id
        self.device = resolve_device(cfg.device)
        self.pipe = pipeline("image-classification", model=source,
                             image_processor=load_image_processor(source), device=self.device)
        self.labels = list(self.pipe.model.config.id2label.values())
        self.revision = (getattr(self.pipe.model.config, "_commit_hash", None)
                         or _manifest_revision(cfg, spec.model_id))
        self.loaded_from = source
        self.batch_size = cfg.batch_size

    def predict(self, images: List[Image.Image]) -> List[Dict[str, float]]:
        if not images:
            return []
        out = self.pipe(images, batch_size=self.batch_size, top_k=len(self.labels))
        if images and isinstance(out, list) and out and isinstance(out[0], dict):
            out = [out]  # single image -> wrap
        return [{d["label"]: float(d["score"]) for d in item} for item in out]

    def features(self, images: List[Image.Image], layers, pos_idx):
        """Frozen backbone features (see head.py) + the model's own positive-class probability."""
        model = self.pipe.model
        return extract(model, self.pipe.image_processor, images, layers, pos_idx,
                       device=model.device, batch_size=self.batch_size)


class HeadBackend:
    """Frozen pretrained backbone + DEEP SHIELD's trained face head (scripts/train_m02_head.py)."""
    name = "transformers+trained_head"

    def __init__(self, base: HFBackend, head: FaceHead, spec: ClassifierSpec):
        self.base, self.head = base, head
        self.labels = base.labels
        self.pos_label, self.neg_label = spec.positive_labels[0], spec.negative_labels[0]
        pos = {l.lower() for l in spec.positive_labels}
        id2label = base.pipe.model.config.id2label
        self.pos_idx = [int(i) for i, l in id2label.items() if l.lower() in pos]
        self.revision = f"{base.revision or 'local'}+head:{head.sha}"
        self.loaded_from = f"{base.loaded_from} + {head.name}"
        self.device = base.device

    def predict(self, images: List[Image.Image]) -> List[Dict[str, float]]:
        if not images:
            return []
        feats, _ = self.base.features(images, self.head.layers, self.pos_idx)
        probs = self.head.predict_proba(feats)
        labels = {self.pos_label, self.neg_label}
        res = []
        for p in probs:
            d = {l: 0.0 for l in self.labels if l not in labels}
            d[self.pos_label], d[self.neg_label] = float(p), float(1.0 - p)
            res.append(d)
        return res


def face_head_path(cfg: EngineConfig) -> Optional[Path]:
    v = os.getenv("DEEPSHIELD_M02_FACE_HEAD")
    if v is not None and v.strip().lower() in ("", "0", "none", "off"):
        return None
    return Path(v) if v else cfg.models_dir / "m02_face_head.npz"


def _apply_head_spec(spec: ClassifierSpec, head: FaceHead) -> ClassifierSpec:
    """Trained head brings its own validated thresholds unless the operator overrides them by env var."""
    lo = spec.threshold_low if os.getenv("DEEPSHIELD_M02_FACE_LOW") else head.threshold_low
    hi = spec.threshold_high if os.getenv("DEEPSHIELD_M02_FACE_HIGH") else head.threshold_high
    lims = [head.meta.get("summary", "Trained DEEP SHIELD face head on a frozen pretrained backbone.")]
    lims += [l for l in spec.limitations if "Internal check on 50" not in l]
    return replace(spec, threshold_low=lo, threshold_high=hi, limitations=tuple(lims))


class StubBackend:
    """Deterministic, model-free scores. FOR TESTS AND UI WIRING ONLY."""
    name = "stub"

    def __init__(self, spec: ClassifierSpec, cfg: EngineConfig):
        self.labels = list(spec.negative_labels) + list(spec.positive_labels)
        self.spec = spec
        self.revision = "stub"
        self.loaded_from = "stub"
        self.device = "cpu"

    def predict(self, images: List[Image.Image]) -> List[Dict[str, float]]:
        res = []
        for im in images:
            a = np.asarray(im.convert("L").resize((16, 16)), dtype=np.float32) / 255.0
            s = float(np.clip(0.5 + (a.std() - 0.2) * 2.0, 0.01, 0.99))
            pos = list(self.spec.positive_labels)
            neg = list(self.spec.negative_labels)
            d = {l: (1 - s) / len(neg) for l in neg}
            d.update({l: s / len(pos) for l in pos})
            res.append(d)
        return res


_BACKENDS: Dict[str, object] = {}


class TaskClassifier:
    def __init__(self, spec: ClassifierSpec, cfg: EngineConfig):
        self.spec = spec
        self.head: Optional[FaceHead] = None
        self.head_note: Optional[str] = None
        if cfg.backend == "stub":
            self.backend = StubBackend(spec, cfg)
        else:  # one loaded copy per checkpoint, shared by tasks that use the same model
            key = f"{spec.model_id}|{cfg.device}"
            if key not in _BACKENDS:
                _BACKENDS[key] = HFBackend(spec, cfg)
            self.backend = _BACKENDS[key]
            if spec.task == TASK_FACE:
                self._maybe_attach_head(cfg)
        self._pos = {l.strip().lower() for l in spec.positive_labels}
        model_labels = {l.lower() for l in self.backend.labels}
        missing = self._pos - model_labels
        if missing:
            raise ValueError(
                f"[{spec.task}] positive labels {sorted(missing)} not found in model labels "
                f"{self.backend.labels}. Set DEEPSHIELD_M02_*_POS to match the checkpoint."
            )

    def _maybe_attach_head(self, cfg: EngineConfig) -> None:
        path = face_head_path(cfg)
        try:
            head = load_head(path)
        except Exception as exc:  # noqa: BLE001 - a broken head file must not break the engine
            self.head_note = f"trained head at {path} could not be loaded ({exc}); using the original model"
            return
        if head is None:
            return
        if head.base_model_id and head.base_model_id != self.spec.model_id:
            self.head_note = (f"trained head {path} was built for {head.base_model_id}, not "
                              f"{self.spec.model_id}; ignored")
            return
        self.head = head
        self.backend = HeadBackend(self.backend, head, self.spec)
        self.spec = _apply_head_spec(self.spec, head)

    @property
    def model_info(self) -> ModelInfo:
        is_stub = self.backend.name == "stub"
        mid = self.spec.model_id + (f" + {self.head.name}" if self.head else "")
        return ModelInfo(
            model_id=("STUB-NOT-A-MODEL" if is_stub else mid),
            revision=self.backend.revision,
            task=self.spec.task,
            license=None if is_stub else self.spec.license,
            source_url=None if is_stub else self.spec.source_url,
            backend=self.backend.name,
        )

    def verdict(self, score: float) -> str:
        ind, inc, neg = self.spec.verdicts
        if score >= self.spec.threshold_high:
            return ind
        if score <= self.spec.threshold_low:
            return neg
        return inc

    def thresholds(self) -> Dict[str, float]:
        return {"low": self.spec.threshold_low, "high": self.spec.threshold_high}

    def classify(self, images: List[Image.Image]) -> List[Prediction]:
        preds = []
        for raw in self.backend.predict(images):
            score = float(sum(p for l, p in raw.items() if l.lower() in self._pos))
            score = min(max(score, 0.0), 1.0)
            top = max(raw.items(), key=lambda kv: kv[1])[0]
            preds.append(Prediction(round(score, 4), {k: round(v, 4) for k, v in raw.items()},
                                    top, self.verdict(score)))
        return preds


_CACHE: Dict[str, TaskClassifier] = {}
_LOCK = threading.Lock()


def get_classifier(task: str, cfg: EngineConfig) -> TaskClassifier:
    key = f"{cfg.backend}:{task}:{SPECS[task].model_id}"
    with _LOCK:
        if key not in _CACHE:
            _CACHE[key] = TaskClassifier(SPECS[task], cfg)
        return _CACHE[key]
