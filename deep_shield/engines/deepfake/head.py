"""Trained face-manipulation head for Module 02.

The pretrained SigLIP backbone stays frozen. We read pooled features from three depths of the
vision encoder (middle, three-quarter and final), standardise them and apply a small head trained by
``scripts/train_m02_head.py`` on DEEP SHIELD's own multi-generator face set. The head is a few MB and
is saved as ``models/m02_face_head.npz``; the engine loads it automatically when the file exists.

Used by both the training script and the engine, so features are computed identically in both.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

FINAL = -1  # marker: the exact input of the model's own classifier layer


def default_layers(num_hidden_layers: int) -> List[int]:
    """Middle and three-quarter encoder layers + the final pooled representation."""
    return sorted({max(1, num_hidden_layers // 2), max(1, (3 * num_hidden_layers) // 4)}) + [FINAL]


def classifier_module(model):
    for name in ("classifier", "head", "score"):
        m = getattr(model, name, None)
        if m is not None and hasattr(m, "register_forward_hook"):
            return m
    raise ValueError(f"cannot find the classifier layer of {type(model).__name__}")


def extract(model, processor, images, layers: Sequence[int], pos_idx: Sequence[int],
            device="cpu", batch_size: int = 16) -> Tuple[np.ndarray, np.ndarray]:
    """-> (features [N, D], original model's positive-class probability [N])."""
    import torch

    feats, probs = [], []
    head = classifier_module(model)
    captured: Dict[str, "torch.Tensor"] = {}
    hook = head.register_forward_hook(lambda m, inp, out: captured.__setitem__("x", inp[0].detach()))
    try:
        model.eval()
        for i in range(0, len(images), batch_size):
            chunk = [im.convert("RGB") for im in images[i:i + batch_size]]
            inputs = processor(images=chunk, return_tensors="pt")
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                out = model(**inputs, output_hidden_states=True)
            hs = out.hidden_states
            parts = []
            for l in layers:
                if l == FINAL:
                    x = captured["x"]
                    if x.dim() == 3:                       # some heads pool internally
                        x = x.mean(dim=1)
                    parts.append(x)
                else:
                    parts.append(hs[l].mean(dim=1))       # mean over patch tokens
            feats.append(torch.cat(parts, dim=1).float().cpu().numpy())
            p = torch.softmax(out.logits.float(), dim=-1)[:, list(pos_idx)].sum(dim=1)
            probs.append(p.cpu().numpy())
    finally:
        hook.remove()
    return np.concatenate(feats), np.concatenate(probs)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40, 40)))


@dataclass
class FaceHead:
    kind: str                      # "linear" | "mlp"
    layers: List[int]
    mean: np.ndarray
    std: np.ndarray
    weights: List[np.ndarray]      # linear: [W (D,1), b (1,)]; mlp: [W1, b1, W2, b2]
    threshold_low: float
    threshold_high: float
    meta: Dict

    @property
    def name(self) -> str:
        return self.meta.get("name", "deepshield-face-head")

    @property
    def base_model_id(self) -> str:
        return self.meta.get("base_model_id", "")

    @property
    def sha(self) -> str:
        return self.meta.get("sha", "")

    def predict_proba(self, feats: np.ndarray) -> np.ndarray:
        x = (feats - self.mean) / self.std
        if self.kind == "linear":
            w, b = self.weights
            return _sigmoid(x @ w + b).ravel()
        w1, b1, w2, b2 = self.weights
        h = np.maximum(0.0, x @ w1 + b1)
        return _sigmoid(h @ w2 + b2).ravel()

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {f"w{i}": w.astype(np.float32) for i, w in enumerate(self.weights)}
        payload = dict(arrays, mean=self.mean.astype(np.float32), std=self.std.astype(np.float32),
                       layers=np.asarray(self.layers, dtype=np.int64))
        digest = hashlib.sha256(b"".join(np.ascontiguousarray(v).tobytes() for v in payload.values())).hexdigest()
        meta = dict(self.meta, sha=digest[:12])
        info = {"kind": self.kind, "threshold_low": self.threshold_low,
                "threshold_high": self.threshold_high, "meta": meta}
        np.savez(path, info=np.asarray(json.dumps(info)), **payload)
        self.meta = meta

    @classmethod
    def load(cls, path: Path) -> "FaceHead":
        z = np.load(path, allow_pickle=False)
        info = json.loads(str(z["info"]))
        n = len([k for k in z.files if k.startswith("w")])
        return cls(kind=info["kind"], layers=[int(v) for v in z["layers"]], mean=z["mean"], std=z["std"],
                   weights=[z[f"w{i}"] for i in range(n)], threshold_low=float(info["threshold_low"]),
                   threshold_high=float(info["threshold_high"]), meta=info["meta"])


def load_head(path: Optional[Path]) -> Optional[FaceHead]:
    if path is None or not Path(path).exists():
        return None
    return FaceHead.load(Path(path))
