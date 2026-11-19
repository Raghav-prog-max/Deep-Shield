"""Trained classification heads for Module 02 (`face_manipulation` and `synthetic_image`).

The pretrained SigLIP2 backbone stays frozen. We read pooled features from three depths of the
vision encoder (middle, three-quarter and final), optionally L2-normalise per layer to suppress
cross-dataset brightness/contrast shortcuts, standardise them, and apply a compact head trained
by ``scripts/train_m02_head.py`` or ``scripts/train_m02_synth_head.py``.

Supported head architectures (all pure numpy at inference time, zero extra dependencies):
  * ``linear``  — L2-regularised logistic regression: weights = [W, b]
  * ``mlp``     — 1-hidden-layer ReLU MLP: weights = [W1, b1, W2, b2]
  * ``mlp_ens`` — K-fold ensemble of 1-hidden-layer ReLU MLPs (4 arrays per fold), with optional
                  per-layer L2 normalisation and validation-calibrated residual blend with the
                  frozen backbone's own classifier logit.
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
    return 1.0 / (1.0 + np.exp(-np.clip(z, -40.0, 40.0)))


def _logit(p: np.ndarray) -> np.ndarray:
    pc = np.clip(p.astype(np.float32), 1e-5, 1.0 - 1e-5)
    return np.log(pc / (1.0 - pc))


def l2_normalise_blocks(feats: np.ndarray, n_blocks: int) -> np.ndarray:
    """Independently L2-normalise each encoder layer slice to reduce domain magnitude shift."""
    if n_blocks <= 1 or feats.shape[1] % n_blocks != 0:
        norms = np.linalg.norm(feats, axis=1, keepdims=True) + 1e-6
        return (feats / norms).astype(np.float32)
    dim = feats.shape[1] // n_blocks
    parts = []
    for i in range(n_blocks):
        sl = feats[:, i * dim : (i + 1) * dim]
        parts.append(sl / (np.linalg.norm(sl, axis=1, keepdims=True) + 1e-6))
    return np.concatenate(parts, axis=1).astype(np.float32)


@dataclass
class FaceHead:
    kind: str                      # "linear" | "mlp" | "mlp_ens"
    layers: List[int]
    mean: np.ndarray
    std: np.ndarray
    weights: List[np.ndarray]      # linear: [W, b]; mlp: [W1, b1, W2, b2]; mlp_ens: K * [W1, b1, W2, b2]
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

    def predict_logits(self, feats: np.ndarray, base_probs: Optional[np.ndarray] = None) -> np.ndarray:
        f = l2_normalise_blocks(feats, len(self.layers)) if self.meta.get("l2_norm_layers") else feats
        x = (f - self.mean) / self.std
        if self.kind == "linear":
            w, b = self.weights
            z = (x @ w + b).ravel()
        elif self.kind == "mlp":
            w1, b1, w2, b2 = self.weights
            h = np.maximum(0.0, x @ w1 + b1)
            z = (h @ w2 + b2).ravel()
        elif self.kind == "mlp_ens":
            k = len(self.weights) // 4
            zs = []
            for i in range(k):
                w1, b1, w2, b2 = self.weights[4 * i : 4 * (i + 1)]
                h = np.maximum(0.0, x @ w1 + b1)
                zs.append((h @ w2 + b2).ravel())
            z = np.mean(zs, axis=0)
        else:
            raise ValueError(f"Unsupported head kind: {self.kind}")

        alpha = float(self.meta.get("base_logit_alpha", 0.0))
        if alpha > 0.0 and base_probs is not None:
            z = z + alpha * _logit(np.asarray(base_probs, dtype=np.float32)).ravel()
        temp = float(self.meta.get("temperature", 1.0))
        if temp > 0.0 and temp != 1.0 and self.kind == "mlp_ens":
            z = z / temp
        return z.astype(np.float32)

    def predict_proba(self, feats: np.ndarray, base_probs: Optional[np.ndarray] = None) -> np.ndarray:
        return _sigmoid(self.predict_logits(feats, base_probs=base_probs))

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
