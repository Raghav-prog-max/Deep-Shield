"""Evaluation metrics (numpy only): accuracy, precision, recall, F1, ROC-AUC, threshold sweep."""
from __future__ import annotations

from typing import Dict, List, Sequence

import numpy as np


def roc_auc(y_true: Sequence[int], scores: Sequence[float]) -> float:
    """Mann-Whitney U formulation with average ranks for ties."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    n_pos, n_neg = int(y.sum()), int((1 - y).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = s.argsort(kind="mergesort")
    ranks = np.empty(len(s), dtype=float)
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def binary_metrics(y_true: Sequence[int], scores: Sequence[float], threshold: float = 0.5) -> Dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    p = (np.asarray(scores, dtype=float) >= threshold).astype(int)
    tp = int(((p == 1) & (y == 1)).sum()); tn = int(((p == 0) & (y == 0)).sum())
    fp = int(((p == 1) & (y == 0)).sum()); fn = int(((p == 0) & (y == 1)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    return {"threshold": threshold, "n": len(y), "n_positive": int(y.sum()), "n_negative": int((1 - y).sum()),
            "accuracy": (tp + tn) / len(y) if len(y) else 0.0, "precision": prec, "recall": rec, "f1": f1,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def three_band(y_true: Sequence[int], scores: Sequence[float], low: float, high: float) -> Dict[str, float]:
    """Metrics on decided cases only + share left inconclusive (how the engine actually reports)."""
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    decided = (s >= high) | (s <= low)
    out = {"low": low, "high": high, "coverage": float(decided.mean()) if len(s) else 0.0,
           "inconclusive": int((~decided).sum())}
    if decided.any():
        m = binary_metrics(y[decided], s[decided], threshold=high)
        out.update({f"decided_{k}": v for k, v in m.items() if k not in ("threshold",)})
    return out


def sweep(y_true: Sequence[int], scores: Sequence[float], thresholds: List[float] = None) -> List[Dict[str, float]]:
    thresholds = thresholds or [round(t, 2) for t in np.arange(0.05, 1.0, 0.05)]
    return [binary_metrics(y_true, scores, t) for t in thresholds]
