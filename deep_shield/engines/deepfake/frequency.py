"""Frequency-domain (spectral) fingerprint of a face crop or image.

GAN / diffusion up-sampling layers often leave excess or periodic energy in the high spatial
frequencies, visible in the azimuthally averaged power spectrum (Durall et al., CVPR 2020,
"Watch your Up-Convolution"). We fit the natural 1/f fall-off on mid frequencies and measure the
residual in the high band. The score is an UNCALIBRATED indicator, never a verdict.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
from PIL import Image

SIZE = 256


def spectral_fingerprint(img: Image.Image, size: int = SIZE) -> Dict:
    g = np.asarray(img.convert("L").resize((size, size), Image.BICUBIC), dtype=np.float64)
    g -= g.mean()
    win = np.outer(np.hanning(size), np.hanning(size))
    F = np.fft.fftshift(np.fft.fft2(g * win))
    logp = np.log1p(np.abs(F) ** 2)

    c = size // 2
    yy, xx = np.indices((size, size))
    r = np.hypot(xx - c, yy - c).astype(int)
    sums = np.bincount(r.ravel(), logp.ravel())
    cnts = np.bincount(r.ravel())
    prof = (sums / np.maximum(cnts, 1))[1:c]
    rr = np.arange(1, c)

    mid = (rr >= 8) & (rr < c // 2)
    hi = rr >= c // 2
    slope, icpt = np.polyfit(np.log(rr[mid]), prof[mid], 1)
    pred = slope * np.log(rr) + icpt
    resid = prof - pred
    excess = float(resid[hi].mean())
    peak = float(resid[hi].max() - np.median(resid[hi]))
    score = float(1.0 / (1.0 + np.exp(-excess / 0.5)))
    return {
        "method": "azimuthal_power_spectrum",
        "size": size,
        "spectral_slope": round(float(slope), 4),
        "hf_excess": round(excess, 4),
        "hf_peak_prominence": round(peak, 4),
        "indicator_score": round(score, 4),
        "_profile": prof, "_pred": pred, "_rr": rr, "_logspec": logp, "_hi_start": int(c // 2),
    }


def public(fp: Dict) -> Dict:
    return {k: v for k, v in fp.items() if not k.startswith("_")}


def save_spectrum_plot(fp: Dict, path: Path, title: str) -> bool:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return False
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(9, 3.4), dpi=110)
    a1.imshow(fp["_logspec"], cmap="magma")
    a1.set_title("log power spectrum (centred)", fontsize=8)
    a1.axis("off")
    rr = fp["_rr"]
    a2.plot(rr, fp["_profile"], lw=1.4, label="observed")
    a2.plot(rr, fp["_pred"], lw=1.0, ls="--", label="1/f fit (mid band)")
    a2.axvspan(fp["_hi_start"], rr[-1], color="#fde9c8", alpha=0.6, lw=0, label="high band")
    a2.set_xscale("log")
    a2.set_xlabel("spatial frequency (radius)", fontsize=8)
    a2.set_ylabel("mean log power", fontsize=8)
    a2.tick_params(labelsize=7)
    a2.legend(fontsize=7)
    a2.set_title(f"HF excess {fp['hf_excess']:+.2f} · peak {fp['hf_peak_prominence']:.2f} "
                 f"· indicator {fp['indicator_score']:.2f}", fontsize=8)
    fig.suptitle(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
    return True
