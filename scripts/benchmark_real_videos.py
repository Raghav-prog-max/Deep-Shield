"""End-to-end REAL VIDEO benchmark across M02 (Celeb-DF v2 + FaceForensics++/DFD) and M01/M03 (UCF-Crime CCTV).

Evaluates:
  1. M02 full video temporal pipeline (YuNet face detection -> IoU/center tracking -> temporal aggregation)
     comparing our trained head against the raw SigLIP2 baseline across 40 actor-disjoint real vs deepfake MP4s.
  2. M01 (`hustvl/yolos-tiny` + `MultiObjectIoUTracker`) and M03 (rules + PyTorch Temporal Kinematic GRU)
     across 12 real UCF-Crime multi-camera CCTV surveillance videos (6 normal + 6 anomaly).

Writes:
  - reports/video_benchmark.json
  - reports/VIDEO_BENCHMARK_REPORT.md
"""
from __future__ import annotations

import json
import os
import sys
import tempfile
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from deep_shield.engines.deepfake import DeepfakeImageIntegrityEngine, EngineConfig  # noqa: E402
from deep_shield.engines.deepfake.classifiers import _CACHE as M02_CACHE  # noqa: E402
from deep_shield.engines.deepfake.metrics import binary_metrics, roc_auc, three_band  # noqa: E402
from deep_shield.engines.threat import ThreatConfig, ThreatIntelligenceEngine  # noqa: E402
from deep_shield.engines.video_events import VideoEventConfig, VideoEventEngine  # noqa: E402


def compute_ece(y_true, y_score, n_bins: int = 8) -> float:
    y = np.asarray(y_true, dtype=np.float64)
    p = np.asarray(y_score, dtype=np.float64)
    if len(y) == 0:
        return 0.0
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        m = (p >= lo) & (p <= hi if i == n_bins - 1 else p < hi)
        if np.any(m):
            ece += (m.sum() / len(y)) * abs(y[m].mean() - p[m].mean())
    return round(float(ece), 4)


def score_video_set(videos: list[tuple[Path, int, str]], use_head: bool) -> dict:
    M02_CACHE.clear()
    prev = os.environ.get("DEEPSHIELD_M02_FACE_HEAD")
    if not use_head:
        os.environ["DEEPSHIELD_M02_FACE_HEAD"] = "none"
    elif "DEEPSHIELD_M02_FACE_HEAD" in os.environ:
        del os.environ["DEEPSHIELD_M02_FACE_HEAD"]

    per_video = []
    model_id = ""
    thresholds = {}
    try:
        with tempfile.TemporaryDirectory(prefix="ds_vid_bench_") as td:
            cfg = EngineConfig(
                backend="hf",
                evidence_dir=Path(td),
                explain_faces=False,
                evidence_plots=False,
                video_sample_fps=1.5,
                video_max_frames=18,
                video_max_duration_s=15.0,
            )
            eng = DeepfakeImageIntegrityEngine(cfg)
            for vp, lbl, src in videos:
                t0 = time.perf_counter()
                res = eng.analyze(vp, tasks=["face_manipulation"])
                ms = round((time.perf_counter() - t0) * 1000, 1)
                fs = [s for s in res.task_summaries if s.task == "face_manipulation"][0]
                if fs.model and not model_id:
                    model_id = fs.model.model_id
                tr = [f for f in res.findings if f.level == "track" and f.task == "face_manipulation"]
                if tr and not thresholds and tr[0].thresholds:
                    thresholds = tr[0].thresholds
                score = fs.score
                if score is None and tr:
                    score = max(f.score for f in tr if f.score is not None)
                per_video.append({
                    "video": vp.name,
                    "source": src,
                    "label": "fake" if lbl == 1 else "real",
                    "y": lbl,
                    "classification": fs.classification,
                    "score": score,
                    "tracks": len(tr),
                    "sampled_frames": res.media_info.get("sampled_frames", 0),
                    "faces_observed": res.media_info.get("faces_observed", 0),
                    "processing_ms": ms,
                })
    finally:
        M02_CACHE.clear()
        if prev is not None:
            os.environ["DEEPSHIELD_M02_FACE_HEAD"] = prev
        elif "DEEPSHIELD_M02_FACE_HEAD" in os.environ:
            del os.environ["DEEPSHIELD_M02_FACE_HEAD"]

    lo = thresholds.get("low", 0.35)
    hi = thresholds.get("high", 0.65)

    def summarize(subset: list[dict]) -> dict:
        scored = [r for r in subset if r["score"] is not None]
        if not scored:
            return {"n_total": len(subset), "n_scored": 0}
        ys = [r["y"] for r in scored]
        ps = [float(r["score"]) for r in scored]
        bm = binary_metrics(ys, ps, 0.5)
        tb = three_band(ys, ps, lo, hi)
        real_ps = [p for y, p in zip(ys, ps) if y == 0]
        fake_ps = [p for y, p in zip(ys, ps) if y == 1]
        return {
            "n_total": len(subset),
            "n_scored": len(scored),
            "n_fake": int(sum(ys)),
            "n_real": int(len(ys) - sum(ys)),
            "roc_auc": round(float(roc_auc(ys, ps)), 4) if len(set(ys)) == 2 else None,
            "acc@0.5": bm["accuracy"],
            "precision@0.5": bm["precision"],
            "recall@0.5": bm["recall"],
            "f1@0.5": bm["f1"],
            "ece": compute_ece(ys, ps),
            "band_low": lo,
            "band_high": hi,
            "coverage": tb["coverage"],
            "decided_accuracy": tb["decided_accuracy"],
            "mean_real_score": round(float(np.mean(real_ps)), 4) if real_ps else None,
            "mean_fake_score": round(float(np.mean(fake_ps)), 4) if fake_ps else None,
            "mean_ms": round(float(np.mean([r["processing_ms"] for r in scored])), 1),
        }

    return {
        "model_id": model_id,
        "thresholds": thresholds,
        "pooled": summarize(per_video),
        "celebdf_v2": summarize([r for r in per_video if r["source"] == "celebdf_v2"]),
        "dfd_ffpp": summarize([r for r in per_video if r["source"] == "dfd_ffpp"]),
        "per_video": per_video,
    }


def benchmark_ucf_surveillance(ucf_dir: Path) -> dict:
    zones = [{"zone_id": "scene_center", "polygon": [[0.15, 0.15], [0.85, 0.15], [0.85, 0.85], [0.15, 0.85]]}]
    clips = []
    for lbl in ("normal", "anomaly"):
        d = ucf_dir / lbl
        if d.is_dir():
            for p in sorted(d.glob("*.mp4")):
                clips.append((p, lbl))

    rows = []
    with tempfile.TemporaryDirectory(prefix="ds_ucf_bench_") as td:
        eng1 = ThreatIntelligenceEngine(ThreatConfig(evidence_dir=Path(td) / "m01", sample_fps=1.0, max_duration_s=12.0))
        eng3 = VideoEventEngine(VideoEventConfig(evidence_dir=Path(td) / "m03", sample_fps=2.0, max_duration_s=12.0))
        for p, lbl in clips:
            t0 = time.perf_counter()
            r1 = eng1.analyze(p, zones=zones)
            m01_ms = round((time.perf_counter() - t0) * 1000, 1)
            t1 = time.perf_counter()
            r3 = eng3.analyze(p, zones=zones)
            m03_ms = round((time.perf_counter() - t1) * 1000, 1)

            gru_findings = [f for f in r3.findings if f.task == "temporal_gru"]
            gru_scores = [f.score for f in gru_findings if f.score is not None]
            gru_flagged = [f for f in gru_findings if f.classification == "kinematic_anomaly_candidate"]
            event_types = sorted({f.classification for f in r3.findings if f.task != "temporal_gru"})
            rows.append({
                "video": p.name,
                "label": lbl,
                "m01_status": r1.status,
                "m01_detections": len([f for f in r1.findings if f.task == "object_detection"]),
                "m01_ms": m01_ms,
                "m03_status": r3.status,
                "m03_tracked_persons": r3.media_info.get("tracked_persons", 0),
                "m03_gru_tracks": len(gru_findings),
                "m03_gru_flagged": len(gru_flagged),
                "m03_gru_max_score": round(max(gru_scores), 4) if gru_scores else None,
                "m03_events": event_types,
                "m03_ms": m03_ms,
            })
            print(f"  [UCF/{lbl}] {p.name}: M01_det={rows[-1]['m01_detections']} tracks={rows[-1]['m03_tracked_persons']} GRU_max={rows[-1]['m03_gru_max_score']} events={event_types}")

    norm_rows = [r for r in rows if r["label"] == "normal"]
    anom_rows = [r for r in rows if r["label"] == "anomaly"]
    return {
        "n_clips": len(rows),
        "n_normal": len(norm_rows),
        "n_anomaly": len(anom_rows),
        "normal_clips_with_detections": sum(1 for r in norm_rows if r["m01_detections"] > 0),
        "anomaly_clips_with_detections": sum(1 for r in anom_rows if r["m01_detections"] > 0),
        "normal_gru_flagged_clips": sum(1 for r in norm_rows if r["m03_gru_flagged"] > 0),
        "anomaly_gru_flagged_clips": sum(1 for r in anom_rows if r["m03_gru_flagged"] > 0),
        "clips": rows,
    }


def write_markdown(report: dict, out_md: Path) -> None:
    tr = report["m02_real_deepfake_videos"]["trained_head"]
    bs = report["m02_real_deepfake_videos"]["original_siglip2_baseline"]
    ucf = report["m01_m03_ucf_crime_cctv"]

    def _fmt(v):
        return "-" if v is None else f"{v:.4f}" if isinstance(v, float) else str(v)

    lines = [
        "# DEEP SHIELD — Real Video Benchmark Report (Celeb-DF v2, FaceForensics++/DFD & UCF-Crime)",
        "",
        f"**Generated:** {report['created_at']}",
        f"**M02 Trained Video Model:** `{tr['model_id']}` (thresholds `{tr['thresholds']}`)",
        f"**M02 Baseline Model:** `{bs['model_id']}`",
        "",
        "---",
        "",
        "## 1. Module 02 — Real Deepfake Video Benchmark (40 Actor-Disjoint Held-Out `.mp4` Videos)",
        "",
        "All videos were evaluated through the full `DeepfakeImageIntegrityEngine.analyze(clip.mp4)` video pipeline "
        "(timestamped frame sampling → YuNet face detection → multi-frame face tracking → track temporal mean aggregation). "
        "Every test video actor/identity is strictly disjoint from training.",
        "",
        "| Video Benchmark Split | Videos Scored (`Fake/Real`) | Baseline SigLIP2 ROC-AUC | Trained Head ROC-AUC | Acc @ 0.5 | F1 @ 0.5 | ECE | Decided Acc | Mean Real Score | Mean Fake Score |",
        "| :--- | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |",
    ]
    for key, title in [("pooled", "**Pooled (40 Real Videos)**"), ("celebdf_v2", "Celeb-DF v2 (Official Test)"), ("dfd_ffpp", "FaceForensics++ / Google DFD")]:
        a, b = tr[key], bs[key]
        lines.append(
            f"| {title} | {a.get('n_fake', 0)} / {a.get('n_real', 0)} | {_fmt(b.get('roc_auc'))} | **{_fmt(a.get('roc_auc'))}** | "
            f"{_fmt(a.get('acc@0.5'))} | {_fmt(a.get('f1@0.5'))} | {_fmt(a.get('ece'))} | {_fmt(a.get('decided_accuracy'))} | "
            f"{_fmt(a.get('mean_real_score'))} | {_fmt(a.get('mean_fake_score'))} |"
        )

    lines.extend([
        "",
        "---",
        "",
        "## 2. Modules 01 & 03 — Real Multi-Camera CCTV Surveillance Benchmark (`UCF-Crime`)",
        "",
        f"Evaluated `hustvl/yolos-tiny` + `MultiObjectIoUTracker` (M01) and calibrated rules + PyTorch Temporal Kinematic GRU (M03) across **{ucf['n_clips']} real UCF-Crime CCTV videos** ({ucf['n_normal']} `Testing_Normal`, {ucf['n_anomaly']} real anomaly categories).",
        "",
        "| Clip | Ground Truth | M01 Detections | M03 Tracks | GRU Max Score | GRU Flagged Tracks | M03 Events Triggered |",
        "| :--- | :---: | :---: | :---: | :---: | :---: | :--- |",
    ])
    for c in ucf["clips"]:
        lines.append(
            f"| `{c['video']}` | `{c['label']}` | {c['m01_detections']} | {c['m03_tracked_persons']} | "
            f"{_fmt(c['m03_gru_max_score'])} | {c['m03_gru_flagged']} | {', '.join(c['m03_events']) or 'none'} |"
        )

    lines.extend([
        "",
        "> **Honest CCTV Note:** M01 + M03 operate on person/bag kinematic tracks only (no violence/weapon/intent classification). "
        "Low-resolution CCTV angles, rapid camera cuts, or non-pedestrian anomalies (e.g., distant indoor shoplifting or occlusion) "
        "do not always exhibit high-speed running kinematics; every alert remains advisory for human review.",
        "",
    ])
    out_md.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    df_root = ROOT / "eval" / "deepfake_videos"
    videos = []
    for src, folder in [("celebdf_v2", df_root / "celebdf"), ("dfd_ffpp", df_root / "dfd")]:
        for lbl_name, lbl_int in [("real", 0), ("fake", 1)]:
            d = folder / lbl_name
            if d.is_dir():
                for p in sorted(d.glob("*.mp4")):
                    videos.append((p, lbl_int, src))

    print(f"Benchmarking M02 on {len(videos)} real actor-disjoint held-out videos (trained head)...")
    trained_res = score_video_set(videos, use_head=True)
    print("  Trained head pooled:", trained_res["pooled"])
    print("  Trained head Celeb-DF v2:", trained_res["celebdf_v2"])
    print("  Trained head FF++/DFD:", trained_res["dfd_ffpp"])

    print(f"\nBenchmarking M02 on {len(videos)} real videos (original SigLIP2 baseline)...")
    base_res = score_video_set(videos, use_head=False)
    print("  Baseline pooled:", base_res["pooled"])

    print("\nBenchmarking M01 + M03 on real UCF-Crime CCTV surveillance videos...")
    ucf_res = benchmark_ucf_surveillance(ROOT / "eval" / "surveillance_ucf")

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "m02_real_deepfake_videos": {
            "trained_head": trained_res,
            "original_siglip2_baseline": base_res,
        },
        "m01_m03_ucf_crime_cctv": ucf_res,
    }
    out_json = ROOT / "reports" / "video_benchmark.json"
    out_md = ROOT / "reports" / "VIDEO_BENCHMARK_REPORT.md"
    out_json.write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_markdown(report, out_md)
    print(f"\nSaved {out_json} and {out_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
