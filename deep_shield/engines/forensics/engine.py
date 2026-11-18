"""DEEP SHIELD Module 04 — Visual Forensics engine.

    from deep_shield.engines.forensics import VisualForensicsEngine
    r = VisualForensicsEngine().analyze("photo.jpg")                 # all image tasks
    r = VisualForensicsEngine().analyze("clip.mp4", frames=[120])    # selected frame(s) of a video

Tasks
  copy_move      keypoint self-matching -> candidate cloned region pairs (validated on synthetic clones)
  ela_noise      Error Level Analysis + local noise-level map (visual, no verdict)
  jpeg_analysis  quantisation-table quality, 8x8 grid alignment, JPEG ghosts (observations + maps)
  metadata       EXIF camera / software / time fields (observations)
Video: copy_move + ela_noise on selected frames (default: 3 evenly spaced keyframes); the frame
index / timestamp are kept so findings link to Modules 01-03 in the cross-module layer.
"""
from __future__ import annotations

import json
import logging
import os
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Sequence

import numpy as np

from deep_shield.common.schema import (BBox, Finding, MediaType, ModelInfo, ModuleResult, ScoreType,
                                       TaskSummary)
from deep_shield.engines.deepfake import manipulation as manip
from deep_shield.engines.deepfake.evidence import EvidenceStore
from deep_shield.engines.deepfake.media import (MediaError, load_image, media_kind, probe_video, read_frames,
                                                sha256_file)

from . import copymove, jpeg

log = logging.getLogger("deep_shield.m04")

MODULE_NAME = "M04_visual_forensics"
ENGINE_VERSION = "m04-0.1.0"
TASK_CM, TASK_EN, TASK_JPEG, TASK_META = "copy_move", "ela_noise", "jpeg_analysis", "metadata"
ALL_TASKS = (TASK_CM, TASK_EN, TASK_JPEG, TASK_META)
VIDEO_TASKS = (TASK_CM, TASK_EN)

CM_LIMITS = [
    "Candidate cloned regions for human review, not proof. Internal synthetic check: 27/28 clones of "
    "120-200 px and 16/28 clones of 50-110 px found; 2/14 untouched photos flagged, both highly "
    "repetitive textures (grass, gravel).",
    "Repeated real structures (tiles, windows, text, foliage) can match; flat regions (sky, walls) "
    "have no keypoints and cannot be checked.",
]
EN_LIMITS = [
    "Exploratory visualizations for human review; no automatic verdict.",
    "ELA is most informative for JPEG sources; texture, edges and dark/flat areas vary naturally.",
]
JPEG_LIMITS = [
    "Observations, not verdicts. Ghost maps are informative only when a region was previously saved at a "
    "LOWER quality than the final save; a shifted 8x8 grid also results from ordinary cropping.",
    "Flat or saturated areas (pure black/white) appear bright in every ghost map; only a region that is "
    "bright at ONE quality and dark at the others is meaningful.",
]
META_LIMITS = ["Metadata is easily stripped or edited; its absence or presence is a lead, never proof."]
RESAMPLING_NOTE = ("Resampling (resize) detection was evaluated and excluded: it did not separate resized "
                   "from original images after JPEG compression in our checks.")


ROOT = Path(__file__).resolve().parents[3]


def _load_m04_params() -> dict:
    p = ROOT / "models" / "m04_forensics_params.json"
    if p.is_file():
        try:
            return json.loads(p.read_text())
        except Exception:  # noqa: BLE001
            pass
    return {}


def _rooted(value: str) -> Path:
    p = Path(value).expanduser()
    return p if p.is_absolute() else (ROOT / p).resolve()


@dataclass
class ForensicsConfig:
    evidence_dir: Path = field(default_factory=lambda: _rooted(os.getenv("DEEPSHIELD_M04_EVIDENCE_DIR", "evidence_store/m04")))
    keyframes: int = field(default_factory=lambda: int(_load_m04_params().get("parameters", {}).get("video_keyframes", 3)))
    ela_quality: int = field(default_factory=lambda: int(_load_m04_params().get("parameters", {}).get("ela_jpeg_quality", 90)))
    sift_ratio: float = field(default_factory=lambda: float(_load_m04_params().get("parameters", {}).get("sift_ratio_threshold", 0.65)))
    min_keypoint_dist: float = field(default_factory=lambda: float(_load_m04_params().get("parameters", {}).get("min_keypoint_distance_px", 28.0)))
    ransac_min_inliers: int = field(default_factory=lambda: int(_load_m04_params().get("parameters", {}).get("ransac_min_inliers", 6)))
    sift_contrast: float = field(default_factory=lambda: float(_load_m04_params().get("parameters", {}).get("sift_contrast_threshold", 0.006)))
    max_upload_mb: int = field(default_factory=lambda: int(os.getenv("DEEPSHIELD_MAX_UPLOAD_MB", "200")))
    keep_uploads: bool = field(default_factory=lambda: os.getenv("DEEPSHIELD_KEEP_UPLOADS", "0") == "1")


def _bbox(b) -> BBox:
    x, y, w, h = b
    return BBox(x=int(x), y=int(y), w=int(w), h=int(h))


def _model(task: str, mid: str) -> ModelInfo:
    return ModelInfo(model_id=f"classical:{mid}", task=task, backend="opencv/numpy")


class VisualForensicsEngine:
    def __init__(self, cfg: Optional[ForensicsConfig] = None):
        self.cfg = cfg or ForensicsConfig()
        self.params = _load_m04_params()

    def health(self):
        import cv2
        return {"module": MODULE_NAME, "engine_version": ENGINE_VERSION, "tasks": list(ALL_TASKS),
                "copy_move_detector": "sift" if hasattr(cv2, "SIFT_create") else "orb",
                "calibrated_parameters": self.params.get("parameters"),
                "benchmark_metrics": self.params.get("benchmark_metrics"),
                "note": RESAMPLING_NOTE}

    def analyze(self, path, tasks: Optional[Sequence[str]] = None, frames: Optional[Sequence[int]] = None,
                source_name: Optional[str] = None) -> ModuleResult:
        t0 = time.perf_counter()
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(path)
        kind = media_kind(path)
        explicit = bool(tasks)
        tasks = list(dict.fromkeys(tasks or (ALL_TASKS if kind == "image" else VIDEO_TASKS)))
        bad = [t for t in tasks if t not in ALL_TASKS]
        if bad:
            raise ValueError(f"Unknown task(s) {bad}; valid: {list(ALL_TASKS)}")
        sha = sha256_file(path)
        res = ModuleResult(module=MODULE_NAME, engine_version=ENGINE_VERSION, media_id=sha[:16], media_sha256=sha,
                           media_type=MediaType(kind), source_name=source_name or path.name,
                           status="completed", tasks_requested=tasks)
        store = EvidenceStore(self.cfg.evidence_dir, res.media_id)
        try:
            if kind == "image":
                bgr, _ = load_image(path)
                res.media_info = {"width": bgr.shape[1], "height": bgr.shape[0]}
                self._frame_tasks(bgr, tasks, store, res, MediaType.IMAGE)
                if TASK_JPEG in tasks:
                    self._jpeg(path, bgr, store, res)
                if TASK_META in tasks:
                    self._metadata(path, res)
            else:
                for t in tasks:
                    if t not in VIDEO_TASKS:
                        res.task_summaries.append(TaskSummary(task=t, classification="not_applicable_video_input",
                                                              basis="file-level analysis; not meaningful for video codecs"))
                self._video(path, [t for t in tasks if t in VIDEO_TASKS], frames, store, res)
            res.warnings.append(RESAMPLING_NOTE)
        except MediaError as exc:
            res.status, res.error = "failed", str(exc)
        except Exception as exc:  # noqa: BLE001
            log.error("M04 analysis failed: %s", traceback.format_exc())
            res.status, res.error = "failed", f"{type(exc).__name__}: {exc}"
        res.processing_ms = int((time.perf_counter() - t0) * 1000)
        return res

    # ------------------------------------------------------------------ per image / frame
    def _frame_tasks(self, bgr, tasks, store, res, mtype, fi: Optional[int] = None, ts: Optional[float] = None):
        pre = f"f{fi:06d}_" if fi is not None else ""
        level = "frame" if fi is not None else "image"
        where = f"frame {fi} ({ts:.2f}s)" if fi is not None else "image"
        extra = (["Video frame: codec compression makes ELA/noise maps less informative."] if fi is not None else [])

        if TASK_CM in tasks:
            cm = copymove.detect(
                bgr,
                ratio=self.cfg.sift_ratio,
                min_dist=self.cfg.min_keypoint_dist,
                min_inliers=self.cfg.ransac_min_inliers,
                contrast=self.cfg.sift_contrast,
            )
            ref = store.save_bgr(copymove.render(bgr, cm), f"{pre}copy_move.jpg", "copy_move_matches",
                                 f"Copy-move keypoint matches, {where} (green = source, red = target)")
            pub = copymove.public(cm)
            model = _model(TASK_CM, "sift_self_matching+ransac")
            if cm["clusters"]:
                for i, c in enumerate(pub["clusters"]):
                    res.findings.append(Finding(
                        module=MODULE_NAME, task=TASK_CM, level=level, media_id=res.media_id, media_type=mtype,
                        classification="copy_move_candidate", score=None, score_type=ScoreType.FORENSIC_INDICATOR,
                        region=_bbox(c["target_bbox"]), track_id=f"{pre}C{i + 1}", frame_index=fi, timestamp_s=ts,
                        model=model, explanation={**{k: v for k, v in pub.items() if k != "clusters"}, **c},
                        evidence=[ref], limitations=CM_LIMITS + extra))
            else:
                res.findings.append(Finding(
                    module=MODULE_NAME, task=TASK_CM, level=level, media_id=res.media_id, media_type=mtype,
                    classification="no_copy_move_candidates", score=None, score_type=ScoreType.FORENSIC_INDICATOR,
                    frame_index=fi, timestamp_s=ts, model=model, explanation=pub, evidence=[ref],
                    limitations=CM_LIMITS + extra))

        if TASK_EN in tasks:
            r = manip.analyse(bgr, jpeg_quality=self.cfg.ela_quality)
            _, _, panel = manip.render(r)
            ref = store.save_bgr(panel, f"{pre}ela_noise_panel.jpg", "integrity_panel",
                                 f"Input | ELA | noise-level map, {where}")
            res.findings.append(Finding(
                module=MODULE_NAME, task=TASK_EN, level=level, media_id=res.media_id, media_type=mtype,
                classification="exploratory_visualization", score=None, score_type=ScoreType.FORENSIC_INDICATOR,
                frame_index=fi, timestamp_s=ts, model=_model(TASK_EN, "ela+immerkaer_noise"),
                explanation=manip.public(r), evidence=[ref], limitations=EN_LIMITS + extra))

        if fi is None:   # image: one summary per task
            self._summaries(tasks, res, where_desc="image")

    def _summaries(self, tasks, res, where_desc: str):
        if TASK_CM in tasks:
            n = sum(1 for f in res.findings if f.task == TASK_CM and f.classification == "copy_move_candidate")
            res.task_summaries.append(TaskSummary(
                task=TASK_CM, classification="copy_move_candidates_found" if n else "no_copy_move_candidates",
                basis=(f"{n} candidate cloned region pair(s) in {where_desc}; human review required" if n
                       else f"no consistent cloned-region matches in {where_desc}"),
                score_type=ScoreType.FORENSIC_INDICATOR, model=_model(TASK_CM, "sift_self_matching+ransac")))
        if TASK_EN in tasks:
            res.task_summaries.append(TaskSummary(
                task=TASK_EN, classification="exploratory_visualization",
                basis=f"ELA + noise-level maps for {where_desc}; for human review, no automatic verdict",
                score_type=ScoreType.FORENSIC_INDICATOR, model=_model(TASK_EN, "ela+immerkaer_noise")))

    # ------------------------------------------------------------------ file-level (images)
    def _jpeg(self, path, bgr, store, res):
        q = jpeg.quality_from_tables(path)
        grid = jpeg.grid_alignment(bgr)
        g = jpeg.ghosts(bgr)
        # show qualities BELOW the file's own last save: an earlier, lower-quality paste appears there
        top = min(90, (q.get("estimated_quality") or 95) - 5)
        show = sorted({int(5 * round(v / 5)) for v in np.linspace(50, max(55, top), 4)})
        ref = store.save_bgr(jpeg.render_ghosts(g, show), "jpeg_ghosts.jpg", "jpeg_ghosts",
                             f"JPEG ghost maps at q{show} (bright = region matches an earlier save at that quality)")
        obs = jpeg.observations(q, grid)
        model = _model(TASK_JPEG, "quant_tables+grid+ghosts")
        res.findings.append(Finding(
            module=MODULE_NAME, task=TASK_JPEG, level="image", media_id=res.media_id, media_type=MediaType.IMAGE,
            classification="observations", score=None, score_type=ScoreType.FORENSIC_INDICATOR, model=model,
            explanation={"quantisation": q, "grid": grid, "ghost_qualities_shown": show, "observations": obs},
            evidence=[ref], limitations=JPEG_LIMITS))
        res.task_summaries.append(TaskSummary(task=TASK_JPEG, classification="observations",
                                              basis="; ".join(obs) or "no JPEG observations",
                                              score_type=ScoreType.FORENSIC_INDICATOR, model=model))

    def _metadata(self, path, res):
        meta = manip.read_metadata(path)
        model = _model(TASK_META, "exif")
        res.findings.append(Finding(
            module=MODULE_NAME, task=TASK_META, level="image", media_id=res.media_id, media_type=MediaType.IMAGE,
            classification="observations", score=None, score_type=ScoreType.FORENSIC_INDICATOR, model=model,
            explanation=meta, limitations=META_LIMITS))
        fields = ", ".join(f"{k}={meta[k]}" for k in ("make", "model", "software") if meta.get(k))
        basis = "; ".join([x for x in [fields] + meta.get("notes", []) if x]) or "no metadata observations"
        res.task_summaries.append(TaskSummary(task=TASK_META, classification="observations", basis=basis,
                                              score_type=ScoreType.FORENSIC_INDICATOR, model=model))

    # ------------------------------------------------------------------ video
    def _video(self, path, tasks, frames, store, res):
        meta = probe_video(path, 2.0)
        n = meta.frame_count
        if frames:
            idx = sorted({int(f) for f in frames if f >= 0 and (not n or f < n)})
        else:
            if not n:
                raise MediaError("Video frame count unknown; pass explicit frame indices.")
            k = max(1, self.cfg.keyframes)
            idx = sorted({int(round(x)) for x in np.linspace(0, n - 1, k + 2)[1:-1]})
        got = read_frames(path, idx)
        if not got:
            raise MediaError("Requested frames could not be decoded.")
        res.media_info = {**meta.as_dict(), "frames_analysed": sorted(got)}
        for fi in sorted(got):
            self._frame_tasks(got[fi], tasks, store, res, MediaType.VIDEO, fi=fi, ts=round(fi / meta.fps, 3))
        self._summaries(tasks, res, where_desc=f"{len(got)} frame(s) {sorted(got)}")
