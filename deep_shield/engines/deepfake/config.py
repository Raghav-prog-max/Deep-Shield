"""Module 02 configuration: model registry, thresholds and runtime settings.

Every value can be overridden with an environment variable so the demo machine,
the DGX node and a CPU laptop can share one codebase.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple

MODULE_NAME = "M02_deepfake_image_integrity"
ENGINE_VERSION = "m02-0.3.0"

TASK_FACE = "face_manipulation"      # model: face deepfake classifier (+ suspicious-region heatmap)
TASK_SYNTH = "synthetic_image"       # model: whole-image AI-generation check (experimental, opt-in)
TASK_FREQ = "frequency_analysis"     # classical: spectral fingerprint of face crops (indicator only)
TASK_MANIP = "image_manipulation"    # classical: ELA + noise-level maps + metadata (visual only)
ALL_TASKS = (TASK_FACE, TASK_SYNTH, TASK_FREQ, TASK_MANIP)
SCORED_TASKS = (TASK_FACE, TASK_SYNTH)
IMAGE_ONLY_TASKS = (TASK_FREQ, TASK_MANIP)
DEFAULT_TASKS = tuple(t for t in os.getenv(
    "DEEPSHIELD_M02_DEFAULT_TASKS", f"{TASK_FACE},{TASK_MANIP},{TASK_FREQ}").split(",") if t)

FREQ_LIMITATIONS = (
    "Visual aid only: the spectral high-frequency indicator did not separate real from StyleGAN faces "
    "in our internal check (ROC-AUC 0.52, 50 faces), so no score or verdict is reported. "
    "Raw spectral metrics are kept in the explanation for transparency.",
    "JPEG compression, resizing, sharpening and sensor noise all change the high-frequency band; "
    "compare against known-real images from the same source before interpreting.",
)
MANIP_LIMITATIONS = (
    "Exploratory visualizations for human review; no automatic manipulation verdict or region flags "
    "(automatic flagging was unreliable in our checks: missed pastes and flagged textured areas).",
    "ELA is most informative for JPEG sources; texture, edges and dark/flat areas vary naturally.",
    "Missing metadata is common after messaging/social-media upload and is not evidence.",
    "Localised splicing/copy-move analysis with region matching is provided by Module 04.",
)


def _env_float(name: str, default: float) -> float:
    v = os.getenv(name)
    return float(v) if v not in (None, "") else default


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name)
    return int(v) if v not in (None, "") else default


@dataclass(frozen=True)
class ClassifierSpec:
    task: str
    model_id: str
    positive_labels: Tuple[str, ...]      # labels whose probabilities sum to the task score
    negative_labels: Tuple[str, ...]
    license: str
    source_url: str
    threshold_low: float                  # score <= low  -> "not indicated"
    threshold_high: float                 # score >= high -> "indicated"; in between -> inconclusive
    verdicts: Tuple[str, str, str]        # (indicated, inconclusive, not_indicated)
    limitations: Tuple[str, ...]


FACE_SPEC = ClassifierSpec(
    task=TASK_FACE,
    model_id=os.getenv("DEEPSHIELD_M02_FACE_MODEL", "prithivMLmods/open-deepfake-detection"),
    positive_labels=tuple(os.getenv("DEEPSHIELD_M02_FACE_POS", "Fake").split(",")),
    negative_labels=tuple(os.getenv("DEEPSHIELD_M02_FACE_NEG", "Real").split(",")),
    license="apache-2.0",
    source_url="https://huggingface.co/prithivMLmods/open-deepfake-detection",
    threshold_low=_env_float("DEEPSHIELD_M02_FACE_LOW", 0.35),
    threshold_high=_env_float("DEEPSHIELD_M02_FACE_HIGH", 0.65),
    verdicts=("face_manipulation_indicated", "inconclusive", "no_face_manipulation_indicated"),
    limitations=(
        "SigLIP2 classifier fine-tuned on the OpenDeepfake-Preview dataset; the model author labels it "
        "experimental. Internal check on 50 FFHQ/StyleGAN faces: ROC-AUC 0.96 (not a formal benchmark; "
        "possible overlap with training sources).",
        "Applied to detected face crops only; low-resolution, heavily compressed, occluded or "
        "profile faces reduce reliability. Face-swap video deepfakes were not evaluated.",
        "Score is an uncalibrated model output, not a probability that the face was manipulated.",
        "The system does not identify or recognise individuals.",
    ),
)

SYNTH_SPEC = ClassifierSpec(
    task=TASK_SYNTH,
    model_id=os.getenv("DEEPSHIELD_M02_SYNTH_MODEL", "prithivMLmods/open-deepfake-detection"),
    positive_labels=tuple(os.getenv("DEEPSHIELD_M02_SYNTH_POS", "Fake").split(",")),
    negative_labels=tuple(os.getenv("DEEPSHIELD_M02_SYNTH_NEG", "Real").split(",")),
    license="apache-2.0",
    source_url="https://huggingface.co/prithivMLmods/open-deepfake-detection",
    threshold_low=_env_float("DEEPSHIELD_M02_SYNTH_LOW", 0.25),
    threshold_high=_env_float("DEEPSHIELD_M02_SYNTH_HIGH", 0.85),
    verdicts=("ai_generation_indicated", "inconclusive", "no_ai_generation_indicated"),
    limitations=(
        "EXPERIMENTAL: whole-image AI-generation check. On a mixed web sample (art, logos, photos) "
        "neither candidate model separated real from AI images reliably (ROC-AUC ~0.5); a wide "
        "inconclusive band (0.25-0.85) is used so the system rarely makes a definite claim.",
        "Best suited to photographic images; artwork, graphics and screenshots are out of scope.",
        "Assesses whole-image generation only. Localised splicing / copy-move tampering is "
        "handled by Module 04 (Visual Forensics).",
        "Score is an uncalibrated model output, not a probability that the image is synthetic.",
    ),
)

SPECS = {TASK_FACE: FACE_SPEC, TASK_SYNTH: SYNTH_SPEC}


@dataclass
class EngineConfig:
    backend: str = field(default_factory=lambda: os.getenv("DEEPSHIELD_M02_BACKEND", "hf"))  # hf | stub
    device: str = field(default_factory=lambda: os.getenv("DEEPSHIELD_DEVICE", "auto"))      # auto | cpu | cuda | cuda:0
    models_dir: Path = field(default_factory=lambda: Path(os.getenv("DEEPSHIELD_MODELS_DIR", "models")))
    evidence_dir: Path = field(default_factory=lambda: Path(os.getenv("DEEPSHIELD_EVIDENCE_DIR", "evidence_store/m02")))
    yunet_path: Optional[Path] = field(default_factory=lambda: Path(p) if (p := os.getenv("DEEPSHIELD_YUNET_PATH")) else None)
    haar_path: Optional[Path] = field(default_factory=lambda: Path(p) if (p := os.getenv("DEEPSHIELD_HAAR_PATH")) else None)

    # face detection
    face_score_threshold: float = field(default_factory=lambda: _env_float("DEEPSHIELD_M02_FACE_DET_THR", 0.75))
    face_min_size: int = field(default_factory=lambda: _env_int("DEEPSHIELD_M02_FACE_MIN", 48))
    face_margin: float = 0.25
    max_faces_per_frame: int = 5

    # video
    video_sample_fps: float = field(default_factory=lambda: _env_float("DEEPSHIELD_M02_SAMPLE_FPS", 2.0))
    video_max_frames: int = field(default_factory=lambda: _env_int("DEEPSHIELD_M02_MAX_FRAMES", 64))
    video_max_duration_s: float = field(default_factory=lambda: _env_float("DEEPSHIELD_M02_MAX_DURATION", 120.0))
    synthetic_keyframes: int = 6
    track_iou: float = 0.3
    track_center_dist: float = 0.6      # also link if centres move < 0.6 x face size
    track_max_gap: int = 3              # sampled frames a track may skip
    min_track_observations: int = 3     # fewer -> track verdict forced to inconclusive
    evidence_frames_per_track: int = 3

    # explainability / classical analyses
    explain_faces: bool = field(default_factory=lambda: os.getenv("DEEPSHIELD_M02_EXPLAIN", "1") == "1")
    occlusion_grid: int = 5
    evidence_plots: bool = True       # spectrum charts / maps; evaluation turns these off for speed
    ela_quality: int = 90

    # inference / io
    batch_size: int = field(default_factory=lambda: _env_int("DEEPSHIELD_M02_BATCH", 16))
    max_upload_mb: int = field(default_factory=lambda: _env_int("DEEPSHIELD_MAX_UPLOAD_MB", 200))
    keep_uploads: bool = field(default_factory=lambda: os.getenv("DEEPSHIELD_KEEP_UPLOADS", "0") == "1")
