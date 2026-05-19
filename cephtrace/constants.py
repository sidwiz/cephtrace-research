"""Pipeline constants for CephTrace v4 inference.

All numerical values are extracted verbatim from the production codebase and
cross-referenced against:

- `ML-v4-Pipeline-Spec.md` (architecture overview)
- `ML-v4-Stage0-Implementation-Spec.md` (per-substage tensor shapes + values)
- `ML-v4-Training-Log.md` (Run 03 training conventions)
- The Phase 0E training notebook (`CONFIDENCE_TIERS`, `make_gaussian`)

Do not change these values without re-exporting the ONNX models — they are
tightly coupled to the trained weights.
"""

from __future__ import annotations

from typing import Tuple

# ── Landmark orderings ───────────────────────────────────────────────────────
#
# Three orderings appear in Stage 0 / Stage 1 code. They MUST NOT be confused.
#
# CANONICAL_25  — Stage 1 heatmap channel index, attention map channel index,
#                 the public output ordering. This is the single source of
#                 truth for "the 25 landmarks".
# ANCHOR_NAMES  — Phase 0D output, Phase 0E MLP input order (7 anchors).
# DERIVED_NAMES — Phase 0E MLP output order (18 derived landmarks).

CANONICAL_25: Tuple[str, ...] = (
    "S", "N", "Or", "Po", "ANS", "PNS", "A", "B", "Pog", "Gn",
    "Me", "Go", "Ar", "Co", "U1_tip", "U1_root", "L1_tip", "L1_root",
    "UL", "LL", "Pm", "Ba", "Pog_soft", "Sn", "Prn",
)

ANCHOR_NAMES: Tuple[str, ...] = (
    "S", "N", "Me", "ANS", "Go", "Pog", "Prn",
)

DERIVED_NAMES: Tuple[str, ...] = (
    "Or", "Po", "PNS", "A", "B", "Gn", "Ar", "Co",
    "U1_tip", "U1_root", "L1_tip", "L1_root",
    "UL", "LL", "Pm", "Ba", "Pog_soft", "Sn",
)

NUM_LANDMARKS: int = len(CANONICAL_25)  # 25
NUM_ANCHORS: int = len(ANCHOR_NAMES)    # 7
NUM_DERIVED: int = len(DERIVED_NAMES)   # 18

# Index lookup tables — useful when reshuffling MLP output into CANONICAL_25
# attention channel order without recomputing on every call.
CANONICAL_INDEX: dict = {name: i for i, name in enumerate(CANONICAL_25)}
ANCHOR_TO_CANONICAL: dict = {
    name: CANONICAL_INDEX[name] for name in ANCHOR_NAMES
}
DERIVED_TO_CANONICAL: dict = {
    name: CANONICAL_INDEX[name] for name in DERIVED_NAMES
}


# ── Confidence tiers and per-landmark Gaussian σ (pixels at 256×256) ─────────
#
# Source: Phase 0E training notebook Cell 10 `CONFIDENCE_TIERS`, cross-
# validated against `attention_batch_3.npz['dental_101']`.
#
# These σ values control how strongly each attention prior biases Stage 1
# toward the predicted location. Smaller σ → tighter Gaussian → stronger prior.
# Landmarks Phase 0D can anchor confidently (S, N, Prn, ...) get small σ;
# landmarks with high inherent anatomical ambiguity (PNS, Ba) get loose σ so
# Stage 1 retains freedom to override the prior.

CONFIDENCE_TIERS: dict = {
    # tier name → (σ in pixels at 256×256, list of landmarks)
    "high_5":   (5.0,  ["Prn"]),
    "high_6":   (6.0,  ["S", "N"]),
    "high_7":   (7.0,  ["Me", "ANS"]),
    "med_8":    (8.0,  ["UL", "LL", "Sn"]),
    "med_10":   (10.0, ["Pog", "Gn", "Pog_soft"]),
    "med_12":   (12.0, ["Go", "Or", "A", "Pm", "U1_tip", "L1_tip"]),
    "med_13":   (13.0, ["Ar"]),
    "low_18":   (18.0, ["Co", "U1_root", "L1_root"]),
    "low_20":   (20.0, ["Po", "B"]),
    "low_22":   (22.0, ["PNS", "Ba"]),
}

# Flat per-landmark σ lookup (pixels at 256×256).
SIGMA_BY_LANDMARK: dict = {
    name: sigma
    for sigma, landmarks in (
        (tier[0], tier[1]) for tier in CONFIDENCE_TIERS.values()
    )
    for name in landmarks
}

# Tier label per landmark (for UI rendering: green / blue / orange).
def _build_tier_label_map() -> dict:
    out: dict = {}
    for tier_name, (_sigma, landmarks) in CONFIDENCE_TIERS.items():
        if tier_name.startswith("high"):
            label = "high"
        elif tier_name.startswith("med"):
            label = "medium"
        else:
            label = "low"
        for name in landmarks:
            out[name] = label
    return out


CONFIDENCE_LABEL_BY_LANDMARK: dict = _build_tier_label_map()


# ── Image resolution constants ───────────────────────────────────────────────
#
# Stage 1 input is 512×512 RGB; attention maps are 256×256 (upsampled to 512).
# Phase 0C zone inputs are 256×256 grayscale. All training and inference code
# uses these values verbatim — do not change without retraining.

INPUT_SIZE: int = 512      # Stage 1 input width and height
HEATMAP_SIZE: int = 256    # Stage 1 output heatmap width and height
ATTENTION_SIZE: int = 256  # Phase 0E attention map native resolution
ZONE_INPUT: int = 256      # Phase 0C per-zone ONNX input resolution
IN_CHANNELS: int = 28      # 3 RGB + 25 attention channels for Stage 1


# ── Stage 1 → original-space back-projection ─────────────────────────────────

# Average cephalogram dimensions in mm (Proffit 7th Edition reference).
# Used for converting Douglas-Peucker tolerances from mm → pixels.
CEPH_WIDTH_MM: float = 200.0
CEPH_HEIGHT_MM: float = 250.0

# Proffit 7th Edition Sella-Nasion mean distance, used for S-N landmark
# calibration in clinical workflows (downstream of this package).
SN_DISTANCE_MM: float = 71.0


# ── Stage 0 model filenames ──────────────────────────────────────────────────
#
# Keys are the internal model identifiers; values are paths relative to the
# weights root directory (e.g., `models/` after `scripts/download_weights.py`
# has populated it).

MODEL_FILES: dict = {
    "stage0_profile":     "stage0/v4_stage0_profile.onnx",
    "stage0_contours_z1": "stage0/z1_cranial_base_contours.onnx",
    "stage0_contours_z2": "stage0/z2_midface_contours.onnx",
    "stage0_contours_z3": "stage0/z3_mandible_contours.onnx",
    "stage0_contours_z4": "stage0/z4_posterior_contours.onnx",
    "stage0_derive":      "stage0/phase0e_model.onnx",
    "stage1":             "stage1/v4_stage1.onnx",
}

MODEL_SIZES_MB: dict = {
    # Expected sizes (used by download script for integrity checks).
    "stage0_profile":     25.55,
    "stage0_contours_z1": 25.55,
    "stage0_contours_z2": 25.55,
    "stage0_contours_z3": 25.55,
    "stage0_contours_z4": 25.55,
    "stage0_derive":      0.43,
    "stage1":             123.93,
}

# Phase 0E is a 114K-parameter MLP whose ONNX is legitimately ~444 KB. All
# other ONNX files embed convolutional weights and must be at least 1 MB.
MIN_MODEL_SIZE_BYTES: int = 1 * 1024 * 1024
MIN_DERIVE_MODEL_SIZE_BYTES: int = 200 * 1024


# ── Phase 0C zone → contour-class mapping ────────────────────────────────────
#
# Source: Phase 0C training notebook (each zone model was trained to emit a
# specific set of contour classes). Do NOT infer from filenames.

ZONE_CONTOUR_MAP: dict = {
    "zone_1_cranial_base": {
        "model_key": "stage0_contours_z1",
        "classes": ["cranial_base"],
    },
    "zone_2_midface": {
        "model_key": "stage0_contours_z2",
        "classes": ["palatal_plane", "upper_incisor_axis"],
    },
    "zone_3_mandible": {
        "model_key": "stage0_contours_z3",
        "classes": [
            "mandibular_border", "mandibular_symphysis", "lower_incisor_axis",
        ],
    },
    "zone_4_posterior": {
        "model_key": "stage0_contours_z4",
        "classes": ["mandibular_border", "cranial_base"],
    },
}

# All distinct contour class names that can appear in Stage 0 output.
CONTOUR_CLASSES: Tuple[str, ...] = (
    "cranial_base",
    "palatal_plane",
    "upper_incisor_axis",
    "mandibular_border",
    "mandibular_symphysis",
    "lower_incisor_axis",
)


# ── Phase 0B CLAHE per-zone parameters ───────────────────────────────────────
#
# Source: ML-v4-Training-Log.md "Phase 0B" section. Each zone uses different
# CLAHE settings tuned for its anatomical content (Z1 cranial base wants
# higher contrast for sella turcica; Z5 soft tissue wants gentle enhancement
# to preserve the air-skin boundary).

CLAHE_PARAMS: dict = {
    "zone_1_cranial_base": {"clipLimit": 3.0, "tileGridSize": (4, 4)},
    "zone_2_midface":      {"clipLimit": 2.5, "tileGridSize": (8, 8)},
    "zone_3_mandible":     {"clipLimit": 3.0, "tileGridSize": (6, 6)},
    "zone_4_posterior":    {"clipLimit": 3.5, "tileGridSize": (4, 4)},
    "zone_5_soft_tissue":  {"clipLimit": 1.5, "tileGridSize": (8, 8)},
}


# ── Phase 0D Douglas-Peucker tolerances ──────────────────────────────────────
#
# Per-contour-class tolerance in millimetres. Tighter (smaller) tolerances
# preserve more vertex detail and are used on contours where landmark
# precision matters most (e.g., mandibular symphysis carries Me, Pog, B).

DP_TOLERANCE_MM: dict = {
    "mandibular_symphysis": 0.5,
    "mandibular_border":    1.0,
    "cranial_base":         2.0,
    "palatal_plane":        1.0,
    "upper_incisor_axis":   0.5,
    "lower_incisor_axis":   0.5,
}


# ── Phase 0B zone geometry guards ────────────────────────────────────────────

EMPTY_MASK_THRESHOLD: int = 50  # Below this many nonzero pixels → Stage 0 fully degraded
ZONE_PADDING: float = 0.10      # → pad = 51 px at 512 res
MIN_ZONE_SIZE: int = 128        # Minimum side length for any zone bbox

# Hard-coded fallback zones used when Phase 0A produces no usable profile.
# Identical across all 1502 training images per `zone_metadata.json` cross-
# validation. Format: (x1, y1, x2, y2) at 512×512 resolution.
Z1_BBOX: Tuple[int, int, int, int] = (77, 0, 512, 486)
Z4_BBOX: Tuple[int, int, int, int] = (25, 0, 512, 486)


# ── Stage 1 confidence thresholds ────────────────────────────────────────────
#
# Source: `v4_pipeline.py` `predict()` method. The per-landmark heatmap peak
# value (after sigmoid) is bucketed into qualitative confidence labels for UI
# display.

STAGE1_CONFIDENCE_HIGH_PEAK: float = 0.5
STAGE1_CONFIDENCE_MEDIUM_PEAK: float = 0.2


# ── Default Hugging Face Hub repository for model weights ────────────────────
#
# Updated when the public release is published. Used by
# `scripts/download_weights.py` and the optional `auto_download` flag on
# `CephTracePredictor`.

HF_REPO_ID: str = "CephTrace/cephtrace-v4"
HF_REPO_REVISION: str = "main"
