"""CephTrace v4 — Anatomy-Guided Cephalometric Landmark Detection.

Inference-only Python package for the CephTrace v4 pipeline described in
arXiv:2605.03358 (Mohapatra & Mohanty 2026). The package is the public
research release of the production pipeline; training notebooks and the
Stage 2 / Stage 3 components are not included.

Quick start:

    from cephtrace import CephTracePredictor

    predictor = CephTracePredictor(model_dir="models/")
    result = predictor.predict("/path/to/cephalogram.jpg")
    for lm in result.landmarks:
        print(lm.name, lm.x, lm.y, lm.confidence_label)

Run `python scripts/download_weights.py` first to fetch the ONNX models
from Hugging Face Hub (~277 MB total).

Citation:
    @article{mohapatra2026cephtrace,
      title   = {Tracing Like a Clinician: Anatomy-Guided Spatial Priors
                 for Cephalometric Landmark Detection},
      author  = {Mohapatra, Sidhartha and Mohanty, Pallavi},
      journal = {arXiv preprint arXiv:2605.03358},
      year    = {2026},
    }
"""

from __future__ import annotations

from .constants import (
    ANCHOR_NAMES,
    CANONICAL_25,
    CONFIDENCE_LABEL_BY_LANDMARK,
    CONFIDENCE_TIERS,
    DERIVED_NAMES,
    HEATMAP_SIZE,
    HF_REPO_ID,
    INPUT_SIZE,
    MODEL_FILES,
    NUM_LANDMARKS,
    SIGMA_BY_LANDMARK,
    SN_DISTANCE_MM,
)
from .dark import dark_postprocess
from .predict import (
    CephTracePredictor,
    LandmarkPrediction,
    PredictionResult,
)

__version__ = "0.4.0"

__all__ = [
    # Constants
    "ANCHOR_NAMES",
    "CANONICAL_25",
    "CONFIDENCE_LABEL_BY_LANDMARK",
    "CONFIDENCE_TIERS",
    "DERIVED_NAMES",
    "HEATMAP_SIZE",
    "HF_REPO_ID",
    "INPUT_SIZE",
    "MODEL_FILES",
    "NUM_LANDMARKS",
    "SIGMA_BY_LANDMARK",
    "SN_DISTANCE_MM",
    # Functional API
    "dark_postprocess",
    # High-level API
    "CephTracePredictor",
    "LandmarkPrediction",
    "PredictionResult",
    # Version
    "__version__",
]
