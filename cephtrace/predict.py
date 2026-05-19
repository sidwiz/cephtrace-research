"""CephTracePredictor — high-level orchestrator for v4 inference.

Loads ONNX models from a local directory, runs Stage 0 (anatomical zone
decomposition) and Stage 1 (HRNet-W32 heatmap regression with DARK
sub-pixel decode), and returns a `PredictionResult` with 25 landmark
coordinates in original-image pixel space.

Stage 2 (per-landmark specialist refinement) is part of the production
deployment but is NOT included in this public research release. The
specialist heads and their training data are retained for ongoing
research and are covered by US Provisional Application 64/037,246.

Typical usage:

    from cephtrace import CephTracePredictor

    predictor = CephTracePredictor(model_dir="models/")
    result = predictor.predict("/path/to/cephalogram.jpg")
    for lm in result.landmarks:
        print(lm["name"], lm["x"], lm["y"], lm["confidence"])
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import onnxruntime as ort

from .constants import (
    CANONICAL_25,
    CONFIDENCE_LABEL_BY_LANDMARK,
    MIN_DERIVE_MODEL_SIZE_BYTES,
    MIN_MODEL_SIZE_BYTES,
    MODEL_FILES,
    NUM_LANDMARKS,
)
from .preprocessing import load_image_bytes, preprocess_rgb_for_stage1
from .stage0 import Stage0Result, run_stage0
from .stage1 import Stage1Output, peak_to_confidence_label, run_stage1

__all__ = ["CephTracePredictor", "PredictionResult", "LandmarkPrediction"]


@dataclass
class LandmarkPrediction:
    """A single landmark prediction.

    Attributes:
        index: Position in `CANONICAL_25` (0..24).
        name: Landmark name (e.g., "S", "N", "Pog").
        x: x-coordinate in original image pixel space.
        y: y-coordinate in original image pixel space.
        peak: Stage 1 heatmap peak value, raw [0, 1] from the network.
        confidence_score: Same as `peak`, clipped to [0, 1].
        confidence_label: Qualitative label derived from `peak`:
            "high" (peak > 0.5), "medium" (peak > 0.2), or "low".
        tier_label: Confidence tier from the Stage 0 attention prior
            ("high" / "medium" / "low"), reflecting the model's expected
            anatomical certainty for this landmark independent of the
            individual prediction.
    """
    index: int
    name: str
    x: float
    y: float
    peak: float
    confidence_score: float
    confidence_label: str
    tier_label: str


@dataclass
class PredictionResult:
    """Output of `CephTracePredictor.predict`.

    Attributes:
        landmarks: List of 25 `LandmarkPrediction` items in CANONICAL_25
            order. Also exposed as a `list[dict]` via `to_list_of_dicts()`
            for easy JSON serialization.
        original_width: Width of the input image in pixels.
        original_height: Height of the input image in pixels.
        stage0_status: Per-substage strict-success flags (5 keys).
        stage0_details: Per-substage diagnostic counts and missing-item lists.
        degraded: True if any Stage 0 substage failed or produced partial
            output. Stage 1 still ran and produced valid coordinates;
            accuracy may be reduced.
    """
    landmarks: List[LandmarkPrediction]
    original_width: int
    original_height: int
    stage0_status: Dict[str, bool] = field(default_factory=dict)
    stage0_details: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    degraded: bool = False

    def to_list_of_dicts(self) -> List[Dict[str, Any]]:
        """Return the landmarks as a list of plain dicts (JSON-serializable)."""
        return [
            {
                "index": lm.index,
                "name": lm.name,
                "x": lm.x,
                "y": lm.y,
                "peak": lm.peak,
                "confidence_score": lm.confidence_score,
                "confidence": lm.confidence_label,
                "tier": lm.tier_label,
            }
            for lm in self.landmarks
        ]

    def to_dict(self) -> Dict[str, Any]:
        """Return the full result as a JSON-serializable dict."""
        return {
            "landmarks": self.to_list_of_dicts(),
            "original_width": self.original_width,
            "original_height": self.original_height,
            "stage0_status": dict(self.stage0_status),
            "stage0_details": dict(self.stage0_details),
            "degraded": self.degraded,
        }


class CephTracePredictor:
    """High-level inference API for the CephTrace v4 pipeline.

    Loads 7 ONNX models on construction (Stage 0: profile + 4 contours + MLP;
    Stage 1: HRNet). Stage 0 models are optional — if any fail to load, the
    pipeline silently falls back to zero-filled attention channels and the
    `degraded` flag is set on each subsequent prediction.

    Args:
        model_dir: Path to the directory containing the `stage0/` and
            `stage1/` subdirectories with the ONNX files. Use
            `scripts/download_weights.py` to populate it.
        use_gpu: When True, uses the CUDAExecutionProvider for all sessions.
            Requires `onnxruntime-gpu` to be installed. Default False (CPU).
        strict: When True, raise `RuntimeError` if any Stage 0 model is
            missing. When False (default), missing Stage 0 models are
            tolerated and predictions run in degraded mode.
    """

    def __init__(
        self,
        model_dir: Union[str, Path] = "models",
        use_gpu: bool = False,
        strict: bool = False,
    ) -> None:
        self._model_dir = Path(model_dir)
        self._use_gpu = bool(use_gpu)
        self._strict = bool(strict)
        self._sessions: Dict[str, ort.InferenceSession] = {}
        self._loaded: Dict[str, bool] = {}
        self._stage0_available: bool = False

        providers = (
            ["CUDAExecutionProvider", "CPUExecutionProvider"]
            if self._use_gpu
            else ["CPUExecutionProvider"]
        )

        for key, relpath in MODEL_FILES.items():
            path = self._model_dir / relpath
            is_required = (key == "stage1")
            min_size = (
                MIN_DERIVE_MODEL_SIZE_BYTES
                if key == "stage0_derive"
                else MIN_MODEL_SIZE_BYTES
            )

            if not path.is_file():
                if is_required or self._strict:
                    raise RuntimeError(
                        f"Required ONNX model not found: {path}\n"
                        f"Run `python scripts/download_weights.py --output {self._model_dir}` "
                        f"to download the weights."
                    )
                self._loaded[key] = False
                continue

            size_bytes = path.stat().st_size
            if size_bytes < min_size:
                msg = (
                    f"Model file too small: {path} is {size_bytes / 1024:.1f} KB "
                    f"(minimum {min_size / 1024:.0f} KB). Weights may not be "
                    f"embedded — re-download the file."
                )
                if is_required or self._strict:
                    raise RuntimeError(msg)
                self._loaded[key] = False
                continue

            sidecar = path.with_suffix(path.suffix + ".data")
            if sidecar.is_file():
                msg = (
                    f"Sidecar weights file found at {sidecar}. The Stage 1 "
                    f"ONNX uses external-data format. Re-download with "
                    f"`scripts/download_weights.py` to get a self-contained file."
                )
                if is_required or self._strict:
                    raise RuntimeError(msg)
                self._loaded[key] = False
                continue

            self._sessions[key] = ort.InferenceSession(
                str(path), providers=providers,
            )
            self._loaded[key] = True

        if "stage1" not in self._sessions:
            raise RuntimeError(
                "Stage 1 model failed to load. CephTracePredictor cannot run "
                "without Stage 1. Check the error messages above."
            )

        stage0_keys = (
            "stage0_profile",
            "stage0_contours_z1", "stage0_contours_z2",
            "stage0_contours_z3", "stage0_contours_z4",
            "stage0_derive",
        )
        self._stage0_available = all(self._loaded.get(k, False) for k in stage0_keys)

    # ── Properties ───────────────────────────────────────────────────────────

    @property
    def model_dir(self) -> Path:
        """Directory containing the loaded ONNX models."""
        return self._model_dir

    @property
    def stage0_available(self) -> bool:
        """True iff all 6 Stage 0 models loaded successfully."""
        return self._stage0_available

    @property
    def loaded_models(self) -> Dict[str, bool]:
        """Per-model load status as a {key: bool} dict."""
        return dict(self._loaded)

    # ── Prediction API ───────────────────────────────────────────────────────

    def predict(self, image_path: Union[str, Path]) -> PredictionResult:
        """Predict 25 landmarks for a single image on disk.

        Args:
            image_path: Filesystem path to a JPEG, PNG, BMP, or any other
                Pillow-supported image format.

        Returns:
            PredictionResult with the 25 landmark predictions and diagnostic
            information.
        """
        image_bytes = load_image_bytes(str(image_path))
        return self._predict_from_bytes(image_bytes)

    def predict_from_array(self, image_bgr: np.ndarray) -> PredictionResult:
        """Predict 25 landmarks from a numpy array.

        Args:
            image_bgr: Numpy array of shape (H, W, 3) RGB uint8, (H, W, 3)
                BGR uint8, or (H, W) grayscale. The pipeline internally
                normalizes to RGB.

                **Note**: The argument is named `image_bgr` for OpenCV
                familiarity, but the function accepts either BGR or RGB
                — it's a 3-channel array and is treated as RGB by Pillow
                downstream. If you have a BGR-ordered OpenCV image, pass
                `image[:, :, ::-1]` to flip channels.

        Returns:
            PredictionResult.
        """
        # Re-encode to bytes via PIL so the rest of the pipeline sees the
        # same code path as `predict()`. This sidesteps a subtle source of
        # numerical drift between array-direct and bytes-decode paths in
        # Phase 0A.
        import io as _io
        from PIL import Image as _Image

        if image_bgr.ndim == 2:
            pil = _Image.fromarray(image_bgr).convert("RGB")
        elif image_bgr.ndim == 3 and image_bgr.shape[2] == 3:
            pil = _Image.fromarray(image_bgr.astype(np.uint8))
        else:
            raise ValueError(
                f"image_bgr must have shape (H, W) or (H, W, 3), got "
                f"{image_bgr.shape}"
            )
        buf = _io.BytesIO()
        pil.save(buf, format="PNG")
        return self._predict_from_bytes(buf.getvalue())

    def predict_batch(
        self, image_paths: Sequence[Union[str, Path]],
    ) -> List[PredictionResult]:
        """Predict 25 landmarks for each image in a sequence.

        Each image is processed independently — ONNX Runtime sessions do
        NOT support cross-image batching for the variable-input-size
        Phase 0A model.

        Args:
            image_paths: Sequence of filesystem paths.

        Returns:
            List of `PredictionResult` in the same order as `image_paths`.
        """
        return [self.predict(p) for p in image_paths]

    # ── Internals ────────────────────────────────────────────────────────────

    def _predict_from_bytes(self, image_bytes: bytes) -> PredictionResult:
        # Stage 1 RGB preprocessing yields the (1, 3, 512, 512) input
        # plus the metadata we need to back-project coordinates.
        rgb_tensor, meta = preprocess_rgb_for_stage1(image_bytes, target_size=512)
        orig_w = int(meta["original_width"])
        orig_h = int(meta["original_height"])

        # Stage 0 — attention priors.
        if self._stage0_available:
            s0: Stage0Result = run_stage0(self._sessions, image_bytes)
            attention_maps_512 = s0.attention_maps
            stage0_status = s0.substage_status
            stage0_details = s0.details
            stage0_degraded = s0.degraded
        else:
            attention_maps_512 = None
            stage0_status = {
                "0A": False, "0B": False, "0C": False, "0D": False, "0E": False,
            }
            stage0_details = {
                "0A": {"reason": "stage0_unavailable"},
                "0B": {}, "0C": {}, "0D": {}, "0E": {},
            }
            stage0_degraded = True

        # Stage 1 — HRNet + DARK.
        s1: Stage1Output = run_stage1(
            self._sessions["stage1"],
            rgb_tensor=rgb_tensor,
            attention_maps_512=attention_maps_512,
            original_width=orig_w,
            original_height=orig_h,
            return_heatmaps=False,
        )

        # Assemble per-landmark predictions.
        landmarks: List[LandmarkPrediction] = []
        for i, name in enumerate(CANONICAL_25):
            peak = float(s1.peaks[i])
            clipped = float(np.clip(s1.peaks[i], 0.0, 1.0))
            x = float(s1.coords_orig[i, 0])
            y = float(s1.coords_orig[i, 1])
            landmarks.append(
                LandmarkPrediction(
                    index=i,
                    name=name,
                    x=x,
                    y=y,
                    peak=peak,
                    confidence_score=clipped,
                    confidence_label=peak_to_confidence_label(peak),
                    tier_label=CONFIDENCE_LABEL_BY_LANDMARK[name],
                )
            )

        # `degraded` is True if attention maps were missing (the only
        # observable difference to the caller) — even if all 5 substage
        # status flags are False, that condition is covered too.
        degraded = stage0_degraded or attention_maps_512 is None

        return PredictionResult(
            landmarks=landmarks,
            original_width=orig_w,
            original_height=orig_h,
            stage0_status=stage0_status,
            stage0_details=stage0_details,
            degraded=degraded,
        )
