"""Stage 1 — HRNet-W32 heatmap regression + DARK sub-pixel decode.

The Stage 1 ONNX model takes a (1, 28, 512, 512) float32 input (3 RGB +
25 attention map channels) and produces (1, 25, 256, 256) heatmaps. Each
channel is the predicted spatial probability map for one of the 25
canonical landmarks.

This module:
    1. Concatenates the RGB tensor with the Stage 0 attention maps (or
       zeros if Stage 0 is degraded).
    2. Runs the Stage 1 ONNX session.
    3. Decodes the heatmaps to sub-pixel coordinates via DARK.
    4. Back-projects from heatmap space to original image pixel space.

The HRNet-W32 backbone, Adaptive Wing Loss training objective, and DARK
decoder are all described in:

    Mohapatra & Mohanty (2026). "Tracing Like a Clinician: Anatomy-Guided
    Spatial Priors for Cephalometric Landmark Detection."
    arXiv:2605.03358

Patent: US Provisional Application No. 64/037,246.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import onnxruntime as ort

from .constants import (
    HEATMAP_SIZE,
    IN_CHANNELS,
    INPUT_SIZE,
    NUM_LANDMARKS,
    STAGE1_CONFIDENCE_HIGH_PEAK,
    STAGE1_CONFIDENCE_MEDIUM_PEAK,
)
from .dark import dark_postprocess

__all__ = [
    "Stage1Output",
    "run_stage1",
    "back_project_heatmap_coords",
    "peak_to_confidence_label",
]


@dataclass
class Stage1Output:
    """Output of `run_stage1`.

    Attributes:
        coords_orig: (25, 2) float64 array of (x, y) pixel coordinates in
            ORIGINAL image space.
        coords_heatmap: (25, 2) float64 array of sub-pixel (x, y) in
            heatmap space (256×256). Useful for downstream refinement.
        peaks: (25,) float32 array of the maximum heatmap value per
            landmark. Acts as a per-landmark confidence score in [0, 1].
        heatmaps: (25, 256, 256) float32 array — the raw heatmap stack,
            retained when `return_heatmaps=True` is passed to `run_stage1`.
            Otherwise None.
    """
    coords_orig: np.ndarray
    coords_heatmap: np.ndarray
    peaks: np.ndarray
    heatmaps: Optional[np.ndarray] = None


def run_stage1(
    session: ort.InferenceSession,
    rgb_tensor: np.ndarray,
    attention_maps_512: Optional[np.ndarray],
    original_width: int,
    original_height: int,
    *,
    return_heatmaps: bool = False,
) -> Stage1Output:
    """Run Stage 1 inference and decode landmark coordinates.

    Args:
        session: Loaded ONNX Runtime session for the Stage 1 model.
        rgb_tensor: (1, 3, 512, 512) float32 array from
            `preprocessing.preprocess_rgb_for_stage1`.
        attention_maps_512: (1, 25, 512, 512) float32 array from
            `stage0.run_stage0`. Pass None to fill the attention channels
            with zeros (Stage 0 degraded path).
        original_width: Original image width in pixels.
        original_height: Original image height in pixels.
        return_heatmaps: When True, the raw heatmap stack is included in
            the returned `Stage1Output.heatmaps`. Default False to save
            memory when only coordinates are needed.

    Returns:
        Stage1Output with sub-pixel landmark coordinates in heatmap space
        and back-projected coordinates in original image space.

    Raises:
        ValueError: If `rgb_tensor` does not have shape (1, 3, 512, 512).
        ValueError: If `attention_maps_512` is supplied but has the wrong shape.
    """
    if rgb_tensor.shape != (1, 3, INPUT_SIZE, INPUT_SIZE):
        raise ValueError(
            f"rgb_tensor must have shape (1, 3, {INPUT_SIZE}, {INPUT_SIZE}), "
            f"got {rgb_tensor.shape}"
        )

    if attention_maps_512 is None:
        attention_maps_512 = np.zeros(
            (1, NUM_LANDMARKS, INPUT_SIZE, INPUT_SIZE), dtype=np.float32,
        )
    elif attention_maps_512.shape != (1, NUM_LANDMARKS, INPUT_SIZE, INPUT_SIZE):
        raise ValueError(
            f"attention_maps_512 must have shape "
            f"(1, {NUM_LANDMARKS}, {INPUT_SIZE}, {INPUT_SIZE}), got "
            f"{attention_maps_512.shape}"
        )

    input_tensor = np.concatenate([rgb_tensor, attention_maps_512], axis=1)
    assert input_tensor.shape == (1, IN_CHANNELS, INPUT_SIZE, INPUT_SIZE), (
        f"Stage 1 input shape mismatch: {input_tensor.shape}"
    )

    input_name = session.get_inputs()[0].name
    raw_output = session.run(None, {input_name: input_tensor})[0]
    # raw_output shape: (1, 25, 256, 256)
    heatmaps = raw_output[0]
    hm_h, hm_w = heatmaps.shape[1], heatmaps.shape[2]
    assert (hm_h, hm_w) == (HEATMAP_SIZE, HEATMAP_SIZE), (
        f"Unexpected heatmap shape: {heatmaps.shape} "
        f"(expected (25, {HEATMAP_SIZE}, {HEATMAP_SIZE}))"
    )

    # DARK sub-pixel decode in heatmap space [0, 256).
    coords_hm = dark_postprocess(heatmaps)

    # Back-project heatmap coords → original-image pixel coords.
    coords_orig = back_project_heatmap_coords(
        coords_hm, original_width, original_height,
    )

    peaks = np.array(
        [float(heatmaps[i].max()) for i in range(NUM_LANDMARKS)],
        dtype=np.float32,
    )

    return Stage1Output(
        coords_orig=coords_orig,
        coords_heatmap=coords_hm,
        peaks=peaks,
        heatmaps=heatmaps if return_heatmaps else None,
    )


def back_project_heatmap_coords(
    coords_hm: np.ndarray, original_width: int, original_height: int,
) -> np.ndarray:
    """Back-project heatmap-space coords (0..256) to original image pixels.

    Args:
        coords_hm: (N, 2) array of sub-pixel (x, y) in [0, HEATMAP_SIZE).
        original_width: Width of the original input image in pixels.
        original_height: Height of the original input image in pixels.

    Returns:
        (N, 2) float64 array of (x, y) in original image pixel space.
    """
    coords = np.asarray(coords_hm, dtype=np.float64)
    out = np.empty_like(coords)
    out[:, 0] = (coords[:, 0] / float(HEATMAP_SIZE)) * float(original_width)
    out[:, 1] = (coords[:, 1] / float(HEATMAP_SIZE)) * float(original_height)
    return out


def peak_to_confidence_label(peak: float) -> str:
    """Bucket a heatmap peak value into a qualitative confidence label.

    Args:
        peak: Heatmap maximum (typically in [0, 1] post-sigmoid).

    Returns:
        "high" if peak > 0.5, "medium" if peak > 0.2, else "low".
    """
    if peak > STAGE1_CONFIDENCE_HIGH_PEAK:
        return "high"
    if peak > STAGE1_CONFIDENCE_MEDIUM_PEAK:
        return "medium"
    return "low"
