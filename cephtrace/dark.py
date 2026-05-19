"""DARK sub-pixel heatmap coordinate decoder.

Implements Distribution-Aware coordinate Representation for Keypoint
(Zhang et al., CVPR 2020). Refines the argmax-pixel coordinate by computing
a second-order Taylor expansion on the log-heatmap to recover the sub-pixel
location of the underlying Gaussian peak.

This is the only coordinate decoder used in Stage 1 inference. Soft-argmax
was evaluated during v3 development and rejected because it produces biased
estimates on heatmaps trained with EWC or Adaptive Wing Loss (the peaks are
too sharp for the soft-argmax assumption of a smooth distribution).

Reference:
    Zhang, F. et al. "Distribution-Aware Coordinate Representation for Human
    Pose Estimation." CVPR 2020. https://arxiv.org/abs/1910.06278
"""

from __future__ import annotations

import numpy as np

__all__ = ["dark_postprocess"]


def dark_postprocess(heatmaps: np.ndarray) -> np.ndarray:
    """Decode (N, H, W) heatmaps into (N, 2) sub-pixel coordinates.

    For each heatmap channel:

      1. Find the integer-pixel argmax (px, py).
      2. Compute first and second numerical derivatives of the log-heatmap at
         (px, py) using central differences on neighboring pixels.
      3. Solve the Taylor expansion `H(p) ≈ H(p*) + 0.5 (p - p*)^T H'' (p - p*)`
         for the sub-pixel offset, clamping each component to [-0.5, +0.5].
      4. Return `(px + offset_x, py + offset_y)` in heatmap pixel coordinates.

    Args:
        heatmaps: float32/float64 array of shape (N, H, W). Values should be
            non-negative (typically post-sigmoid in [0, 1]). Zeros are clipped
            to 1e-10 before taking the log to avoid `-inf`.

    Returns:
        coords: float64 array of shape (N, 2). Each row is (x, y) in heatmap
            pixel space — to back-project to the original image, divide by
            (W, H) and multiply by the original image (width, height).

    Raises:
        ValueError: If `heatmaps` is not a 3D array.
    """
    if heatmaps.ndim != 3:
        raise ValueError(
            f"dark_postprocess expects (N, H, W) array, got shape "
            f"{heatmaps.shape!r}"
        )

    n, h, w = heatmaps.shape
    coords = np.zeros((n, 2), dtype=np.float64)

    # Work in float64 for stable log/division.
    heatmaps_f = heatmaps.astype(np.float64, copy=False)

    for i in range(n):
        hm = heatmaps_f[i]

        # 1. Integer-pixel argmax.
        flat_idx = int(np.argmax(hm))
        py, px = divmod(flat_idx, w)

        # Clamp to interior so the central-difference stencil has all 4
        # neighbors available. The clamp is harmless for typical anatomical
        # landmarks (peak is rarely at the very edge); when it does happen
        # (degraded prediction near image boundary), the sub-pixel offset
        # collapses to ~0 and we return the boundary pixel.
        py = max(1, min(py, h - 2))
        px = max(1, min(px, w - 2))

        # 2. Log-heatmap with floor at 1e-10 to avoid log(0).
        log_hm = np.log(np.clip(hm, 1e-10, None))

        dx = (log_hm[py, px + 1] - log_hm[py, px - 1]) * 0.5
        dy = (log_hm[py + 1, px] - log_hm[py - 1, px]) * 0.5
        dxx = log_hm[py, px + 1] - 2.0 * log_hm[py, px] + log_hm[py, px - 1]
        dyy = log_hm[py + 1, px] - 2.0 * log_hm[py, px] + log_hm[py - 1, px]

        # 3. Newton step in log-space. The +1e-10 in the denominator is a
        # numerical guard against perfectly flat heatmaps (dxx == 0).
        offset_x = float(np.clip(-dx / (dxx + 1e-10), -0.5, 0.5))
        offset_y = float(np.clip(-dy / (dyy + 1e-10), -0.5, 0.5))

        # 4. Sub-pixel coordinate in heatmap space.
        coords[i, 0] = px + offset_x
        coords[i, 1] = py + offset_y

    return coords
