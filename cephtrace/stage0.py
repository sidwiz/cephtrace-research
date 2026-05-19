"""Stage 0 — Anatomical Zone Decomposition.

Stage 0 produces (25, 256, 256) Gaussian attention maps in CANONICAL_25
channel order. These are concatenated with the (1, 3, 512, 512) RGB tensor
to form Stage 1's (1, 28, 512, 512) input.

The substages run sequentially with graceful degradation at each step:

    Phase 0A — Soft-tissue profile binary mask (ONNX, 1-channel grayscale in)
    Phase 0B — Profile features, zone bboxes, per-zone CLAHE, soft-tissue
               landmark extraction (no ONNX)
    Phase 0C — Per-zone contour segmentation (4× ONNX, 1-channel grayscale in)
    Phase 0D — Douglas-Peucker simplification, 7 anchor extraction (no ONNX)
    Phase 0E — MLP for 18 derived landmarks, Gaussian attention rendering

If any substage fails, the chain returns `None` for the attention maps. The
caller (CephTracePredictor) then runs Stage 1 with zero-filled attention
channels and flags `degraded=True` on the result.

This file is a self-contained extraction of the production Stage 0 pipeline
that ships in CephTrace v4. The training notebooks are not part of this
public release.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import cv2
import numpy as np
import onnxruntime as ort
from scipy.ndimage import gaussian_filter1d, label as _scipy_label
from scipy.signal import argrelextrema

from .constants import (
    ANCHOR_NAMES,
    ATTENTION_SIZE,
    CANONICAL_25,
    CEPH_HEIGHT_MM,
    CEPH_WIDTH_MM,
    CLAHE_PARAMS,
    DERIVED_NAMES,
    DP_TOLERANCE_MM,
    EMPTY_MASK_THRESHOLD,
    INPUT_SIZE,
    MIN_ZONE_SIZE,
    NUM_LANDMARKS,
    SIGMA_BY_LANDMARK,
    Z1_BBOX,
    Z4_BBOX,
    ZONE_CONTOUR_MAP,
    ZONE_INPUT,
    ZONE_PADDING,
)

__all__ = [
    "Stage0Result",
    "run_stage0",
    "render_attention_maps_256",
]


# ── Result container ─────────────────────────────────────────────────────────


@dataclass
class Stage0Result:
    """Output of `run_stage0`.

    Attributes:
        attention_maps: (1, 25, 512, 512) float32 array in CANONICAL_25
            channel order, ready to concat with the Stage 1 RGB input.
            None when Stage 0 produced no usable output.
        substage_status: Per-substage strict-success flags. True iff that
            substage produced its full expected output (e.g., all 7 anchors,
            all 4 primary contours).
        details: Per-substage diagnostic counts and missing-item lists.
        degraded: True if any substage failed or produced partial output.
    """
    attention_maps: Optional[np.ndarray]
    substage_status: Dict[str, bool]
    details: Dict[str, Dict[str, Any]]
    degraded: bool


# ── Phase 0B geometric helpers ───────────────────────────────────────────────


def _extract_profile_contour(mask_512: np.ndarray) -> np.ndarray:
    """Largest connected component's outer contour, sorted by y.

    Returns (N, 2) int array of (x, y) points in proc512 coordinates.
    Returns an empty (0, 2) array if no connected component is found.
    """
    labeled, n = _scipy_label(mask_512)
    if n == 0:
        return np.empty((0, 2), dtype=np.int64)
    sizes = [int((labeled == i).sum()) for i in range(1, n + 1)]
    largest = (labeled == (int(np.argmax(sizes)) + 1)).astype(np.uint8)
    cnts, _ = cv2.findContours(largest, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    if not cnts:
        return np.empty((0, 2), dtype=np.int64)
    cnt = max(cnts, key=len).squeeze()
    if cnt.ndim != 2:
        return np.empty((0, 2), dtype=np.int64)
    return cnt[np.argsort(cnt[:, 1])]


def _compute_orientation_features(
    contour: np.ndarray,
) -> Optional[Dict[str, Any]]:
    """Bbox, centroid, nose tip, chin bottom from the profile contour.

    Returns None when the contour has fewer than 10 points.
    """
    if len(contour) < 10:
        return None

    xs = contour[:, 0].astype(np.float64)
    ys = contour[:, 1].astype(np.float64)
    x_min, x_max = int(xs.min()), int(xs.max())
    y_min, y_max = int(ys.min()), int(ys.max())
    cx, cy = float(xs.mean()), float(ys.mean())

    upper_mask = ys < (y_min + 0.4 * (y_max - y_min))
    if upper_mask.sum() > 0:
        ux, uy = xs[upper_mask], ys[upper_mask]
        i = int(np.argmin(ux))
        nose_tip = (int(ux[i]), int(uy[i]))
    else:
        i = int(np.argmin(xs))
        nose_tip = (int(xs[i]), int(ys[i]))

    chin_idx = int(np.argmax(ys))
    chin_bottom = (int(xs[chin_idx]), int(ys[chin_idx]))
    mai = int(np.argmin(xs))
    most_anterior = (int(xs[mai]), int(ys[mai]))

    return {
        "nose_tip": nose_tip,
        "chin_bottom": chin_bottom,
        "most_anterior": most_anterior,
        "centroid": (cx, cy),
        "bbox": (x_min, y_min, x_max, y_max),
        "vertical_extent": (y_min, y_max),
        "horizontal_extent": (x_min, x_max),
        "profile_height": float(y_max - y_min),
        "y_mid": float((y_min + y_max) / 2.0),
    }


def _extract_anterior_profile(contour: np.ndarray) -> np.ndarray:
    """For each distinct y, take the minimum-x point — the face-side edge."""
    if len(contour) < 5:
        return contour
    y_to_min_x: Dict[int, int] = {}
    for x, y in contour:
        x_i = int(x)
        y_i = int(y)
        if y_i not in y_to_min_x or x_i < y_to_min_x[y_i]:
            y_to_min_x[y_i] = x_i
    ys_sorted = sorted(y_to_min_x.keys())
    return np.array([(y_to_min_x[y], y) for y in ys_sorted], dtype=np.int64)


def _extract_soft_tissue_landmarks(
    contour: np.ndarray, features: Dict[str, Any],
) -> Dict[str, Tuple[int, int]]:
    """Extract Prn, Sn, UL, LL, Pog_soft, Pm from the profile contour.

    Returns a dict mapping landmark name to (x, y) in proc512 pixels.
    Landmarks may be absent from the dict when the contour shape doesn't
    support reliable extraction (e.g., no local maximum in the expected
    arc-length window).
    """
    if features is None or len(contour) < 20:
        return {}
    anterior = _extract_anterior_profile(contour)
    if len(anterior) < 20:
        return {}

    xs = anterior[:, 0].astype(np.float64)
    ys = anterior[:, 1].astype(np.float64)
    y_min, y_max = ys.min(), ys.max()
    profile_h = y_max - y_min
    if profile_h < 20:
        return {}

    sigma = max(3.0, profile_h * 0.015)
    xs_smooth = gaussian_filter1d(xs, sigma=sigma)
    y_norm = (ys - y_min) / profile_h
    landmarks: Dict[str, Tuple[int, int]] = {}

    # Prn — most anterior point in the upper 35% of the profile.
    m = y_norm < 0.35
    if m.sum() > 3:
        rx, ry = xs_smooth[m], ys[m]
        i = int(np.argmin(rx))
        landmarks["Prn"] = (int(rx[i]), int(ry[i]))

    # Sn — deepest concavity between nose and lip, arc-length in [0.25, 0.50].
    m = (y_norm > 0.25) & (y_norm < 0.50)
    if m.sum() > 5:
        rx, ry = xs_smooth[m], ys[m]
        order = max(3, len(rx) // 8)
        local_max = argrelextrema(rx, np.greater, order=order)[0]
        if len(local_max) > 0:
            j = int(local_max[0])
            landmarks["Sn"] = (int(rx[j]), int(ry[j]))
        else:
            i = int(np.argmax(rx))
            landmarks["Sn"] = (int(rx[i]), int(ry[i]))

    # UL — most anterior between Sn and Sn+0.15 arc.
    sn_yn = (
        (landmarks.get("Sn", (0, int(y_min + 0.35 * profile_h)))[1] - y_min)
        / profile_h
    )
    m = (y_norm > sn_yn) & (y_norm < min(sn_yn + 0.15, 0.55))
    if m.sum() > 3:
        rx, ry = xs_smooth[m], ys[m]
        i = int(np.argmin(rx))
        landmarks["UL"] = (int(rx[i]), int(ry[i]))

    # LL — most anterior between UL+0.02 and UL+0.18 arc.
    ul_yn = (
        (landmarks.get("UL", (0, int(y_min + 0.45 * profile_h)))[1] - y_min)
        / profile_h
    )
    m = (y_norm > ul_yn + 0.02) & (y_norm < min(ul_yn + 0.18, 0.70))
    if m.sum() > 3:
        rx, ry = xs_smooth[m], ys[m]
        i = int(np.argmin(rx))
        landmarks["LL"] = (int(rx[i]), int(ry[i]))

    # Pog_soft — most anterior in the lower 25% of the profile.
    m = y_norm > 0.75
    if m.sum() > 3:
        rx, ry = xs_smooth[m], ys[m]
        i = int(np.argmin(rx))
        landmarks["Pog_soft"] = (int(rx[i]), int(ry[i]))

    # Pm — inflection between LL and Pog_soft (sign change in second derivative).
    ll_y = landmarks.get("LL", (0, int(y_min + 0.55 * profile_h)))[1]
    pog_y = landmarks.get("Pog_soft", (0, int(y_min + 0.85 * profile_h)))[1]
    m = (ys > ll_y) & (ys < pog_y)
    if m.sum() > 10:
        rx, ry = xs_smooth[m], ys[m]
        if len(rx) > 5:
            ddx = np.gradient(np.gradient(rx))
            sign_changes = np.where(np.diff(np.sign(ddx)))[0]
            if len(sign_changes) > 0:
                mid = len(rx) // 2
                best = int(sign_changes[int(np.argmin(np.abs(sign_changes - mid)))])
                landmarks["Pm"] = (int(rx[best]), int(ry[best]))
            else:
                upper = max(1, len(rx) // 3)
                i = int(np.argmax(rx[:upper]))
                landmarks["Pm"] = (int(rx[i]), int(ry[i]))

    return landmarks


def _clamp_zone_bbox(
    x1: int, y1: int, x2: int, y2: int, img_size: int,
) -> Tuple[int, int, int, int]:
    """Clamp a zone bbox to image bounds and enforce minimum side length."""
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(img_size, x2)
    y2 = min(img_size, y2)
    if x2 - x1 < MIN_ZONE_SIZE:
        mid = (x1 + x2) // 2
        x1 = max(0, mid - MIN_ZONE_SIZE // 2)
        x2 = min(img_size, x1 + MIN_ZONE_SIZE)
    if y2 - y1 < MIN_ZONE_SIZE:
        mid = (y1 + y2) // 2
        y1 = max(0, mid - MIN_ZONE_SIZE // 2)
        y2 = min(img_size, y1 + MIN_ZONE_SIZE)
    return int(x1), int(y1), int(x2), int(y2)


def _compute_zone_bboxes(
    features: Dict[str, Any],
) -> Dict[str, Tuple[int, int, int, int]]:
    """Compute the 5 zone bboxes from profile orientation features.

    Zone 1 (Cranial base) and Zone 4 (Posterior) use hard-coded geometry —
    they are identical across all 1502 training images. Zones 2, 3, 5 are
    computed from the profile contour.
    """
    pad = int(INPUT_SIZE * ZONE_PADDING)  # = 51 px at 512 res
    cx, cy = features["centroid"]
    prof_x_min, prof_x_max = features["horizontal_extent"]
    prof_y_min, prof_y_max = features["vertical_extent"]

    y_split = int(cy * 0.95)
    x_anterior_limit = int(prof_x_max + 0.25 * (INPUT_SIZE - prof_x_max))

    return {
        "zone_1_cranial_base": _clamp_zone_bbox(
            int(INPUT_SIZE * 0.25) - pad, 0,
            INPUT_SIZE, int(INPUT_SIZE * 0.85) + pad,
            INPUT_SIZE,
        ),
        "zone_2_midface": _clamp_zone_bbox(
            0, 0,
            x_anterior_limit + 3 * pad, y_split + 4 * pad,
            INPUT_SIZE,
        ),
        "zone_3_mandible": _clamp_zone_bbox(
            0, y_split - 4 * pad,
            INPUT_SIZE, INPUT_SIZE,
            INPUT_SIZE,
        ),
        "zone_4_posterior": _clamp_zone_bbox(
            int(INPUT_SIZE * 0.15) - pad, 0,
            INPUT_SIZE, int(INPUT_SIZE * 0.85) + pad,
            INPUT_SIZE,
        ),
        "zone_5_soft_tissue": _clamp_zone_bbox(
            max(0, prof_x_min - 4 * pad),
            max(0, prof_y_min - 3 * pad),
            prof_x_max + 4 * pad,
            min(INPUT_SIZE, prof_y_max + 3 * pad),
            INPUT_SIZE,
        ),
    }


def _fallback_zones() -> Dict[str, Tuple[int, int, int, int]]:
    """Hard-coded zones used when Phase 0A produces no usable profile."""
    return {
        "zone_1_cranial_base": Z1_BBOX,
        "zone_2_midface":      (0, 0, int(INPUT_SIZE * 0.75), int(INPUT_SIZE * 0.60)),
        "zone_3_mandible":     (0, int(INPUT_SIZE * 0.30), INPUT_SIZE, INPUT_SIZE),
        "zone_4_posterior":    Z4_BBOX,
        "zone_5_soft_tissue":  (0, int(INPUT_SIZE * 0.02), int(INPUT_SIZE * 0.45), int(INPUT_SIZE * 0.98)),
    }


# ── Phase 0D geometric helpers ───────────────────────────────────────────────


def _orient_polyline_far_then_near(
    poly: np.ndarray, ref: Tuple[float, float],
) -> np.ndarray:
    """Reverse the polyline so its last endpoint is the one closer to `ref`.

    Canonicalizes contour direction against a known anatomical reference
    (e.g., the chin or nose tip). Returns a new (N, 2) array.
    """
    if len(poly) < 2:
        return poly
    r = np.array(ref, dtype=np.float64)
    d_start = float(np.linalg.norm(poly[0] - r))
    d_end = float(np.linalg.norm(poly[-1] - r))
    if d_start < d_end:
        return poly[::-1].copy()
    return poly


def _anchor_by_chord_window(
    poly: np.ndarray, frac_lo: float, frac_hi: float,
) -> Optional[Tuple[float, float]]:
    """Vertex with maximum perpendicular chord deviation in [frac_lo, frac_hi]."""
    n = len(poly)
    if n < 3:
        return None
    segs = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(segs)])
    total = float(arc[-1])
    if total <= 0.0:
        return None

    fracs = arc / total
    mask = (fracs >= frac_lo) & (fracs <= frac_hi)
    idxs = np.where(mask)[0]
    if len(idxs) == 0:
        mid_frac = 0.5 * (frac_lo + frac_hi)
        idxs = np.array([int(np.argmin(np.abs(fracs - mid_frac)))])

    p0, pN = poly[0], poly[-1]
    ab = pN - p0
    ab_norm = float(np.linalg.norm(ab))
    if ab_norm < 1e-9:
        mid_idx = int(idxs[len(idxs) // 2])
        return float(poly[mid_idx, 0]), float(poly[mid_idx, 1])

    best_idx = -1
    best_dev = -1.0
    for i in idxs:
        ap = poly[i] - p0
        dev = abs(float(ab[0] * ap[1] - ab[1] * ap[0])) / ab_norm
        if dev > best_dev:
            best_dev = dev
            best_idx = int(i)
    if best_idx < 0:
        return None
    return float(poly[best_idx, 0]), float(poly[best_idx, 1])


def _anchor_by_curvature_window(
    poly: np.ndarray, frac_lo: float, frac_hi: float,
) -> Optional[Tuple[float, float]]:
    """Vertex with maximum discrete curvature in [frac_lo, frac_hi]."""
    n = len(poly)
    if n < 5:
        return None
    segs = np.linalg.norm(np.diff(poly, axis=0), axis=1)
    arc = np.concatenate([[0.0], np.cumsum(segs)])
    total = float(arc[-1])
    if total <= 0.0:
        return None

    fracs = arc / total
    best_idx = -1
    best_ang = -1.0
    for i in range(1, n - 1):
        if not (frac_lo <= fracs[i] <= frac_hi):
            continue
        e1 = poly[i] - poly[i - 1]
        e2 = poly[i + 1] - poly[i]
        n1 = float(np.linalg.norm(e1))
        n2 = float(np.linalg.norm(e2))
        if n1 < 1e-9 or n2 < 1e-9:
            continue
        cos_t = float(np.clip(np.dot(e1, e2) / (n1 * n2), -1.0, 1.0))
        ang = float(np.arccos(cos_t))
        if ang > best_ang:
            best_ang = ang
            best_idx = i

    if best_idx < 0:
        mid_frac = 0.5 * (frac_lo + frac_hi)
        best_idx = int(np.argmin(np.abs(fracs - mid_frac)))
    return float(poly[best_idx, 0]), float(poly[best_idx, 1])


# ── Phase 0A: soft-tissue profile binary mask ────────────────────────────────


def _run_phase0a(
    session: ort.InferenceSession, image_bytes: bytes,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[Dict[str, Any]], int]:
    """Run Phase 0A ONNX → binary soft-tissue profile mask.

    Returns:
        mask_512: (512, 512) uint8 {0, 1}, or None if mask is empty.
        img_512_u8: (512, 512) uint8 grayscale (used downstream by 0B).
        meta: Dict with scale factors back to original image coordinates.
        nonzero: Number of nonzero pixels in the mask (for diagnostics).
    """
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img_orig_gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if img_orig_gray is None:
        return None, None, None, 0
    orig_h, orig_w = img_orig_gray.shape

    img_512_u8 = cv2.resize(
        img_orig_gray, (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR,
    )
    tensor = (img_512_u8.astype(np.float32) / 255.0)[np.newaxis, np.newaxis, :, :]

    input_name = session.get_inputs()[0].name
    out = session.run(None, {input_name: tensor})[0]

    if out.shape[1] == 1:
        probs = 1.0 / (1.0 + np.exp(-out[0, 0]))
        mask_512 = (probs > 0.5).astype(np.uint8)
    elif out.shape[1] == 2:
        mask_512 = (out[0, 1] > out[0, 0]).astype(np.uint8)
    else:
        raise ValueError(
            f"Unexpected Phase 0A output channels: {out.shape[1]} "
            f"(expected 1 or 2)"
        )

    nonzero = int(np.count_nonzero(mask_512))
    if nonzero < EMPTY_MASK_THRESHOLD:
        return None, None, None, nonzero

    meta = {
        "orig_w": int(orig_w),
        "orig_h": int(orig_h),
        "scale_x": orig_w / float(INPUT_SIZE),
        "scale_y": orig_h / float(INPUT_SIZE),
    }
    return mask_512, img_512_u8, meta, nonzero


# ── Phase 0B: features, zones, CLAHE, soft-tissue landmarks ──────────────────


def _run_phase0b(
    mask_512: np.ndarray, img_512_u8: np.ndarray,
) -> Tuple[
    Dict[str, Dict[str, Any]],
    Optional[Dict[str, Any]],
    Dict[str, Tuple[int, int]],
    bool,
]:
    """Compute orientation features, 5 zones, CLAHE-enhanced crops, and the
    6 soft-tissue landmarks. All outputs in proc512 coordinates.

    Returns:
        zone_data: Per-zone {"bbox": (x1, y1, x2, y2), "enhanced": uint8 crop}.
        features: Profile orientation features, or None if fallback used.
        soft_tissue_512: Dict of soft-tissue landmark positions in proc512.
        used_fallback: True iff `features` is None and fallback zones were used.
    """
    contour = _extract_profile_contour(mask_512)
    features = _compute_orientation_features(contour) if len(contour) > 0 else None

    soft_tissue_512: Dict[str, Tuple[int, int]] = {}
    used_fallback = False

    if features is None:
        zones = _fallback_zones()
        used_fallback = True
    else:
        zones = _compute_zone_bboxes(features)
        soft_tissue_512 = _extract_soft_tissue_landmarks(contour, features)

    zone_data: Dict[str, Dict[str, Any]] = {}
    for zone_name, bbox in zones.items():
        x1, y1, x2, y2 = bbox
        crop = img_512_u8[y1:y2, x1:x2]
        cfg = CLAHE_PARAMS[zone_name]
        clahe = cv2.createCLAHE(
            clipLimit=cfg["clipLimit"], tileGridSize=cfg["tileGridSize"],
        )
        enhanced = clahe.apply(crop)
        zone_data[zone_name] = {"bbox": bbox, "enhanced": enhanced}

    return zone_data, features, soft_tissue_512, used_fallback


# ── Phase 0C: per-zone contour segmentation ──────────────────────────────────


def _run_phase0c(
    sessions: Dict[str, ort.InferenceSession],
    zone_data: Dict[str, Dict[str, Any]],
    meta: Dict[str, Any],
) -> Tuple[Dict[str, np.ndarray], bool, list]:
    """Run the 4 zone ONNX models and extract polylines in original coords.

    Applies cross-zone merge policy: Z3 is primary for `mandibular_border`,
    Z1 is primary for `cranial_base`, Z4 serves as fragment fallback.

    Returns:
        contour_polylines_orig: dict mapping class name to (N, 2) float64
            polyline in original-image pixel coordinates.
        full_success: True iff all 4 primary contour classes are non-empty.
        missing_primary: List of primary class names that came back empty.
    """
    scale_x_o = meta["scale_x"]
    scale_y_o = meta["scale_y"]

    per_zone: Dict[str, Dict[str, np.ndarray]] = {}

    for zone_name, zone_info in zone_data.items():
        zone_def = ZONE_CONTOUR_MAP.get(zone_name)
        if zone_def is None:
            continue  # Zone 5 has no contour model
        class_names = zone_def["classes"]
        model_key = zone_def["model_key"]

        enhanced = zone_info["enhanced"]
        crop_256 = cv2.resize(
            enhanced, (ZONE_INPUT, ZONE_INPUT), interpolation=cv2.INTER_LINEAR,
        )
        tensor = (crop_256.astype(np.float32) / 255.0)[np.newaxis, np.newaxis, :, :]

        session = sessions[model_key]
        input_name = session.get_inputs()[0].name
        logits = session.run(None, {input_name: tensor})[0]

        if logits.shape[1] != len(class_names):
            per_zone[zone_name] = {}
            continue

        probs = 1.0 / (1.0 + np.exp(-logits[0]))
        binary = (probs > 0.5).astype(np.uint8)  # (n_cls, 256, 256)

        x1, y1, x2, y2 = zone_info["bbox"]
        h_z, w_z = y2 - y1, x2 - x1
        polylines: Dict[str, np.ndarray] = {}

        for c, cls_name in enumerate(class_names):
            m_native = cv2.resize(
                binary[c], (w_z, h_z), interpolation=cv2.INTER_NEAREST,
            )
            m_full = np.zeros((INPUT_SIZE, INPUT_SIZE), dtype=np.uint8)
            m_full[y1:y2, x1:x2] = m_native

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
            m_clean = cv2.morphologyEx(m_full, cv2.MORPH_CLOSE, kernel, iterations=1)
            m_clean = cv2.morphologyEx(m_clean, cv2.MORPH_OPEN, kernel, iterations=1)

            cnts, _ = cv2.findContours(
                m_clean, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE,
            )
            keep: list = []
            for cnt in cnts:
                if cv2.contourArea(cnt) >= 20 or len(cnt) >= 20:
                    keep.append(cnt.reshape(-1, 2).astype(np.float64))
            if not keep:
                polylines[cls_name] = np.empty((0, 2), dtype=np.float64)
                continue

            keep.sort(key=lambda c: len(c), reverse=True)
            poly_orig = keep[0] * np.array([scale_x_o, scale_y_o], dtype=np.float64)
            polylines[cls_name] = poly_orig

        per_zone[zone_name] = polylines

    # Cross-zone merge policy.
    merged: Dict[str, np.ndarray] = {}

    cb_z1 = per_zone.get("zone_1_cranial_base", {}).get(
        "cranial_base", np.empty((0, 2)),
    )
    cb_z4 = per_zone.get("zone_4_posterior", {}).get(
        "cranial_base", np.empty((0, 2)),
    )
    merged["cranial_base"] = cb_z1 if len(cb_z1) else cb_z4

    mb_z3 = per_zone.get("zone_3_mandible", {}).get(
        "mandibular_border", np.empty((0, 2)),
    )
    mb_z4 = per_zone.get("zone_4_posterior", {}).get(
        "mandibular_border", np.empty((0, 2)),
    )
    merged["mandibular_border"] = mb_z3 if len(mb_z3) else mb_z4

    merged["palatal_plane"] = per_zone.get("zone_2_midface", {}).get(
        "palatal_plane", np.empty((0, 2)),
    )
    merged["upper_incisor_axis"] = per_zone.get("zone_2_midface", {}).get(
        "upper_incisor_axis", np.empty((0, 2)),
    )
    merged["mandibular_symphysis"] = per_zone.get("zone_3_mandible", {}).get(
        "mandibular_symphysis", np.empty((0, 2)),
    )
    merged["lower_incisor_axis"] = per_zone.get("zone_3_mandible", {}).get(
        "lower_incisor_axis", np.empty((0, 2)),
    )

    primary_names = (
        "cranial_base", "mandibular_border", "palatal_plane", "mandibular_symphysis",
    )
    missing_primary = [n for n in primary_names if len(merged[n]) <= 1]
    full_success = len(missing_primary) == 0
    return merged, full_success, missing_primary


# ── Phase 0D: Douglas-Peucker + anchor extraction ────────────────────────────


def _run_phase0d(
    contour_polylines_orig: Dict[str, np.ndarray],
    soft_tissue_orig: Dict[str, Tuple[float, float]],
    meta: Dict[str, Any],
    features: Optional[Dict[str, Any]],
) -> Dict[str, Tuple[float, float]]:
    """Simplify contours and extract 7 anchor landmarks.

    Returns dict mapping anchor name → (x, y) in original image coords.
    Anchors that could not be extracted are absent from the returned dict.
    """
    orig_w = meta["orig_w"]
    orig_h = meta["orig_h"]
    px_per_mm = (
        (orig_w / CEPH_WIDTH_MM) + (orig_h / CEPH_HEIGHT_MM)
    ) / 2.0

    # Simplify each contour with class-specific Douglas-Peucker tolerance.
    simplified: Dict[str, np.ndarray] = {}
    for cls_name, poly in contour_polylines_orig.items():
        if len(poly) < 2:
            simplified[cls_name] = poly
            continue
        tol_mm = DP_TOLERANCE_MM.get(cls_name, 1.0)
        eps = max(1e-3, tol_mm * px_per_mm)
        s = cv2.approxPolyDP(
            poly.astype(np.float32).reshape(-1, 1, 2),
            epsilon=eps, closed=False,
        ).reshape(-1, 2).astype(np.float64)
        simplified[cls_name] = s

    anchors: Dict[str, Tuple[float, float]] = {}

    # Prn — soft tissue passthrough (Phase 0A/B detected it on the profile).
    if "Prn" in soft_tissue_orig:
        anchors["Prn"] = soft_tissue_orig["Prn"]

    def _in_orig(pt_proc512: Tuple[int, int]) -> Tuple[float, float]:
        return (
            pt_proc512[0] * meta["scale_x"],
            pt_proc512[1] * meta["scale_y"],
        )

    ref_chin = _in_orig(features["chin_bottom"]) if features else None
    ref_nose = _in_orig(features["nose_tip"]) if features else None

    # Cranial base: orient with far-endpoint=Ba-side, near-endpoint=N-side.
    cb = simplified.get("cranial_base", np.empty((0, 2)))
    if len(cb) >= 3 and ref_nose is not None:
        cb = _orient_polyline_far_then_near(cb, ref_nose)
        anchors["N"] = (float(cb[-1, 0]), float(cb[-1, 1]))
        s_pt = _anchor_by_chord_window(cb, frac_lo=0.20, frac_hi=0.60)
        if s_pt is not None:
            anchors["S"] = s_pt
    elif len(cb) >= 2 and ref_nose is None:
        anchors["N"] = (float(cb[-1, 0]), float(cb[-1, 1]))

    # Mandibular symphysis: orient with far=B-side, near=Me-side (chin bottom).
    ms = simplified.get("mandibular_symphysis", np.empty((0, 2)))
    if len(ms) >= 3 and ref_chin is not None:
        ms = _orient_polyline_far_then_near(ms, ref_chin)
        anchors["Me"] = (float(ms[-1, 0]), float(ms[-1, 1]))
        pog_pt = _anchor_by_chord_window(ms, frac_lo=0.10, frac_hi=0.65)
        if pog_pt is not None:
            anchors["Pog"] = pog_pt

    # Palatal plane: orient with far=PNS-side (posterior), near=ANS-side.
    pp = simplified.get("palatal_plane", np.empty((0, 2)))
    if len(pp) >= 2 and ref_nose is not None:
        pp = _orient_polyline_far_then_near(pp, ref_nose)
        anchors["ANS"] = (float(pp[-1, 0]), float(pp[-1, 1]))

    # Mandibular border: orient far=Cd-side, near=B-side; Go = max curvature.
    mb = simplified.get("mandibular_border", np.empty((0, 2)))
    if len(mb) >= 5 and ref_chin is not None:
        mb = _orient_polyline_far_then_near(mb, ref_chin)
        go_pt = _anchor_by_curvature_window(mb, frac_lo=0.15, frac_hi=0.50)
        if go_pt is not None:
            anchors["Go"] = go_pt

    return anchors


# ── Phase 0E: MLP + attention map rendering ──────────────────────────────────


def _run_phase0e(
    session: ort.InferenceSession,
    anchors_orig: Dict[str, Tuple[float, float]],
    soft_tissue_orig: Dict[str, Tuple[float, float]],
    meta: Dict[str, Any],
) -> Optional[np.ndarray]:
    """Run Phase 0E MLP, merge derived + anchor + soft-tissue positions,
    render the 25 Gaussian attention maps at 256×256.

    Returns (25, 256, 256) float32 in CANONICAL_25 channel order, or None
    on MLP failure.
    """
    orig_w = float(meta["orig_w"])
    orig_h = float(meta["orig_h"])

    # Build (1, 14) MLP input in ANCHOR_NAMES order, normalized to [0, 1].
    anc_in: list = []
    for name in ANCHOR_NAMES:
        if name in anchors_orig:
            x, y = anchors_orig[name]
            anc_in.append(float(x) / orig_w)
            anc_in.append(float(y) / orig_h)
        else:
            anc_in.extend([0.5, 0.5])  # missing → centered fallback

    inp = np.array([anc_in], dtype=np.float32)  # (1, 14)

    try:
        input_name = session.get_inputs()[0].name
        out = session.run(None, {input_name: inp})[0]  # (1, 36)
    except Exception:
        return None

    derived = out[0]  # (36,)

    # Build normalized positions for all 25 landmarks in CANONICAL_25 order.
    positions_norm: Dict[str, Tuple[float, float]] = {}
    for i, name in enumerate(DERIVED_NAMES):
        positions_norm[name] = (float(derived[2 * i]), float(derived[2 * i + 1]))

    # Anchor re-insertion overrides any DERIVED overlap (there shouldn't be any).
    for i, name in enumerate(ANCHOR_NAMES):
        positions_norm[name] = (float(anc_in[2 * i]), float(anc_in[2 * i + 1]))

    # Soft-tissue overrides for UL, LL, Sn, Pog_soft, Pm.
    for name in ("UL", "LL", "Sn", "Pog_soft", "Pm"):
        if name in soft_tissue_orig:
            sx, sy = soft_tissue_orig[name]
            positions_norm[name] = (float(sx) / orig_w, float(sy) / orig_h)

    return render_attention_maps_256(positions_norm)


def render_attention_maps_256(
    positions_norm: Dict[str, Tuple[float, float]],
) -> np.ndarray:
    """Render the 25-channel attention map stack at 256×256.

    Args:
        positions_norm: Dict mapping landmark name → (x_norm, y_norm) in
            [0, 1] coordinates. Missing landmarks produce zero channels.

    Returns:
        attn: (25, 256, 256) float32 in CANONICAL_25 channel order. Each
        channel is a 2D Gaussian with peak 1.0 at the landmark location and
        the per-landmark σ from `SIGMA_BY_LANDMARK`. Missing landmarks
        produce an all-zero channel.
    """
    attn = np.zeros(
        (NUM_LANDMARKS, ATTENTION_SIZE, ATTENTION_SIZE), dtype=np.float32,
    )
    yy, xx = np.mgrid[0:ATTENTION_SIZE, 0:ATTENTION_SIZE].astype(np.float32)

    for ch, name in enumerate(CANONICAL_25):
        if name not in positions_norm:
            continue  # absent → all-zero channel
        nx, ny = positions_norm[name]
        cx = nx * ATTENTION_SIZE
        cy = ny * ATTENTION_SIZE
        sigma = SIGMA_BY_LANDMARK[name]
        attn[ch] = np.exp(
            -((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma * sigma)
        )

    return attn


def _upsample_attention_to_512(attn_256: np.ndarray) -> np.ndarray:
    """Bilinear upsample (25, 256, 256) → (1, 25, 512, 512) for Stage 1 input."""
    out = np.zeros(
        (NUM_LANDMARKS, INPUT_SIZE, INPUT_SIZE), dtype=np.float32,
    )
    for c in range(NUM_LANDMARKS):
        out[c] = cv2.resize(
            attn_256[c], (INPUT_SIZE, INPUT_SIZE), interpolation=cv2.INTER_LINEAR,
        )
    return out[np.newaxis, :, :, :]


# ── Orchestrator ─────────────────────────────────────────────────────────────


def run_stage0(
    sessions: Dict[str, ort.InferenceSession],
    image: Union[bytes, str],
) -> Stage0Result:
    """Run the full Stage 0 chain on a single image.

    Args:
        sessions: Dict mapping model keys to loaded ONNX Runtime sessions.
            Required keys: stage0_profile, stage0_contours_z1..z4,
            stage0_derive. If any are missing, Stage 0 returns a degraded
            result with `attention_maps=None`.
        image: File path or raw image bytes. Decoded to grayscale via
            OpenCV at native resolution (Phase 0A) and at 512×512 for
            zone processing (Phase 0B onward).

    Returns:
        Stage0Result. When degraded, `attention_maps` is None and the
        caller should run Stage 1 with zero-filled attention channels.
    """
    if isinstance(image, str):
        with open(image, "rb") as f:
            image_bytes = f.read()
    elif isinstance(image, bytes):
        image_bytes = image
    else:
        raise TypeError(
            f"image must be str or bytes, got {type(image).__name__}"
        )

    substage_status = {"0A": False, "0B": False, "0C": False, "0D": False, "0E": False}
    details: Dict[str, Dict[str, Any]] = {
        "0A": {}, "0B": {}, "0C": {}, "0D": {}, "0E": {},
    }

    required_keys = (
        "stage0_profile",
        "stage0_contours_z1", "stage0_contours_z2",
        "stage0_contours_z3", "stage0_contours_z4",
        "stage0_derive",
    )
    for key in required_keys:
        if key not in sessions:
            details["0A"]["reason"] = f"missing session: {key}"
            return Stage0Result(
                attention_maps=None,
                substage_status=substage_status,
                details=details,
                degraded=True,
            )

    # Phase 0A
    mask_512, img_512_u8, meta, nonzero = _run_phase0a(
        sessions["stage0_profile"], image_bytes,
    )
    details["0A"]["mask_nonzero_pixels"] = nonzero
    details["0A"]["empty_mask_threshold"] = EMPTY_MASK_THRESHOLD
    if mask_512 is None:
        return Stage0Result(
            attention_maps=None,
            substage_status=substage_status,
            details=details,
            degraded=True,
        )
    substage_status["0A"] = True

    # Phase 0B
    zone_data, features, soft_tissue_512, b_used_fallback = _run_phase0b(
        mask_512, img_512_u8,
    )
    soft_tissue_expected = ("Prn", "Sn", "UL", "LL", "Pog_soft", "Pm")
    st_missing = [n for n in soft_tissue_expected if n not in soft_tissue_512]
    details["0B"] = {
        "soft_tissue_found": len(soft_tissue_512),
        "soft_tissue_expected": len(soft_tissue_expected),
        "missing": st_missing,
        "used_fallback_zones": b_used_fallback,
    }
    substage_status["0B"] = (not b_used_fallback) and (len(st_missing) == 0)

    # Phase 0C
    contour_polylines_orig, c_full_success, c_missing = _run_phase0c(
        sessions, zone_data, meta,
    )
    details["0C"] = {
        "primary_present": 4 - len(c_missing),
        "primary_expected": 4,
        "missing": c_missing,
    }
    substage_status["0C"] = c_full_success

    # Rescale soft-tissue landmarks to original-image coords for 0D + 0E.
    scale_x_o = meta["scale_x"]
    scale_y_o = meta["scale_y"]
    soft_tissue_orig: Dict[str, Tuple[float, float]] = {
        name: (float(xy[0] * scale_x_o), float(xy[1] * scale_y_o))
        for name, xy in soft_tissue_512.items()
    }

    # Phase 0D
    anchors_orig = _run_phase0d(
        contour_polylines_orig, soft_tissue_orig, meta, features,
    )
    d_missing = [n for n in ANCHOR_NAMES if n not in anchors_orig]
    details["0D"] = {
        "anchors_found": len(anchors_orig),
        "anchors_expected": len(ANCHOR_NAMES),
        "missing": d_missing,
    }
    substage_status["0D"] = len(d_missing) == 0

    # Phase 0E
    attn_256 = _run_phase0e(
        sessions["stage0_derive"], anchors_orig, soft_tissue_orig, meta,
    )
    if attn_256 is None:
        details["0E"] = {"mlp_ran": False}
        return Stage0Result(
            attention_maps=None,
            substage_status=substage_status,
            details=details,
            degraded=True,
        )
    substage_status["0E"] = True
    details["0E"] = {"mlp_ran": True}

    attention_512 = _upsample_attention_to_512(attn_256)

    degraded = not all(substage_status.values())
    return Stage0Result(
        attention_maps=attention_512,
        substage_status=substage_status,
        details=details,
        degraded=degraded,
    )
