"""Evaluate CephTrace v4 against per-image ground-truth landmark annotations.

Computes Mean Radial Error (MRE) in millimetres and Successful Detection Rate
(SDR) at configurable thresholds (defaults 2.0, 2.5, 3.0, 4.0 mm). Calibration
is done per-image using the Proffit-mean Sella-Nasion distance:

    mm_per_pixel = SN_DISTANCE_MM / pixel_distance(S, N)

The S-N distance is computed from ground-truth coordinates so the calibration
itself does not depend on model predictions.

Ground-truth file format (one JSON per image, same stem as the image):

    {
        "S":   [x, y],
        "N":   [x, y],
        "Or":  [x, y],
        ...
    }

Coordinates may be ints or floats; missing landmarks (key absent or value
null) are skipped during MRE/SDR aggregation. Landmark names must match the
`CANONICAL_25` ordering used by the predictor.

Usage:
    python scripts/evaluate.py \\
        --image-dir test_images/ \\
        --gt-dir test_landmarks/ \\
        --output results.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cephtrace import CephTracePredictor  # noqa: E402
from cephtrace.constants import CANONICAL_25, SN_DISTANCE_MM  # noqa: E402


DEFAULT_SDR_THRESHOLDS_MM = (2.0, 2.5, 3.0, 4.0)


def _load_gt(path: Path) -> Dict[str, Tuple[float, float]]:
    """Load a ground-truth JSON file and return a {name: (x, y)} dict."""
    data = json.loads(path.read_text())
    out: Dict[str, Tuple[float, float]] = {}
    for name, value in data.items():
        if value is None:
            continue
        if isinstance(value, dict):
            x = value.get("x")
            y = value.get("y")
            if x is None or y is None:
                continue
            out[name] = (float(x), float(y))
        elif isinstance(value, (list, tuple)) and len(value) >= 2:
            out[name] = (float(value[0]), float(value[1]))
    return out


def _compute_mm_per_pixel(gt: Dict[str, Tuple[float, float]]) -> Optional[float]:
    """Compute S-N landmark calibration. Returns None if S or N is absent."""
    if "S" not in gt or "N" not in gt:
        return None
    sx, sy = gt["S"]
    nx, ny = gt["N"]
    pixel_dist = math.hypot(sx - nx, sy - ny)
    if pixel_dist <= 1.0:
        return None
    return SN_DISTANCE_MM / pixel_dist


def _per_landmark_errors_mm(
    pred: Dict[str, Tuple[float, float]],
    gt: Dict[str, Tuple[float, float]],
    mm_per_pixel: float,
) -> Dict[str, float]:
    """Return {landmark_name: error_mm} for landmarks present in BOTH dicts."""
    errors: Dict[str, float] = {}
    for name in CANONICAL_25:
        if name not in gt or name not in pred:
            continue
        px, py = pred[name]
        gx, gy = gt[name]
        err_px = math.hypot(px - gx, py - gy)
        errors[name] = err_px * mm_per_pixel
    return errors


def _gather_pairs(
    image_dir: Path, gt_dir: Path,
) -> List[Tuple[Path, Path]]:
    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    images = sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in image_exts and p.is_file()
    )
    pairs: List[Tuple[Path, Path]] = []
    for img in images:
        gt_path = gt_dir / f"{img.stem}.json"
        if gt_path.is_file():
            pairs.append((img, gt_path))
    return pairs


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate CephTrace v4 predictions against ground-truth landmark "
            "annotations. Reports MRE and SDR per-landmark and overall."
        ),
    )
    parser.add_argument(
        "--image-dir", type=Path, required=True,
        help="Directory of cephalogram images.",
    )
    parser.add_argument(
        "--gt-dir", type=Path, required=True,
        help="Directory of ground-truth landmark JSON files (one per image).",
    )
    parser.add_argument(
        "--model-dir", type=Path, default=Path("models"),
        help="Directory containing downloaded ONNX models (default: ./models).",
    )
    parser.add_argument(
        "--output", "-o", type=Path,
        help="Output JSON path for the full evaluation report.",
    )
    parser.add_argument(
        "--sdr-thresholds-mm", type=float, nargs="+",
        default=list(DEFAULT_SDR_THRESHOLDS_MM),
        help="SDR thresholds in mm (default: 2.0 2.5 3.0 4.0).",
    )
    parser.add_argument(
        "--gpu", action="store_true",
        help="Use CUDA execution provider (requires onnxruntime-gpu).",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Fail if any Stage 0 model is missing.",
    )
    parser.add_argument(
        "--limit", type=int, default=None,
        help="Limit evaluation to the first N image/GT pairs (for smoke tests).",
    )
    args = parser.parse_args(argv)

    pairs = _gather_pairs(args.image_dir, args.gt_dir)
    if not pairs:
        print(
            f"No matching image/GT pairs found in {args.image_dir} / "
            f"{args.gt_dir}",
            file=sys.stderr,
        )
        return 1
    if args.limit is not None:
        pairs = pairs[: args.limit]

    print(f"Loading models from {args.model_dir} ...", flush=True)
    predictor = CephTracePredictor(
        model_dir=args.model_dir,
        use_gpu=args.gpu,
        strict=args.strict,
    )
    if not predictor.stage0_available:
        print("[warn] Stage 0 unavailable — running in degraded mode.", file=sys.stderr)

    # Accumulate per-landmark errors across the whole test set.
    per_landmark_errors: Dict[str, List[float]] = {name: [] for name in CANONICAL_25}
    skipped_no_calibration: List[str] = []

    n_total = len(pairs)
    print(f"Evaluating {n_total} image(s) ...", flush=True)

    for i, (img_path, gt_path) in enumerate(pairs, start=1):
        gt = _load_gt(gt_path)
        mm_per_pixel = _compute_mm_per_pixel(gt)
        if mm_per_pixel is None:
            skipped_no_calibration.append(str(img_path))
            print(
                f"  [{i:>4}/{n_total}] {img_path.name}  -- skipped "
                f"(no S/N in GT for calibration)",
                flush=True,
            )
            continue

        result = predictor.predict(img_path)
        pred = {lm.name: (lm.x, lm.y) for lm in result.landmarks}
        errors = _per_landmark_errors_mm(pred, gt, mm_per_pixel)

        for name, err_mm in errors.items():
            per_landmark_errors[name].append(err_mm)

        mean_err = (
            float(np.mean(list(errors.values()))) if errors else float("nan")
        )
        print(
            f"  [{i:>4}/{n_total}] {img_path.name}  "
            f"mm/px={mm_per_pixel:.4f}  MRE={mean_err:.3f} mm  "
            f"(n_lm={len(errors)})",
            flush=True,
        )

    # Aggregate.
    per_lm_summary: Dict[str, Dict] = {}
    all_errors: List[float] = []
    for name in CANONICAL_25:
        errs = per_landmark_errors[name]
        if not errs:
            per_lm_summary[name] = {
                "n": 0, "mre_mm": None,
                **{f"sdr_{t}mm": None for t in args.sdr_thresholds_mm},
            }
            continue
        e = np.array(errs, dtype=np.float64)
        per_lm_summary[name] = {
            "n": int(e.size),
            "mre_mm": float(np.mean(e)),
            "median_mm": float(np.median(e)),
            "p95_mm": float(np.percentile(e, 95.0)),
        }
        for t in args.sdr_thresholds_mm:
            per_lm_summary[name][f"sdr_{t}mm"] = float(np.mean(e <= t))
        all_errors.extend(errs)

    overall: Dict[str, float] = {}
    if all_errors:
        a = np.array(all_errors, dtype=np.float64)
        overall["mre_mm"] = float(np.mean(a))
        overall["median_mm"] = float(np.median(a))
        overall["p95_mm"] = float(np.percentile(a, 95.0))
        for t in args.sdr_thresholds_mm:
            overall[f"sdr_{t}mm"] = float(np.mean(a <= t))

    # Print summary.
    print("\n=== Per-landmark MRE (sorted by error) ===")
    sortable = [
        (name, info["mre_mm"], info["n"])
        for name, info in per_lm_summary.items()
        if info["mre_mm"] is not None
    ]
    sortable.sort(key=lambda x: x[1], reverse=True)
    print(f"{'Landmark':<10}  {'MRE (mm)':>10}  {'n':>5}")
    print("-" * 30)
    for name, mre, n in sortable:
        print(f"{name:<10}  {mre:>10.3f}  {n:>5}")

    print("\n=== Overall ===")
    for k, v in overall.items():
        if k.startswith("sdr_"):
            print(f"  {k:<14}  {v * 100:>6.2f}%")
        else:
            print(f"  {k:<14}  {v:>6.3f} mm")
    print(f"  total_landmark_observations: {len(all_errors)}")
    print(f"  images_evaluated: {n_total - len(skipped_no_calibration)}")
    print(f"  images_skipped_no_calibration: {len(skipped_no_calibration)}")

    report = {
        "model_dir": str(args.model_dir),
        "image_dir": str(args.image_dir),
        "gt_dir": str(args.gt_dir),
        "n_images": n_total,
        "n_images_evaluated": n_total - len(skipped_no_calibration),
        "skipped_no_calibration": skipped_no_calibration,
        "sdr_thresholds_mm": list(args.sdr_thresholds_mm),
        "per_landmark": per_lm_summary,
        "overall": overall,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2))
        print(f"\nWrote full report to {args.output}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
