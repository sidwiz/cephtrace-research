"""Run CephTrace v4 inference on one or many cephalogram images.

Usage:
    # Single image, print to stdout
    python scripts/run_inference.py --image /path/to/ceph.jpg

    # Single image, write JSON
    python scripts/run_inference.py --image ceph.jpg --output result.json

    # Single image, save annotated overlay
    python scripts/run_inference.py --image ceph.jpg --visualize --output overlay.png

    # Batch processing
    python scripts/run_inference.py --image-dir cephs/ --output results.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable, List

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from cephtrace import (  # noqa: E402
    CephTracePredictor,
    PredictionResult,
)
from cephtrace.constants import CONFIDENCE_LABEL_BY_LANDMARK  # noqa: E402


# Colors for the three Stage 0 confidence tiers (RGB tuples for PIL).
TIER_COLORS = {
    "high":   (16, 185, 129),   # emerald-500
    "medium": (59, 130, 246),   # blue-500
    "low":    (245, 158, 11),   # amber-500
}


def _print_one(result: PredictionResult, image_path: str) -> None:
    print(f"\n=== {image_path} ===")
    print(
        f"Image: {result.original_width} x {result.original_height}  "
        f"degraded={result.degraded}  "
        f"stage0_status={result.stage0_status}"
    )
    print(
        f"{'Idx':>4}  {'Name':<10}  {'Tier':<7}  {'Conf':<6}  "
        f"{'Peak':>7}  {'X':>10}  {'Y':>10}"
    )
    print("-" * 70)
    for lm in result.landmarks:
        icon = {"high": "●", "medium": "◆", "low": "○"}[lm.confidence_label]
        print(
            f"{lm.index:>4}  {lm.name:<10}  {lm.tier_label:<7}  "
            f"{icon} {lm.confidence_label:<4}  "
            f"{lm.peak:>7.3f}  {lm.x:>10.2f}  {lm.y:>10.2f}"
        )


def _render_overlay(
    image_path: Path, result: PredictionResult, output_path: Path,
) -> None:
    from PIL import Image, ImageDraw, ImageFont  # local import — optional

    img = Image.open(image_path).convert("RGB")
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.load_default()
    except Exception:  # pragma: no cover
        font = None

    radius = max(4, int(0.005 * min(img.width, img.height)))
    label_offset = radius + 4

    for lm in result.landmarks:
        color = TIER_COLORS[lm.tier_label]
        x, y = float(lm.x), float(lm.y)
        # Filled circle.
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=color,
            outline=(255, 255, 255),
            width=2,
        )
        # Landmark label.
        if font is not None:
            draw.text(
                (x + label_offset, y - label_offset),
                lm.name,
                fill=color,
                font=font,
            )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(str(output_path))


def _gather_images(image_dir: Path) -> List[Path]:
    exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    return sorted(
        p for p in image_dir.iterdir()
        if p.suffix.lower() in exts and p.is_file()
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Run CephTrace v4 landmark detection on a single image or a "
            "directory of images."
        ),
    )
    src = parser.add_mutually_exclusive_group(required=True)
    src.add_argument(
        "--image", type=Path, help="Path to a single cephalogram image."
    )
    src.add_argument(
        "--image-dir", type=Path,
        help="Directory of cephalogram images (JPEG / PNG / BMP / TIFF).",
    )

    parser.add_argument(
        "--model-dir", type=Path, default=Path("models"),
        help="Directory containing the downloaded ONNX models (default: ./models)",
    )
    parser.add_argument(
        "--output", "-o", type=Path,
        help=(
            "Output file (.json for predictions, .png for overlay when "
            "--visualize is set). Required when --image-dir is used or "
            "--visualize is set."
        ),
    )
    parser.add_argument(
        "--visualize", action="store_true",
        help="Render an annotated overlay PNG instead of (or in addition to) JSON.",
    )
    parser.add_argument(
        "--gpu", action="store_true",
        help="Use CUDA execution provider (requires onnxruntime-gpu).",
    )
    parser.add_argument(
        "--strict", action="store_true",
        help="Fail if any Stage 0 model is missing (default: degrade gracefully).",
    )
    parser.add_argument(
        "--quiet", "-q", action="store_true",
        help="Suppress per-landmark printout.",
    )

    args = parser.parse_args(argv)

    if args.visualize and args.image_dir is not None and args.output is None:
        parser.error("--visualize with --image-dir requires --output to be a directory")

    if args.visualize and args.image is not None and args.output is None:
        parser.error("--visualize requires --output (a .png path)")

    print(f"Loading models from {args.model_dir} ...", flush=True)
    predictor = CephTracePredictor(
        model_dir=args.model_dir,
        use_gpu=args.gpu,
        strict=args.strict,
    )
    if not predictor.stage0_available:
        print(
            "[warn] Stage 0 models not fully loaded — predictions will run in "
            "degraded mode (no attention priors).",
            file=sys.stderr,
        )

    targets: List[Path] = (
        [args.image] if args.image is not None else _gather_images(args.image_dir)
    )
    if not targets:
        print("No images found.", file=sys.stderr)
        return 1

    results: dict = {}
    for img_path in targets:
        result = predictor.predict(img_path)
        if not args.quiet:
            _print_one(result, str(img_path))
        results[str(img_path)] = result.to_dict()

        if args.visualize and args.output is not None:
            if args.image_dir is not None:
                # Output a separate PNG per image inside the output directory.
                out_dir = args.output
                out_dir.mkdir(parents=True, exist_ok=True)
                _render_overlay(img_path, result, out_dir / f"{img_path.stem}_overlay.png")
            else:
                _render_overlay(img_path, result, args.output)

    if args.output is not None and not args.visualize:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # If only one image was processed, flatten the dict to its inner record
        # for cleaner JSON output.
        payload = next(iter(results.values())) if len(results) == 1 else results
        args.output.write_text(json.dumps(payload, indent=2))
        print(f"\nWrote predictions to {args.output}")

    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
