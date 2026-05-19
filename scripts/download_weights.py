#!/usr/bin/env python3
"""Download CephTrace v4 ONNX model weights from Hugging Face Hub.

Usage:
    python scripts/download_weights.py
    python scripts/download_weights.py --output ./my_models
    python scripts/download_weights.py --force
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

try:
    from huggingface_hub import hf_hub_download
except ImportError as e:
    print(
        "huggingface-hub is required. Install with: pip install huggingface-hub",
        file=sys.stderr,
    )
    raise SystemExit(1) from e

# Allow running as a script from the repo root without installing the package.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# --------------------------------------------------------------------------
# Configuration
# --------------------------------------------------------------------------

HF_REPO_ID = "CephTrace/cephtrace-v4"
HF_REVISION = "main"

# local_path -> huggingface_filename
MODEL_FILES = {
    "stage0/v4_stage0_profile.onnx": "v4_stage0_profile.onnx",
    "stage0/z1_cranial_base_contours.onnx": "z1_cranial_base_contours.onnx",
    "stage0/z2_midface_contours.onnx": "z2_midface_contours.onnx",
    "stage0/z3_mandible_contours.onnx": "z3_mandible_contours.onnx",
    "stage0/z4_posterior_contours.onnx": "z4_posterior_contours.onnx",
    "stage0/phase0e_model.onnx": "phase0e_model.onnx",
    "stage1/v4_stage1.onnx": "v4_stage1.onnx",
}

MODEL_SIZES_MB = {
    "stage0/v4_stage0_profile.onnx": 25.6,
    "stage0/z1_cranial_base_contours.onnx": 25.6,
    "stage0/z2_midface_contours.onnx": 25.6,
    "stage0/z3_mandible_contours.onnx": 25.6,
    "stage0/z4_posterior_contours.onnx": 25.6,
    "stage0/phase0e_model.onnx": 0.4,
    "stage1/v4_stage1.onnx": 123.9,
}


def _humanize_mb(size_bytes: int) -> str:
    return f"{size_bytes / (1024 * 1024):.1f} MB"


def download_all(
    output_dir: Path,
    repo_id: str = HF_REPO_ID,
    revision: str = HF_REVISION,
    keys: Iterable[str] = tuple(MODEL_FILES.keys()),
    force: bool = False,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "stage0").mkdir(exist_ok=True)
    (output_dir / "stage1").mkdir(exist_ok=True)

    total_planned = 0
    total_downloaded = 0
    total_skipped = 0

    for relpath in keys:
        dest = output_dir / relpath
        hf_filename = MODEL_FILES[relpath]
        expected_mb = MODEL_SIZES_MB.get(relpath, 0.0)
        total_planned += 1

        if dest.is_file() and not force:
            size = dest.stat().st_size
            size_mb = size / (1024 * 1024)
            if size_mb > expected_mb * 0.5:
                total_skipped += 1
                print(f"  [skip] {relpath:46s}  {_humanize_mb(size)}  (already exists)")
                continue
            else:
                print(f"  [warn] {relpath:46s}  {_humanize_mb(size)}  (too small, redownloading)")

        print(f"  [get]  {relpath:46s}  ~{expected_mb:.1f} MB")

        local_path = hf_hub_download(
            repo_id=repo_id,
            filename=hf_filename,
            revision=revision,
            local_dir=str(output_dir),
            local_dir_use_symlinks=False,
        )

        # huggingface-hub may place the file at output_dir/hf_filename (flat)
        # but we need it at output_dir/relpath (with subdirectory)
        local_path = Path(local_path)
        if not dest.is_file():
            dest.parent.mkdir(parents=True, exist_ok=True)
            if local_path.is_file():
                local_path.rename(dest)

        if dest.is_file():
            print(f"         -> saved to {dest}  ({_humanize_mb(dest.stat().st_size)})")
            total_downloaded += 1
        else:
            print(f"         -> WARNING: file not found at {dest}")

    print(f"\nDone. {total_downloaded} downloaded, {total_skipped} skipped, {total_planned} total.")
    print(f"Models directory: {output_dir.resolve()}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Download CephTrace v4 ONNX model weights from Hugging Face Hub.",
    )
    parser.add_argument(
        "--output",
        default=Path("models"),
        help="Output directory (default: ./models)",
    )
    parser.add_argument(
        "--repo-id",
        default=HF_REPO_ID,
        help=f"Hugging Face repo id (default: {HF_REPO_ID})",
    )
    parser.add_argument(
        "--revision",
        default=HF_REVISION,
        help=f"Repo revision/tag (default: {HF_REVISION})",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        choices=list(MODEL_FILES.keys()),
        help="Download only the specified model keys.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-download files even if they already exist locally.",
    )
    args = parser.parse_args(argv)

    keys = tuple(args.only) if args.only else tuple(MODEL_FILES.keys())

    try:
        download_all(
            output_dir=Path(args.output),
            repo_id=args.repo_id,
            revision=args.revision,
            keys=keys,
            force=args.force,
        )
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        return 130
    except Exception as e:
        print(f"\nError: {e}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
