"""Image preprocessing for CephTrace v4 inference.

Two preprocessing paths are exposed:

1. `preprocess_rgb_for_stage1` — produces the (1, 3, 512, 512) float32 RGB
   tensor that Stage 1 consumes (after concatenation with attention maps).

2. `to_grayscale_tensor_512` — produces the (1, 1, 512, 512) float32
   grayscale tensor that Phase 0A (soft-tissue profile detector) consumes.
   It is intentionally derived directly from the original image rather than
   by reusing the RGB tensor — Phase 0A was trained on `cv2.imdecode` output
   and reusing the PIL-resized RGB tensor introduces a small but real
   numerical drift in the mask boundary.

Both paths use simple `/255` normalization. The pipeline does NOT use
ImageNet-style mean/std subtraction because cephalometric X-rays have a
very different intensity distribution from natural photographs, and Stage 1
was trained with `/255` normalization end-to-end. Changing this here without
retraining will degrade accuracy.

Coordinate spaces produced:
    - `proc512`        : 512×512 grayscale, [0, 255] uint8 or [0, 1] float32
    - `stage1_input`   : 512×512 RGB,       [0, 1] float32, channel-first
"""

from __future__ import annotations

import io
from typing import Dict, Tuple, Union

import numpy as np
from PIL import Image

from .constants import INPUT_SIZE

__all__ = [
    "preprocess_rgb_for_stage1",
    "to_grayscale_tensor_512",
    "to_grayscale_orig",
    "load_image_bytes",
    "decode_to_array",
]


def load_image_bytes(image_path: str) -> bytes:
    """Read an image file from disk into raw bytes.

    Args:
        image_path: Filesystem path to a JPEG, PNG, or any other format
            supported by Pillow.

    Returns:
        The file contents as raw bytes.
    """
    with open(image_path, "rb") as f:
        return f.read()


def decode_to_array(image: Union[bytes, np.ndarray, str]) -> np.ndarray:
    """Decode an image input into a (H, W, 3) uint8 RGB numpy array.

    Args:
        image: One of:
            - File path (str) — opened with Pillow.
            - Raw image bytes — opened with Pillow.
            - Numpy array of shape (H, W, 3) RGB uint8 — passed through.
            - Numpy array of shape (H, W) — broadcast to 3 channels.
            - Numpy array of shape (H, W, 1) — broadcast to 3 channels.

    Returns:
        rgb: uint8 array of shape (H, W, 3) in RGB channel order.

    Raises:
        TypeError: If `image` is not one of the supported types.
    """
    if isinstance(image, str):
        pil = Image.open(image).convert("RGB")
        return np.asarray(pil, dtype=np.uint8)
    if isinstance(image, bytes):
        pil = Image.open(io.BytesIO(image)).convert("RGB")
        return np.asarray(pil, dtype=np.uint8)
    if isinstance(image, np.ndarray):
        arr = image
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        elif arr.ndim == 3 and arr.shape[2] == 1:
            arr = np.concatenate([arr] * 3, axis=-1)
        if arr.dtype != np.uint8:
            arr = arr.astype(np.uint8)
        return arr
    raise TypeError(
        f"image must be str, bytes, or np.ndarray, got {type(image).__name__}"
    )


def preprocess_rgb_for_stage1(
    image: Union[bytes, np.ndarray, str],
    target_size: int = INPUT_SIZE,
) -> Tuple[np.ndarray, Dict]:
    """Resize and normalize an image for Stage 1 ONNX input.

    Pipeline:
        1. Decode to (H, W, 3) uint8 RGB numpy array.
        2. Resize to `(target_size, target_size)` using PIL bilinear
           interpolation. PIL is used (rather than OpenCV) to match the
           training-time preprocessing exactly.
        3. Normalize from [0, 255] to [0, 1] by dividing by 255.0.
        4. Transpose HWC → CHW.
        5. Add batch dimension → (1, 3, target_size, target_size).

    Args:
        image: File path, raw bytes, or (H, W, 3) uint8 array.
        target_size: Output spatial dimension. Default 512.

    Returns:
        tensor: float32 array of shape (1, 3, target_size, target_size),
            values in [0.0, 1.0].
        metadata: Dictionary with keys:
            - original_width (int)
            - original_height (int)
            - scale_x (float): target_size / original_width
            - scale_y (float): target_size / original_height
            - target_size (int)

        The metadata is used to back-project Stage 1 heatmap-space
        coordinates back to original-image pixel space.
    """
    rgb = decode_to_array(image)
    original_height, original_width = rgb.shape[:2]

    pil_img = Image.fromarray(rgb).resize(
        (target_size, target_size), Image.Resampling.BILINEAR,
    )
    arr = np.asarray(pil_img, dtype=np.float32) / 255.0

    tensor = arr.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)

    metadata: Dict = {
        "original_width": int(original_width),
        "original_height": int(original_height),
        "scale_x": float(target_size) / float(original_width),
        "scale_y": float(target_size) / float(original_height),
        "target_size": int(target_size),
    }
    return tensor, metadata


def to_grayscale_tensor_512(
    image: Union[bytes, np.ndarray, str],
    target_size: int = INPUT_SIZE,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Decode an image as grayscale and resize to (target_size, target_size).

    Produces the input tensor for Phase 0A (soft-tissue profile segmentation).
    Phase 0A expects 1-channel grayscale, not RGB — do NOT reuse the tensor
    from `preprocess_rgb_for_stage1`.

    Args:
        image: File path, raw bytes, or numpy array.
        target_size: Output spatial dimension. Default 512.

    Returns:
        tensor: float32 array of shape (1, 1, target_size, target_size),
            values in [0.0, 1.0].
        img_uint8: uint8 array of shape (target_size, target_size) — the
            same grayscale image at uint8 resolution, used by Phase 0B for
            CLAHE enhancement.
        metadata: Same shape as `preprocess_rgb_for_stage1` metadata, with
            an additional `orig_w`/`orig_h` alias for convenience.
    """
    import cv2  # local import keeps Pillow-only callers lightweight

    if isinstance(image, str):
        with open(image, "rb") as f:
            data = f.read()
    elif isinstance(image, bytes):
        data = image
    elif isinstance(image, np.ndarray):
        # Already a numpy array — go through PIL to ensure consistent grayscale.
        rgb = decode_to_array(image)
        gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
        orig_h, orig_w = gray.shape
        img_uint8 = cv2.resize(
            gray, (target_size, target_size), interpolation=cv2.INTER_LINEAR,
        )
        tensor = (img_uint8.astype(np.float32) / 255.0)[np.newaxis, np.newaxis, :, :]
        return tensor, img_uint8, _make_metadata(orig_w, orig_h, target_size)
    else:
        raise TypeError(
            f"image must be str, bytes, or np.ndarray, got {type(image).__name__}"
        )

    arr = np.frombuffer(data, dtype=np.uint8)
    gray_orig = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    if gray_orig is None:
        raise ValueError(
            "cv2.imdecode returned None — the image bytes could not be decoded "
            "as a valid image."
        )

    orig_h, orig_w = gray_orig.shape
    img_uint8 = cv2.resize(
        gray_orig, (target_size, target_size), interpolation=cv2.INTER_LINEAR,
    )
    tensor = (img_uint8.astype(np.float32) / 255.0)[np.newaxis, np.newaxis, :, :]
    return tensor, img_uint8, _make_metadata(orig_w, orig_h, target_size)


def to_grayscale_orig(image: Union[bytes, np.ndarray, str]) -> np.ndarray:
    """Decode an image as grayscale at its NATIVE resolution.

    Used for downstream specialist-refinement steps that need original
    pixel detail (Stage 2 patches in the production pipeline). This module
    only exposes the function; the public release ships Stage 0 + Stage 1
    inference and stops here.

    Args:
        image: File path, raw bytes, or numpy array.

    Returns:
        float32 array of shape (H_orig, W_orig) with values in [0, 255].
    """
    import cv2

    if isinstance(image, str):
        with open(image, "rb") as f:
            data = f.read()
        arr = np.frombuffer(data, dtype=np.uint8)
        gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    elif isinstance(image, bytes):
        arr = np.frombuffer(image, dtype=np.uint8)
        gray = cv2.imdecode(arr, cv2.IMREAD_GRAYSCALE)
    elif isinstance(image, np.ndarray):
        rgb = decode_to_array(image)
        gray = np.dot(rgb[..., :3], [0.299, 0.587, 0.114]).astype(np.uint8)
    else:
        raise TypeError(
            f"image must be str, bytes, or np.ndarray, got {type(image).__name__}"
        )

    if gray is None:
        raise ValueError("Failed to decode image to grayscale.")

    return gray.astype(np.float32)


def _make_metadata(orig_w: int, orig_h: int, target_size: int) -> Dict:
    return {
        "original_width": int(orig_w),
        "original_height": int(orig_h),
        "orig_w": int(orig_w),
        "orig_h": int(orig_h),
        "scale_x": float(target_size) / float(orig_w),
        "scale_y": float(target_size) / float(orig_h),
        "target_size": int(target_size),
    }
