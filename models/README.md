# CephTrace v4 ONNX Model Weights

This directory holds the seven ONNX files that make up the CephTrace v4
inference pipeline. They are **not** committed to git — the repository's
`.gitignore` excludes `*.onnx` files. Run the download script to populate
this directory.

## Download

```bash
python scripts/download_weights.py --output ./models
```

The script fetches files from the Hugging Face Hub repository
`cephtrace/cephtrace-v4` (configurable via `--repo-id`). Already-present
files with valid size are skipped, so re-running the script is cheap.

## Expected file sizes

| Relative path                                  | Stage  | Phase | Size       | Description                            |
|------------------------------------------------|--------|-------|------------|----------------------------------------|
| `stage0/v4_stage0_profile.onnx`                | 0      | 0A    | 25.55 MB   | Soft-tissue profile binary segmenter   |
| `stage0/z1_cranial_base_contours.onnx`         | 0      | 0C    | 25.55 MB   | Zone 1 cranial-base contour            |
| `stage0/z2_midface_contours.onnx`              | 0      | 0C    | 25.55 MB   | Zone 2 midface contours                |
| `stage0/z3_mandible_contours.onnx`             | 0      | 0C    | 25.55 MB   | Zone 3 mandible contours               |
| `stage0/z4_posterior_contours.onnx`            | 0      | 0C    | 25.55 MB   | Zone 4 posterior contours              |
| `stage0/phase0e_model.onnx`                    | 0      | 0E    | 0.43 MB    | Anchor → derived landmark MLP          |
| `stage1/v4_stage1.onnx`                        | 1      | —     | 123.93 MB  | HRNet-W32 28-channel heatmap regressor |

Total: **~277 MB**

## Loading models manually

The package loads these files automatically via `CephTracePredictor`. If
you need a single ONNX file for a custom pipeline:

```python
import onnxruntime as ort
session = ort.InferenceSession("models/stage1/v4_stage1.onnx")
```

Stage 1 input is `(1, 28, 512, 512)` float32 (3 RGB + 25 attention channels,
all normalised to `[0, 1]`). Output is `(1, 25, 256, 256)` float32 heatmaps.

## License

The ONNX weights are released under CC BY-NC-SA 4.0. Commercial use of the
underlying methods may require a separate patent licence — see the top-level
`LICENSE` for the full notice.
