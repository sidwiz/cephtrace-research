# CephTrace v4

**Anatomy-Guided Spatial Priors for Cephalometric Landmark Detection**

[![arXiv](https://img.shields.io/badge/arXiv-2605.03358-b31b1b)](https://arxiv.org/abs/2605.03358)
[![License](https://img.shields.io/badge/License-MIT-blue)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org)
[![ONNX](https://img.shields.io/badge/Inference-ONNX%20Runtime-orange)](https://onnxruntime.ai)

The CephTrace v4 inference pipeline as released for the arXiv paper
[2605.03358](https://arxiv.org/abs/2605.03358) (Mohapatra & Mohanty, 2026).
This repository contains the Python code, model-download scripts, and
documentation needed to reproduce the published landmark-detection results
on lateral cephalogram images.

> The training notebooks, the Stage 2 per-landmark specialists, and the
> Stage 3 recursive auto-optimizer are part of the production CephTrace
> deployment and are NOT included in this research release.

## Results

Evaluated on a 151-image held-out test set drawn from ISBI 2015 (n=400),
CEPHA29 / Aariz (n=1000), and the DentalCepha dataset (n=102).

| Metric                      | CephTrace v4 |  Prior SOTA   | Improvement |
|-----------------------------|-------------:|--------------:|------------:|
| Mean Radial Error (MRE)     |    **1.04 mm** |       1.23 mm |    **17 %** |
| Successful Detection @ 2 mm |     **88.4 %** |        85.5 % |    **+2.9 pp** |
| Successful Detection @ 4 mm |     **96.8 %** |             — |              — |
| Landmarks                   |          **25** |            19 |    +6 landmarks |
| CPU inference (Railway)     |       **~ 600 ms** |             — |              — |

Prior SOTA: CephRes-MHNet (19 landmarks, Aariz). See the paper for the full
methodology, ablations, and cross-dataset generalization analysis.

## Architecture

The pipeline composes four stages; **this repository implements Stage 0
and Stage 1** (the components needed to reproduce the published numbers).

```
Input image (any resolution)
      │
      ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 0 — Anatomical Zone Decomposition                             │
│  ────────────────────────────────────────────────────────────────    │
│  Phase 0A  Soft-tissue profile binary mask    (1-ch grayscale ONNX)  │
│  Phase 0B  Profile features, 5 anatomical zones, per-zone CLAHE      │
│  Phase 0C  Per-zone contour segmentation      (4 × 1-ch grayscale)   │
│  Phase 0D  Douglas-Peucker simplification + 7 anchor extraction      │
│  Phase 0E  18 derived landmarks via MLP + 25 Gaussian attention maps │
└──────────────────────────────────────────────────────────────────────┘
                              │  (1, 25, 512, 512) attention prior
                              ▼
┌──────────────────────────────────────────────────────────────────────┐
│  Stage 1 — HRNet-W32 Heatmap Regression                              │
│  ────────────────────────────────────────────────────────────────    │
│  Input :  (1, 28, 512, 512)  = 3 RGB + 25 attention channels         │
│  Model :  HRNet-W32 + transposed-conv head                           │
│  Output:  (1, 25, 256, 256)  heatmaps in CANONICAL_25 channel order  │
│  Decode:  DARK sub-pixel argmax (Zhang et al. CVPR 2020)             │
└──────────────────────────────────────────────────────────────────────┘
      │
      ▼
25 landmark predictions in ORIGINAL image pixel space
+ per-landmark heatmap-peak confidence + Stage 0 substage diagnostics
```

The Stage 0 contribution is the core scientific novelty: a learned
anatomy-aware spatial prior derived from clinically-defined zone
decomposition, contour topology, and a 114K-param MLP that predicts the
remaining 18 landmarks from 7 high-confidence anchor positions. The
resulting (25, 256, 256) Gaussian attention prior is concatenated with
the input image for Stage 1 — the same physical pipeline a human clinician
uses (find the easy bony anchors first, then locate the harder landmarks
relative to those anchors).

## Installation

```bash
git clone https://github.com/cephtrace-research/cephtrace-v4-public.git
cd cephtrace-v4-public
pip install -r requirements.txt
```

Then download the ONNX model weights (~277 MB) from Hugging Face Hub:

```bash
python scripts/download_weights.py --output ./models
```

For GPU inference, install `onnxruntime-gpu` instead of `onnxruntime`:

```bash
pip uninstall onnxruntime
pip install onnxruntime-gpu
```

## Quick start

### Python API

```python
from cephtrace import CephTracePredictor

predictor = CephTracePredictor(model_dir="models/")

result = predictor.predict("/path/to/cephalogram.jpg")

print(f"Image: {result.original_width} x {result.original_height}")
print(f"Degraded: {result.degraded}")
for lm in result.landmarks:
    print(
        f"  {lm.name:<10} ({lm.x:.1f}, {lm.y:.1f})  "
        f"confidence={lm.confidence_label} (peak={lm.peak:.2f})"
    )
```

### Command-line interface

```bash
# Single image, print to stdout
python scripts/run_inference.py --image cephalogram.jpg

# Single image, write JSON
python scripts/run_inference.py --image cephalogram.jpg --output result.json

# Single image, save an annotated overlay PNG
python scripts/run_inference.py \
    --image cephalogram.jpg \
    --visualize \
    --output overlay.png

# Batch process a directory
python scripts/run_inference.py --image-dir cephs/ --output results.json

# Run on GPU
python scripts/run_inference.py --image cephalogram.jpg --gpu
```

### Reproducing published metrics

```bash
python scripts/evaluate.py \
    --image-dir /path/to/test_images \
    --gt-dir /path/to/test_landmarks \
    --output evaluation_report.json
```

Ground-truth files are one JSON per image, named `{image_stem}.json`:

```json
{
    "S":   [962, 668],
    "N":   [712, 670],
    "Or":  [617, 913],
    "...": "..."
}
```

The evaluator calibrates each image's mm-per-pixel scale from the
ground-truth S-N distance (Proffit 7th Ed. mean = 71.0 mm), then computes
MRE and SDR per landmark and overall.

## Landmark set

The 25-landmark set used throughout this repository (`CANONICAL_25`):

| Idx | Name      | Zone | Type            | Description                                        |
|-----|-----------|------|-----------------|----------------------------------------------------|
| 0   | S         | 1    | Skeletal anchor | Sella turcica (midpoint of pituitary fossa)        |
| 1   | N         | 2    | Skeletal anchor | Nasion (most anterior frontonasal suture)          |
| 2   | Or        | 2    | Skeletal        | Orbitale (lowest orbital rim point)                |
| 3   | Po        | 4    | Skeletal        | Porion (top of external auditory meatus)           |
| 4   | ANS       | 2    | Skeletal anchor | Anterior nasal spine                               |
| 5   | PNS       | 2    | Skeletal        | Posterior nasal spine                              |
| 6   | A         | 2    | Skeletal        | Subspinale (deepest premaxillary point)            |
| 7   | B         | 3    | Skeletal        | Supramentale (deepest mandibular point)            |
| 8   | Pog       | 3    | Skeletal anchor | Pogonion (most anterior chin)                      |
| 9   | Gn        | 3    | Skeletal        | Gnathion (midpoint Pog–Me)                         |
| 10  | Me        | 3    | Skeletal anchor | Menton (most inferior symphysis)                   |
| 11  | Go        | 3    | Skeletal anchor | Gonion (mandibular angle)                          |
| 12  | Ar        | 4    | Skeletal        | Articulare (cranial base × ramus)                  |
| 13  | Co        | 4    | Skeletal        | Condylion (top of condyle)                         |
| 14  | U1_tip    | 2    | Dental          | Upper incisor tip                                  |
| 15  | U1_root   | 2    | Dental          | Upper incisor root apex                            |
| 16  | L1_tip    | 3    | Dental          | Lower incisor tip                                  |
| 17  | L1_root   | 3    | Dental          | Lower incisor root apex                            |
| 18  | UL        | 5    | Soft tissue     | Upper lip                                          |
| 19  | LL        | 5    | Soft tissue     | Lower lip                                          |
| 20  | Pm        | 5    | Soft tissue     | Suprapogonion (inflection chin–lower lip)          |
| 21  | Ba        | 1    | Skeletal        | Basion                                             |
| 22  | Pog_soft  | 5    | Soft tissue     | Soft-tissue pogonion                               |
| 23  | Sn        | 5    | Soft tissue     | Subnasale                                          |
| 24  | Prn       | 5    | Soft tissue     | Pronasale (nose tip)                               |

The 7 Stage 0 anchors are `S, N, Me, ANS, Go, Pog, Prn`. The remaining 18
are derived by Phase 0E and refined by Stage 1.

## Datasets

The published results were obtained on a combined corpus of three public
datasets. CephTrace does **not** redistribute the source images — fetch
them from the original sources.

| Dataset                | Images | Landmarks | Where to get it                                                  |
|------------------------|-------:|----------:|------------------------------------------------------------------|
| ISBI 2015 senior       |    400 |        19 | <https://figshare.com/articles/dataset/Cephalometric/3471833>   |
| CEPHA29 / Aariz        |  1,000 |        29 | <https://figshare.com/articles/dataset/Aariz/22149727>          |
| DentalCepha            |    102 |        19 | <https://github.com/dental-cepha/dentalcepha>                   |

The 1,502-image dataset is split 1201 / 150 / 151 (train / val / test)
with stratification by source. Landmark names from the source files are
normalized into the `CANONICAL_25` vocabulary at load time — see the
paper for the alias table.

## Pipeline status (what's in this release vs. production)

| Component          | In this repository | In production | Patent          |
|--------------------|:------------------:|:-------------:|:----------------|
| Stage 0 (all five) |         ✅          |       ✅       | 64/039,042      |
| Stage 1 HRNet      |         ✅          |       ✅       | 64/037,246      |
| DARK decoder       |         ✅          |       ✅       | —               |
| Stage 2 specialists |        ❌          |       ✅       | 64/037,246      |
| Stage 3 optimizer  |         ❌          |   (offline)   | 64/037,252      |
| Training notebooks |         ❌          |   (offline)   | —               |
| Web application    |         ❌          |       ✅       | —               |

The Stage 0 + Stage 1 pipeline alone reproduces the published 1.04 mm MRE
on the 25-landmark canonical set — Stage 2 contributes a further
sub-pixel refinement on 7 priority landmarks (Δ = +0.008 mm aggregate)
that is documented in the paper but not part of this research release.

## Citation

```bibtex
@article{mohapatra2026cephtrace,
  title        = {Tracing Like a Clinician: Anatomy-Guided Spatial Priors
                  for Cephalometric Landmark Detection},
  author       = {Mohapatra, Sidhartha and Mohanty, Pallavi},
  journal      = {arXiv preprint arXiv:2605.03358},
  year         = {2026},
  url          = {https://arxiv.org/abs/2605.03358}
}
```

## Patent notice

The methods, architectures, and techniques implemented in this software are
the subject of three pending US provisional patent applications filed in
April 2026:

- **64/037,246** — Two-Stage Heatmap Regression with Per-Landmark Specialist
  Networks (Stage 1 + Stage 2).
- **64/037,252** — Recursive Self-Optimization Framework (Stage 3 auto-
  optimizer).
- **64/039,042** — Anatomy-Guided Landmark Initialization via Zone
  Decomposition (Stage 0).

The MIT licence on the source code does not extend a patent licence to
commercial uses of the underlying methods. Academic and reproducibility use
is fully permitted. For commercial licensing, contact
`sidhartha@cephtrace.com`.

## Acknowledgments

The CephTrace pipeline was developed by **Sidhartha Mohapatra** with
clinical input from **Dr. Pallavi Mohanty, DDS**. The DARK sub-pixel
decoder is adapted from Zhang et al., CVPR 2020. The HRNet-W32 backbone
is adapted from Sun et al., CVPR 2019. The Aariz / CEPHA29 dataset is
the work of multiple authors at the National University of Sciences and
Technology, Islamabad. The ISBI 2015 dataset is from Wang et al.

## License

MIT for the source code; CC BY-NC-SA 4.0 for the model weights distributed
via Hugging Face. See [LICENSE](LICENSE) for the full notice including
patent terms.
