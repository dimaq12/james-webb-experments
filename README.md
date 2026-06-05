# james-webb-experments

Spectral Flow Transform (SFT) analysis of JWST deep-field and cluster images.
Universal pipeline for extracting hidden spectral-geometric structure from any image:
κ_W curvature, tail exponent α, Omega vector diagnostics, iterated residuals.

## Install

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
# Analyze a photo
python sft_pipeline.py photo.jpg

# Analyze JWST FITS with 3-level residual regression
python sft_pipeline.py image.fits --residual-depth 3

# Download JWST data from MAST
python download_jwst.py smacs --filter f090w --outdir data/
```

## Pipeline Flags

| Flag | Default | Description |
|---|---|---|
| `--residual-depth, -d` | 2 | Iterated regression depth (0–3) |
| `--max-regions, -n` | 200 | Max scanned blocks |
| `--stride, -s` | 64 | Scan stride (pixels) |
| `--region-size, -r` | 64 | Block size (pixels) |
| `--n-grid` | 8 | Operator grid size N |
| `--m-params` | 4 | Basis parameters M |
| `--perturbation` | off | Use perturbation kappa (slower, more accurate) |
| `--output-dir, -o` | `pipeline_out/` | Output directory |

## Outputs

```
pipeline_out/
├── image_metrics.json    # All metrics (κ_W, α, Ω, R² layers)
└── image_reveal.png      # Original + κ_W overlay + residual maps
```

## Metrics

| Name | Description |
|---|---|
| **κ_W** | Spectral curvature of the W-operator |
| **α** | Tail exponent of the eigenvalue spectrum |
| **Ω** | Omega vector — multi-scale tail diagnostic (4 components) |
| **\|\|Ω\|\|** | Omega vector norm |
| **cv(α)** | Coefficient of variation of α — spectral heterogeneity index |
| **cv(\|\|Ω\|\|)** | Spatial variation of Omega norm |
| **Cumulative R²** | Total explained variance of κ_W across regression layers |
| **Compression ratio** | σ(residual_final) / σ(κ_W) |

## Download JWST Images

```bash
python download_jwst.py list              # list available programs
python download_jwst.py smacs --list       # list SMACS 0723 files
python download_jwst.py ceers --filter f277w  # download CEERS deep field
python download_jwst.py nebula --outdir data/  # Southern Ring Nebula
```

Supported targets: `smacs`, `ceers`, `nebula`.
HUDF requires external download.

## Published Data Points

Analysis of 14 measurements across real and synthetic image fields:

| Field | cv(α) | Category |
|---|---|---|
| Empty sky (synthetic) | 0.002 | uniform |
| HUDF JWST (ultra-deep) | 0.003 | uniform |
| Clouds (synthetic) | 0.056 | natural |
| Forest-like (synthetic) | 0.044 | natural |
| City blocks (synthetic) | 0.071 | natural |
| DSCF1809 (outdoor photo) | 0.078 | natural |
| IMG_2169 (complex photo) | 0.099 | natural |
| CEERS JWST (wide survey) | 0.143 | rich structure |
| SMACS F200W (cluster) | 0.158 | rich structure |
| Southern Ring Nebula | 0.185 | rich structure |
| SMACS F090W (cluster) | 0.199 | rich structure |

cv(α) — the coefficient of variation of the tail exponent — appears to measure
**spectral heterogeneity**: how much the local spectral regime varies across an image.
It orders fields from empty/uniform (cv ≈ 0.001–0.01) through natural scenes
(cv ≈ 0.03–0.10) to morphologically rich astronomical fields (cv ≈ 0.14–0.20).

## License

MIT License — see [LICENSE](LICENSE).

The Spectral Flow Transform method originates from:
*"Spectral Flow on the space of operators"*, D. Sierikov, 2025.
Please cite this reference when using κ_W, α, or Ω diagnostics.
