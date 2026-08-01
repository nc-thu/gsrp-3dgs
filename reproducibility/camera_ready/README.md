# Camera-ready aggregate reproduction

This directory reproduces the GSRP camera-ready aggregate evidence without
access to the private GPU/EDA environment.

## Run

From the repository root:

```powershell
python -m pip install matplotlib numpy
python reproducibility/camera_ready/reproduce_camera_ready.py --out reproduced_camera_ready
```

The output directory contains:

- `fig3_quality_validation.pdf/.svg/.png`;
- `fig4_hardware_value.pdf/.svg/.png`;
- `table_quality.csv`;
- `table_hardware.csv`;
- `reproduction_manifest.json` with SHA-256 hashes.

The figures and tables are regenerated from the sanitized aggregate CSV/JSON
files in `source_data/`. Internal RTL run names, server paths, model paths,
and raw per-camera traces are not part of this bundle.

## Evidence boundary

This script reproduces the released aggregate values for Fig. 3, Fig. 4, the
quality table, and the hardware table. It does not rerun the 573-camera CUDA
renderer. A fresh renderer run requires a user-supplied trained model, a legal
copy of the relevant standard dataset, a camera list, and the matching public
profile. Dataset source names and citation information are documented in the
repository documentation.
