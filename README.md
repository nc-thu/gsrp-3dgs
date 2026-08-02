# GSRP: Structure-Preserving Fixed-Point Rasterization

GSRP is a profile-driven research framework for evaluating 3D Gaussian
Splatting raster quality and camera-level performance under hardware-oriented
numeric formats.

This public snapshot contains two independent simulators:

- `simulators/performance/` — camera-level raster timing, FPS, and system-model
  evaluation (`sard-v11`).
- `simulators/precision/` — fixed-point raster quality, overflow, and conic
  structure evaluation (`gs-raster-precision-v2`).
- `simulators/precision/profiles/conic21_quality_safe_v1.json` — the released
  21-bit `UQ3.18/SQ2.18` structure-preserving raster profile.
- `docs/` — model boundaries, data formats, and reproducibility notes.
- `examples/` — small synthetic, dependency-free fixtures.

The camera-ready paper is maintained separately. The repository intentionally
does not contain a `paper/` directory. The project link is:

<https://github.com/nc-thu/gsrp-3dgs>

The release boundary contains simulators, profiles, tests, aggregate examples,
and reproduction scripts. It does not contain private models, datasets, raw or
full traces, server paths, credentials, complete RTL, netlists, DDC files, or
private experiment logs.

## Camera-ready aggregate reproduction

The public snapshot also contains a self-contained aggregate-data reproduction
under `reproducibility/camera_ready/`. From a clean clone, install the small
plotting dependencies and run:

```powershell
python -m pip install matplotlib numpy
python reproducibility/camera_ready/reproduce_camera_ready.py --out reproduced_camera_ready
```

This regenerates Fig. 3, Fig. 4, the quality summary table, and the hardware
summary table as PDF/SVG/PNG/CSV artifacts. The script consumes only released
aggregate source data and writes a hash manifest. It does not claim to rerun
the original 573-camera CUDA renderer: that experiment requires the user's
trained models and dataset copies, which are intentionally not distributed.

## Install

The simulators remain separate packages so either research question can be
reproduced independently:

```powershell
python -m pip install -e simulators/performance
python -m pip install -e simulators/precision
```

The performance package has no mandatory third-party dependency. The precision
package uses NumPy, Pillow, and PyYAML; optional tests use pytest.

## Minimal interfaces

The performance evaluator accepts a public workload directory containing
camera/tile aggregate inputs:

```powershell
sard-v11 validate-camera path/to/camera_tiles.csv
sard-v11 replay-camera path/to/camera_tiles.csv --camera 0
sard-v11 report path/to/results --out report.html
```

The precision evaluator is profile-driven:

```powershell
gs-raster-precision-v2 --help
gs-raster-precision-v2 camera-ready-report --results path/to/results --profile path/to/profile.json
```

Users supply their own trained-model adapter, dataset source, camera list, and
profile. Both paths emit machine-readable results and a manifest containing
the profile version, input semantics, source identifier, and output hashes.
Performance reports focus on camera-level cycles/FPS and stage timing;
precision reports focus on PSNR/SSIM/LPIPS, saturation, and conic-structure
checks.

The public repository currently contains complete public Python implementations
of both simulators, profiles, tests, aggregate result examples, and the
camera-ready figure/table reproduction script. It does not contain pretrained
models, dataset copies, private traces, complete RTL, or DC scripts. Standard
dataset sources and user-side acquisition notes are listed in `docs/`.

## Public scene naming

The paper uses `Train` and `Truck` for the two Tanks and Temples scenes, and
keeps `Lego`, `Hotdog`, `DrJohnson`, `Playroom`, `Bicycle`, and `Garden` as
the other public scene names.

## Citation

See `CITATION.cff`, the package-level citation files, and the project
documentation. The manuscript link is intentionally kept outside this code
repository; the software release and paper use the same project URL.
