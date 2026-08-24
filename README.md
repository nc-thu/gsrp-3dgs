# GSRP: Structure-Preserving Fixed-Point Rasterization

GSRP is a public research artifact containing two independent, profile-driven
simulators for 3D Gaussian Splatting. The public tree contains software,
profiles, tests, aggregate evidence, and reproduction scripts. It does not
contain the manuscript or private hardware infrastructure.

- `simulators/performance/` — camera-level raster timing and system-model evaluation.
- `simulators/precision/` — fixed-point raster quality, structure-aware profiles, and workload profile selection.
- `docs/` — model boundaries, the workload protocol, and reproducibility notes.
- `examples/` — small, synthetic, dependency-free examples.

The public repository is:

<https://github.com/nc-thu/gsrp-3dgs>

The release boundary contains simulators, profiles, tests, aggregate source
data, and reproduction scripts. It does not contain the manuscript, private
models, dataset copies, raw traces, server paths, credentials, complete RTL,
netlists, or DDC files.

## Install

The two simulators remain separate packages:

```powershell
python -m pip install -e simulators/performance
python -m pip install -e simulators/precision
```

The precision package uses NumPy, Pillow, and PyYAML; its optional tests use
pytest.

## Minimal interfaces

```powershell
sard-v11 validate-camera path/to/camera_tiles.csv
sard-v11 replay-camera path/to/camera_tiles.csv --camera 0
sard-v11 report path/to/results --out report.html
```

```powershell
gs-raster-precision-v2 --help
gs-raster-precision-v2 camera-ready-report `
  --results path/to/results `
  --profile path/to/profile.json
```

Both paths emit machine-readable results and a manifest containing the profile
version, input semantics, source identifier, and output hashes.

## Workload-aware profile selection

Camera 0 from each scene is used for rapid range calibration and candidate
generation. The complete 573-camera, eight-scene workload is then used for
profile selection and validation; it is not presented as an unseen test set.
The selector minimizes measured raster-core area subject to the published
constraints in
`simulators/precision/profiles/workload_profile_selection_v1.json`.

```powershell
gs-raster-precision-v2 select-workload-profile `
  --results path/to/profile20_camera_results.csv path/to/profile21_camera_results.csv `
  --costs simulators/precision/profiles/workload_profile_costs_081v.json `
  --contract simulators/precision/profiles/workload_profile_selection_v1.json `
  --output selected_profile_report
```

The output records constraint failures, measured area, selection rank, and
the reason each candidate was selected or rejected. A new model or dataset is
a separate workload; this artifact makes no generalization claim beyond the
reported benchmark workload.

## Evidence boundary

The headline quality, functional, and pre-layout synthesis evidence is
preserved from completed experiment assets. The public profile-selection
summary is a re-aggregation of camera-level results; it does not claim a new
CUDA, RTL, or DC run.

## Citation

See `CITATION.cff` and the two package-level citation files. Dataset and
renderer acquisition instructions are documented without distributing model
or dataset files.
