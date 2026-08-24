# gs_raster_precision_v2

This project determines static fixed-point formats for the standard 3DGS
rasterizer. The formal image-quality baseline is the unmodified
`renderCUDA`; the allocator, tile list, Gaussian depth order, camera and model
remain frozen while numerical formats change.

Version `1.1.0rc1` contains the public numerical experiment interface:

- camera 0 from each of eight scenes generates a shared candidate profile;
- the complete 573-camera workload selects and validates the implementation
  point;
- raster inputs, quadratic intermediates, EXP LUT and blending states are
  evaluated as separate stages;
- the selection contract reports both quality and structural failures.

The recommended software profile uses an 8-bit EXP address with a 10-bit
output, 10-bit alpha, UQ1.15 transmittance and a 16-bit RGB accumulator.
The naive quadratic form currently needs 40-bit cancellation-safe weighted
intermediates. This is a validated numerical upper bound, not an area-optimal
RTL claim.

The first public stage evaluates seven raster inputs independently:

`mean_x`, `mean_y`, `conic_A`, `conic_B`, `conic_C`, `opacity`, and `RGB`.

Projected means are localized to the current 16x16 tile before quantization.
Conic A/B/C use independent formats. Fixed-point operations use
round-to-nearest-even and saturation. Camera 0 determines candidate formats;
the complete benchmark workload selects and validates them. It is not an
unseen test set and does not support a generalization claim beyond this
workload.

Evidence levels are kept separate:

- real CUDA full-image rendering measures quality;
- the analytical bit-width proxy is not area or power;
- RTL simulation is required for bit-exact hardware evidence;
- synthesis is required for area and timing evidence.

The historical SARD recurrence and the invalid custom-FP16 image baseline are
not part of v2.

## Workload profile selection

A format is never selected from PSNR@GT alone. The public selector first
requires a complete eight-scene, 573-camera workload and then checks:

1. zero final non-PD outputs;
2. worst-camera delta PSNR of at least -1 dB;
3. eight-scene equal-weight delta PSNR/SSIM/LPIPS thresholds;
4. zero unexplained dangerous saturation.

Among passing candidates, measured raster-core area is minimized. A candidate
without measured area is reported as software-only and cannot be selected as a
hardware point. The contract is stored in
`profiles/workload_profile_selection_v1.json` and the 0.81-V area data are in
`profiles/workload_profile_costs_081v.json`.

Run the selector with:

```powershell
gs-raster-precision-v2 select-workload-profile `
  --results profile20_camera_results.csv profile21_camera_results.csv `
  --costs profiles/workload_profile_costs_081v.json `
  --contract profiles/workload_profile_selection_v1.json `
  --output workload_selection
```

This matters because quantization can occasionally move an image closer to the
ground truth even while changing the renderer output substantially.

## Current workload result

Camera 0 alone is not a representative workload-selection set. It is used to
bound ranges and generate candidates. The complete 573-camera workload is the
source of the final tail-quality and area decision. The high-precision
fixed-point control converges to the standard renderer, which checks the
simulator contract; it is not itself a hardware selection.

Build the offline report with:

```text
gs-raster-precision-v2 report --results <results_tt107k_inputs_v2>
```

Build the eight-scene camera-ready report with:

```text
gs-raster-precision-v2 camera-ready-report \
  --results <results_camera_ready_v1> \
  --profile <profiles/camera_ready_quality_safe_v1.json>
```

No model, dataset, private image, absolute server path, RTL, area, timing or
power claim is distributed in the public package.
