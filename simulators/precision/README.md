# gs_raster_precision_v2

This project determines static fixed-point formats for the standard 3DGS
rasterizer. The formal image-quality baseline is the unmodified
`renderCUDA`; the allocator, tile list, Gaussian depth order, camera and model
remain frozen while numerical formats change.

Version `1.0.0rc1` contains the full camera-ready numerical experiment:

- camera 0 from each of eight scenes calibrates one shared static profile;
- all remaining 573 test cameras validate that frozen profile;
- raster inputs, quadratic intermediates, EXP LUT and blending states are
  evaluated as separate stages;
- the final candidate has zero unexplained saturation and passes the PSNR,
  SSIM and LPIPS quality gates.

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
held-out cameras validate them and never move the binary point.

Evidence levels are kept separate:

- real CUDA full-image rendering measures quality;
- the analytical bit-width proxy is not area or power;
- RTL simulation is required for bit-exact hardware evidence;
- synthesis is required for area and timing evidence.

The historical SARD recurrence and the invalid custom-FP16 image baseline are
not part of v2.

## Quality gates

A format is never selected from PSNR@GT alone. Three independent checks are
required:

1. task quality relative to the standard renderer (`delta_psnr_vs_standard`
   and `delta_ssim_vs_standard`);
2. direct candidate-to-standard similarity (`psnr_standard_ref`);
3. zero unexplained saturation on held-out cameras.

This matters because quantization can occasionally move an image closer to the
ground truth even while changing the renderer output substantially.

## Current Train result

Camera 0 alone is not a representative range-calibration set. Its selected
formats saturate on other Train cameras because projected local means expand to
about `[-6663, 3917]` and RGB reaches `4.55`. The high-precision fixed-point
control converges to the standard renderer, proving that the simulator
contract is valid. The Train diagnostic balanced profile passes the measured
quality gates, but is not a frozen cross-scene hardware profile.

Build the offline report with:

```text
gs-raster-precision-v2 report --results <results_train_inputs_v2>
```

Build the eight-scene camera-ready report with:

```text
gs-raster-precision-v2 camera-ready-report \
  --results <results_camera_ready_v1> \
  --profile <profiles/camera_ready_quality_safe_v1.json>
```

No model, dataset, private image, absolute server path, RTL, area, timing or
power claim is distributed in the public package.
