# Reproducibility boundary

GSRP has two complementary software paths:

1. The performance path replays per-camera/per-tile workload aggregates and
   models the 16x16 raster engine, four W16 cores, T termination, and the
   established system stages.
2. The precision path keeps the standard renderer semantics fixed and evaluates
   a user-supplied quantization profile against the FP16 reference.

Both paths record a profile identifier, public scene/camera labels, arithmetic
semantics, and output hashes in their manifests. The public repository contains
synthetic examples and aggregate experiment source data, but not private
models, dataset copies, full traces, server addresses, the manuscript, or
hardware implementation files.

## Profile-selection protocol

The eight camera-0 views are a rapid calibration and candidate-generation set.
The complete 573-camera benchmark workload is the profile-selection and
validation workload. It is not called a held-out test set, and the artifact
does not claim generalization to an unseen scene, model, or camera distribution.

The published selector minimizes measured raster-core area subject to:

- zero final non-PD conics;
- worst-camera delta PSNR of at least -1 dB;
- eight-scene equal-weight delta PSNR of at least -0.2 dB;
- eight-scene equal-weight delta SSIM of at least -0.002;
- mean LPIPS increase of at most 0.005;
- no unexplained dangerous saturation.

The selector reports pooled-camera statistics and equal-weight scene statistics.
Missing camera rows or missing measured area never silently pass.

For a new dataset, first create a public adapter that emits the documented
camera/tile aggregate schema or the precision-renderer input schema. Then run
the relevant CLI from the package README and preserve the generated manifest
with the result directory.

## Evidence labels

- `software-model`: Python or CUDA adapter quality/timing evidence.
- `rtl-simulation`: bit-accurate functional evidence supplied by a separate
  private hardware track.
- `pre-layout-synthesis`: area/timing/power evidence from a mapped netlist.

These labels should not be merged when reporting a new experiment.
