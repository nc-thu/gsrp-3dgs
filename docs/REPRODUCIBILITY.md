# Reproducibility boundary

GSRP has two complementary software paths:

1. The performance path replays per-camera/per-tile workload aggregates and
   models the 16x16 raster engine, four W16 cores, T termination, and the
   established system stages.
2. The precision path keeps the standard renderer semantics fixed and evaluates
   a user-supplied quantization profile against the FP16 reference.

Both paths record a profile identifier, public scene/camera labels, arithmetic
semantics, and output hashes in their manifests. The public repository contains
synthetic examples and aggregate paper source data, but not private models,
datasets, full traces, server addresses, or hardware implementation files.

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

