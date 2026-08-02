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

## What can be reproduced from a clean clone

The directory `reproducibility/camera_ready/` contains sanitized aggregate
source data and `reproduce_camera_ready.py`. At the final public commit, a
clean clone regenerates the paper's Conic21 quality summary, hardware summary,
Fig. 3, and Fig. 4 without a server or a GPU. The generated manifest records the input
and output hashes.

This is an aggregate-data reproduction claim. Re-running the original
573-camera CUDA renderer requires user-supplied trained models and dataset
copies. The repository provides standard dataset names and acquisition notes,
but does not provide pretrained models or dataset copies. The performance model
and both public Python simulator packages are included; complete RTL, DC
scripts, mapped netlists, and private server environments are not.

| Artifact | Current public status |
| --- | --- |
| Performance simulator | Complete Python package and CLI |
| Precision simulator | Complete Python package and CLI |
| Performance model | Included and documented |
| Profiles and tests | Included |
| Fig. 3, Fig. 4, quality/hardware tables | Regenerated from released aggregate data |
| Original 573-camera CUDA rerun | Requires user models and datasets |
| Pretrained models and dataset copies | Not distributed |
| Complete RTL and DC scripts | Not distributed |

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
