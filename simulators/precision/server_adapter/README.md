# Renderer adapter

The adapter adds precision profiles to the standard 3D Gaussian Splatting
`renderCUDA` path without changing tile allocation or Gaussian depth order.

The public patch contains:

- `renderer_patch/forward_current.cu`: the instrumented forward raster path;
- `renderer_patch/gsrp_naive_precision.cuh`: RNE, saturation, quadratic,
  EXP-LUT and blending helpers;
- `run_camera_ready.py`: range collection and full-image profile evaluation;
- `BUILD_PROVENANCE.json`: hashes of the patch and the private experiment
  extension binary.

Integration is deliberately manual because upstream renderer forks differ.
Compare `forward_current.cu` with your renderer's `cuda_rasterizer/forward.cu`,
apply only the GSRP blocks, rebuild the extension, and confirm the mode-5
standard-renderer contract before collecting results.

The adapter does not contain models, datasets, source-image paths, experiment
images, server addresses or compiled CUDA extensions.
