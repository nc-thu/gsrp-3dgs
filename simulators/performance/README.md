# gs_arch_sim v1.1.1

A compact, reproducible camera-level raster timing model for the SARD target
streaming architecture.  The formal profile is fixed to 16x16 tiles, four
continuous-W16 render cores at 400 MHz, two KVP contexts, and a 32-row FIFO.

The real renderer adapter emits only per-tile aggregate counts.  T termination
stops future depth-ordered KVPs only after all valid pixels in a tile reach
`T < 1e-4`; an already started KVP is retained.

```bash
python -m pip install -e .
sard-v11 validate-camera workloads_real_v11_train/baseline_aabb/train/tiles.csv
sard-v11 replay-camera workloads_real_v11_train/baseline_aabb/train/tiles.csv --camera 0
sard-v11 run-experiments workloads_real_v11_train --out results_camera_ready_v11_train
sard-v11 report results_camera_ready_v11_train --out results_camera_ready_v11_train/report.html
```

Reported FPS is camera-level SARD raster FPS.  Preprocess, sorting, cache,
DRAM, and energy models are outside the v1.1 formal result.
