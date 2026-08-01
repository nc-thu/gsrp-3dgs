# SARD v1.1 revision log

## Simulator

- Replaced the historical variable-width event replay with a separate
  continuous-W16 camera-level package.
- Added exact tile-summary validation, deterministic four-core scheduling,
  single-camera replay, full experiment execution, 2x2 aggregation, manifests,
  and a self-contained Chinese report.
- Fixed the formal configuration to 16x16 tiles, four cores at 400 MHz, two
  KVP contexts, FIFO depth 32, and 16 cycles per retained KVP.
- Kept standard tile-wide transmittance termination as an input behavior rather
  than a claimed SARD contribution.

## Real workload

- Added a summary-only CUDA instrumentation path that records raw/retained KVPs
  and producer cycles without persisting event traces.
- Completed eight scenes under AABB and ATAE-equivalent SNUGBOX allocation:
  581 cameras per allocator and 1162 camera-allocator runs in total.
- Verified camera alignment, per-tile closure, allocator sanity, repeated
  Lego/Hotdog byte identity, and deterministic simulator output hashes.

## Manuscript

- Created a separate `sard0318_v11.tex` and `sections_v11/`; the prior draft is
  preserved unchanged.
- Reframed the paper around the continuous W16 target architecture, exact
  workload summaries, allocator effects, and standard tile-T termination.
- Removed variable-width and fine-grained event-replay performance claims from
  the v1.1 main text.
- Replaced unsupported full-system headlines with camera-level raster results
  and explicit model/RTL boundaries.
- Compiled the revised PDF with no undefined references/citations and no
  overfull horizontal boxes.
