# SARD target streaming model v1.1.1

## Scope

The formal result is camera-level raster latency. It does not include
preprocessing, sorting, cache, DRAM, host execution, or energy.

## Fixed configuration

- Tile: 16x16 pixels.
- Render cores: four, freely scheduled.
- Clock: 400 MHz.
- KVP contexts: two, ping-pong.
- Row FIFO: 32 entries.
- Consumer: continuous W16, 16 cycles per retained KVP.

## Real renderer summary

For each tile, the input records raw and retained KVP counts, raw and retained
RIE producer cycles, and the first depth-ordered KVP after which every valid
tile pixel has reached `T < 1e-4`. The KVP that causes the last pixel to finish
is retained; only later KVPs are stopped.

## Timing

`C_tile = max(C_RIE, 16 * N_KVP) + 30` for non-empty tiles. Tiles are assigned
in renderer order to the earliest-free core, with lower core id breaking ties.
Camera latency is the largest final core load. FPS is `400e6 / camera_cycles`.

The 30-cycle term is the fixed target-pipeline fill/drain contract and is
reported separately from the measured workload counts.
