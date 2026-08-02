# Conic21 quality-safe profile

`conic21_quality_safe_v1` is the released profile corresponding to the
21-bit software and RTL validation track.

- A/C: unsigned `UQ3.18<21>`
- B: signed `SQ2.18<21>`
- coordinates: signed 18-bit `Q14.3`
- opacity: unsigned 8-bit `UQ1.7`
- RGB: unsigned 10-bit `UQ3.7`
- quadratic intermediate path: 40/40/42/20 bits
- EXP: 8-bit address, 10-bit output LUT
- blending: alpha 10-bit, transmittance/visibility 16-bit, RGB accumulator 16-bit
- rounding: round-to-nearest-even; overflow: saturation

The eigen-floor and integer positive-definiteness repair are preprocessing
semantics. The raster simulator consumes only the repaired fixed-point conic.
The public package contains the profile and aggregate evidence, but not models,
dataset copies, private traces, RTL, or synthesis scripts.
