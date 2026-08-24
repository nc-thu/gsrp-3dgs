# Experiment protocol

The precision path uses a fixed trained model, camera list, FP16 renderer
semantics, allocator, depth order, and arithmetic order for every profile.
Each candidate rebuilds its tile list after the guarded and quantized conic is
created, so a changed footprint is not evaluated with a stale tile list.

The protocol has three roles:

- **Candidate generation:** one camera-0 view per scene estimates ranges and
  produces a small set of formats for hardware exploration.
- **Profile selection and validation:** all 573 cameras measure the quality
  tail, structure counters, and hardware cost constraints.
- **Evidence boundary:** no result in this release is an unseen-distribution
  generalization test; new scenes and models require a separate workload
  manifest.

The released selection contract and area-cost table are in
`simulators/precision/profiles/`. Camera-level CSV rows remain the source of
truth for pooled and equal-weight scene summaries.
