# Public release scope

The camera-ready release is deliberately reproducible without disclosing
private research infrastructure. It includes:

- both profile-driven simulators;
- public model specifications and quantization profiles;
- unit tests and synthetic fixtures;
- paper figures, source-data summaries, and citation metadata;
- sanitized aggregate source data and a clean-clone reproduction script for
  the camera-ready quality/hardware summaries and Fig. 3/Fig. 4;
- installation and regeneration instructions.

It excludes trained models, dataset copies, raw or full workload traces,
server/GPU/EDA paths, credentials, complete RTL, DDC files, netlists, and
private logs. A user with legal access to the relevant model and dataset can
regenerate the workload inputs through the documented adapter interfaces.

The public repository is the artifact release. It does not claim that the
original 573-camera CUDA render can be rerun without user-provided models and
datasets; it does claim that the listed aggregate tables and figures can be
regenerated from the released source data.
