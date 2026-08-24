# Workload-aware profile selection

GSRP separates fast candidate generation from final workload selection.
Camera 0 from each of the eight scenes supplies value ranges and removes
obviously impossible formats. The complete 573-camera benchmark workload then
measures the quality tail and chooses the smallest measured raster-core area
that meets the published constraints.

The 573 cameras are therefore a profile-selection and validation workload, not
an unseen test set. The artifact does not claim that the selected profile is
optimal for an arbitrary future model, scene, or camera distribution.

## Selection rule

For each candidate profile, the selector requires:

1. 573 camera rows across eight scenes;
2. zero final non-PD conics;
3. worst-camera delta PSNR of at least -1 dB;
4. eight-scene equal-weight delta PSNR of at least -0.2 dB;
5. eight-scene equal-weight delta SSIM of at least -0.002;
6. mean LPIPS increase of at most 0.005;
7. zero unexplained or dangerous saturation.

Among candidates that pass, the selector minimizes measured raster-core area
at the declared synthesis corner. A candidate without measured area remains a
software-only result and cannot win the hardware selection.

The current aggregate evidence rejects the 20-bit guarded point because its
worst camera is about -1.88 dB. The 21-bit point passes with a worst camera of
about -0.89 dB and an area increase of about 0.16% over the 20-bit guarded
point.
