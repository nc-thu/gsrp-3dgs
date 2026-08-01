/*
 * Copyright (C) 2023, Inria
 * GRAPHDECO research group, https://team.inria.fr/graphdeco
 * All rights reserved.
 *
 * This software is free for non-commercial, research and evaluation use 
 * under the terms of the LICENSE.md file.
 *
 * For inquiries contact  george.drettakis@inria.fr
 */

#include "forward.h"
#include "auxiliary.h"
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <cstdlib>
#include <cuda_fp16.h>
#include "sard_exp_lut_10bit.inc"
namespace cg = cooperative_groups;

// SARD: alpha is gated at 1/255 in the renderer. Since
// alpha = opacity * exp(-sigma) <= exp(-sigma), any pixel with
// sigma > tau_sigma is guaranteed alpha < 1/255 (a non-contributor).
// tau_sigma = ln(255) ~= 5.5413 is a conservative bound valid for any
// opacity <= 1, so the early-stop only ever skips true non-contributors.
#define SARD_TAU_SIGMA 5.5412945f

// Exact ES-v2 producer recurrence used by the Tile311 RTL reference.  This
// stays separate from alpha/opacity chunk gating: RIE determines how many x
// steps are produced, while RSPA receives only the non-empty alpha chunks.
__device__ __forceinline__ unsigned int sardRieProducerCycles(
	const float2 xy, const float4 con_o, const uint2 pix_min)
{
	const __half half = __float2half_rn(0.5f);
	const __half one = __float2half_rn(1.0f);
	const __half tau = __ushort_as_half(0x458A); // FP16 ln(255) threshold from ES-v2 reference
	const __half xg = __float2half_rn(xy.x);
	const __half yg = __float2half_rn(xy.y);
	const __half xbase = __float2half_rn((float)pix_min.x);
	const __half ybase = __float2half_rn((float)pix_min.y);
	const __half c0 = __float2half_rn(con_o.x);
	const __half c1 = __float2half_rn(con_o.y);
	const __half c2 = __float2half_rn(con_o.z);
	const __half half_c0 = __hmul(c0, half);
	const __half dx0 = __hsub(xg, xbase);
	const __half dx1 = __hsub(dx0, one);
	const __half term_x0 = __hmul(half_c0, __hmul(dx0, dx0));
	const __half term_x1 = __hmul(half_c0, __hmul(dx1, dx1));

	__half sigma[BLOCK_Y], d1[BLOCK_Y], previous[BLOCK_Y];
	bool seen_valid[BLOCK_Y], inc_from_start[BLOCK_Y], done[BLOCK_Y];
	for (int y = 0; y < BLOCK_Y; ++y) {
		const __half ycur = __hadd(ybase, __float2half_rn((float)y));
		const __half dy = __hsub(yg, ycur);
		const __half term_y = __hmul(c2, __hmul(dy, dy));
		const __half c1_dy = __hmul(c1, dy);
		const __half sigma0 = __hadd(__hadd(term_x0, __hmul(c1_dy, dx0)), term_y);
		const __half sigma1 = __hadd(__hadd(term_x1, __hmul(c1_dy, dx1)), term_y);
		sigma[y] = sigma0;
		d1[y] = __hsub(sigma1, sigma0);
		previous[y] = __float2half_rn(0.0f);
		seen_valid[y] = false;
		inc_from_start[y] = true;
		done[y] = false;
	}

	for (unsigned int x = 0; x < BLOCK_X; ++x) {
		bool all_done = true;
		for (int y = 0; y < BLOCK_Y; ++y) {
			if (done[y]) continue;
			all_done = false;
			const __half value = sigma[y];
			if (__hle(value, tau)) seen_valid[y] = true;
			if (x > 0 && __hlt(value, previous[y])) inc_from_start[y] = false;
			previous[y] = value;
			if (__hgt(value, tau) && (seen_valid[y] || inc_from_start[y])) done[y] = true;
		}
		if (all_done) return x + 1;
		bool now_all_done = true;
		for (int y = 0; y < BLOCK_Y; ++y) now_all_done = now_all_done && done[y];
		if (now_all_done) return x + 1;
		if (x + 1 < BLOCK_X) {
			for (int y = 0; y < BLOCK_Y; ++y) {
				sigma[y] = __hadd(sigma[y], d1[y]);
				d1[y] = __hadd(d1[y], c0);
			}
		}
	}
	return BLOCK_X;
}

// Forward method for converting the input spherical harmonics
// coefficients of each Gaussian to a simple RGB color.
__device__ glm::vec3 computeColorFromSH(int idx, int deg, int max_coeffs, const glm::vec3* means, glm::vec3 campos, const float* shs, bool* clamped)
{
	// The implementation is loosely based on code for 
	// "Differentiable Point-Based Radiance Fields for 
	// Efficient View Synthesis" by Zhang et al. (2022)
	glm::vec3 pos = means[idx];
	glm::vec3 dir = pos - campos;
	dir = dir / glm::length(dir);

	glm::vec3* sh = ((glm::vec3*)shs) + idx * max_coeffs;
	glm::vec3 result = SH_C0 * sh[0];

	if (deg > 0)
	{
		float x = dir.x;
		float y = dir.y;
		float z = dir.z;
		result = result - SH_C1 * y * sh[1] + SH_C1 * z * sh[2] - SH_C1 * x * sh[3];

		if (deg > 1)
		{
			float xx = x * x, yy = y * y, zz = z * z;
			float xy = x * y, yz = y * z, xz = x * z;
			result = result +
				SH_C2[0] * xy * sh[4] +
				SH_C2[1] * yz * sh[5] +
				SH_C2[2] * (2.0f * zz - xx - yy) * sh[6] +
				SH_C2[3] * xz * sh[7] +
				SH_C2[4] * (xx - yy) * sh[8];

			if (deg > 2)
			{
				result = result +
					SH_C3[0] * y * (3.0f * xx - yy) * sh[9] +
					SH_C3[1] * xy * z * sh[10] +
					SH_C3[2] * y * (4.0f * zz - xx - yy) * sh[11] +
					SH_C3[3] * z * (2.0f * zz - 3.0f * xx - 3.0f * yy) * sh[12] +
					SH_C3[4] * x * (4.0f * zz - xx - yy) * sh[13] +
					SH_C3[5] * z * (xx - yy) * sh[14] +
					SH_C3[6] * x * (xx - 3.0f * yy) * sh[15];
			}
		}
	}
	result += 0.5f;

	// RGB colors are clamped to positive values. If values are
	// clamped, we need to keep track of this for the backward pass.
	clamped[3 * idx + 0] = (result.x < 0);
	clamped[3 * idx + 1] = (result.y < 0);
	clamped[3 * idx + 2] = (result.z < 0);
	return glm::max(result, 0.0f);
}

// Forward version of 2D covariance matrix computation
__device__ float3 computeCov2D(const float3& mean, float focal_x, float focal_y, float tan_fovx, float tan_fovy, const float* cov3D, const float* viewmatrix)
{
	// The following models the steps outlined by equations 29
	// and 31 in "EWA Splatting" (Zwicker et al., 2002). 
	// Additionally considers aspect / scaling of viewport.
	// Transposes used to account for row-/column-major conventions.
	float3 t = transformPoint4x3(mean, viewmatrix);

	const float limx = 1.3f * tan_fovx;
	const float limy = 1.3f * tan_fovy;
	const float txtz = t.x / t.z;
	const float tytz = t.y / t.z;
	t.x = min(limx, max(-limx, txtz)) * t.z;
	t.y = min(limy, max(-limy, tytz)) * t.z;

	glm::mat3 J = glm::mat3(
		focal_x / t.z, 0.0f, -(focal_x * t.x) / (t.z * t.z),
		0.0f, focal_y / t.z, -(focal_y * t.y) / (t.z * t.z),
		0, 0, 0);

	glm::mat3 W = glm::mat3(
		viewmatrix[0], viewmatrix[4], viewmatrix[8],
		viewmatrix[1], viewmatrix[5], viewmatrix[9],
		viewmatrix[2], viewmatrix[6], viewmatrix[10]);

	glm::mat3 T = W * J;

	glm::mat3 Vrk = glm::mat3(
		cov3D[0], cov3D[1], cov3D[2],
		cov3D[1], cov3D[3], cov3D[4],
		cov3D[2], cov3D[4], cov3D[5]);

	glm::mat3 cov = glm::transpose(T) * glm::transpose(Vrk) * T;

	// Apply low-pass filter: every Gaussian should be at least
	// one pixel wide/high. Discard 3rd row and column.
	cov[0][0] += 0.3f;
	cov[1][1] += 0.3f;
	return { float(cov[0][0]), float(cov[0][1]), float(cov[1][1]) };
}

// Forward method for converting scale and rotation properties of each
// Gaussian to a 3D covariance matrix in world space. Also takes care
// of quaternion normalization.
__device__ void computeCov3D(const glm::vec3 scale, float mod, const glm::vec4 rot, float* cov3D)
{
	// Create scaling matrix
	glm::mat3 S = glm::mat3(1.0f);
	S[0][0] = mod * scale.x;
	S[1][1] = mod * scale.y;
	S[2][2] = mod * scale.z;

	// Normalize quaternion to get valid rotation
	glm::vec4 q = rot;// / glm::length(rot);
	float r = q.x;
	float x = q.y;
	float y = q.z;
	float z = q.w;

	// Compute rotation matrix from quaternion
	glm::mat3 R = glm::mat3(
		1.f - 2.f * (y * y + z * z), 2.f * (x * y - r * z), 2.f * (x * z + r * y),
		2.f * (x * y + r * z), 1.f - 2.f * (x * x + z * z), 2.f * (y * z - r * x),
		2.f * (x * z - r * y), 2.f * (y * z + r * x), 1.f - 2.f * (x * x + y * y)
	);

	glm::mat3 M = S * R;

	// Compute 3D world covariance matrix Sigma
	glm::mat3 Sigma = glm::transpose(M) * M;

	// Covariance is symmetric, only store upper right
	cov3D[0] = Sigma[0][0];
	cov3D[1] = Sigma[0][1];
	cov3D[2] = Sigma[0][2];
	cov3D[3] = Sigma[1][1];
	cov3D[4] = Sigma[1][2];
	cov3D[5] = Sigma[2][2];
}

// Perform initial steps for each Gaussian prior to rasterization.
template<int C>
__global__ void preprocessCUDA(int P, int D, int M,
	const float* orig_points,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float tan_fovx, float tan_fovy,
	const float focal_x, float focal_y,
	int* radii,
	float2* points_xy_image,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool baseline_aabb)
{
	auto idx = cg::this_grid().thread_rank();
	if (idx >= P)
		return;

	// Initialize radius and touched tiles to 0. If this isn't changed,
	// this Gaussian will not be processed further.
	radii[idx] = 0;
	tiles_touched[idx] = 0;

	// Perform near culling, quit if outside.
	float3 p_view;
	if (!in_frustum(idx, orig_points, viewmatrix, projmatrix, prefiltered, p_view))
		return;

	// Transform point by projecting
	float3 p_orig = { orig_points[3 * idx], orig_points[3 * idx + 1], orig_points[3 * idx + 2] };
	float4 p_hom = transformPoint4x4(p_orig, projmatrix);
	float p_w = 1.0f / (p_hom.w + 0.0000001f);
	float3 p_proj = { p_hom.x * p_w, p_hom.y * p_w, p_hom.z * p_w };

	// If 3D covariance matrix is precomputed, use it, otherwise compute
	// from scaling and rotation parameters. 
	const float* cov3D;
	if (cov3D_precomp != nullptr)
	{
		cov3D = cov3D_precomp + idx * 6;
	}
	else
	{
		computeCov3D(scales[idx], scale_modifier, rotations[idx], cov3Ds + idx * 6);
		cov3D = cov3Ds + idx * 6;
	}

	// Compute 2D screen-space covariance matrix
	float3 cov = computeCov2D(p_orig, focal_x, focal_y, tan_fovx, tan_fovy, cov3D, viewmatrix);

	// Invert covariance (EWA algorithm)
	float det = (cov.x * cov.z - cov.y * cov.y);
	if (det == 0.0f)
		return;
	float det_inv = 1.f / det;
	float3 conic = { cov.z * det_inv, -cov.y * det_inv, cov.x * det_inv };
       // Compute extent in screen space (by finding eigenvalues of
       // 2D covariance matrix). Use extent to compute a bounding rectangle
       // of screen-space tiles that this Gaussian overlaps with. Quit if
       // rectangle covers 0 tiles. 
       float mid = 0.5f * (cov.x + cov.z);
       float lambda1 = mid + sqrt(max(0.1f, mid * mid - det));
       float lambda2 = mid - sqrt(max(0.1f, mid * mid - det));
       float my_radius = ceil(3.f * sqrt(max(lambda1, lambda2)));

  // Updated: Compute extent in screen space by identifying exact
  // screen-space tile.overlap with Gaussian.
  // No longer need radius
	float2 point_image = { ndc2Pix(p_proj.x, W), ndc2Pix(p_proj.y, H) };
	float4 con_o = { conic.x, conic.y, conic.z, opacities[idx] };
  // Only counts tiles touched when nullptr is passed as array argment.
  uint32_t tiles_count = baseline_aabb
      ? duplicateAABBToTilesTouched(point_image, my_radius, grid, 0, 0, 0, nullptr, nullptr)
      : duplicateToTilesTouched(point_image, con_o, grid, 0, 0, 0, nullptr, nullptr);
  if (tiles_count == 0)
    return;

	// If colors have been precomputed, use them, otherwise convert
	// spherical harmonics coefficients to RGB color.
	if (colors_precomp == nullptr)
	{
		glm::vec3 result = computeColorFromSH(idx, D, M, (glm::vec3*)orig_points, *cam_pos, shs, clamped);
		rgb[idx * C + 0] = result.x;
		rgb[idx * C + 1] = result.y;
		rgb[idx * C + 2] = result.z;
	}

	// Store some useful helper data for the next steps.
	depths[idx] = p_view.z;
       radii[idx] = my_radius;
	points_xy_image[idx] = point_image;
	// Inverse 2D covariance and opacity neatly pack into one float4
	conic_opacity[idx] = con_o;
  tiles_touched[idx] = tiles_count;
}

// Main rasterization method. Collaboratively works on one tile per
// block, each thread treats one pixel. Alternates between fetching 
// and rasterizing data.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	unsigned long long* __restrict__ counters)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
	unsigned int lc[7] = { 0, 0, 0, 0, 0, 0, 0 };   // SARD workload counters (per-thread regs)

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;
			lc[0]++;   // [0] total pixel-Gaussian pairs iterated

			// Resample using conic matrix (cf. "Surface 
			// Splatting" by Zwicker et al., 2001)
			float2 xy = collected_xy[j];
			float2 d = { xy.x - pixf.x, xy.y - pixf.y };
			float4 con_o = collected_conic_opacity[j];
			float power = -0.5f * (con_o.x * d.x * d.x + con_o.z * d.y * d.y) - con_o.y * d.x * d.y;
			if (power > 0.0f)
			{
				lc[1]++;   // [1] sigma < 0 (degenerate) skip
				continue;
			}

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			// Avoid numerical instabilities (see paper appendix). 
			lc[3]++;   // [3] exp() evaluation
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
			{
				lc[4]++;   // [4] alpha < 1/255 skip
				continue;
			}
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}
			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// SARD workload counters: flush per-thread register counts to global once
	// per block (gated by counters != nullptr, i.e. SARD_STATS=1 at the host).
	if (inside) lc[6]++;
	if (counters)
		for (int k = 0; k < 7; k++)
			atomicAdd(&counters[k], (unsigned long long)lc[k]);

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}

// ===== SARD path: SOD recurrence + Rule 2.1 row-level early-stop =====
// Identical to renderCUDA except the inner per-Gaussian loop. The dense
// per-pixel conic quadratic (-0.5*(inv_xx*dx^2 + inv_yy*dy^2) - inv_xy*dx*dy)
// is replaced by a row-wise second-order difference (SOD) recurrence closed
// form, and a monotonicity-based early-stop (Rule 2.1) skips exp()+alpha-blend
// for whole rows that are guaranteed non-contributing. Output is numerically
// equivalent to renderCUDA: SOD is algebraically exact, and the early-stop only
// removes pixels that renderCUDA would also discard via alpha < 1/255.

// ===== gs_precision_lab: target RIE recurrence + per-operation fixed point =====
// Enabled only by SARD_PRECISION_QUANT=1.  This path is deliberately separate
// from timing/workload modes. counters[0..6] become saturation counters for
// [xy, conic, recurrence/sigma, opacity, RGB input, blend/T, output].
__device__ __forceinline__ float gsp_qfix(
	float x, int bits, int frac, bool is_signed, unsigned int* sat)
{
	const float scale = ldexpf(1.0f, frac);
	const float qlo = is_signed ? -ldexpf(1.0f, bits - 1) : 0.0f;
	const float qhi = is_signed ? ldexpf(1.0f, bits - 1) - 1.0f : ldexpf(1.0f, bits) - 1.0f;
	float scaled = x * scale;
	float clipped = fminf(qhi, fmaxf(qlo, scaled));
	if (clipped != scaled && sat != nullptr) (*sat)++;
	return nearbyintf(clipped) / scale;
}

// Split recurrence diagnostics.  The legacy seven counters remain unchanged;
// in precision mode counters[7..11] carry product/d1/sigma sign-separated
// saturation counts.  These slots are already allocated by the v3/v4
// instrumentation ABI for workload counters and are intentionally reused only
// when precision mode skips that workload post-pass.
__device__ __forceinline__ float gsp_qfix_recur(
	float x, int bits, int frac, bool is_signed, unsigned int* sat,
	unsigned int pos[3], unsigned int neg[3], int kind)
{
	const float scale = ldexpf(1.0f, frac);
	const float qlo = is_signed ? -ldexpf(1.0f, bits - 1) : 0.0f;
	const float qhi = is_signed ? ldexpf(1.0f, bits - 1) - 1.0f : ldexpf(1.0f, bits) - 1.0f;
	const float scaled = x * scale;
	const float clipped = fminf(qhi, fmaxf(qlo, scaled));
	if (clipped != scaled)
	{
		if (sat != nullptr) (*sat)++;
		if (kind >= 0 && kind < 3)
		{
			if (scaled > qhi) pos[kind]++;
			if (scaled < qlo) neg[kind]++;
		}
	}
	return nearbyintf(clipped) / scale;
}

// Optional v3.1 aggregate range recorder.  The extra counter slots are
// enabled only by SARD_RANGE_STATS=1 and are intentionally kept outside the
// legacy 12 workload/saturation counters.  Each signal owns six slots:
// min/max (double bit patterns), operation count, positive count, negative
// count, and reserved underflow count.
enum SardRangeSignal {
	SARD_R_LOCAL_DX0 = 0, SARD_R_LOCAL_DY0, SARD_R_LOCAL_DX,
	SARD_R_LOCAL_DY, SARD_R_CONIC0, SARD_R_CONIC1, SARD_R_CONIC2,
	SARD_R_OPACITY, SARD_R_COLOR, SARD_R_PRODUCT, SARD_R_D1,
	SARD_R_SIGMA, SARD_R_EXP_INPUT, SARD_R_ALPHA, SARD_R_TRANS,
	SARD_R_VISIBILITY, SARD_R_COLOR_PRODUCT, SARD_R_COLOR_ACCUM,
	SARD_RANGE_SIGNAL_COUNT
};
#define SARD_RANGE_BASE 12
#define SARD_RANGE_STRIDE 6

__device__ __forceinline__ void gsp_range_min(unsigned long long* addr, float x)
{
	if (!isfinite(x)) return;
	const double value = (double)x;
	const unsigned long long bits = __double_as_longlong(value);
	unsigned long long old = *addr;
	for (;;) {
		if (old == 0ULL) {
			const unsigned long long prior = atomicCAS(addr, 0ULL, bits);
			if (prior == 0ULL) return;
			old = prior;
			continue;
		}
		const double old_value = __longlong_as_double(old);
		if (value >= old_value) return;
		const unsigned long long prior = atomicCAS(addr, old, bits);
		if (prior == old) return;
		old = prior;
	}
}

__device__ __forceinline__ void gsp_range_max(unsigned long long* addr, float x)
{
	if (!isfinite(x)) return;
	const double value = (double)x;
	const unsigned long long bits = __double_as_longlong(value);
	unsigned long long old = *addr;
	for (;;) {
		if (old == 0ULL) {
			const unsigned long long prior = atomicCAS(addr, 0ULL, bits);
			if (prior == 0ULL) return;
			old = prior;
			continue;
		}
		const double old_value = __longlong_as_double(old);
		if (value <= old_value) return;
		const unsigned long long prior = atomicCAS(addr, old, bits);
		if (prior == old) return;
		old = prior;
	}
}

__device__ __forceinline__ void gsp_range_record(
	unsigned long long* counters, bool enabled, int signal, float value)
{
	if (!enabled || counters == nullptr || !isfinite(value)) return;
	const int base = SARD_RANGE_BASE + signal * SARD_RANGE_STRIDE;
	gsp_range_min(&counters[base + 0], value);
	gsp_range_max(&counters[base + 1], value);
	atomicAdd(&counters[base + 2], 1ULL);
	if (value > 0.0f) atomicAdd(&counters[base + 3], 1ULL);
	if (value < 0.0f) atomicAdd(&counters[base + 4], 1ULL);
}

__device__ __forceinline__ float gsp_exp_lut10(float sigma)
{
	if (sigma < 0.0f || sigma >= SARD_TAU_SIGMA) return 0.0f;
	const float scaled = sigma * (1023.0f / SARD_TAU_SIGMA);
	int addr = __float2int_rn(scaled);
	addr = max(0, min(1023, addr));
	return ((float)sard_exp_lut_10bit[addr]) * (1.0f / 32768.0f);
}

#include "gsrp_naive_precision.cuh"

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA_SARD_PRECISION(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	unsigned long long* __restrict__ counters,
	int opacity_bits, int rgb_bits, int conic_frac,
	int product_bits, int product_frac, int d1_frac, int sigma_frac, int local_frac,
	int state_bits, bool exp_lut10, bool range_stats, bool safe_sigma_mode)
{
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	bool inside = pix.x < W && pix.y < H;
	bool done = !inside;
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];
	float T = 1.0f;
	float C[CHANNELS] = { 0 };
	uint32_t contributor = 0, last_contributor = 0;
	unsigned int sat[7] = {0,0,0,0,0,0,0};
	unsigned int recur_pos[3] = {0,0,0};
	unsigned int recur_neg[3] = {0,0,0};
	unsigned int safe_sigma_bypasses = 0;
	const int rgb_frac = rgb_bits - 2;       // UQ2.(B-2)
	const int opacity_frac = opacity_bits - 1; // UQ1.(B-1), represents 1.0

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		if (__syncthreads_count(done) == BLOCK_SIZE) break;
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			contributor++;
			float2 xy = collected_xy[j];
			float4 co = collected_conic_opacity[j];
			// v3: subtract the tile origin before quantizing. The old path
			// quantized global x/y first and lost local fractional precision.
			const float raw_dx0 = xy.x - (float)pix_min.x;
			const float raw_dy0 = xy.y - (float)pix_min.y;
			gsp_range_record(counters, range_stats, SARD_R_LOCAL_DX0, raw_dx0);
			gsp_range_record(counters, range_stats, SARD_R_LOCAL_DY0, raw_dy0);
			float dx0 = gsp_qfix(raw_dx0, 16, local_frac, true, &sat[0]);
			float dy0 = gsp_qfix(raw_dy0, 16, local_frac, true, &sat[0]);
			gsp_range_record(counters, range_stats, SARD_R_CONIC0, co.x);
			gsp_range_record(counters, range_stats, SARD_R_CONIC1, co.y);
			gsp_range_record(counters, range_stats, SARD_R_CONIC2, co.z);
			float c0 = gsp_qfix(co.x, 24, conic_frac, true, &sat[1]);
			float c1 = gsp_qfix(co.y, 24, conic_frac, true, &sat[1]);
			float c2 = gsp_qfix(co.z, 24, conic_frac, true, &sat[1]);
			gsp_range_record(counters, range_stats, SARD_R_OPACITY, co.w);
			float opacity = gsp_qfix(co.w, opacity_bits, opacity_frac, false, &sat[3]);
			float u = (float)block.thread_index().x;
			gsp_range_record(counters, range_stats, SARD_R_LOCAL_DY,
				raw_dy0 - (float)block.thread_index().y);
			const float raw_dx_unquant = raw_dx0 - u;
			const float raw_dx = raw_dx_unquant;
			gsp_range_record(counters, range_stats, SARD_R_LOCAL_DX, raw_dx);
			// Safe-sigma fallback: use a wide, unquantized shadow recurrence
			// only to prove that this Gaussian is already non-contributing and
			// that sigma cannot decrease across the remaining columns.  This is
			// a bypass guard, not a replacement for the fixed-point datapath.
			if (safe_sigma_mode)
			{
				const float sh_t0 = co.z * raw_dy0;
				const float sh_t1 = co.y * raw_dx_unquant;
				const float sh_t2 = co.x * raw_dx_unquant;
				float sh_d1 = 0.5f * co.z - (sh_t0 + sh_t1);
				float sh_sigma = 0.5f * (raw_dy0 * sh_t0 + raw_dx_unquant * sh_t2) + raw_dy0 * sh_t1;
				for (int v = 0; v < (int)block.thread_index().y; v++)
				{
					sh_sigma += sh_d1;
					sh_d1 += co.z;
				}
				if (sh_sigma >= SARD_TAU_SIGMA && sh_d1 >= 0.0f && co.z >= 0.0f)
				{
					safe_sigma_bypasses++;
					continue;
				}
			}
			float dx = gsp_qfix(raw_dx, 16, local_frac, true, &sat[0]);
			const float raw_t0 = c2 * dy0;
			gsp_range_record(counters, range_stats, SARD_R_PRODUCT, raw_t0);
			gsp_range_record(counters, range_stats, SARD_R_D1, raw_t0);
			float t0 = gsp_qfix_recur(raw_t0, product_bits, product_frac, true, &sat[2], recur_pos, recur_neg, 0);
			t0 = gsp_qfix_recur(t0, state_bits, d1_frac, true, &sat[2], recur_pos, recur_neg, 1);
			gsp_range_record(counters, range_stats, SARD_R_D1, t0);
			const float raw_a = 0.5f * c2;
			gsp_range_record(counters, range_stats, SARD_R_D1, raw_a);
			float a = gsp_qfix_recur(raw_a, state_bits, d1_frac, true, &sat[2], recur_pos, recur_neg, 1);
			gsp_range_record(counters, range_stats, SARD_R_D1, a);
			const float raw_dyt0 = dy0 * t0;
			gsp_range_record(counters, range_stats, SARD_R_PRODUCT, raw_dyt0);
			gsp_range_record(counters, range_stats, SARD_R_D1, raw_dyt0);
			float dyt0 = gsp_qfix_recur(raw_dyt0, product_bits, product_frac, true, &sat[2], recur_pos, recur_neg, 0);
			dyt0 = gsp_qfix_recur(dyt0, state_bits, d1_frac, true, &sat[2], recur_pos, recur_neg, 1);
			gsp_range_record(counters, range_stats, SARD_R_D1, dyt0);
			const float raw_t1 = c1 * dx;
			gsp_range_record(counters, range_stats, SARD_R_PRODUCT, raw_t1);
			gsp_range_record(counters, range_stats, SARD_R_D1, raw_t1);
			float t1 = gsp_qfix_recur(raw_t1, product_bits, product_frac, true, &sat[2], recur_pos, recur_neg, 0);
			t1 = gsp_qfix_recur(t1, state_bits, d1_frac, true, &sat[2], recur_pos, recur_neg, 1);
			gsp_range_record(counters, range_stats, SARD_R_D1, t1);
			const float raw_t2 = c0 * dx;
			gsp_range_record(counters, range_stats, SARD_R_PRODUCT, raw_t2);
			gsp_range_record(counters, range_stats, SARD_R_D1, raw_t2);
			float t2 = gsp_qfix_recur(raw_t2, product_bits, product_frac, true, &sat[2], recur_pos, recur_neg, 0);
			t2 = gsp_qfix_recur(t2, state_bits, d1_frac, true, &sat[2], recur_pos, recur_neg, 1);
			gsp_range_record(counters, range_stats, SARD_R_D1, t2);
			const float raw_b0 = t0 + t1;
			gsp_range_record(counters, range_stats, SARD_R_D1, raw_b0);
			float b = gsp_qfix_recur(raw_b0, state_bits, d1_frac, true, &sat[2], recur_pos, recur_neg, 1);
			const float raw_b1 = -b;
			gsp_range_record(counters, range_stats, SARD_R_D1, raw_b1);
			b = gsp_qfix_recur(raw_b1, state_bits, d1_frac, true, &sat[2], recur_pos, recur_neg, 1);
			const float raw_d1 = a + b;
			gsp_range_record(counters, range_stats, SARD_R_D1, raw_d1);
			float d1 = gsp_qfix_recur(raw_d1, state_bits, d1_frac, true, &sat[2], recur_pos, recur_neg, 1);
			gsp_range_record(counters, range_stats, SARD_R_D1, d1);
			const float raw_dx_t2 = dx * t2;
			gsp_range_record(counters, range_stats, SARD_R_PRODUCT, raw_dx_t2);
			gsp_range_record(counters, range_stats, SARD_R_D1, raw_dx_t2);
			float dx_t2 = gsp_qfix_recur(raw_dx_t2, product_bits, product_frac, true, &sat[2], recur_pos, recur_neg, 0);
			dx_t2 = gsp_qfix_recur(dx_t2, state_bits, d1_frac, true, &sat[2], recur_pos, recur_neg, 1);
			gsp_range_record(counters, range_stats, SARD_R_D1, dx_t2);
			const float raw_sum1 = dyt0 + dx_t2;
			gsp_range_record(counters, range_stats, SARD_R_SIGMA, raw_sum1);
			float sum1 = gsp_qfix_recur(raw_sum1, state_bits, sigma_frac, true, &sat[2], recur_pos, recur_neg, 2);
			const float raw_sigma0 = 0.5f * sum1;
			gsp_range_record(counters, range_stats, SARD_R_SIGMA, raw_sigma0);
			float sigma = gsp_qfix_recur(raw_sigma0, state_bits, sigma_frac, true, &sat[2], recur_pos, recur_neg, 2);
			gsp_range_record(counters, range_stats, SARD_R_SIGMA, sigma);
			bool sigma_noncontrib = safe_sigma_mode && sigma >= SARD_TAU_SIGMA && d1 >= 0.0f && c2 >= 0.0f;
			const float raw_dy_t1 = dy0 * t1;
			gsp_range_record(counters, range_stats, SARD_R_PRODUCT, raw_dy_t1);
			gsp_range_record(counters, range_stats, SARD_R_SIGMA, raw_dy_t1);
			float dy_t1 = gsp_qfix_recur(raw_dy_t1, product_bits, product_frac, true, &sat[2], recur_pos, recur_neg, 0);
			dy_t1 = gsp_qfix_recur(dy_t1, state_bits, sigma_frac, true, &sat[2], recur_pos, recur_neg, 2);
			gsp_range_record(counters, range_stats, SARD_R_SIGMA, dy_t1);
			sigma = gsp_qfix_recur(sigma + dy_t1, state_bits, sigma_frac, true, &sat[2], recur_pos, recur_neg, 2);
			gsp_range_record(counters, range_stats, SARD_R_SIGMA, sigma);
			for (int v = 0; v < (int)block.thread_index().y; v++)
			{
				const float raw_sigma_next = sigma + d1;
				sigma = gsp_qfix_recur(raw_sigma_next, state_bits, sigma_frac, true, &sat[2], recur_pos, recur_neg, 2);
				const float raw_d1_next = d1 + c2;
				if (safe_sigma_mode && raw_sigma_next >= SARD_TAU_SIGMA && raw_d1_next >= 0.0f && c2 >= 0.0f)
					sigma_noncontrib = true;
				d1 = gsp_qfix_recur(raw_d1_next, state_bits, d1_frac, true, &sat[2], recur_pos, recur_neg, 1);
				gsp_range_record(counters, range_stats, SARD_R_SIGMA, raw_sigma_next);
				gsp_range_record(counters, range_stats, SARD_R_D1, raw_d1_next);
				gsp_range_record(counters, range_stats, SARD_R_SIGMA, sigma);
				gsp_range_record(counters, range_stats, SARD_R_D1, d1);
			}
			if (sigma_noncontrib)
			{
				safe_sigma_bypasses++;
				continue;
			}
			float power = gsp_qfix_recur(-sigma, state_bits, sigma_frac, true, &sat[2], recur_pos, recur_neg, 2);
			gsp_range_record(counters, range_stats, SARD_R_EXP_INPUT, power);
			if (power > 0.0f) continue;
			float expv = exp_lut10 ? gsp_exp_lut10(sigma) : gsp_qfix(expf(power), 16, 15, false, &sat[5]);
			float alpha = gsp_qfix(opacity * expv, 16, 15, false, &sat[5]);
			gsp_range_record(counters, range_stats, SARD_R_ALPHA, opacity * expv);
			alpha = gsp_qfix(fminf(0.99f, alpha), 16, 15, false, &sat[5]);
			if (alpha < 1.0f / 255.0f) continue;
			float omt = gsp_qfix(1.0f - alpha, 16, 15, false, &sat[5]);
			float test_T = gsp_qfix(T * omt, 16, 15, false, &sat[5]);
			float vis = gsp_qfix(alpha * T, 16, 15, false, &sat[5]);
			gsp_range_record(counters, range_stats, SARD_R_TRANS, test_T);
			gsp_range_record(counters, range_stats, SARD_R_VISIBILITY, alpha * T);
			for (int ch = 0; ch < CHANNELS; ch++)
			{
				float color = gsp_qfix(features[collected_id[j] * CHANNELS + ch], rgb_bits, rgb_frac, false, &sat[4]);
				float prod = gsp_qfix(vis * color, 16, 14, false, &sat[5]);
				gsp_range_record(counters, range_stats, SARD_R_COLOR, features[collected_id[j] * CHANNELS + ch]);
				gsp_range_record(counters, range_stats, SARD_R_COLOR_PRODUCT, vis * color);
				gsp_range_record(counters, range_stats, SARD_R_COLOR_ACCUM, C[ch] + prod);
				C[ch] = gsp_qfix(C[ch] + prod, 16, 13, false, &sat[5]);
			}
			T = test_T;
			last_contributor = contributor;
			// The Gaussian that crosses the T threshold still contributes.
			// T-stop only suppresses later depth-ordered KVPs.
			if (T < 0.0001f) done = true;
		}
	}
	if (counters && inside)
	{
		for (int k = 0; k < 7; k++) atomicAdd(&counters[k], (unsigned long long)sat[k]);
		atomicAdd(&counters[7], (unsigned long long)recur_pos[0]);
		atomicAdd(&counters[8], (unsigned long long)recur_neg[0]);
		atomicAdd(&counters[9], (unsigned long long)recur_pos[1]);
		atomicAdd(&counters[10], (unsigned long long)recur_neg[1]);
		atomicAdd(&counters[11], ((unsigned long long)recur_pos[2] << 32) | recur_neg[2]);
		atomicAdd(&counters[6], (unsigned long long)safe_sigma_bypasses);
	}
	if (inside)
	{
		final_T[pix_id] = T; n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
		{
			float bgprod = gsp_qfix(T * bg_color[ch], 16, 14, false, &sat[6]);
			out_color[ch * H * W + pix_id] = gsp_qfix(C[ch] + bgprod, 16, 13, false, &sat[6]);
		}
	}
}

template <uint32_t CHANNELS, bool EARLYSTOP = true>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA_SARD(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	unsigned long long* __restrict__ counters,
	unsigned int* __restrict__ t_lane_trace_entries)
{
	// Identify current tile and associated min/max pixel range.
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	// Check if this thread is associated with a valid pixel or outside.
	bool inside = pix.x < W&& pix.y < H;
	// Done threads can help with fetching, but don't rasterize
	bool done = !inside;

	// Load start/end range of IDs to process in bit sorted list.
	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	// Allocate storage for batches of collectively fetched data.
	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	// Initialize helper variables
	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
	unsigned int lc[7] = { 0, 0, 0, 0, 0, 0, 0 };   // SARD workload counters (per-thread regs)

	// Iterate over batches until all done or range is complete
	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		// End if entire block votes that it is done rasterizing
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		// Collectively fetch per-Gaussian data from global to shared
		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		// Iterate over current batch
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); j++)
		{
			// Keep track of current position in range
			contributor++;
			lc[0]++;   // [0] total pixel-Gaussian pairs iterated

			float2 xy = collected_xy[j];
			float4 con_o = collected_conic_opacity[j];

			// --- SARD: row-wise second-order difference (SOD) recurrence ---
			// For a fixed row v, sigma(u,v) is quadratic in column index u.
			// Build the row-init bundle: s0 = sigma(0,v), d1_0 = sigma(1,v)-sigma(0,v),
			// and the constant second difference d2 = inv_xx (= con_o.x, >= 0).
			// Each thread computes its own row's bundle here (init per-pixel);
			// a shared-memory broadcast across the row is a later tuning step.
			float dx0 = xy.x - (float)pix_min.x;       // mu_x - tile x origin (tile-constant)
			float dy_v = xy.y - pixf.y;                 // row-constant vertical offset
			float term_y = 0.5f * con_o.z * dy_v * dy_v;
			float s0 = 0.5f * con_o.x * dx0 * dx0 + term_y + con_o.y * dx0 * dy_v;
			// d1_0 = sigma(1,v) - sigma(0,v), derived analytically (avoids forming s1).
			float d1_0 = -con_o.x * dx0 - con_o.y * dy_v + 0.5f * con_o.x;
			float d2 = con_o.x;                          // constant second difference, >= 0

			// --- SARD Rule 2.1: row-level monotonic early-stop ---
			// If the row is monotonically non-decreasing (d1_0 >= 0, d2 >= 0)
			// starting from s0 > tau_sigma, every column has alpha < 1/255 and
			// contributes nothing. Skip the expensive exp()+alpha-blend for the
			// whole row. Conservative: only skips pixels renderCUDA also skips.
			if (EARLYSTOP && s0 > SARD_TAU_SIGMA && d1_0 >= 0.0f)
			{
				lc[2]++;   // [2] Rule 2.1 row-level early-stop skip
				continue;
			}

			// Recover sigma at this thread's column via the SOD closed form.
			float uf = (float)block.thread_index().x;   // column index 0..BLOCK_X-1
			float sigma = s0 + uf * d1_0 + 0.5f * uf * (uf - 1.0f) * d2;
			float power = -sigma;
			if (power > 0.0f)                            // sigma < 0: degenerate
				continue;

			// Eq. (2) from 3D Gaussian splatting paper.
			// Obtain alpha by multiplying with Gaussian opacity
			// and its exponential falloff from mean.
			lc[3]++;   // [3] exp() evaluation
			float alpha = min(0.99f, con_o.w * exp(power));
			if (alpha < 1.0f / 255.0f)
			{
				lc[4]++;   // [4] alpha < 1/255 skip
				continue;
			}
			float test_T = T * (1 - alpha);
			if (test_T < 0.0001f)
			{
				done = true;
				continue;
			}
			// This is the exact RSPA-input mask: prior T state was still live,
			// and this Gaussian's contribution occurs before it updates T.
			if (t_lane_trace_entries != nullptr)
				atomicOr(&t_lane_trace_entries[(size_t)(range.x + i * BLOCK_SIZE + j) * 32 + 16 + block.thread_index().y], 1u << block.thread_index().x);

			// Eq. (3) from 3D Gaussian splatting paper.
			for (int ch = 0; ch < CHANNELS; ch++)
				C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

			T = test_T;

			// Keep track of last range entry to update this
			// pixel.
			last_contributor = contributor;
		}
	}

	// SARD workload counters: flush per-thread register counts to global once
	// per block (gated by counters != nullptr, i.e. SARD_STATS=1 at the host).
	if (inside) lc[6]++;
	if (counters)
		for (int k = 0; k < 7; k++)
			atomicAdd(&counters[k], (unsigned long long)lc[k]);

	// All threads that treat valid pixel write out their final
	// rendering data to the frame and auxiliary buffers.
	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}

// ===== SARD v3: SOD + warp-uniform Rule 2.1 early-stop =====
// The per-pixel early-stop in renderCUDA_SARD is inert on SIMT GPUs: a warp
// covers two rows (BLOCK_X=16, warp=32), and if ANY pixel in the warp still
// needs exp() the whole warp pays for it, so masking off the skippable pixels
// saves no wall-clock time. This variant makes the skip warp-uniform: exp()+
// alpha-blend are skipped for the entire warp only when EVERY still-active
// pixel in the warp is provably non-contributing for this Gaussian.
//
// Voting every iteration requires the warp to stay converged, so the !done
// early-skip is turned into a predicate: done threads skip the sigma compute
// and the exp()/blend but still take part in the ballot/all vote (cheap). The
// block-level early-exit (__syncthreads_count(done) == BLOCK_SIZE) is kept.
// Numerically equivalent to renderCUDA: the gate only skips Gaussians that
// contribute alpha < 1/255 in every active pixel of the warp.
template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA_SARD_WG(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float* __restrict__ features,
	const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T,
	uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color,
	float* __restrict__ out_color,
	unsigned long long* __restrict__ counters)
{
	auto block = cg::this_thread_block();
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y };
	uint2 pix_max = { min(pix_min.x + BLOCK_X, W), min(pix_min.y + BLOCK_Y , H) };
	uint2 pix = { pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y };
	uint32_t pix_id = W * pix.y + pix.x;
	float2 pixf = { (float)pix.x, (float)pix.y };

	bool inside = pix.x < W&& pix.y < H;
	bool done = !inside;

	uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	__shared__ int collected_id[BLOCK_SIZE];
	__shared__ float2 collected_xy[BLOCK_SIZE];
	__shared__ float4 collected_conic_opacity[BLOCK_SIZE];

	float T = 1.0f;
	uint32_t contributor = 0;
	uint32_t last_contributor = 0;
	float C[CHANNELS] = { 0 };
	unsigned int lc[7] = { 0, 0, 0, 0, 0, 0, 0 };   // SARD workload counters (per-thread regs)

	for (int i = 0; i < rounds; i++, toDo -= BLOCK_SIZE)
	{
		int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
			break;

		int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y)
		{
			int coll_id = point_list[range.x + progress];
			collected_id[block.thread_rank()] = coll_id;
			collected_xy[block.thread_rank()] = points_xy_image[coll_id];
			collected_conic_opacity[block.thread_rank()] = conic_opacity[coll_id];
		}
		block.sync();

		for (int j = 0; j < min(BLOCK_SIZE, toDo); j++)
		{
			float4 con_o = collected_conic_opacity[j];

			// SOD row-init. Done threads skip it (their vote is excluded by the
			// active mask anyway), but they still reach the warp vote below so
			// the warp stays converged. d2 = con_o.x is implicit in sigma below.
			float s0 = 0.0f, d1_0 = 0.0f;
			bool row_skip = false;
			if (!done)
			{
				float2 xy = collected_xy[j];
				float dx0 = xy.x - (float)pix_min.x;
				float dy_v = xy.y - pixf.y;
				float term_y = 0.5f * con_o.z * dy_v * dy_v;
				s0 = 0.5f * con_o.x * dx0 * dx0 + term_y + con_o.y * dx0 * dy_v;
				d1_0 = -con_o.x * dx0 - con_o.y * dy_v + 0.5f * con_o.x;
				row_skip = (s0 > SARD_TAU_SIGMA) && (d1_0 >= 0.0f);
			}

			// Warp-uniform gate: skip exp()+blend for the whole warp iff every
			// still-active pixel is skippable. __ballot_sync / __all_sync are
			// convergent collectives; all lanes (incl. done) call them, and the
			// mask restricts the AND to active lanes. Empty mask (whole warp
			// done) would be UB for __all_sync, so guard it.
			unsigned active = __ballot_sync(0xffffffff, !done);
			bool warp_all_skip = (active != 0) && __all_sync(active, row_skip);

			if (!done && !warp_all_skip)
			{
				contributor++;
				float uf = (float)block.thread_index().x;
				float sigma = s0 + uf * d1_0 + 0.5f * uf * (uf - 1.0f) * con_o.x;
				float power = -sigma;
				if (power > 0.0f)
					continue;

				float alpha = min(0.99f, con_o.w * exp(power));
				if (alpha < 1.0f / 255.0f)
					continue;
				float test_T = T * (1 - alpha);
				if (test_T < 0.0001f)
				{
					done = true;
					continue;
				}

				lc[5]++;   // [5] actually blended into the pixel
				for (int ch = 0; ch < CHANNELS; ch++)
					C[ch] += features[collected_id[j] * CHANNELS + ch] * alpha * T;

				T = test_T;
				last_contributor = contributor;
			}
		}
	}

	if (inside)
	{
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ch++)
			out_color[ch * H * W + pix_id] = C[ch] + T * bg_color[ch];
	}
}

// Offline workload characterization for the sparse SARD back end.  This
// kernel consumes the exact post-preprocess/post-binning arrays produced by
// the renderer and counts row/chunk occupancy without changing the rendered
// image or the KVP stream.  It is launched only when SARD_STATS=1.
__global__ void countNonemptySARD(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	unsigned long long* __restrict__ counters,
	unsigned long long* __restrict__ tile_counters,
	unsigned char* __restrict__ fifo_trace_entries,
	unsigned int* __restrict__ t_lane_trace_entries)
{
	uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	uint2 pix_min = { blockIdx.x * BLOCK_X, blockIdx.y * BLOCK_Y };
	uint2 range = ranges[blockIdx.y * horizontal_blocks + blockIdx.x];

	__shared__ unsigned long long block_counts[5];
	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		for (int i = 0; i < 5; ++i) block_counts[i] = 0;
	}
	__syncthreads();

	unsigned long long local_rows = 0;
	unsigned long long local_w4 = 0;
	unsigned long long local_w8 = 0;
	unsigned long long local_w16 = 0;
	unsigned long long local_rie_cycles = 0;
	const int lane = (int)threadIdx.y * BLOCK_X + (int)threadIdx.x;
	const int tile_pixels = BLOCK_X * BLOCK_Y;

	for (uint32_t p = range.x + lane; p < range.y; p += tile_pixels)
	{
		const int coll_id = point_list[p];
		const float2 xy = points_xy_image[coll_id];
		const float4 con_o = conic_opacity[coll_id];
		const unsigned int producer_cycles = sardRieProducerCycles(xy, con_o, pix_min);
		local_rie_cycles += producer_cycles;
		unsigned int row_masks[BLOCK_Y];
		for (int y = 0; y < BLOCK_Y; ++y) row_masks[y] = 0;

		for (int y = 0; y < BLOCK_Y; ++y)
		{
			if (pix_min.y + y >= H) continue;
			const float dy_v = xy.y - (float)(pix_min.y + y);
			const float dx0 = xy.x - (float)pix_min.x;
			const float s0 = 0.5f * con_o.x * dx0 * dx0
				+ con_o.z * dy_v * dy_v + con_o.y * dx0 * dy_v;
			const float d1_0 = -con_o.x * dx0 - con_o.y * dy_v + 0.5f * con_o.x;
			if (s0 > SARD_TAU_SIGMA && d1_0 >= 0.0f) continue;

			for (int x = 0; x < BLOCK_X; ++x)
			{
				if (pix_min.x + x >= W) continue;
				const float dx = xy.x - (float)(pix_min.x + x);
				const float sigma = 0.5f * con_o.x * dx * dx
					+ con_o.z * dy_v * dy_v + con_o.y * dx * dy_v;
				if (sigma < 0.0f) continue;
				const float alpha = min(0.99f, con_o.w * expf(-sigma));
				if (alpha < 1.0f / 255.0f) continue;
				row_masks[y] |= (1u << x);
			}
		}

		if (fifo_trace_entries != nullptr)
		{
			const size_t base = (size_t)p * 9;
			fifo_trace_entries[base] = (unsigned char)producer_cycles;
			for (int y = 0; y < BLOCK_Y; ++y)
			{
				const unsigned int row_mask = row_masks[y];
				unsigned char w4_mask = 0;
				if (row_mask & 0x0000000Fu) w4_mask |= 0x1;
				if (row_mask & 0x000000F0u) w4_mask |= 0x2;
				if (row_mask & 0x00000F00u) w4_mask |= 0x4;
				if (row_mask & 0x0000F000u) w4_mask |= 0x8;
				const size_t packed_index = base + 1 + (size_t)(y >> 1);
				if ((y & 1) == 0)
					fifo_trace_entries[packed_index] = w4_mask;
				else
					fifo_trace_entries[packed_index] |= (unsigned char)(w4_mask << 4);
			}
		}
		if (t_lane_trace_entries != nullptr)
			for (int y = 0; y < BLOCK_Y; ++y)
				t_lane_trace_entries[(size_t)p * 32 + y] = row_masks[y];

		for (int y = 0; y < BLOCK_Y; ++y)
		{
			const unsigned int mask = row_masks[y];
			if (mask == 0) continue;
			local_rows++;
			local_w16++;
			if (mask & 0x0000000Fu) local_w4++;
			if (mask & 0x000000F0u) local_w4++;
			if (mask & 0x00000F00u) local_w4++;
			if (mask & 0x0000F000u) local_w4++;
			if (mask & 0x000000FFu) local_w8++;
			if (mask & 0x0000FF00u) local_w8++;
		}
	}

	atomicAdd(&block_counts[0], local_rie_cycles);
	atomicAdd(&block_counts[1], local_rows);
	atomicAdd(&block_counts[2], local_w4);
	atomicAdd(&block_counts[3], local_w8);
	atomicAdd(&block_counts[4], local_w16);
	__syncthreads();
	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		const uint32_t tile_id = blockIdx.y * horizontal_blocks + blockIdx.x;
		tile_counters[tile_id * 6 + 0] = (unsigned long long)(range.y - range.x);
		tile_counters[tile_id * 6 + 1] = block_counts[0];
		tile_counters[tile_id * 6 + 2] = block_counts[1];
		tile_counters[tile_id * 6 + 3] = block_counts[2];
		tile_counters[tile_id * 6 + 4] = block_counts[3];
		tile_counters[tile_id * 6 + 5] = block_counts[4];
		atomicAdd(&counters[7], block_counts[1]);
		atomicAdd(&counters[8], block_counts[2]);
		atomicAdd(&counters[9], block_counts[3]);
		atomicAdd(&counters[10], block_counts[4]);
		atomicAdd(&counters[11], block_counts[0]);
	}
}

// gs_arch_sim v1.1 summary-only instrumentation.  One 16x16 CUDA block
// replays a tile's depth-ordered KVP list using the same alpha and T gates as
// renderCUDA.  It records only seven aggregate values per tile and never
// materializes row, chunk, or lane traces.
__global__ void summarizeTileTStop(
	const uint2* __restrict__ ranges,
	const uint32_t* __restrict__ point_list,
	int W, int H,
	const float2* __restrict__ points_xy_image,
	const float4* __restrict__ conic_opacity,
	unsigned long long* __restrict__ tile_t_summary)
{
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint32_t tile_id = blockIdx.y * horizontal_blocks + blockIdx.x;
	const uint2 pix_min = { blockIdx.x * BLOCK_X, blockIdx.y * BLOCK_Y };
	const uint2 pix = { pix_min.x + threadIdx.x, pix_min.y + threadIdx.y };
	const bool inside = pix.x < W && pix.y < H;
	bool done = !inside;
	float T = 1.0f;
	const uint2 range = ranges[tile_id];
	const unsigned int raw = range.y - range.x;

	__shared__ unsigned long long producer_raw;
	__shared__ unsigned long long producer_retained;
	__shared__ unsigned int retained;
	__shared__ unsigned int stop_after;
	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		producer_raw = 0;
		producer_retained = 0;
		retained = raw;
		stop_after = raw;
		for (uint32_t p = range.x; p < range.y; ++p)
		{
			const int coll_id = point_list[p];
			producer_raw += sardRieProducerCycles(
				points_xy_image[coll_id], conic_opacity[coll_id], pix_min);
		}
	}
	__syncthreads();

	for (unsigned int k = 0; k < raw; ++k)
	{
		const int coll_id = point_list[range.x + k];
		const float2 xy = points_xy_image[coll_id];
		const float4 con_o = conic_opacity[coll_id];
		if (threadIdx.x == 0 && threadIdx.y == 0)
			producer_retained += sardRieProducerCycles(xy, con_o, pix_min);

		if (!done)
		{
			const float dx = xy.x - (float)pix.x;
			const float dy = xy.y - (float)pix.y;
			const float power = -0.5f * (con_o.x * dx * dx + con_o.z * dy * dy)
				- con_o.y * dx * dy;
			if (power <= 0.0f)
			{
				const float alpha = min(0.99f, con_o.w * expf(power));
				if (alpha >= 1.0f / 255.0f)
				{
					const float test_T = T * (1.0f - alpha);
					if (test_T < 0.0001f)
						done = true;
					else
						T = test_T;
				}
			}
		}

		const int num_done = __syncthreads_count(done);
		if (num_done == BLOCK_SIZE)
		{
			if (threadIdx.x == 0 && threadIdx.y == 0)
			{
				retained = k + 1;
				stop_after = k + 1;
			}
			break;
		}
	}
	__syncthreads();

	if (threadIdx.x == 0 && threadIdx.y == 0)
	{
		const size_t base = (size_t)tile_id * 7;
		tile_t_summary[base + 0] = raw;
		tile_t_summary[base + 1] = retained;
		tile_t_summary[base + 2] = raw - retained;
		tile_t_summary[base + 3] = producer_raw;
		tile_t_summary[base + 4] = producer_retained;
		tile_t_summary[base + 5] = stop_after;
		tile_t_summary[base + 6] =
			min((int)BLOCK_X, W - (int)pix_min.x) *
			min((int)BLOCK_Y, H - (int)pix_min.y);
	}
}

void FORWARD::render(
	const dim3 grid, dim3 block,
	const uint2* ranges,
	const uint32_t* point_list,
	int W, int H,
	const float2* means2D,
	const float* colors,
	const float4* conic_opacity,
	float* final_T,
	uint32_t* n_contrib,
	const float* bg_color,
	float* out_color,
	unsigned long long* counters,
	unsigned long long* tile_counters,
	unsigned long long* tile_t_summary,
	unsigned char* fifo_trace_entries,
	unsigned int* t_lane_trace_entries)
{
	// Independent gs_raster_precision path.  This branch always evaluates the
	// original per-pixel quadratic and never enters the SARD recurrence kernels.
	const char* gsrp_env = getenv("GSRP_MODE");
	const int gsrp_mode = gsrp_env ? atoi(gsrp_env) : 0;
	GsrpFormats gsrp_cfg{};
	gsrp_cfg.mode = gsrp_mode;
	const char* gsrp_mask_env = getenv("GSRP_ACTIVE_MASK");
	gsrp_cfg.active_mask = gsrp_mask_env ? (unsigned int)strtoul(gsrp_mask_env, nullptr, 0) : 0u;
	const char* gsrp_names[7] = {"GSRP_MEAN_X", "GSRP_MEAN_Y", "GSRP_CONIC_A", "GSRP_CONIC_B", "GSRP_CONIC_C", "GSRP_OPACITY", "GSRP_RGB"};
	for (int i = 0; i < 7; ++i) {
		char bits_name[64], frac_name[64];
		snprintf(bits_name, sizeof(bits_name), "%s_BITS", gsrp_names[i]);
		snprintf(frac_name, sizeof(frac_name), "%s_FRAC", gsrp_names[i]);
		const char* bits_env = getenv(bits_name);
		const char* frac_env = getenv(frac_name);
		gsrp_cfg.bits[i] = bits_env ? atoi(bits_env) : 16;
		gsrp_cfg.frac[i] = frac_env ? atoi(frac_env) : 8;
	}
	const char* gsrp_power_bits = getenv("GSRP_POWER_BITS");
	const char* gsrp_power_frac = getenv("GSRP_POWER_FRAC");
	const char* gsrp_exp_addr = getenv("GSRP_EXP_ADDR_BITS");
	const char* gsrp_exp_out = getenv("GSRP_EXP_OUT_BITS");
	gsrp_cfg.power_bits = gsrp_power_bits ? atoi(gsrp_power_bits) : 24;
	gsrp_cfg.power_frac = gsrp_power_frac ? atoi(gsrp_power_frac) : 18;
	gsrp_cfg.exp_addr_bits = gsrp_exp_addr ? atoi(gsrp_exp_addr) : 10;
	gsrp_cfg.exp_out_bits = gsrp_exp_out ? atoi(gsrp_exp_out) : 16;
	const char* gsrp_quad_fixed = getenv("GSRP_QUAD_FIXED");
	const char* gsrp_quad_geom_bits = getenv("GSRP_QUAD_GEOM_BITS");
	const char* gsrp_quad_geom_frac = getenv("GSRP_QUAD_GEOM_FRAC");
	const char* gsrp_quad_term_bits = getenv("GSRP_QUAD_TERM_BITS");
	const char* gsrp_quad_term_frac = getenv("GSRP_QUAD_TERM_FRAC");
	const char* gsrp_quad_sum_bits = getenv("GSRP_QUAD_SUM_BITS");
	const char* gsrp_quad_sum_frac = getenv("GSRP_QUAD_SUM_FRAC");
	const char* gsrp_exp_lut_enable = getenv("GSRP_EXP_LUT_ENABLE");
	gsrp_cfg.quad_fixed = gsrp_quad_fixed && gsrp_quad_fixed[0] == '1';
	gsrp_cfg.quad_geom_bits = gsrp_quad_geom_bits ? atoi(gsrp_quad_geom_bits) : 30;
	gsrp_cfg.quad_geom_frac = gsrp_quad_geom_frac ? atoi(gsrp_quad_geom_frac) : 2;
	gsrp_cfg.quad_term_bits = gsrp_quad_term_bits ? atoi(gsrp_quad_term_bits) : 28;
	gsrp_cfg.quad_term_frac = gsrp_quad_term_frac ? atoi(gsrp_quad_term_frac) : 16;
	gsrp_cfg.quad_sum_bits = gsrp_quad_sum_bits ? atoi(gsrp_quad_sum_bits) : 30;
	gsrp_cfg.quad_sum_frac = gsrp_quad_sum_frac ? atoi(gsrp_quad_sum_frac) : 14;
	gsrp_cfg.exp_lut_enable = gsrp_exp_lut_enable && gsrp_exp_lut_enable[0] == '1';
	const char* gsrp_blend = getenv("GSRP_BLEND_FIXED");
	gsrp_cfg.blend_mode = gsrp_blend && gsrp_blend[0] == '1';
	const char* gsrp_blend_mask = getenv("GSRP_BLEND_MASK");
	gsrp_cfg.blend_mask = gsrp_blend_mask ? (unsigned int)strtoul(gsrp_blend_mask, nullptr, 0) : 0xFu;
	const char* gsrp_raw_alpha = getenv("GSRP_BLEND_RAW_ALPHA_TSTOP");
	gsrp_cfg.raw_alpha_for_tstop = gsrp_raw_alpha && gsrp_raw_alpha[0] == '1';
	const int blend_default_bits[4] = {8, 16, 16, 20};
	const int blend_default_frac[4] = {8, 15, 15, 16};
	const char* blend_names[4] = {"ALPHA", "T", "VIS", "ACC"};
	for (int i = 0; i < 4; ++i) {
		char bits_name[64], frac_name[64];
		snprintf(bits_name, sizeof(bits_name), "GSRP_BLEND_%s_BITS", blend_names[i]);
		snprintf(frac_name, sizeof(frac_name), "GSRP_BLEND_%s_FRAC", blend_names[i]);
		const char* bits_env = getenv(bits_name);
		const char* frac_env = getenv(frac_name);
		gsrp_cfg.blend_bits[i] = bits_env ? atoi(bits_env) : blend_default_bits[i];
		gsrp_cfg.blend_frac[i] = frac_env ? atoi(frac_env) : blend_default_frac[i];
	}
	const char* gsrp_range_env = getenv("GSRP_RANGE_STATS");
	const bool gsrp_range_stats = gsrp_range_env && gsrp_range_env[0] == '1';
	if (getenv("GSRP_DEBUG"))
		fprintf(stderr, "GSRP cfg mode=%d mask=0x%x meanx=%d/%d meany=%d/%d A=%d/%d B=%d/%d C=%d/%d op=%d/%d rgb=%d/%d\\n",
			gsrp_cfg.mode, gsrp_cfg.active_mask,
			gsrp_cfg.bits[0], gsrp_cfg.frac[0], gsrp_cfg.bits[1], gsrp_cfg.frac[1],
			gsrp_cfg.bits[2], gsrp_cfg.frac[2], gsrp_cfg.bits[3], gsrp_cfg.frac[3],
			gsrp_cfg.bits[4], gsrp_cfg.frac[4], gsrp_cfg.bits[5], gsrp_cfg.frac[5],
			gsrp_cfg.bits[6], gsrp_cfg.frac[6]);
	if (gsrp_mode < 0 || gsrp_mode > 7 || gsrp_cfg.exp_addr_bits < 8 || gsrp_cfg.exp_addr_bits > 10 || gsrp_cfg.exp_out_bits < 8 || gsrp_cfg.exp_out_bits > 16 ||
		gsrp_cfg.quad_geom_bits < 16 || gsrp_cfg.quad_geom_bits > 50 ||
		gsrp_cfg.quad_term_bits < 16 || gsrp_cfg.quad_term_bits > 50 ||
		gsrp_cfg.quad_sum_bits < 16 || gsrp_cfg.quad_sum_bits > 50) {
		fprintf(stderr, "invalid GSRP precision configuration\n");
		return;
	}
	// SARD runtime switch (recoverable), read every call so the path can be
	// toggled within a single process for A/B benchmarking (cost: one getenv
	// per rendered image, negligible):
	//   SARD_SOD unset/0 -> original renderCUDA
	//   SARD_SOD=1       -> SOD recurrence + Rule 2.1 early-stop (full SARD)
	//   SARD_SOD=2       -> SOD recurrence only, no early-stop (diagnostic:
	//                       isolates SOD arithmetic cost from early-stop gain)
	//   SARD_SOD=3       -> SOD + warp-uniform early-stop (restructured so the
	//                       skip can actually save exp() time on SIMT GPUs)
	const char* sard_env = getenv("SARD_SOD");
	int sard_mode = (sard_env != nullptr && sard_env[0] >= '0' && sard_env[0] <= '9')
		? (sard_env[0] - '0') : 0;
	const char* precision_env = getenv("SARD_PRECISION_QUANT");
	const bool precision_quant = precision_env != nullptr && precision_env[0] == '1';
	int opacity_bits = 6, rgb_bits = 10;
	int product_bits = 24;
	int conic_frac = 20, product_frac = 11, d1_frac = 7, sigma_frac = 7;
	int local_frac = 3, state_bits = 20;
	bool exp_lut10 = false;
	bool safe_sigma_mode = false;
	const char* opacity_env = getenv("SARD_PRECISION_OPACITY_BITS");
	const char* rgb_env = getenv("SARD_PRECISION_RGB_BITS");
	const char* conic_frac_env = getenv("SARD_PRECISION_CONIC_FRAC");
	const char* product_frac_env = getenv("SARD_PRECISION_PRODUCT_FRAC");
	const char* product_bits_env = getenv("SARD_PRECISION_PRODUCT_BITS");
	const char* d1_frac_env = getenv("SARD_PRECISION_D1_FRAC");
	const char* sigma_frac_env = getenv("SARD_PRECISION_SIGMA_FRAC");
	const char* local_frac_env = getenv("SARD_PRECISION_LOCAL_FRAC");
	const char* state_bits_env = getenv("SARD_PRECISION_STATE_BITS");
	const char* exp_lut_env = getenv("SARD_PRECISION_EXP_LUT10");
	const char* safe_sigma_env = getenv("SARD_PRECISION_SAFE_SIGMA");
	if (opacity_env) opacity_bits = atoi(opacity_env);
	if (rgb_env) rgb_bits = atoi(rgb_env);
	if (conic_frac_env) conic_frac = atoi(conic_frac_env);
	if (product_frac_env) product_frac = atoi(product_frac_env);
	if (product_bits_env) product_bits = atoi(product_bits_env);
	if (d1_frac_env) d1_frac = atoi(d1_frac_env);
	if (sigma_frac_env) sigma_frac = atoi(sigma_frac_env);
	if (local_frac_env) local_frac = atoi(local_frac_env);
	if (state_bits_env) state_bits = atoi(state_bits_env);
	if (exp_lut_env) exp_lut10 = atoi(exp_lut_env) != 0;
	if (safe_sigma_env) safe_sigma_mode = atoi(safe_sigma_env) != 0;
	if (opacity_bits < 4 || opacity_bits > 6 || rgb_bits < 8 || rgb_bits > 10)
	{
		fprintf(stderr, "gs_precision_lab expects opacity 4..6 and RGB 8..10 bits\n");
		return;
	}
	if (conic_frac != 20 || (product_bits != 24 && product_bits != 28 && product_bits != 32) || product_frac < 2 || product_frac > 20 || d1_frac < 2 || d1_frac > 20 ||
		sigma_frac < 2 || sigma_frac > 20 || local_frac < 0 || local_frac > 10 ||
		(state_bits != 20 && state_bits != 24))
	{
		fprintf(stderr, "gs_precision_lab invalid conic/recurrence fractional bits\n");
		return;
	}

	// SARD workload counters: only count when SARD_STATS=1 (zero overhead
	// otherwise, keeps timing runs uncontaminated). counters is a fresh
	// zero-initialized buffer per call from the host side.
	const char* stats_env = getenv("SARD_STATS");
	const char* range_env = getenv("SARD_RANGE_STATS");
	const bool range_stats = range_env != nullptr && range_env[0] == '1';
	unsigned long long* ctr = ((stats_env != nullptr && stats_env[0] == '1') || range_stats) ? counters : nullptr;

	if (gsrp_mode)
	{
		if (gsrp_range_stats)
			collectGsrpInputRanges<NUM_CHANNELS> << <grid, block >> > (
				ranges, point_list, W, means2D, colors, conic_opacity, counters);
		renderCUDA_GSRP_NAIVE<NUM_CHANNELS> << <grid, block >> > (
			ranges, point_list, W, H, means2D, colors, conic_opacity,
			final_T, n_contrib, bg_color, out_color, counters, gsrp_cfg, gsrp_range_stats);
	}
	else if (precision_quant)
	{
		renderCUDA_SARD_PRECISION<NUM_CHANNELS> << <grid, block >> > (
			ranges, point_list, W, H, means2D, colors, conic_opacity,
			final_T, n_contrib, bg_color, out_color, ctr, opacity_bits, rgb_bits, conic_frac, product_bits, product_frac, d1_frac, sigma_frac, local_frac, state_bits, exp_lut10, range_stats, safe_sigma_mode);
	}
	else if (sard_mode == 2)
	{
		renderCUDA_SARD<NUM_CHANNELS, false> << <grid, block >> > (
			ranges, point_list, W, H, means2D, colors, conic_opacity,
			final_T, n_contrib, bg_color, out_color, ctr, t_lane_trace_entries);
	}
	else if (sard_mode == 1)
	{
		renderCUDA_SARD<NUM_CHANNELS, true> << <grid, block >> > (
			ranges, point_list, W, H, means2D, colors, conic_opacity,
			final_T, n_contrib, bg_color, out_color, ctr, t_lane_trace_entries);
	}
	else if (sard_mode == 3)
	{
		renderCUDA_SARD_WG<NUM_CHANNELS> << <grid, block >> > (
			ranges, point_list, W, H, means2D, colors, conic_opacity,
			final_T, n_contrib, bg_color, out_color, ctr);
	}
	else
	{
		renderCUDA<NUM_CHANNELS> << <grid, block >> > (
			ranges, point_list, W, H, means2D, colors, conic_opacity,
			final_T, n_contrib, bg_color, out_color, ctr);
	}

	if (ctr && !precision_quant)
	{
		countNonemptySARD<<<grid, block>>>(
			ranges, point_list, W, H, means2D, conic_opacity, ctr, tile_counters, fifo_trace_entries, t_lane_trace_entries);
	}

	const char* v11_env = getenv("SARD_V11_SUMMARY");
	if (v11_env != nullptr && v11_env[0] == '1')
	{
		summarizeTileTStop<<<grid, block>>>(
			ranges, point_list, W, H, means2D, conic_opacity, tile_t_summary);
	}
}

void FORWARD::preprocess(int P, int D, int M,
	const float* means3D,
	const glm::vec3* scales,
	const float scale_modifier,
	const glm::vec4* rotations,
	const float* opacities,
	const float* shs,
	bool* clamped,
	const float* cov3D_precomp,
	const float* colors_precomp,
	const float* viewmatrix,
	const float* projmatrix,
	const glm::vec3* cam_pos,
	const int W, int H,
	const float focal_x, float focal_y,
	const float tan_fovx, float tan_fovy,
	int* radii,
	float2* means2D,
	float* depths,
	float* cov3Ds,
	float* rgb,
	float4* conic_opacity,
	const dim3 grid,
	uint32_t* tiles_touched,
	bool prefiltered,
	bool baseline_aabb)
{
	preprocessCUDA<NUM_CHANNELS> << <(P + 255) / 256, 256 >> > (
		P, D, M,
		means3D,
		scales,
		scale_modifier,
		rotations,
		opacities,
		shs,
		clamped,
		cov3D_precomp,
		colors_precomp,
		viewmatrix, 
		projmatrix,
		cam_pos,
		W, H,
		tan_fovx, tan_fovy,
		focal_x, focal_y,
		radii,
		means2D,
		depths,
		cov3Ds,
		rgb,
		conic_opacity,
		grid,
		tiles_touched,
		prefiltered,
		baseline_aabb
		);
}
