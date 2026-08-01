// Independent naive per-pixel precision kernel for gs_raster_precision.
// Included by forward.cu after gsp_qfix/gsp_range_record definitions.
#pragma once

__device__ __forceinline__ float gsrp_h(float x) { return __half2float(__float2half_rn(x)); }
__device__ __forceinline__ float gsrp_hadd(float a, float b) { return __half2float(__hadd(__float2half_rn(a), __float2half_rn(b))); }
__device__ __forceinline__ float gsrp_hsub(float a, float b) { return __half2float(__hsub(__float2half_rn(a), __float2half_rn(b))); }
__device__ __forceinline__ float gsrp_hmul(float a, float b) { return __half2float(__hmul(__float2half_rn(a), __float2half_rn(b))); }
__device__ __forceinline__ float gsrp_hexp(float a) { return gsrp_h(expf(gsrp_h(a))); }

struct GsrpFormats {
	int mode;          // 1..4=historical, 5=float baseline, 6=input Q+float, 7=staged camera-ready pipeline
	unsigned int active_mask;
	int bits[7];       // mean_x, mean_y, A, B, C, opacity, RGB
	int frac[7];
	int power_bits;
	int power_frac;
	int exp_addr_bits;
	int exp_out_bits;
	int quad_fixed;
	int quad_geom_bits;
	int quad_geom_frac;
	int quad_term_bits;
	int quad_term_frac;
	int quad_sum_bits;
	int quad_sum_frac;
	int exp_lut_enable;
	int blend_mode;
	unsigned int blend_mask; // alpha=1<<0, T=1<<1, visibility=1<<2, accumulator=1<<3
	int raw_alpha_for_tstop;
	int blend_bits[4];
	int blend_frac[4];
};

template <uint32_t CHANNELS>
__global__ void collectGsrpInputRanges(
	const uint2* __restrict__ ranges, const uint32_t* __restrict__ point_list,
	int W, const float2* __restrict__ points_xy_image,
	const float* __restrict__ features, const float4* __restrict__ conic_opacity,
	unsigned long long* __restrict__ counters)
{
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint32_t tile = blockIdx.y * horizontal_blocks + blockIdx.x;
	const uint2 range = ranges[tile];
	const float tile_origin_x = (float)(blockIdx.x * BLOCK_X);
	const float tile_origin_y = (float)(blockIdx.y * BLOCK_Y);
	for (uint32_t pos = range.x + threadIdx.y * blockDim.x + threadIdx.x;
		 pos < range.y; pos += blockDim.x * blockDim.y) {
		const uint32_t id = point_list[pos];
		const float2 xy = points_xy_image[id];
		const float4 co = conic_opacity[id];
		gsp_range_record(counters, true, 0, xy.x - tile_origin_x);
		gsp_range_record(counters, true, 1, xy.y - tile_origin_y);
		gsp_range_record(counters, true, 2, co.x);
		gsp_range_record(counters, true, 3, co.y);
		gsp_range_record(counters, true, 4, co.z);
		gsp_range_record(counters, true, 5, co.w);
		for (int ch = 0; ch < CHANNELS; ++ch)
			gsp_range_record(counters, true, 6, features[id * CHANNELS + ch]);
	}
}

__device__ __forceinline__ float gsrp_input(float x, int signal, const GsrpFormats cfg, unsigned int sat[7])
{
	// Mode 5 is the corrected comparison path.  It keeps the original
	// renderCUDA arithmetic in float, so only explicitly requested blending
	// quantization is measured against the standard renderer.  Modes 1--4 are
	// historical diagnostics and intentionally retain their old FP16/input-Q
	// semantics.
	if (cfg.mode == 5)
		return x;
	if (cfg.active_mask & (1u << signal))
		return gsp_qfix(x, cfg.bits[signal], cfg.frac[signal], signal == 0 || signal == 1 || signal == 3, &sat[signal]);
	if (cfg.mode == 6)
		return x;
	if (cfg.mode == 7)
		return x;
	return gsrp_h(x);
}

__device__ __forceinline__ float gsrp_exp_lut(float power, const GsrpFormats cfg)
{
	const float lo = -5.5412945f;
	if (power > 0.0f || power <= lo) return 0.0f;
	const int size = 1 << cfg.exp_addr_bits;
	int address = __float2int_rn((power - lo) * ((float)(size - 1) / -lo));
	address = max(0, min(size - 1, address));
	const float sample_power = lo + ((float)address) * (-lo / (float)(size - 1));
	return gsp_qfix(expf(sample_power), cfg.exp_out_bits, cfg.exp_out_bits - 1, false, nullptr);
}

__device__ __forceinline__ float gsrp_blend_q(
	float value, int signal, const GsrpFormats cfg, unsigned int* sat = nullptr)
{
	return cfg.blend_mode && (cfg.blend_mask & (1u << signal))
		? gsp_qfix(value, cfg.blend_bits[signal], cfg.blend_frac[signal], false, sat) : value;
}

template <uint32_t CHANNELS>
__global__ void __launch_bounds__(BLOCK_X * BLOCK_Y)
renderCUDA_GSRP_NAIVE(
	const uint2* __restrict__ ranges, const uint32_t* __restrict__ point_list,
	int W, int H, const float2* __restrict__ points_xy_image,
	const float* __restrict__ features, const float4* __restrict__ conic_opacity,
	float* __restrict__ final_T, uint32_t* __restrict__ n_contrib,
	const float* __restrict__ bg_color, float* __restrict__ out_color,
	unsigned long long* __restrict__ counters, GsrpFormats cfg, bool range_stats)
{
	auto block = cg::this_thread_block();
	const uint32_t horizontal_blocks = (W + BLOCK_X - 1) / BLOCK_X;
	const uint2 pix_min = {block.group_index().x * BLOCK_X, block.group_index().y * BLOCK_Y};
	const uint2 pix = {pix_min.x + block.thread_index().x, pix_min.y + block.thread_index().y};
	const uint32_t pix_id = W * pix.y + pix.x;
	const bool inside = pix.x < W && pix.y < H;
	bool done = !inside;
	const uint2 range = ranges[block.group_index().y * horizontal_blocks + block.group_index().x];
	const int rounds = ((range.y - range.x + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int toDo = range.y - range.x;

	__shared__ int ids[BLOCK_SIZE];
	__shared__ float2 xy_shared[BLOCK_SIZE];
	__shared__ float4 co_shared[BLOCK_SIZE];
	// 0..6: input saturation; 7: geometry products; 8: weighted
	// terms; 9: protected sum; 10: final power writeback;
	// 11..14: alpha, T, visibility,
	// and RGB accumulator.
	unsigned int sat[15] = {0,0,0,0,0,0,0,0,0,0,0,0,0,0,0};
	float T = gsrp_blend_q(gsrp_h(1.0f), 1, cfg, &sat[12]);
	float C[CHANNELS] = {0};
	uint32_t contributor = 0, last_contributor = 0;
	unsigned int t_stop_count = 0;

	for (int i = 0; i < rounds; ++i, toDo -= BLOCK_SIZE)
	{
		if (__syncthreads_count(done) == BLOCK_SIZE) break;
		const int progress = i * BLOCK_SIZE + block.thread_rank();
		if (range.x + progress < range.y) {
			const int id = point_list[range.x + progress];
			ids[block.thread_rank()] = id;
			xy_shared[block.thread_rank()] = points_xy_image[id];
			co_shared[block.thread_rank()] = conic_opacity[id];
		}
		block.sync();
		for (int j = 0; !done && j < min(BLOCK_SIZE, toDo); ++j)
		{
			++contributor;
			const float2 xy0 = xy_shared[j];
			const float4 co0 = co_shared[j];
			// Localize before quantization.  Quantizing a global coordinate (which
			// can be thousands of pixels) loses fractional bits and amplifies the
			// conic cross term through dx*dy.
			const float mx = gsrp_input(xy0.x - (float)pix_min.x, 0, cfg, sat);
			const float my = gsrp_input(xy0.y - (float)pix_min.y, 1, cfg, sat);
			const float A = gsrp_input(co0.x, 2, cfg, sat);
			const float B = gsrp_input(co0.y, 3, cfg, sat);
			const float CC = gsrp_input(co0.z, 4, cfg, sat);
			const float opacity = gsrp_input(co0.w, 5, cfg, sat);

			float power;
			if (cfg.mode == 7 && cfg.quad_fixed) {
				const float dx = mx - (float)block.thread_index().x;
				const float dy = my - (float)block.thread_index().y;
				unsigned int* geom_sat = counters ? &sat[7] : nullptr;
				unsigned int* term_sat = counters ? &sat[8] : nullptr;
				unsigned int* sum_sat = counters ? &sat[9] : nullptr;
				unsigned int* power_sat = counters ? &sat[10] : nullptr;
				const float dx2 = gsp_qfix(dx * dx, cfg.quad_geom_bits, cfg.quad_geom_frac, false, geom_sat);
				const float dy2 = gsp_qfix(dy * dy, cfg.quad_geom_bits, cfg.quad_geom_frac, false, geom_sat);
				const float dxdy = gsp_qfix(dx * dy, cfg.quad_geom_bits, cfg.quad_geom_frac, true, geom_sat);
				const float term_a = gsp_qfix(A * dx2, cfg.quad_term_bits, cfg.quad_term_frac, true, term_sat);
				const float term_c = gsp_qfix(CC * dy2, cfg.quad_term_bits, cfg.quad_term_frac, true, term_sat);
				const float term_b = gsp_qfix(B * dxdy, cfg.quad_term_bits, cfg.quad_term_frac, true, term_sat);
				const float diag = gsp_qfix(term_a + term_c, cfg.quad_sum_bits, cfg.quad_sum_frac, true, sum_sat);
				const float half_diag = gsp_qfix(-0.5f * diag, cfg.quad_sum_bits, cfg.quad_sum_frac, true, sum_sat);
				const float raw_power = half_diag - term_b;
				// Hardware-safe terminal classification. Values outside the EXP
				// domain never need to be represented in the narrow power
				// register: positive power is invalid, while power at or below
				// -ln(255) is below the renderer contribution threshold.
				if (raw_power > 0.0f)
					power = raw_power;
				else if (raw_power <= -5.5412945f)
					power = -5.5412945f;
				else
					power = gsp_qfix(raw_power, cfg.power_bits, cfg.power_frac, true, power_sat);
			} else if (cfg.mode == 5 || cfg.mode == 6 || cfg.mode == 7) {
				const float dx = mx - (float)block.thread_index().x;
				const float dy = my - (float)block.thread_index().y;
				power = -0.5f * (A * dx * dx + CC * dy * dy) - B * dx * dy;
			} else if (cfg.mode <= 2) {
				const float dx = gsrp_hsub(mx, (float)block.thread_index().x);
				const float dy = gsrp_hsub(my, (float)block.thread_index().y);
				const float diag = gsrp_hadd(gsrp_hmul(A, gsrp_hmul(dx, dx)), gsrp_hmul(CC, gsrp_hmul(dy, dy)));
				power = gsrp_hsub(gsrp_hmul(-0.5f, diag), gsrp_hmul(B, gsrp_hmul(dx, dy)));
			} else {
				const float dx = mx - (float)block.thread_index().x;
				const float dy = my - (float)block.thread_index().y;
				const float raw = -0.5f * (A * dx * dx + CC * dy * dy) - B * dx * dy;
				power = gsp_qfix(raw, cfg.power_bits, cfg.power_frac, true, nullptr);
			}
			if (power > 0.0f) continue;
			const bool corrected_float = cfg.mode == 5 || cfg.mode == 6 || cfg.mode == 7;
			const float expv = (cfg.mode == 4 || (cfg.mode == 7 && cfg.exp_lut_enable)) ? gsrp_exp_lut(power, cfg) :
				(corrected_float ? expf(power) : gsrp_hexp(power));
			const float alpha_raw = corrected_float ? opacity * expv : gsrp_hmul(opacity, expv);
			const float alpha_raw_capped = min(corrected_float ? 0.99f : gsrp_h(0.99f), alpha_raw);
			// Corrected semantics: clamp the one-shot alpha first, then quantize
			// it if requested.  Alpha is not a persistent state.
			float alpha = gsrp_blend_q(alpha_raw_capped, 0, cfg, &sat[11]);
			if (alpha < 1.0f / 255.0f) continue;
			const float alpha_for_tstop = cfg.raw_alpha_for_tstop ? alpha_raw_capped : alpha;
			const float visibility_raw = corrected_float ? alpha * T : gsrp_hmul(alpha, T);
			const float visibility = gsrp_blend_q(visibility_raw, 2, cfg, &sat[13]);
			const float test_t_raw = corrected_float ? T * (1.0f - alpha_for_tstop) :
				gsrp_hmul(T, gsrp_hsub(1.0f, alpha_for_tstop));
			const float test_T = gsrp_blend_q(test_t_raw, 1, cfg, &sat[12]);
			// Keep the original renderer's termination order for exact bypass.
			if (test_T < 0.0001f) { done = true; ++t_stop_count; continue; }
			for (int ch = 0; ch < CHANNELS; ++ch) {
				const float color = gsrp_input(features[ids[j] * CHANNELS + ch], 6, cfg, sat);
				const float contribution = corrected_float ? color * visibility : gsrp_hmul(color, visibility);
				const float accumulated = corrected_float ? C[ch] + contribution : gsrp_hadd(C[ch], contribution);
				C[ch] = gsrp_blend_q(accumulated, 3, cfg, &sat[14]);
			}
			T = test_T;
			last_contributor = contributor;
		}
	}
	if (counters) {
		for (int k = 0; k < 15; ++k) atomicAdd(&counters[k], (unsigned long long)sat[k]);
		if (inside) {
			atomicAdd(&counters[110], (unsigned long long)contributor);
			atomicAdd(&counters[111], (unsigned long long)last_contributor);
			atomicAdd(&counters[112], (unsigned long long)t_stop_count);
		}
	}
	if (inside) {
		final_T[pix_id] = T;
		n_contrib[pix_id] = last_contributor;
		for (int ch = 0; ch < CHANNELS; ++ch) {
			const bool corrected_float = cfg.mode == 5 || cfg.mode == 6 || cfg.mode == 7;
			const float background = corrected_float ? T * bg_color[ch] : gsrp_hmul(T, bg_color[ch]);
			const float output = corrected_float ? C[ch] + background : gsrp_hadd(C[ch], background);
			out_color[ch * H * W + pix_id] = gsrp_blend_q(output, 3, cfg);
		}
	}
}
