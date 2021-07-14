#include "CUDAFLERP.h"

__global__ void CUDAFLERP_kernel(const cudaTextureObject_t d_img_tex, const float gxs, const float gys, float* __restrict const d_out, const int neww) {
	uint32_t x = (blockIdx.x << 9) + (threadIdx.x << 1);
	const uint32_t y = blockIdx.y;
	const float fy = (y + 0.5f)*gys - 0.5f;
	const float wt_y = fy - floor(fy);
	const float invwt_y = 1.0f - wt_y;
#pragma unroll
	for (int i = 0; i < 2; ++i, ++x) {
		const float fx = (x + 0.5f)*gxs - 0.5f;
		const float4 f = tex2Dgather<float4>(d_img_tex, fx + 0.5f, fy + 0.5f);
		const float wt_x = fx - floor(fx);
		const float invwt_x = 1.0f - wt_x;
		const float xa = invwt_x*f.w + wt_x*f.z;
		const float xb = invwt_x*f.x + wt_x*f.y;
		const float res = invwt_y*xa + wt_y*xb;
		if (x < neww) d_out[y*neww + x] = res;
	}
}

void CUDAFLERP(const cudaTextureObject_t d_img_tex, const int oldw, const int oldh, float* __restrict const d_out, const uint32_t neww, const uint32_t newh) {
	const float gxs = static_cast<float>(oldw) / static_cast<float>(neww);
	const float gys = static_cast<float>(oldh) / static_cast<float>(newh);
	CUDAFLERP_kernel<<<{((neww - 1) >> 9) + 1, newh}, 256>>>(d_img_tex, gxs, gys, d_out, neww);
	cudaDeviceSynchronize();
}