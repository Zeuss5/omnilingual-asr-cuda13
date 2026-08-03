// Bandwidth-optimal bf16 GEMV for autoregressive decode: y[M,N] = x[M,K] @ W[N,K]^T
// with M tiny (1..8). The whole thing is a streaming read of W, so the only
// job is to issue perfectly coalesced 16-byte loads and keep enough warps in
// flight to saturate HBM.
//
// One warp owns one output row n: lane l walks W[n, :] with a stride of
// 32*8 bf16 values, loading 8 at a time (one 16-byte transaction per lane, so
// a warp reads a contiguous 512 B line per step). x[m, :] is reused by every
// warp and stays resident in L1/L2.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#define WARP 32

template <int MMAX>
__global__ __launch_bounds__(256) void gemv_bf16_kernel(
    const __nv_bfloat16 *__restrict__ X,   // [M, K]
    const __nv_bfloat16 *__restrict__ W,   // [N, K]
    __nv_bfloat16 *__restrict__ Y,         // [M, N]
    int M, int N, int K, int ldx, int ldw, int ldy)
{
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / WARP;
    const int lane = threadIdx.x % WARP;
    if (warp_id >= N) return;

    const __nv_bfloat16 *w_row = W + (size_t)warp_id * ldw;

    float acc[MMAX];
#pragma unroll
    for (int m = 0; m < MMAX; ++m) acc[m] = 0.f;

    // 8 bf16 = 16 B per lane per step
    const int vecK = K / 8;
    const float4 *w4 = reinterpret_cast<const float4 *>(w_row);

    for (int i = lane; i < vecK; i += WARP) {
        float4 wv = w4[i];
        const __nv_bfloat16 *wb = reinterpret_cast<const __nv_bfloat16 *>(&wv);
#pragma unroll
        for (int m = 0; m < MMAX; ++m) {
            if (m < M) {
                const float4 *x4 = reinterpret_cast<const float4 *>(X + (size_t)m * ldx);
                float4 xv = x4[i];
                const __nv_bfloat16 *xb = reinterpret_cast<const __nv_bfloat16 *>(&xv);
#pragma unroll
                for (int j = 0; j < 8; ++j)
                    acc[m] = fmaf(__bfloat162float(wb[j]), __bfloat162float(xb[j]), acc[m]);
            }
        }
    }
    // tail (K not a multiple of 8)
    for (int i = vecK * 8 + lane; i < K; i += WARP) {
        float wv = __bfloat162float(w_row[i]);
#pragma unroll
        for (int m = 0; m < MMAX; ++m)
            if (m < M) acc[m] = fmaf(wv, __bfloat162float(X[(size_t)m * ldx + i]), acc[m]);
    }

#pragma unroll
    for (int m = 0; m < MMAX; ++m) {
        if (m >= M) continue;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1)
            acc[m] += __shfl_down_sync(0xffffffff, acc[m], off);
        if (lane == 0) Y[(size_t)m * ldy + warp_id] = __float2bfloat16(acc[m]);
    }
}

torch::Tensor gemv_bf16(torch::Tensor x, torch::Tensor w)
{
    TORCH_CHECK(x.is_cuda() && w.is_cuda(), "cuda tensors required");
    TORCH_CHECK(x.scalar_type() == torch::kBFloat16, "bf16 only");
    const int M = x.size(0), K = x.size(1), N = w.size(0);
    TORCH_CHECK(w.size(1) == K, "shape mismatch");

    auto y = torch::empty({M, N}, x.options());
    const int threads = 256;
    const int warps_per_block = threads / WARP;
    const int blocks = (N + warps_per_block - 1) / warps_per_block;
    auto stream = at::cuda::getCurrentCUDAStream();

#define LAUNCH(MM)                                                                    \
    gemv_bf16_kernel<MM><<<blocks, threads, 0, stream>>>(                             \
        reinterpret_cast<const __nv_bfloat16 *>(x.data_ptr<at::BFloat16>()),          \
        reinterpret_cast<const __nv_bfloat16 *>(w.data_ptr<at::BFloat16>()),          \
        reinterpret_cast<__nv_bfloat16 *>(y.data_ptr<at::BFloat16>()),                \
        M, N, K, x.stride(0), w.stride(0), y.stride(0));

    if (M == 1)      { LAUNCH(1) }
    else if (M <= 2) { LAUNCH(2) }
    else if (M <= 4) { LAUNCH(4) }
    else if (M <= 8) { LAUNCH(8) }
    else             { TORCH_CHECK(false, "M > 8 not supported"); }
#undef LAUNCH
    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) { m.def("gemv_bf16", &gemv_bf16, "bf16 GEMV"); }
