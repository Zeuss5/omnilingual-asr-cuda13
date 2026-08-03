// Persistent cooperative megakernel for one omniASR Llama decode step.
//
// The entire step — 12 layers of {RMSNorm, packed QKV + RoPE + KV write,
// single-query attention, o_proj, RMSNorm, gate/up + SwiGLU, down_proj} plus
// the final norm — runs inside ONE cooperative kernel. Kernel boundaries are
// replaced by cg::grid_group::sync().
//
// Layout notes:
//  * one warp owns one output row of a GEMV (or a RoPE *pair* of rows in QKV,
//    so the rotation stays inside a single warp),
//  * o_proj / down_proj fold the residual add into their epilogue: each warp
//    owns a distinct element of x, so `x[n] += acc` is race-free,
//  * RMSNorm needs a full-row reduction, so block 0 does it alone while the
//    rest of the grid waits at the barrier — x is only 8 KB.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

#define WARP 32
#define ATT_SPLITS 16

struct LayerPtrs {
    const __nv_bfloat16 *w_qkv;      // [3D, D]
    const __nv_bfloat16 *w_o;        // [D, D]
    const __nv_bfloat16 *w_gate_up;  // [2I, D]
    const __nv_bfloat16 *w_down;     // [D, I]
    const __nv_bfloat16 *attn_norm;  // [D]
    const __nv_bfloat16 *ffn_norm;   // [D]
    __nv_bfloat16 *k_cache;          // [H, MAX, DH]
    __nv_bfloat16 *v_cache;          // [H, MAX, DH]
};

__device__ __forceinline__ float warp_dot(
    const __nv_bfloat16 *__restrict__ w, const __nv_bfloat16 *__restrict__ v,
    int K, int lane)
{
    float acc = 0.f;
    const int vecK = K >> 3;
    const float4 *w4 = reinterpret_cast<const float4 *>(w);
    const float4 *v4 = reinterpret_cast<const float4 *>(v);
    for (int i = lane; i < vecK; i += WARP) {
        float4 wv = w4[i], xv = v4[i];
        const __nv_bfloat16 *wb = reinterpret_cast<const __nv_bfloat16 *>(&wv);
        const __nv_bfloat16 *xb = reinterpret_cast<const __nv_bfloat16 *>(&xv);
#pragma unroll
        for (int j = 0; j < 8; ++j)
            acc = fmaf(__bfloat162float(wb[j]), __bfloat162float(xb[j]), acc);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) acc += __shfl_down_sync(0xffffffff, acc, off);
    return acc;   // lane 0
}

// block 0 only: y = rmsnorm(x) * wn
__device__ void block0_rmsnorm(
    const __nv_bfloat16 *__restrict__ x, const __nv_bfloat16 *__restrict__ wn,
    __nv_bfloat16 *__restrict__ y, int D, float eps)
{
    __shared__ float red[32];
    float p = 0.f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = __bfloat162float(x[i]);
        p += v * v;
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) p += __shfl_down_sync(0xffffffff, p, off);
    int warp = threadIdx.x / WARP, lane = threadIdx.x % WARP;
    if (lane == 0) red[warp] = p;
    __syncthreads();
    if (threadIdx.x == 0) {
        float s = 0.f;
        for (int i = 0; i < blockDim.x / WARP; ++i) s += red[i];
        red[31] = rsqrtf(s / D + eps);
    }
    __syncthreads();
    const float rstd = red[31];
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        y[i] = __float2bfloat16(__bfloat162float(x[i]) * rstd * __bfloat162float(wn[i]));
}

extern "C" __global__ void decode_step_mega(
    const LayerPtrs *__restrict__ layers, int L,
    __nv_bfloat16 *__restrict__ x,        // [D] residual stream, in/out
    __nv_bfloat16 *__restrict__ nrm,      // [D]
    __nv_bfloat16 *__restrict__ qkv,      // [3D]
    __nv_bfloat16 *__restrict__ attn,     // [D]
    __nv_bfloat16 *__restrict__ sw,       // [I]
    float *__restrict__ part_o,           // [H, SPLITS, DH]
    float *__restrict__ part_m,           // [H, SPLITS]
    float *__restrict__ part_l,           // [H, SPLITS]
    __nv_bfloat16 *__restrict__ out,      // [D]
    const __nv_bfloat16 *__restrict__ final_norm,
    const int *__restrict__ pos_ptr, const int *__restrict__ len_ptr,
    int D, int I, int H, int DH, int MAXLEN, float eps, float log_theta, float scale)
{
    cg::grid_group grid = cg::this_grid();

    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int gw = tid / WARP;                       // global warp id
    const int lane = threadIdx.x % WARP;
    const int nwarps = (gridDim.x * blockDim.x) / WARP;

    const int pos = *pos_ptr;
    const int klen = *len_ptr;

    for (int l = 0; l < L; ++l) {
        const LayerPtrs lp = layers[l];

        // ---- 1. attention RMSNorm ----
        if (blockIdx.x == 0) block0_rmsnorm(x, lp.attn_norm, nrm, D, eps);
        grid.sync();

        // ---- 2. QKV GEMV + RoPE + KV cache write ----
        // A warp owns the row pair (2p, 2p+1) so a RoPE rotation is warp-local.
        const int npairs = (3 * D) >> 1;
        for (int p = gw; p < npairs; p += nwarps) {
            const int r0 = p << 1;
            float a0 = warp_dot(lp.w_qkv + (size_t)r0 * D, nrm, D, lane);
            float a1 = warp_dot(lp.w_qkv + (size_t)(r0 + 1) * D, nrm, D, lane);
            if (lane != 0) continue;

            if (r0 < 2 * D) {                          // q or k -> rotate
                const int loc = (r0 < D) ? r0 : r0 - D;
                const int h = loc / DH, j = loc % DH;
                const float ang = pos * __expf(-((float)j / DH) * log_theta);
                float c, s;
                __sincosf(ang, &s, &c);
                const float o0 = a0 * c - a1 * s;
                const float o1 = a0 * s + a1 * c;
                if (r0 < D) {
                    qkv[r0] = __float2bfloat16(o0);
                    qkv[r0 + 1] = __float2bfloat16(o1);
                } else {
                    __nv_bfloat16 *kc = lp.k_cache + ((size_t)h * MAXLEN + pos) * DH + j;
                    kc[0] = __float2bfloat16(o0);
                    kc[1] = __float2bfloat16(o1);
                }
            } else {                                   // v -> straight to cache
                const int loc = r0 - 2 * D;
                const int h = loc / DH, j = loc % DH;
                __nv_bfloat16 *vc = lp.v_cache + ((size_t)h * MAXLEN + pos) * DH + j;
                vc[0] = __float2bfloat16(a0);
                vc[1] = __float2bfloat16(a1);
            }
        }
        grid.sync();

        // ---- 3a. attention, split over the key axis ----
        // one block per (head, split); each does an online softmax over its range
        const int njobs = H * ATT_SPLITS;
        for (int job = blockIdx.x; job < njobs; job += gridDim.x) {
            const int h = job / ATT_SPLITS, sp = job % ATT_SPLITS;
            const int per = (klen + ATT_SPLITS - 1) / ATT_SPLITS;
            const int lo = sp * per, hi = min(lo + per, klen);

            __shared__ float sm_m, sm_l;
            extern __shared__ float sacc[];            // [DH]
            for (int d = threadIdx.x; d < DH; d += blockDim.x) sacc[d] = 0.f;
            if (threadIdx.x == 0) { sm_m = -INFINITY; sm_l = 0.f; }
            __syncthreads();

            for (int t = lo; t < hi; ++t) {
                const __nv_bfloat16 *kk = lp.k_cache + ((size_t)h * MAXLEN + t) * DH;
                // dot(q_h, k_t) across the block
                float p = 0.f;
                for (int d = threadIdx.x; d < DH; d += blockDim.x)
                    p += __bfloat162float(qkv[h * DH + d]) * __bfloat162float(kk[d]);
#pragma unroll
                for (int off = 16; off > 0; off >>= 1) p += __shfl_down_sync(0xffffffff, p, off);
                __shared__ float red[32];
                if (threadIdx.x % WARP == 0) red[threadIdx.x / WARP] = p;
                __syncthreads();
                __shared__ float sc;
                if (threadIdx.x == 0) {
                    float s = 0.f;
                    for (int i = 0; i < blockDim.x / WARP; ++i) s += red[i];
                    sc = s * scale;
                }
                __syncthreads();

                // online softmax update
                __shared__ float alpha, pe;
                if (threadIdx.x == 0) {
                    const float mnew = fmaxf(sm_m, sc);
                    alpha = __expf(sm_m - mnew);
                    pe = __expf(sc - mnew);
                    sm_l = sm_l * alpha + pe;
                    sm_m = mnew;
                }
                __syncthreads();

                const __nv_bfloat16 *vv = lp.v_cache + ((size_t)h * MAXLEN + t) * DH;
                for (int d = threadIdx.x; d < DH; d += blockDim.x)
                    sacc[d] = sacc[d] * alpha + pe * __bfloat162float(vv[d]);
                __syncthreads();
            }

            float *po = part_o + ((size_t)h * ATT_SPLITS + sp) * DH;
            for (int d = threadIdx.x; d < DH; d += blockDim.x) po[d] = sacc[d];
            if (threadIdx.x == 0) {
                part_m[h * ATT_SPLITS + sp] = (hi > lo) ? sm_m : -INFINITY;
                part_l[h * ATT_SPLITS + sp] = (hi > lo) ? sm_l : 0.f;
            }
            __syncthreads();
        }
        grid.sync();

        // ---- 3b. combine the splits ----
        for (int idx = gw; idx < H * DH; idx += nwarps) {
            if (lane != 0) continue;
            const int h = idx / DH, d = idx % DH;
            float m = -INFINITY;
            for (int s = 0; s < ATT_SPLITS; ++s)
                if (part_l[h * ATT_SPLITS + s] > 0.f) m = fmaxf(m, part_m[h * ATT_SPLITS + s]);
            float acc = 0.f, tot = 0.f;
            for (int s = 0; s < ATT_SPLITS; ++s) {
                const float lv = part_l[h * ATT_SPLITS + s];
                if (lv <= 0.f) continue;
                const float w = __expf(part_m[h * ATT_SPLITS + s] - m);
                acc += part_o[((size_t)h * ATT_SPLITS + s) * DH + d] * w;
                tot += lv * w;
            }
            attn[idx] = __float2bfloat16(acc / tot);
        }
        grid.sync();

        // ---- 4. o_proj, residual add folded into the epilogue ----
        for (int n = gw; n < D; n += nwarps) {
            float a = warp_dot(lp.w_o + (size_t)n * D, attn, D, lane);
            if (lane == 0) x[n] = __float2bfloat16(__bfloat162float(x[n]) + a);
        }
        grid.sync();

        // ---- 5. FFN RMSNorm ----
        if (blockIdx.x == 0) block0_rmsnorm(x, lp.ffn_norm, nrm, D, eps);
        grid.sync();

        // ---- 6. gate/up GEMV + SwiGLU ----
        for (int n = gw; n < I; n += nwarps) {
            float g = warp_dot(lp.w_gate_up + (size_t)n * D, nrm, D, lane);
            float u = warp_dot(lp.w_gate_up + (size_t)(I + n) * D, nrm, D, lane);
            if (lane == 0) sw[n] = __float2bfloat16((g / (1.f + __expf(-g))) * u);
        }
        grid.sync();

        // ---- 7. down_proj + residual ----
        for (int n = gw; n < D; n += nwarps) {
            float a = warp_dot(lp.w_down + (size_t)n * I, sw, I, lane);
            if (lane == 0) x[n] = __float2bfloat16(__bfloat162float(x[n]) + a);
        }
        grid.sync();
    }

    if (blockIdx.x == 0) block0_rmsnorm(x, final_norm, out, D, eps);
}


// ---- diagnostic: the GEMV stages only, still cooperative + grid.sync().
// Isolates "does a persistent cooperative grid hurt the GEMVs themselves?"
// from any inefficiency in the attention stage.
extern "C" __global__ void gemv_only_mega(
    const LayerPtrs *__restrict__ layers, int L,
    __nv_bfloat16 *__restrict__ x, __nv_bfloat16 *__restrict__ nrm,
    __nv_bfloat16 *__restrict__ qkv, __nv_bfloat16 *__restrict__ attn,
    __nv_bfloat16 *__restrict__ sw, int D, int I)
{
    cg::grid_group grid = cg::this_grid();
    const int tid = blockIdx.x * blockDim.x + threadIdx.x;
    const int gw = tid / WARP, lane = threadIdx.x % WARP;
    const int nwarps = (gridDim.x * blockDim.x) / WARP;

    for (int l = 0; l < L; ++l) {
        const LayerPtrs lp = layers[l];
        for (int n = gw; n < 3 * D; n += nwarps) {
            float a = warp_dot(lp.w_qkv + (size_t)n * D, nrm, D, lane);
            if (lane == 0) qkv[n] = __float2bfloat16(a);
        }
        grid.sync();
        for (int n = gw; n < D; n += nwarps) {
            float a = warp_dot(lp.w_o + (size_t)n * D, attn, D, lane);
            if (lane == 0) x[n] = __float2bfloat16(a);
        }
        grid.sync();
        for (int n = gw; n < I; n += nwarps) {
            float g = warp_dot(lp.w_gate_up + (size_t)n * D, nrm, D, lane);
            float u = warp_dot(lp.w_gate_up + (size_t)(I + n) * D, nrm, D, lane);
            if (lane == 0) sw[n] = __float2bfloat16(g + u);
        }
        grid.sync();
        for (int n = gw; n < D; n += nwarps) {
            float a = warp_dot(lp.w_down + (size_t)n * I, sw, I, lane);
            if (lane == 0) x[n] = __float2bfloat16(a);
        }
        grid.sync();
    }
}

int64_t gemv_only_max_blocks(int64_t threads)
{
    int n = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, (const void *)gemv_only_mega, (int)threads, 0);
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    return (int64_t)n * sms;
}

void gemv_only_launch(torch::Tensor layer_table, int64_t L, torch::Tensor x,
                      torch::Tensor nrm, torch::Tensor qkv, torch::Tensor attn,
                      torch::Tensor sw, int64_t D, int64_t I,
                      int64_t blocks, int64_t threads)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    const LayerPtrs *lt = reinterpret_cast<const LayerPtrs *>(layer_table.data_ptr());
    int Li = (int)L, Di = (int)D, Ii = (int)I;
    auto xp = reinterpret_cast<__nv_bfloat16 *>(x.data_ptr<at::BFloat16>());
    auto np = reinterpret_cast<__nv_bfloat16 *>(nrm.data_ptr<at::BFloat16>());
    auto qp = reinterpret_cast<__nv_bfloat16 *>(qkv.data_ptr<at::BFloat16>());
    auto ap = reinterpret_cast<__nv_bfloat16 *>(attn.data_ptr<at::BFloat16>());
    auto sp = reinterpret_cast<__nv_bfloat16 *>(sw.data_ptr<at::BFloat16>());
    void *args[] = {&lt, &Li, &xp, &np, &qp, &ap, &sp, &Di, &Ii};
    cudaError_t e = cudaLaunchCooperativeKernel((void *)gemv_only_mega,
        dim3((unsigned)blocks), dim3((unsigned)threads), args, 0, stream);
    TORCH_CHECK(e == cudaSuccess, "coop launch failed: ", cudaGetErrorString(e));
}

// ---------------------------------------------------------------- host side

static int g_max_blocks = 0;

int64_t mega_max_blocks(int64_t threads)
{
    int n = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, (const void *)decode_step_mega,
                                                  (int)threads, 512 * sizeof(float));
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    g_max_blocks = n * sms;
    return g_max_blocks;
}

void mega_launch(
    torch::Tensor layer_table, int64_t L,
    torch::Tensor x, torch::Tensor nrm, torch::Tensor qkv, torch::Tensor attn,
    torch::Tensor sw, torch::Tensor part_o, torch::Tensor part_m, torch::Tensor part_l,
    torch::Tensor out, torch::Tensor final_norm,
    torch::Tensor pos, torch::Tensor len,
    int64_t D, int64_t I, int64_t H, int64_t DH, int64_t MAXLEN,
    double eps, double theta, int64_t blocks, int64_t threads)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    const size_t shmem = (size_t)DH * sizeof(float);

    const LayerPtrs *lt = reinterpret_cast<const LayerPtrs *>(layer_table.data_ptr());
    int Li = (int)L, Di = (int)D, Ii = (int)I, Hi = (int)H, DHi = (int)DH, Mi = (int)MAXLEN;
    float epsf = (float)eps, logth = logf((float)theta), scale = 1.0f / sqrtf((float)DH);

    auto xp = reinterpret_cast<__nv_bfloat16 *>(x.data_ptr<at::BFloat16>());
    auto np = reinterpret_cast<__nv_bfloat16 *>(nrm.data_ptr<at::BFloat16>());
    auto qp = reinterpret_cast<__nv_bfloat16 *>(qkv.data_ptr<at::BFloat16>());
    auto ap = reinterpret_cast<__nv_bfloat16 *>(attn.data_ptr<at::BFloat16>());
    auto sp = reinterpret_cast<__nv_bfloat16 *>(sw.data_ptr<at::BFloat16>());
    auto pop = part_o.data_ptr<float>();
    auto pmp = part_m.data_ptr<float>();
    auto plp = part_l.data_ptr<float>();
    auto op = reinterpret_cast<__nv_bfloat16 *>(out.data_ptr<at::BFloat16>());
    auto fnp = reinterpret_cast<const __nv_bfloat16 *>(final_norm.data_ptr<at::BFloat16>());
    auto pp = pos.data_ptr<int>();
    auto lnp = len.data_ptr<int>();

    void *args[] = {&lt, &Li, &xp, &np, &qp, &ap, &sp, &pop, &pmp, &plp, &op, &fnp,
                    &pp, &lnp, &Di, &Ii, &Hi, &DHi, &Mi, &epsf, &logth, &scale};

    cudaError_t e = cudaLaunchCooperativeKernel(
        (void *)decode_step_mega, dim3((unsigned)blocks), dim3((unsigned)threads),
        args, shmem, stream);
    TORCH_CHECK(e == cudaSuccess, "cooperative launch failed: ", cudaGetErrorString(e));
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("mega_launch", &mega_launch);
    m.def("mega_max_blocks", &mega_max_blocks);
    m.def("gemv_only_launch", &gemv_only_launch);
    m.def("gemv_only_max_blocks", &gemv_only_max_blocks);
    m.def("layer_struct_bytes", [] { return (int64_t)sizeof(LayerPtrs); });
}
