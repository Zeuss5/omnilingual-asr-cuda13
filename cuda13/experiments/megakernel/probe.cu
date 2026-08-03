// Decisive micro-experiment for a persistent megakernel:
//   A) what does a kernel boundary cost inside an already-captured CUDA graph?
//   B) what does a cooperative-groups grid.sync() cost at the same grid size?
//
// A megakernel replaces ~121 boundaries with ~84 grid syncs, so it only wins if
// (121 * boundary) - (84 * sync) is a meaningful fraction of the 1.91 ms step.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cooperative_groups.h>
#include <cuda_runtime.h>

namespace cg = cooperative_groups;

// Minimal kernel: touches memory so it cannot be optimized away, but does
// essentially no work — what remains is the boundary cost itself.
__global__ void tiny_kernel(float *__restrict__ p)
{
    if (threadIdx.x == 0 && blockIdx.x == 0) p[0] += 1.0f;
}

__global__ void coop_sync_kernel(float *__restrict__ p, int iters)
{
    cg::grid_group grid = cg::this_grid();
    for (int i = 0; i < iters; ++i) {
        if (threadIdx.x == 0 && blockIdx.x == 0) p[0] += 1.0f;
        grid.sync();
    }
}

void launch_tiny(torch::Tensor buf, int64_t n, int64_t blocks, int64_t threads)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    for (int64_t i = 0; i < n; ++i)
        tiny_kernel<<<blocks, threads, 0, stream>>>(buf.data_ptr<float>());
}

double launch_coop(torch::Tensor buf, int64_t iters, int64_t blocks, int64_t threads)
{
    auto stream = at::cuda::getCurrentCUDAStream();
    void *p = buf.data_ptr<float>();
    int it = (int)iters;
    void *args[] = {&p, &it};
    cudaError_t e = cudaLaunchCooperativeKernel(
        (void *)coop_sync_kernel, dim3(blocks), dim3(threads), args, 0, stream);
    TORCH_CHECK(e == cudaSuccess, "cooperative launch failed: ", cudaGetErrorString(e));
    return 0.0;
}

int64_t max_coop_blocks(int64_t threads)
{
    int n = 0;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&n, (void *)coop_sync_kernel, threads, 0);
    int sms = 0;
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, 0);
    return (int64_t)n * sms;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)
{
    m.def("launch_tiny", &launch_tiny);
    m.def("launch_coop", &launch_coop);
    m.def("max_coop_blocks", &max_coop_blocks);
}
