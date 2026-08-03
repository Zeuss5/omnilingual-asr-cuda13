"""Is a persistent megakernel worth it? Compare kernel-boundary cost inside a
CUDA graph against cooperative grid.sync() cost at the same grid size."""
import os

import torch
from torch.utils.cpp_extension import load

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13.0")
here = os.path.dirname(os.path.abspath(__file__))
ext = load(
    name="mega_probe",
    sources=[os.path.join(here, "probe.cu")],
    extra_cuda_cflags=["-O3", "-gencode=arch=compute_120,code=sm_120"],
    extra_ldflags=["-lcudadevrt"],
    verbose=False,
)

buf = torch.zeros(4, device="cuda")
THREADS = 256
print(f"max co-resident blocks @ {THREADS} threads: {ext.max_coop_blocks(THREADS)}")


def graph_time(fn, it=30):
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        fn()
    torch.cuda.current_stream().wait_stream(s)
    torch.cuda.synchronize()
    g = torch.cuda.CUDAGraph()
    with torch.cuda.graph(g):
        fn()
    torch.cuda.synchronize()
    for _ in range(5):
        g.replay()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(it):
        g.replay()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / it


def stream_time(fn, it=30):
    for _ in range(5):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(it):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / it


for blocks in (188, 512, 1024):
    # A: per-kernel-boundary cost inside a graph (slope over kernel count)
    t100 = graph_time(lambda: ext.launch_tiny(buf, 100, blocks, THREADS))
    t300 = graph_time(lambda: ext.launch_tiny(buf, 300, blocks, THREADS))
    per_kernel = (t300 - t100) / 200 * 1e3  # us

    # B: per-grid.sync() cost (slope over sync count)
    s100 = stream_time(lambda: ext.launch_coop(buf, 100, blocks, THREADS))
    s300 = stream_time(lambda: ext.launch_coop(buf, 300, blocks, THREADS))
    per_sync = (s300 - s100) / 200 * 1e3  # us

    print(f"\nblocks={blocks} ({blocks*THREADS//32} warps)")
    print(f"  kernel boundary in graph : {per_kernel:6.2f} us")
    print(f"  cooperative grid.sync()  : {per_sync:6.2f} us")
    print(f"  -> 121 boundaries = {121*per_kernel/1e3:.3f} ms   "
          f"84 syncs = {84*per_sync/1e3:.3f} ms   "
          f"potential saving {(121*per_kernel - 84*per_sync)/1e3:+.3f} ms")
