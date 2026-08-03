"""Measure achievable HBM bandwidth on this GPU and derive the speed-of-light
latency for one autoregressive decode step of the omniASR Llama decoder.

    CUDA_VISIBLE_DEVICES=3 .venv/bin/python cuda13/bench/roofline.py [card]
"""

from __future__ import annotations

import sys

import torch
import torch.nn as nn

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline


def timed(fn, iters=50, warmup=10):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    start, end = torch.cuda.Event(True), torch.cuda.Event(True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


print(f"device: {torch.cuda.get_device_name(0)}")
props = torch.cuda.get_device_properties(0)
print(f"  SMs={props.multi_processor_count}  mem={props.total_memory/1e9:.0f} GB  L2={props.L2_cache_size/1e6:.0f} MB")

# --- peak bandwidth: streaming copy far larger than L2 ---
n = 1 << 30  # 1G elements = 2 GiB bf16
a = torch.empty(n, dtype=torch.bfloat16, device="cuda").normal_()
b = torch.empty(n, dtype=torch.bfloat16, device="cuda")
ms = timed(lambda: b.copy_(a), iters=20)
bw_copy = 2 * a.numel() * a.element_size() / (ms * 1e-3) / 1e12
print(f"  2GiB d2d copy : {ms:.3f} ms -> {bw_copy:.2f} TB/s (read+write)")

# pure-read: reduction over the same buffer
ms = timed(lambda: a.sum(), iters=20)
bw_read = a.numel() * a.element_size() / (ms * 1e-3) / 1e12
print(f"  2GiB reduction: {ms:.3f} ms -> {bw_read:.2f} TB/s (read only)")
del a, b
torch.cuda.empty_cache()

card = sys.argv[1] if len(sys.argv) > 1 else "omniASR_LLM_300M_v2"
pipe = ASRInferencePipeline(model_card=card)
model = pipe.model
dec = model.llama_decoder

dec_bytes = sum(p.numel() * p.element_size() for p in dec.parameters())
proj_bytes = sum(p.numel() * p.element_size() for p in model.final_proj.parameters())
total = dec_bytes + proj_bytes

print(f"\n{card}")
print(f"  decoder weights   : {dec_bytes/1e9:.3f} GB over {len(dec.layers)} layers")
print(f"  final_proj weights: {proj_bytes/1e9:.3f} GB")
print(f"  weight bytes per decode step (batch=1): {total/1e9:.3f} GB")

# --- practical floor: replay every Linear in the decode path as a bare GEMV,
#     in place, with no framework overhead around it. This keeps the real
#     weight footprint (so L2 cannot hide it) and is the best case any fused
#     implementation can reach for batch=1.
linears: list[nn.Linear] = [m for m in dec.modules() if isinstance(m, nn.Linear)]
linears += [m for m in model.final_proj.modules() if isinstance(m, nn.Linear)]
print(f"  Linear layers in decode path: {len(linears)}")
shapes: dict[tuple[int, int], int] = {}
for m in linears:
    shapes[tuple(m.weight.shape)] = shapes.get(tuple(m.weight.shape), 0) + 1
for s, c in sorted(shapes.items(), key=lambda kv: -kv[1]):
    print(f"    {c:3d} x  weight{list(s)}")

for bs in (1, 5):
    xs = [torch.randn(bs, m.in_features, dtype=torch.bfloat16, device="cuda") for m in linears]
    ws = [m.weight for m in linears]

    def step():
        for x, w in zip(xs, ws):
            torch.nn.functional.linear(x, w)

    ms = timed(step, iters=30, warmup=10)
    eff = total / (ms * 1e-3) / 1e12
    print(f"\n  batch={bs}: bare GEMV chain = {ms:.3f} ms/step "
          f"({eff:.2f} TB/s effective)")
    print(f"    -> pure-bandwidth floor @ {bw_read:.2f} TB/s = "
          f"{total/(bw_read*1e12)*1e3:.3f} ms/step")
