"""Kernel-level breakdown of the fused decode step, and a cold-weight GEMV
roofline for the four projections in a layer.

    CUDA_VISIBLE_DEVICES=3 .venv/bin/python cuda13/bench/profile_fused.py [card] [rows]
"""

from __future__ import annotations

import sys

import torch
import torch.nn.functional as F
from torch.profiler import ProfilerActivity, profile

from fairseq2.nn import BatchLayout, IncrementalStateBag
from omnilingual_asr.fused.decoder import FusedLlamaDecoder
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

CARD = sys.argv[1] if len(sys.argv) > 1 else "omniASR_LLM_300M_v2"
ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 1
CTX, STEPS = 200, 50

pipe = ASRInferencePipeline(model_card=CARD)
model = pipe.model
D, dev, dt = model.model_dim, pipe.device, pipe.dtype
fused = FusedLlamaDecoder(model.llama_decoder, max_seq_len=CTX + STEPS + 16,
                          use_cuda_graph=False).to(dev)

ctx = torch.randn(ROWS, CTX, D, device=dev, dtype=dt) * 0.1
tok = torch.randn(ROWS, 1, D, device=dev, dtype=dt) * 0.1


@torch.inference_mode()
def decode_loop(n):
    bag = IncrementalStateBag(max_num_steps=8192)
    fused(seqs=ctx, seqs_layout=BatchLayout.of(ctx, [CTX] * ROWS), state_bag=bag)
    bag.increment_step_nr(CTX)
    for _ in range(n):
        fused(seqs=tok, seqs_layout=BatchLayout.of(tok, [1] * ROWS), state_bag=bag)
        bag.increment_step_nr(1)


decode_loop(8)
torch.cuda.synchronize()

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    decode_loop(STEPS)
    torch.cuda.synchronize()

evs = [e for e in prof.key_averages() if e.self_device_time_total > 0 and "aten::" not in e.key]
tot = sum(e.self_device_time_total for e in evs)
print(f"\n== fused decode: CUDA kernels over {STEPS} steps (total {tot/1e3:.2f} ms, "
      f"{tot/1e3/STEPS:.3f} ms/step) ==")
print(f"  {'kernel':<52} {'ms/step':>9} {'%':>6} {'calls/step':>11}")
for e in sorted(evs, key=lambda x: -x.self_device_time_total)[:14]:
    print(f"  {e.key[:52]:<52} {e.self_device_time_total/1e3/STEPS:9.3f} "
          f"{100*e.self_device_time_total/tot:5.1f}% {e.count/STEPS:11.1f}")

# --- cold-weight GEMV roofline for one layer's four projections ---
print("\n== GEMV roofline (weights too large for L2, as in the real step) ==")
shapes = [("qkv", 3 * D, D), ("o", D, D),
          ("gate_up", 2 * fused.inner_dim, D), ("down", D, fused.inner_dim)]
# Allocate a fresh copy per layer so nothing is L2-resident across iterations.
banks = [[torch.randn(n, k, device=dev, dtype=dt) for _, n, k in shapes]
         for _ in range(fused.n_layers)]
xs = {k: torch.randn(ROWS, k, device=dev, dtype=dt) for _, _, k in shapes}

def gemv_chain():
    for bank in banks:
        for (name, n, k), w in zip(shapes, bank):
            F.linear(xs[k], w)

for _ in range(5):
    gemv_chain()
torch.cuda.synchronize()
ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
ev0.record()
for _ in range(20):
    gemv_chain()
ev1.record()
torch.cuda.synchronize()
ms = ev0.elapsed_time(ev1) / 20
wb = sum(w.numel() * w.element_size() for bank in banks for w in bank)
print(f"  all {fused.n_layers} layers' projections: {ms:.3f} ms  "
      f"({wb/1e9:.3f} GB -> {wb/(ms*1e-3)/1e12:.2f} TB/s)")
print("  ^ this is the floor any fused decoder can reach for the GEMM part")
