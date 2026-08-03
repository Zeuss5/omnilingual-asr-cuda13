"""Microbenchmark for one autoregressive decode step of the omniASR Llama decoder.

Isolates the step that dominates end-to-end latency and separates GPU-busy time
from CPU launch overhead, so we can tell what fusion vs. graph capture can win.

    CUDA_VISIBLE_DEVICES=3 .venv/bin/python cuda13/bench/bench_decode.py [card] [batch]
"""

from __future__ import annotations

import sys
import time

import torch

from fairseq2.nn import BatchLayout, IncrementalStateBag
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

CARD = sys.argv[1] if len(sys.argv) > 1 else "omniASR_LLM_300M_v2"
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 1
CTX = int(sys.argv[3]) if len(sys.argv) > 3 else 200
STEPS = 64

pipe = ASRInferencePipeline(model_card=CARD)
model = pipe.model
dec = model.llama_decoder
D = model.model_dim
dtype = pipe.dtype
dev = pipe.device

print(f"{CARD}  batch={BATCH}  ctx={CTX}  dim={D}  layers={len(dec.layers)}")

weight_bytes = sum(p.numel() * p.element_size() for p in dec.parameters())
weight_bytes += sum(p.numel() * p.element_size() for p in model.final_proj.parameters())
print(f"  weights streamed per step: {weight_bytes/1e9:.3f} GB")


@torch.inference_mode()
def make_state():
    """Prefill a state bag with CTX tokens of context."""
    bag = IncrementalStateBag(max_num_steps=model.max_generation_length)
    prefill = torch.randn(BATCH, CTX, D, device=dev, dtype=dtype)
    layout = BatchLayout.of(prefill, [CTX] * BATCH)
    dec(seqs=prefill, seqs_layout=layout, state_bag=bag)
    bag.increment_step_nr(CTX)
    return bag


@torch.inference_mode()
def decode_steps(bag, n):
    x = torch.randn(BATCH, 1, D, device=dev, dtype=dtype)
    layout = BatchLayout.of(x, [1] * BATCH)
    for _ in range(n):
        out = dec(seqs=x, seqs_layout=layout, state_bag=bag)
        model.final_proj(out)
        bag.increment_step_nr(1)


# warmup
bag = make_state()
decode_steps(bag, 8)
torch.cuda.synchronize()

# --- wall clock ---
bag = make_state()
torch.cuda.synchronize()
t0 = time.perf_counter()
decode_steps(bag, STEPS)
torch.cuda.synchronize()
wall = (time.perf_counter() - t0) / STEPS * 1e3

# --- CPU-side time only (how long the python/dispatch work takes with no waiting) ---
bag = make_state()
torch.cuda.synchronize()
t0 = time.perf_counter()
decode_steps(bag, STEPS)
cpu = (time.perf_counter() - t0) / STEPS * 1e3   # returns before GPU finishes
torch.cuda.synchronize()

# --- GPU busy time via events around the whole run ---
bag = make_state()
torch.cuda.synchronize()
ev0, ev1 = torch.cuda.Event(True), torch.cuda.Event(True)
ev0.record()
decode_steps(bag, STEPS)
ev1.record()
torch.cuda.synchronize()
gpu = ev0.elapsed_time(ev1) / STEPS

print(f"\n  wall        : {wall:.3f} ms/step")
print(f"  cpu enqueue : {cpu:.3f} ms/step")
print(f"  gpu span    : {gpu:.3f} ms/step")
bw = weight_bytes / (wall * 1e-3) / 1e12
print(f"  effective weight bandwidth: {bw:.2f} TB/s")
if cpu > gpu * 0.9:
    print("  -> CPU-BOUND: dispatch/launch overhead dominates (CUDA graphs will help most)")
else:
    print("  -> GPU-BOUND: kernel efficiency dominates (fusion will help most)")
