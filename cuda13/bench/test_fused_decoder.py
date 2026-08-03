"""Check the fused decoder is numerically equivalent to fairseq2's decoder,
and measure the speedup.

    CUDA_VISIBLE_DEVICES=3 .venv/bin/python cuda13/bench/test_fused_decoder.py [card] [rows]
"""

from __future__ import annotations

import sys
import time

import torch

from fairseq2.nn import BatchLayout, IncrementalStateBag
from omnilingual_asr.fused.decoder import FusedLlamaDecoder
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

CARD = sys.argv[1] if len(sys.argv) > 1 else "omniASR_LLM_300M_v2"
ROWS = int(sys.argv[2]) if len(sys.argv) > 2 else 1
CTX, STEPS = 200, 64

torch.manual_seed(0)
pipe = ASRInferencePipeline(model_card=CARD)
model = pipe.model
ref = model.llama_decoder
D, dev, dt = model.model_dim, pipe.device, pipe.dtype

fused = FusedLlamaDecoder(ref, max_seq_len=CTX + STEPS + 16).to(dev)
print(f"{CARD} rows={ROWS} dim={D} layers={fused.n_layers} "
      f"heads={fused.n_heads}x{fused.head_dim} inner={fused.inner_dim}")

ctx = torch.randn(ROWS, CTX, D, device=dev, dtype=dt) * 0.1
toks = [torch.randn(ROWS, 1, D, device=dev, dtype=dt) * 0.1 for _ in range(8)]


@torch.inference_mode()
def run(dec, n_steps):
    bag = IncrementalStateBag(max_num_steps=8192)
    out = dec(seqs=ctx, seqs_layout=BatchLayout.of(ctx, [CTX] * ROWS), state_bag=bag)
    bag.increment_step_nr(CTX)
    outs = [out[:, -1]]
    for i in range(n_steps):
        t = toks[i % len(toks)]
        o = dec(seqs=t, seqs_layout=BatchLayout.of(t, [1] * ROWS), state_bag=bag)
        bag.increment_step_nr(1)
        outs.append(o[:, 0])
    return torch.stack(outs)


print("\n== numerical equivalence ==")
a = run(ref, 8)
b = run(fused, 8)
rel = ((a.float() - b.float()).abs().max() / a.float().abs().max()).item()
print(f"  max rel error over prefill + 8 decode steps: {rel:.2e}")
ok_num = rel < 3e-2
print(f"  [{'PASS' if ok_num else 'FAIL'}] fused decoder matches reference")

# Logit-level agreement is what actually matters for beam search.
with torch.inference_mode():
    la = model.final_proj(a[-1]).float()
    lb = model.final_proj(b[-1]).float()
top_a = la.argmax(-1)
top_b = lb.argmax(-1)
agree = bool((top_a == top_b).all())
print(f"  [{'PASS' if agree else 'FAIL'}] argmax token agrees: {top_a.tolist()} vs {top_b.tolist()}")
k = min(10, la.shape[-1])
print(f"  top-{k} overlap: "
      f"{len(set(la.topk(k, -1).indices[0].tolist()) & set(lb.topk(k, -1).indices[0].tolist()))}/{k}")


@torch.inference_mode()
def bench(dec, label):
    """Time only the decode steps; prefill is charged separately."""
    for _ in range(2):
        run(dec, 8)
    torch.cuda.synchronize()

    bag = IncrementalStateBag(max_num_steps=8192)
    t0 = time.perf_counter()
    dec(seqs=ctx, seqs_layout=BatchLayout.of(ctx, [CTX] * ROWS), state_bag=bag)
    torch.cuda.synchronize()
    pre_ms = (time.perf_counter() - t0) * 1e3
    bag.increment_step_nr(CTX)

    t0 = time.perf_counter()
    for i in range(STEPS):
        t = toks[i % len(toks)]
        dec(seqs=t, seqs_layout=BatchLayout.of(t, [1] * ROWS), state_bag=bag)
        bag.increment_step_nr(1)
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / STEPS * 1e3
    print(f"  {label:<22} {ms:7.3f} ms/step   (prefill {pre_ms:6.2f} ms)")
    return ms


print("\n== decode step latency ==")
ref_ms = bench(ref, "fairseq2 reference")
fused_nog = FusedLlamaDecoder(ref, max_seq_len=CTX + STEPS + 16, use_cuda_graph=False).to(dev)
nog_ms = bench(fused_nog, "fused (no graph)")
g_ms = bench(fused, "fused + CUDA graph")

wb = sum(p.numel() * p.element_size() for p in ref.parameters())
print(f"\n  weights per step: {wb/1e9:.3f} GB")
for name, ms in (("reference", ref_ms), ("fused", nog_ms), ("fused+graph", g_ms)):
    print(f"  {name:<12} {ms:6.3f} ms  -> {wb/(ms*1e-3)/1e12:.2f} TB/s   "
          f"speedup {ref_ms/ms:.2f}x")

sys.exit(0 if (ok_num and agree) else 1)
