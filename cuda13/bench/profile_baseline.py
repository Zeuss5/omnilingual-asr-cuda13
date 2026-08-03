"""Profile omnilingual-asr inference: split encoder vs autoregressive decode,
and rank CUDA kernels by total time.

    CUDA_VISIBLE_DEVICES=3 .venv/bin/python cuda13/bench/profile_baseline.py [card]
"""

from __future__ import annotations

import sys
import time
from collections import defaultdict
from pathlib import Path

import torch
from torch.profiler import ProfilerActivity, profile

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

CARD = sys.argv[1] if len(sys.argv) > 1 else "omniASR_LLM_300M_v2"
AUDIO = str(Path(__file__).resolve().parents[1] / "assets" / "voices_sample.wav")
LANG = "eng_Latn"
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 1

pipe = ASRInferencePipeline(model_card=CARD)
model = pipe.model

n_enc = sum(p.numel() for p in model.encoder.parameters())
n_dec = sum(p.numel() for p in model.llama_decoder.parameters())
print(f"card={CARD} batch={BATCH}")
print(f"  encoder params {n_enc/1e9:.3f}B | llama decoder params {n_dec/1e9:.3f}B")
lc = model.llama_decoder.layers
print(f"  decoder layers={len(lc)} model_dim={model.model_dim} nbest={pipe.beam_search_generator.config.nbest}")


def build_batch():
    builder = pipe._build_audio_wavform_pipeline([AUDIO] * BATCH)
    waves = list(builder.and_return())
    return pipe._create_batch_simple([(w, LANG) for w in waves])


@torch.inference_mode()
def run_stages(batch):
    """Return (encoder_ms, decode_ms, n_steps)."""
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    ctx, ctx_lens, _ = model(batch, return_decoder_inputs=True)
    torch.cuda.synchronize()
    t1 = time.perf_counter()
    toks, lens = pipe.beam_search_generator.generate_hypotheses(
        decoder_context_inputs=ctx, decoder_context_seq_lens=ctx_lens,
        audio_embeddings=None, batch=None)
    torch.cuda.synchronize()
    t2 = time.perf_counter()
    return (t1 - t0) * 1e3, (t2 - t1) * 1e3, int(max(lens))


batch = build_batch()

# Warmup
for _ in range(3):
    run_stages(batch)

N = 10
enc, dec, steps = [], [], 0
for _ in range(N):
    e, d, s = run_stages(batch)
    enc.append(e); dec.append(d); steps = s

enc_ms = sum(enc) / N
dec_ms = sum(dec) / N
print(f"\n== wall clock (mean of {N}) ==")
print(f"  encoder+prefill : {enc_ms:8.2f} ms")
print(f"  decode loop     : {dec_ms:8.2f} ms  over {steps} steps"
      f"  -> {dec_ms/max(steps,1):.3f} ms/step")
print(f"  total           : {enc_ms+dec_ms:8.2f} ms")

# ---- kernel-level profile of the decode loop only ----
with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    with torch.inference_mode():
        ctx, ctx_lens, _ = model(batch, return_decoder_inputs=True)
        torch.cuda.synchronize()
        pipe.beam_search_generator.generate_hypotheses(
            decoder_context_inputs=ctx, decoder_context_seq_lens=ctx_lens,
            audio_embeddings=None, batch=None)
        torch.cuda.synchronize()

evs = prof.key_averages()
tot_cuda = sum(e.self_device_time_total for e in evs)
print(f"\n== top CUDA kernels (total self device time {tot_cuda/1e3:.1f} ms) ==")
print(f"  {'kernel':<58} {'ms':>8} {'%':>6} {'calls':>7}")
for e in sorted(evs, key=lambda x: -x.self_device_time_total)[:22]:
    if e.self_device_time_total <= 0:
        continue
    print(f"  {e.key[:58]:<58} {e.self_device_time_total/1e3:8.2f} "
          f"{100*e.self_device_time_total/tot_cuda:5.1f}% {e.count:7d}")

n_launch = sum(e.count for e in evs if e.self_device_time_total > 0)
print(f"\n  distinct kernel launches in one full inference: {n_launch}")
print(f"  approx launches per decode step: {n_launch/max(steps,1):.0f}")
