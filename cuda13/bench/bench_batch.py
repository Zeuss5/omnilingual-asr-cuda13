"""Throughput vs batch size, baseline against fused.

At batch 1 the decode step is bound by streaming weights; those same weights
serve every sequence in a batch, so throughput should climb with batch size.

    CUDA_VISIBLE_DEVICES=3 .venv/bin/python cuda13/bench/bench_batch.py [card]
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import torch

from omnilingual_asr.fused.pipeline import enable_fused_decoding, keep_allocator_cache
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

CARD = sys.argv[1] if len(sys.argv) > 1 else "omniASR_LLM_300M_v2"
AUDIO = str(Path(__file__).resolve().parents[1] / "assets" / "voices_sample.wav")
DUR = 3.4
BATCHES = [1, 2, 4, 8, 16]
REPS = 6


def run(pipe, n):
    inp, lang = [AUDIO] * n, ["eng_Latn"] * n
    for _ in range(2):
        pipe.transcribe(inp, lang=lang, batch_size=n)
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        pipe.transcribe(inp, lang=lang, batch_size=n)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


print(f"card={CARD}  {DUR}s clips, median of {REPS}\n")
print(f"  {'batch':>5} | {'baseline':>10} {'RTFx':>7} | {'fused':>10} {'RTFx':>7} | {'speedup':>8}")
print("  " + "-" * 62)

base = ASRInferencePipeline(model_card=CARD)
keep_allocator_cache(base)          # compare against a baseline without the stall
base_ms = {n: run(base, n) for n in BATCHES}
del base
torch.cuda.empty_cache()

fast = ASRInferencePipeline(model_card=CARD)
enable_fused_decoding(fast)
for n in BATCHES:
    f = run(fast, n)
    b = base_ms[n]
    print(f"  {n:5d} | {b:9.1f}ms {n*DUR*1000/b:6.1f}x | "
          f"{f:9.1f}ms {n*DUR*1000/f:6.1f}x | {b/f:7.2f}x")

print("\n  RTFx = seconds of audio transcribed per second of wall clock")
