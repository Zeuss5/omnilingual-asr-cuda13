"""End-to-end latency ablation and transcription-equality check.

Measures each optimization separately so the speedup is attributable, and
verifies every variant produces byte-identical transcriptions.

    CUDA_VISIBLE_DEVICES=3 .venv/bin/python cuda13/bench/bench_e2e.py [card] [batch]
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
BATCH = int(sys.argv[2]) if len(sys.argv) > 2 else 1
AUDIO = str(Path(__file__).resolve().parents[1] / "assets" / "voices_sample.wav")
LANG = "eng_Latn"
REPS = 12
WARMUP = 3


def run(pipe, n):
    inp, lang = [AUDIO] * n, [LANG] * n
    for _ in range(WARMUP):
        out = pipe.transcribe(inp, lang=lang, batch_size=n)
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        out = pipe.transcribe(inp, lang=lang, batch_size=n)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return out, statistics.median(ts), min(ts), statistics.stdev(ts)


VARIANTS = [
    ("baseline (upstream)", lambda p: None),
    ("+ allocator fix", keep_allocator_cache),
    ("+ fused decoder", lambda p: enable_fused_decoding(p, use_cuda_graph=False, fast_beam_search=False)),
    ("+ CUDA graph", lambda p: enable_fused_decoding(p, fast_beam_search=False)),
    ("+ sync-free beam search", lambda p: enable_fused_decoding(p)),
]

print(f"card={CARD} batch={BATCH}  median of {REPS} (after {WARMUP} warmup)\n")
print(f"  {'variant':<26} {'median':>9} {'best':>8} {'stdev':>8} {'vs base':>9}")

results, ref_txt, base_ms = [], None, None
for name, apply in VARIANTS:
    pipe = ASRInferencePipeline(model_card=CARD)
    apply(pipe)
    txt, med, best, sd = run(pipe, BATCH)
    if ref_txt is None:
        ref_txt, base_ms = txt, med
    results.append((name, med, best, sd, txt == ref_txt))
    print(f"  {name:<26} {med:8.1f}ms {best:7.1f}ms {sd:7.1f}ms {base_ms/med:8.2f}x")
    del pipe
    torch.cuda.empty_cache()

print(f"\n  transcription: {ref_txt[0]!r}")
bad = [n for n, _, _, _, same in results if not same]
if bad:
    print(f"  [FAIL] output changed for: {bad}")
    sys.exit(1)
print("  [PASS] all variants produce identical transcriptions")
