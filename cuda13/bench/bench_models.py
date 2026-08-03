"""Transcription latency by model size — what a user actually experiences.

    CUDA_VISIBLE_DEVICES=3 .venv/bin/python cuda13/bench/bench_models.py [batch]

Reports, for each LLM-ASR model size, the wall-clock cost of transcribing clips
of several lengths, before and after the fused decode path. CTC models are
included for reference: they have no autoregressive loop, so the fused path does
not apply to them.
"""

from __future__ import annotations

import statistics
import sys
import time
from pathlib import Path

import soundfile as sf
import torch

from omnilingual_asr.fused.pipeline import enable_fused_decoding, keep_allocator_cache
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

BATCH = int(sys.argv[1]) if len(sys.argv) > 1 else 1
ASSETS = Path(__file__).resolve().parents[1] / "assets"
SRC = ASSETS / "voices_sample.wav"
DURATIONS = [3.4, 10.0, 30.0]
REPS, WARMUP = 6, 2

LLM_CARDS = ["omniASR_LLM_300M_v2", "omniASR_LLM_1B_v2",
             "omniASR_LLM_3B_v2", "omniASR_LLM_7B_v2"]
CTC_CARDS = ["omniASR_CTC_300M_v2", "omniASR_CTC_7B_v2"]

# Build clips by tiling the reference sample.
wav, sr = sf.read(SRC)
clips = {}
for d in DURATIONS:
    need = int(d * sr)
    buf = wav
    while len(buf) < need:
        buf = torch.cat([torch.tensor(buf), torch.tensor(wav)]).numpy()
    p = ASSETS / f"_bench_{d:g}s.wav"
    sf.write(p, buf[:need], sr)
    clips[d] = str(p)


def bench(pipe, path, n, lang=True):
    inp = [path] * n
    kw = {"lang": ["eng_Latn"] * n} if lang else {}
    for _ in range(WARMUP):
        pipe.transcribe(inp, batch_size=n, **kw)
    torch.cuda.synchronize()
    ts = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        pipe.transcribe(inp, batch_size=n, **kw)
        torch.cuda.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    return statistics.median(ts)


def params(pipe):
    return sum(p.numel() for p in pipe.model.parameters())


print(f"batch={BATCH}, median of {REPS} runs, clocks locked\n")
hdr = "  " + f"{'model':<22}{'params':>8}" + "".join(f"{f'{d:g}s':>20}" for d in DURATIONS)
print(hdr)
print("  " + f"{'':<22}{'':>8}" + "".join(f"{'base -> fused':>20}" for _ in DURATIONS))
print("  " + "-" * (30 + 20 * len(DURATIONS)))

for card in LLM_CARDS:
    base = ASRInferencePipeline(model_card=card)
    keep_allocator_cache(base)          # honest baseline: no allocator stalls
    n_par = params(base)
    b = {d: bench(base, clips[d], BATCH) for d in DURATIONS}
    del base
    torch.cuda.empty_cache()

    fast = ASRInferencePipeline(model_card=card)
    enable_fused_decoding(fast)
    f = {d: bench(fast, clips[d], BATCH) for d in DURATIONS}
    del fast
    torch.cuda.empty_cache()

    cells = "".join(f"{b[d]:7.0f} ->{f[d]:6.0f}ms" for d in DURATIONS)
    print(f"  {card:<22}{n_par/1e9:7.2f}B{cells}")

print()
for card in CTC_CARDS:
    p = ASRInferencePipeline(model_card=card)
    keep_allocator_cache(p)
    n_par = params(p)
    cells = "".join(f"{bench(p, clips[d], BATCH, lang=False):15.0f}ms" for d in DURATIONS)
    print(f"  {card:<22}{n_par/1e9:7.2f}B{cells}   (CTC, no decode loop)")
    del p
    torch.cuda.empty_cache()

print("\n  RTFx (fused, LLM) = audio seconds per wall-clock second")
for p in ASSETS.glob("_bench_*.wav"):
    p.unlink()
