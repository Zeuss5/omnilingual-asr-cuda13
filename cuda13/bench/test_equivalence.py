"""Equivalence test: fused path must reproduce upstream transcriptions exactly
across audio lengths, batch sizes, ragged batches and missing language codes.

    CUDA_VISIBLE_DEVICES=3 .venv/bin/python cuda13/bench/test_equivalence.py [card]
"""

from __future__ import annotations

import sys
from pathlib import Path

import soundfile as sf
import torch

from omnilingual_asr.fused.pipeline import enable_fused_decoding
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

CARD = sys.argv[1] if len(sys.argv) > 1 else "omniASR_LLM_300M_v2"
ASSETS = Path(__file__).resolve().parents[1] / "assets"
SRC = ASSETS / "voices_sample.wav"

wav, sr = sf.read(SRC)
print(f"source: {len(wav)/sr:.2f}s @ {sr}Hz")

# Build clips of different lengths, including a long one that stresses the KV
# cache reservation (the pipeline caps input at 40 s).
clips = {}
for name, secs in [("full", None), ("half", 1.7), ("short", 0.8)]:
    w = wav if secs is None else wav[: int(secs * sr)]
    p = ASSETS / f"_clip_{name}.wav"
    sf.write(p, w, sr)
    clips[name] = str(p)

# ~30 s by tiling, to exercise a long encoder context
long_w = wav
while len(long_w) / sr < 30:
    long_w = torch.cat([torch.tensor(long_w), torch.tensor(wav)]).numpy()
long_p = ASSETS / "_clip_long.wav"
sf.write(long_p, long_w[: int(30 * sr)], sr)
clips["long30s"] = str(long_p)
print("clips:", {k: f"{sf.info(v).duration:.1f}s" for k, v in clips.items()})

CASES = [
    ("single full",        [clips["full"]],                              ["eng_Latn"], 1),
    ("single long 30s",    [clips["long30s"]],                           ["eng_Latn"], 1),
    ("no lang code",       [clips["full"]],                              None,         1),
    ("batch 2 same",       [clips["full"]] * 2,                          ["eng_Latn"] * 2, 2),
    ("batch 3 ragged",     [clips["full"], clips["half"], clips["short"]],["eng_Latn"] * 3, 3),
    ("batch 4 ragged+long",[clips["short"], clips["long30s"], clips["full"], clips["half"]],
                                                                          ["eng_Latn"] * 4, 4),
    ("bs=2 over 3 items",  [clips["full"], clips["short"], clips["half"]], ["eng_Latn"] * 3, 2),
]


def transcribe_all(pipe):
    res = {}
    for name, inp, lang, bs in CASES:
        res[name] = pipe.transcribe(inp, lang=lang, batch_size=bs)
    return res


print("\nrunning baseline...")
base = ASRInferencePipeline(model_card=CARD)
ref = transcribe_all(base)
del base
torch.cuda.empty_cache()

print("running fused...")
fast = ASRInferencePipeline(model_card=CARD)
enable_fused_decoding(fast)
got = transcribe_all(fast)

print()
fails = []
for name, _, _, _ in CASES:
    ok = ref[name] == got[name]
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}")
    if not ok:
        fails.append(name)
        for i, (a, b) in enumerate(zip(ref[name], got[name])):
            if a != b:
                print(f"        [{i}] base : {a!r}")
                print(f"        [{i}] fused: {b!r}")

for p in ASSETS.glob("_clip_*.wav"):
    p.unlink()

print()
if fails:
    print(f"FAILED: {fails}")
    sys.exit(1)
print("Fused path is transcription-identical to upstream on all cases.")
