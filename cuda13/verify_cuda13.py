"""Smoke test for the CUDA 13.0 / torch 2.10 port of omnilingual-asr.

Run with a single visible GPU, e.g.:
    CUDA_VISIBLE_DEVICES=3 .venv/bin/python cuda13/verify_cuda13.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import torch

import fairseq2n
from fairseq2.data.data_pipeline import read_sequence
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

AUDIO = Path(__file__).parent / "assets" / "voices_sample.wav"
# Ground truth for the VOiCES sample shipped with the torchaudio tutorials.
REFERENCE = "i had that curiosity beside me at this moment"

failures: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> None:
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}{(' — ' + detail) if detail else ''}")
    if not ok:
        failures.append(name)


print("== environment ==")
print(f"  torch          {torch.__version__} (CUDA {torch.version.cuda})")
print(f"  fairseq2n      built for torch {fairseq2n.torch_version()} / {fairseq2n.torch_variant()}")
print(f"  device         {torch.cuda.get_device_name(0)} sm_{''.join(map(str, torch.cuda.get_device_capability(0)))}")
check("torch is a CUDA 13 build", torch.version.cuda.startswith("13"), torch.version.cuda)
check("fairseq2n matches torch", fairseq2n.torch_version().split("+")[0] == torch.__version__.split("+")[0])
check("fairseq2n has CUDA kernels", fairseq2n.supports_cuda() and fairseq2n.cuda_version()[0] == 13)

print("== fairseq2n tensor round-trip (the THPVariable regression) ==")
(out,) = list(read_sequence([torch.arange(4.0)]).map(lambda x: x * 2).and_return())
check("tensor survives a native data pipeline", torch.equal(out, torch.arange(4.0) * 2))

print("== LLM model (omniASR_LLM_300M_v2) ==")
t0 = time.time()
llm = ASRInferencePipeline(model_card="omniASR_LLM_300M_v2")
print(f"  loaded in {time.time() - t0:.1f}s")
res = llm.transcribe([str(AUDIO)], lang=["eng_Latn"], batch_size=1)
print(f"  -> {res[0]!r}")
check("LLM transcription matches reference", res[0].strip() == REFERENCE)

batch = llm.transcribe([str(AUDIO)] * 4, lang=["eng_Latn"] * 4, batch_size=2)
check("batched LLM decode is self-consistent",
      len({r.strip() for r in batch}) == 1 and batch[0].strip() == REFERENCE,
      f"{len(batch)} results")

del llm
torch.cuda.empty_cache()

print("== CTC model (omniASR_CTC_300M_v2) ==")
ctc = ASRInferencePipeline(model_card="omniASR_CTC_300M_v2")
res_ctc = ctc.transcribe([str(AUDIO)], batch_size=1)
print(f"  -> {res_ctc[0]!r}")
check("CTC produces a plausible transcription",
      "curiosity" in res_ctc[0].lower() and "moment" in res_ctc[0].lower())

print()
if failures:
    print(f"FAILED ({len(failures)}): {', '.join(failures)}")
    sys.exit(1)
print("All checks passed — omnilingual-asr runs on CUDA 13.0.")
