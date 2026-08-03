# Fused decode path for omniASR LLM models

`src/omnilingual_asr/fused/` replaces the autoregressive half of LLM-ASR
inference with a fused, CUDA-graph-captured decoder. Output is unchanged —
the equivalence suite asserts byte-identical transcriptions.

```python
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
from omnilingual_asr.fused.pipeline import enable_fused_decoding

pipe = ASRInferencePipeline(model_card="omniASR_LLM_300M_v2")
enable_fused_decoding(pipe)
pipe.transcribe(["audio.wav"], lang=["eng_Latn"])
```

## Where the time was going

Profiled on an RTX PRO 6000 Blackwell (sm_120, 188 SMs, 134 MB L2,
1.53 TB/s measured read bandwidth), `omniASR_LLM_300M_v2`, 3.4 s of audio:

| stage | ms | share |
| --- | --- | --- |
| encoder + prefill | 21 | 10% |
| autoregressive decode (45 steps) | 185 | 90% |

So the decode loop is the whole problem. It ran at **4.12 ms/step** while
issuing **936 kernel launches per step**. The decoder alone measured
3.10 ms/step, with CPU enqueue time (2.93 ms) essentially equal to the GPU span
(3.10 ms) — the GPU was starved, not busy.

Speed of light for this step is set by weight traffic, not math: at batch 1 each
token must stream all 2.44 GB of decoder weights from HBM, giving a floor of
**1.60 ms/step**. The baseline achieved 0.78 TB/s of the 1.53 TB/s available.

## What was done

**1. The allocator flush (`keep_allocator_cache`).** Upstream `_apply_model`
calls `torch.cuda.empty_cache()` after every batch, returning every block to the
driver so the next utterance re-`cudaMalloc`s its working set — including a
67 MB decoder-input buffer. This alone caused latency to swing between 133 ms
and 490 ms run to run. Not a kernel problem, but it dominated the variance and
had to go first to make anything else measurable.

**2. Fused decoder (`fused/decoder.py`, `fused/kernels.py`).**
- q/k/v pack into one GEMM and gate/up into another: 7 GEMMs per layer → 4.
- `add_rmsnorm` folds each sublayer's residual add into the following RMSNorm,
  so the residual stream is read once per sublayer instead of three times.
- `rope_write_kv` applies rotary embedding to q and k *and* appends k/v to the
  KV cache in one kernel.
- `swiglu` is a single pass over the packed `[gate | up]` buffer.
- `decode_attention` is a flash-decoding kernel: online softmax, split across
  the sequence so batch×heads (8-40 programs) does not leave 188 SMs idle.

**3. CUDA graph capture.** The decode step runs entirely against preallocated
buffers. The KV length is held in a *device* tensor (`len_dev`) rather than a
Python int, which is what makes a fixed-shape capture possible at all — the
attention kernel reads how many keys are valid from device memory. The KV cache
is allocated once and reused across utterances so the captured addresses stay
valid; only `cur_len` resets.

**4. Sync-free beam search (`fused/beamsearch.py`).** The upstream loop uses
boolean-mask assignment (`candidate_scores[eos_mask] = -inf`,
`out_tokens[no_token_mask, t] = pad`). Each lowers to `nonzero()` + index_put,
and `nonzero` must copy its result size back to the host — four device→host
syncs per token, each draining the launch queue the graph is trying to keep
full. Rewritten with `torch.where`/`masked_fill_`, leaving exactly one sync per
step (the termination check). The beam-reorder gathers are also skipped when
`nbest == 1`, where the permutation is the identity by construction.

## Results

`omniASR_LLM_300M_v2`, batch 1, 3.4 s audio, median of 12 runs, clocks locked:

| variant | median | best | stdev | speedup |
| --- | --- | --- | --- | --- |
| baseline (upstream) | 299.1 ms | 212.5 ms | 66.3 ms | 1.00× |
| + allocator fix | 211.1 ms | 209.1 ms | 9.1 ms | 1.42× |
| + fused decoder | 161.6 ms | 161.1 ms | 0.5 ms | 1.85× |
| + CUDA graph | 148.3 ms | 147.6 ms | 2.1 ms | 2.02× |
| + sync-free beam search | **133.1 ms** | 132.8 ms | 1.3 ms | **2.25×** |

`omniASR_LLM_7B_v2`, same conditions:

| variant | median | best | stdev | speedup |
| --- | --- | --- | --- | --- |
| baseline (upstream) | 258.0 ms | 219.9 ms | 36.2 ms | 1.00× |
| + allocator fix | 218.7 ms | 214.9 ms | 2.5 ms | 1.18× |
| + fused decoder | 166.8 ms | 165.6 ms | 2.0 ms | 1.55× |
| + CUDA graph | 153.1 ms | 151.1 ms | 1.8 ms | 1.68× |
| + sync-free beam search | **138.0 ms** | 137.3 ms | 1.0 ms | **1.87×** |

The 7B lands within 5 ms of the 300M because the two share the *same* decoder
(12 layers, dim 4096) — the parameter count difference is almost entirely in the
wav2vec2 encoder, which runs once per utterance while the decoder runs once per
token. That is also why optimizing the decode loop pays off across the whole
model family.

Decode step in isolation (200-token context):

| | ms/step | effective bandwidth |
| --- | --- | --- |
| fairseq2 reference | 3.14 | 0.78 TB/s |
| fused | 2.25 | 1.09 TB/s |
| fused + CUDA graph (cuBLAS) | 2.01 | 1.21 TB/s |
| fused + CUDA graph + raw-CUDA GEMV | **1.91** | **1.28 TB/s** |
| *cold-weight GEMV floor* | *1.86* | *1.31 TB/s* |
| *pure bandwidth roofline* | *1.60* | *1.53 TB/s* |

Against the upstream median the end-to-end win is 2.25×; against its *best*
run (which excludes the allocator stalls) it is 1.60×. Run-to-run spread drops
from 66 ms to 1.3 ms, which matters more than the mean for a latency target.

### Latency by model size

What a user actually waits for. Batch 1, median of 6, clocks locked; the
baseline column already has the allocator fix, so this is the honest
kernel-level comparison.

| model | params | 3.4 s clip | 10 s clip | 30 s clip |
| --- | --- | --- | --- | --- |
| `omniASR_LLM_300M_v2` | 1.63 B | 211 → **128 ms** (1.65×) | 652 → **331 ms** (1.97×) | 1568 → **624 ms** (2.51×) |
| `omniASR_LLM_1B_v2` | 2.28 B | 197 → **116 ms** (1.70×) | 621 → **323 ms** (1.92×) | 1519 → **618 ms** (2.46×) |
| `omniASR_LLM_3B_v2` | 4.38 B | 200 → **121 ms** (1.65×) | 639 → **329 ms** (1.94×) | 1552 → **644 ms** (2.41×) |
| `omniASR_LLM_7B_v2` | 7.80 B | 216 → **133 ms** (1.62×) | 656 → **348 ms** (1.89×) | 1593 → **695 ms** (2.29×) |
| `omniASR_CTC_300M_v2` | 0.33 B | 26 ms | 28 ms | 35 ms |
| `omniASR_CTC_7B_v2` | 6.51 B | 31 ms | 48 ms | 108 ms |

Two things worth reading off this table:

**Latency barely depends on model size.** 300M and 7B are within 10% of each
other at every clip length, because all four cards share the *same* decoder —
12 layers, dim 4096, inner 2816. The parameter count lives almost entirely in
the wav2vec2 encoder, which runs once per utterance, while the decoder runs once
per generated token and therefore sets the latency. Picking 7B over 300M costs
~5% wall clock (and 30 GB instead of 6 GB of weights), not 4×.

**The speedup grows with clip length** — 1.6× at 3.4 s, 1.9× at 10 s, 2.3-2.5×
at 30 s — because longer audio means more decode steps, and the decode step is
what was optimized. The fixed encoder cost is what dilutes it on short clips.

For reference, at 30 s the fused LLM path runs at ~43-48× real time, and CTC
(no autoregressive loop at all, so untouched by this work) at ~860×.

### Batch scaling

`omniASR_LLM_300M_v2`, 3.4 s clips. The baseline column already includes the
allocator fix, so this isolates the kernel work:

| batch | baseline | RTFx | fused | RTFx | speedup |
| --- | --- | --- | --- | --- | --- |
| 1 | 213.1 ms | 16.0× | 134.1 ms | 25.4× | 1.59× |
| 2 | 236.7 ms | 28.7× | 145.0 ms | 46.9× | 1.63× |
| 4 | 255.5 ms | 53.2× | 155.9 ms | 87.2× | 1.64× |
| 8 | 287.6 ms | 94.6× | 176.6 ms | 154.1× | 1.63× |
| 16 | 371.8 ms | 146.3× | 221.0 ms | 246.2× | 1.68× |

RTFx = seconds of audio per second of wall clock. The gain holds at ~1.6×
across the range rather than eroding, because the fused step removes per-step
overhead that does not shrink with batching.

## How close to speed of light

The fused step is at **1.21 TB/s of a 1.53 TB/s roofline**, and within 8% of the
measured cold-weight GEMV floor of 1.86 ms. Counting the KV cache as well as the
weights, SoL at ctx 200 is 1.62 ms against the measured 2.01 ms — **81%**. The
kernel breakdown:

| | ms/step | share |
| --- | --- | --- |
| packed projection GEMMs | 1.77 | 87% |
| attention + norms + RoPE + SwiGLU | 0.20 | 10% |

Everything that is not a GEMM now costs 10% of the step. The GEMMs themselves
run at ~90% of achievable bandwidth, so further gains have to come from moving
*fewer bytes* (weight quantization) rather than from more fusion — batch-1
decode is bandwidth-bound by definition, and the arithmetic is already free.

### Raw CUDA beats both cuBLAS and Triton

`fused/csrc/gemv.cu` is a hand-written bf16 GEMV: one warp per output row, each
lane walking `W[n, :]` with vectorized 16-byte loads so a warp pulls a
contiguous 512 B line per step, then a shuffle reduction. There is nothing
clever in it — at batch 1 the kernel is a pure streaming read of `W`, so the
only job is perfect coalescing and enough warps in flight.

That is enough to beat cuBLAS everywhere, by the most on the shapes where
cuBLAS falls back to its `gemvx` path:

| shape | cuBLAS | raw CUDA | |
| --- | --- | --- | --- |
| qkv `[12288,4096]` | 0.837 ms (1.44 TB/s, 94% pk) | 0.819 ms (1.47, 96%) | 1.02× |
| gate_up `[5632,4096]` | 0.418 ms (1.33, 87%) | 0.394 ms (1.41, 92%) | 1.06× |
| o `[4096,4096]` | 0.334 ms (1.21, 79%) | 0.295 ms (1.36, 89%) | 1.13× |
| down `[4096,2816]` | 0.261 ms (1.06, **69%**) | 0.221 ms (1.25, 82%) | 1.18× |
| **total** | **1.850 ms (86% pk)** | **1.729 ms (92% pk)** | **1.07×** |

At M=2 (the pipeline's default batch) the margin is larger still: 1.952 →
1.755 ms, 1.11×. Ranking on these shapes is therefore
**raw CUDA (1.47 TB/s) > cuBLAS (1.44) > Triton (1.32)**.

The extension is JIT-compiled on first use via `torch.utils.cpp_extension` and
is entirely optional: if nvcc or ninja is missing, `cuda_ops.gemv` returns
`None` and the decoder falls back to `F.linear`.

Fusing RMSNorm into this kernel was considered and rejected for a structural
reason rather than a measured one: the norm needs a full-row reduction before
any output column exists, so all ~4096 warps would have to redo it — ~32 MB of
L2 traffic per call, more than the ~2 µs standalone kernel it would replace.
The elementwise kernels are latency-bound, not bandwidth-bound, and the only
way to remove them is to remove the kernel boundary itself.

### What was tried and rejected: a persistent megakernel

Built, measured, and kept in `cuda13/experiments/megakernel/` — the full step in
one cooperative kernel, every boundary replaced by `cg::grid_group::sync()`. It
is correct (8.2e-03 rel. error vs the fused decoder) and **0.64x the speed**:
2.935 ms against 1.891 ms.

| | megakernel | as separate kernels |
| --- | --- | --- |
| GEMV stages | 2.164 ms | 1.770 ms |
| attention + norms + syncs | 0.771 ms | 0.175 ms |

The decisive diagnostic is `gemv_only_mega`: the same 48 GEMVs, still in a
persistent cooperative grid, with attention and norms deleted. It is *still*
0.82x. So this is structural, not an artifact of the attention stage.

The GEMVs are 87% of the step and bandwidth-bound, so they want maximum
memory-level parallelism — and a cooperative grid removes it twice. A
megakernel's register footprint is the union of all its stages (64 regs, 4
blocks/SM, 32 warps/SM) where the standalone GEMV uses 32-48 and reaches ~64
warps/SM; and cooperative launch requires every block co-resident (752-940
blocks) where the standalone GEMV launches 1536 and lets the scheduler run waves
at full occupancy. Even a *free* attention stage leaves a 2.164 ms floor, 1.14x
the current total.

The premise was never sound either. Inside an already-captured CUDA graph a
kernel boundary costs **0.9-1.1 µs, flat in grid size**, because the graph has
already removed launch dispatch; a `grid.sync()` is a software barrier over all
resident blocks and *grows* with them (1.03 µs at 188 blocks, 1.39 at 512, 2.96
at 1024). And summing per-kernel self-times (1.852 ms) against the measured step
(1.907 ms) puts *all* boundary overhead at 0.055 ms — 2.9%. That was the entire
prize.

### What was tried and rejected: a Triton fused GEMV

The obvious next step is to fold RMSNorm into a custom GEMV prologue and SwiGLU
into its epilogue, removing the 24 norm and 12 SwiGLU kernels. It was
implemented and swept over `BN × BK × warps × stages`, in the *real*
configuration (residual delta present, weights sized so nothing is L2-resident).
**It lost on every shape:**

| shape | cuBLAS + separate kernels | best Triton fused | |
| --- | --- | --- | --- |
| qkv `[12288,4096]` | 0.868 ms (1.39 TB/s) | 0.913 ms (1.32 TB/s) | 0.95× |
| gate_up `[5632,4096]` | 0.473 ms (1.17 TB/s) | 0.505 ms (1.10 TB/s) | 0.94× |
| o `[4096,4096]` | 0.323 ms (1.25 TB/s) | 0.344 ms (1.17 TB/s) | 0.94× |
| down `[4096,2816]` | 0.256 ms (1.08 TB/s) | 0.274 ms (1.01 TB/s) | 0.94× |

cuBLAS's cutlass bf16 kernels reach 1.39 TB/s where the hand-written GEMV tops
out at 1.32, and the kernels it would have absorbed cost only ~2 µs each — far
less than the 5-6% GEMM regression. Fusing SwiGLU makes it worse still: it
halves the grid (2816/BN instead of 5632/BN), leaving 88 programs on 188 SMs.

A caution about measuring this: an early version of the sweep showed Triton
*winning* by 6-13%. That was an artifact — the cuBLAS baseline included a
`.clone()` per iteration. With the clone removed the result inverts. The
projections therefore stay on cuBLAS, and `_decode_body` documents why.

### Attention tuning

Sweeping the decode-attention split has to be done *inside a CUDA graph*.
Measured normally, every configuration lands within 1% of every other because
launch overhead (~18 µs/call) swamps the 8 µs kernel. Under graph capture the
differences are real, and small blocks with many splits win — batch×heads is
only 8 programs, so parallelism has to come from the sequence axis:

| ctx | traffic floor | `SPLITS=8, BLOCK_N=64` | `SPLITS=32, BLOCK_N=16, warps=2` |
| --- | --- | --- | --- |
| 200 | 0.026 ms | 0.098 ms | 0.064 ms (1.55×) |
| 500 | 0.064 ms | 0.110 ms | 0.079 ms (1.39×) |
| 1500 | 0.193 ms | 0.342 ms | 0.273 ms (1.25×) |

Attention is ~5% of the step, so this moves the decode step from 2.03 to
2.01 ms — real, but end-to-end it is inside the noise.

### Where that leaves speed of light

Including KV traffic, SoL at ctx 200 is 1.62 ms/step and the fused decoder sits
at 1.91 ms — **85%**. With the raw-CUDA GEMV the projections are ~96% efficient,
so almost all of the remaining 0.29 ms is the ~110 small kernels' fixed cost:
attention, RoPE, the norms and SwiGLU together need ~0.03 ms of traffic but take
~0.15 ms, because each is latency-bound at ~2 us regardless of how little work
it does.

Those kernels are latency-bound, not bandwidth-bound, so the only way to remove
their cost is to remove the kernel boundaries. That was built and measured — see
the megakernel section: it comes out 0.64x, because a persistent cooperative grid
costs the bandwidth-bound GEMVs (87% of the step) more occupancy than the
boundaries were ever worth.

So the kernel-side work is essentially finished. The remaining lever is bytes,
not kernels: at batch 1 the floor is set by streaming 2.44 GB of weights per
token, and int8/fp8 weights would roughly halve the 1.62 ms floor itself.

Batching is the other lever that is already there: weights are streamed once per
step regardless of how many sequences share it, so throughput scales with batch
size until the GEMMs stop being memory-bound.

## Limitations

- LLM-ASR cards only. CTC cards have no autoregressive loop;
  `enable_fused_decoding` raises on them.
- Assumes the omniASR decoder shape: pre-norm, RMSNorm, full-width interleaved
  RoPE, SwiGLU, no grouped-query attention, no biases. `_validate` checks all of
  this at construction and raises rather than silently computing something else.
- `max_seq_len` (default 4096) reserves the KV cache. It must cover encoder
  context plus generated tokens; ~50 frames/s means the 40 s input cap needs
  ~2000 for context. Overflow raises with a clear message.
- Assumes one active state bag at a time, which is how the ASR beam search
  drives it. Concurrent decodes sharing a decoder instance would alias the cache.
- A distinct batch size triggers one graph capture (~100 ms) the first time.

## Reproducing

```bash
CUDA_VISIBLE_DEVICES=3 .venv/bin/python cuda13/bench/roofline.py          # hardware limits
CUDA_VISIBLE_DEVICES=3 .venv/bin/python cuda13/bench/profile_baseline.py  # where time goes
CUDA_VISIBLE_DEVICES=3 .venv/bin/python cuda13/bench/test_kernels.py      # kernel correctness
CUDA_VISIBLE_DEVICES=3 .venv/bin/python cuda13/bench/test_fused_decoder.py
CUDA_VISIBLE_DEVICES=3 .venv/bin/python cuda13/bench/test_equivalence.py  # identical output
CUDA_VISIBLE_DEVICES=3 .venv/bin/python cuda13/bench/bench_e2e.py         # the table above
CUDA_VISIBLE_DEVICES=3 .venv/bin/python cuda13/bench/bench_batch.py       # batch scaling
```

`test_equivalence.py` covers single clips, 30 s clips, a missing language code,
and ragged batches; it passes on both `omniASR_LLM_300M_v2` and
`omniASR_LLM_7B_v2`.

Benchmarks assume an idle GPU with locked clocks
(`nvidia-smi -i 3 --lock-gpu-clocks=2430`); the card otherwise idles at 180 MHz
and the first runs of a burst measure the clock ramp rather than the code.
