# Persistent cooperative megakernel (built, measured, rejected)

Working code, kept because the negative result is worth more than the guess that
preceded it. **Do not re-attempt without reading the numbers below.**

`mega.cu` runs an entire omniASR decode step — 12 layers of {RMSNorm, packed QKV
+ RoPE + KV write, single-query attention, o_proj, RMSNorm, gate/up + SwiGLU,
down_proj} plus the final norm — inside **one** cooperative kernel, replacing
every kernel boundary with `cg::grid_group::sync()`.

It is correct: max relative error 8.2e-03 against the fused decoder, which is
bf16 noise.

## Result

| | ms/step |
| --- | --- |
| current fused path + CUDA graph | **1.891** |
| megakernel | 2.935 (**0.64×**) |

## Why, and why it is not an implementation bug

Splitting the megakernel's time apart:

| | megakernel | as separate kernels |
| --- | --- | --- |
| GEMV stages | 2.164 ms | 1.770 ms |
| attention + norms + syncs | 0.771 ms | 0.175 ms |

`gemv_only_mega` in `mega.cu` is the diagnostic: the same 48 GEMVs, still in a
persistent cooperative grid, with attention and the norms deleted entirely. It
still runs **0.82× (22% slower)** than the same GEMVs as 48 ordinary launches.

That is the whole story. The GEMVs are 87% of the step and are bandwidth-bound,
so they need maximum memory-level parallelism, and a cooperative grid takes it
away twice over:

* **Occupancy.** A megakernel's register footprint is the *union* of all its
  stages — 64 registers here, capping it at 4 blocks/SM (32 warps/SM). The
  standalone GEMV uses ~32-48 and reaches ~64 warps/SM.
* **Grid size.** Cooperative launch requires every block co-resident: 752 blocks
  for the full kernel, 940 for the GEMV-only variant. The standalone GEMV
  launches 1536 blocks and lets the scheduler run them in waves at full
  occupancy. The persistent version must grid-stride instead, serializing rows
  within a warp.

So the structure penalizes exactly the part that dominates. Even with a
*perfect, free* attention stage the megakernel could not go below 2.164 ms —
already 1.14× the current total of 1.891 ms.

The attention stage here is also genuinely slow (0.771 ms vs 0.175 ms), being
one block per (head, split) with a `__syncthreads()` per key. That part is
fixable; it does not matter, because of the line above.

## The budget was never large

`probe.cu` / `probe.py` measure the premise directly. Inside an already-captured
CUDA graph a kernel boundary costs **0.9-1.1 µs and is flat in grid size** — the
graph already removed launch dispatch, leaving a pipeline drain the hardware does
cheaply. A cooperative `grid.sync()` is a software barrier across all resident
blocks and *grows*: 1.03 µs at 188 blocks, 1.39 µs at 512, 2.96 µs at 1024.

Independently, summing the profiler's per-kernel self-times gives 1.852 ms
against a measured 1.907 ms step, so all boundary and gap overhead in the
current path is **0.055 ms — 2.9%**. That was the entire prize.

## Running it

```bash
CUDA_HOME=/usr/local/cuda-13.0 CUDA_VISIBLE_DEVICES=3 \
  .venv/bin/python cuda13/experiments/megakernel/run.py    # correctness + timing
CUDA_HOME=/usr/local/cuda-13.0 CUDA_VISIBLE_DEVICES=3 \
  .venv/bin/python cuda13/experiments/megakernel/probe.py  # boundary vs grid.sync
```
