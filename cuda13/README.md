# Running omnilingual-asr on CUDA 13.0

Upstream omnilingual-asr cannot run on CUDA 13: it requires `fairseq2 <= 0.6.0`,
which requires `fairseq2n == 0.6`, whose PyPI wheel pins `torch == 2.8.0` and links
`libcudart.so.12`. No published `fairseq2n` build targets CUDA 13 (the newest,
0.8.1, still pins torch 2.9.1). The fix is to build `fairseq2n` 0.6 from source
against a cu130 torch, with two corrections.

Verified on an NVIDIA RTX PRO 6000 Blackwell (sm_120), driver 580.126.09.

## Resulting stack

| component | version |
| --- | --- |
| CUDA toolkit | 13.0.88 (`/usr/local/cuda-13.0`) |
| torch | 2.10.0+cu130 |
| torchaudio | 2.10.0+cu130 |
| fairseq2 / fairseq2n | 0.6 (built from source) |
| Python | 3.12 |

## The two upstream problems

**1. `THPVariable` layout drift — the important one.**
`fairseq2n` does not use torch's public tensor API. It re-declares torch's
internal `THPVariable` struct in
`native/python/src/fairseq2n/bindings/type_casters/torch.cc` and reaches into it
directly:

```cpp
struct THPVariable { PyObject_HEAD at::MaybeOwned<at::Tensor> cdata; };
...
value = *reinterpret_cast<THPVariable *>(ptr)->cdata;
```

In torch 2.10 that member is a plain `at::Tensor`, not an
`at::MaybeOwned<at::Tensor>`. The old declaration therefore reads a differently
shaped object and dereferences garbage, so **every** Python→C++ tensor
conversion segfaults. In practice that means any tensor entering a fairseq2 data
pipeline — which is the first thing `ASRInferencePipeline.transcribe` does.
Notably it compiles and links without warning; it only fails at runtime.

`patches/0001-fairseq2n-thpvariable-layout-torch210.patch` mirrors the current
layout behind a `TORCH_VERSION` guard, so the source still builds against older
torch. The real header cannot simply be included: it pulls in
`torch/csrc/utils/pybind.h`, which declares a competing
`type_caster<at::Tensor>`.

**2. Vendored pybind11 is older than torch's.**
`fairseq2n` 0.6 vendors pybind11 2.11.1 while torch 2.10 ships 3.0.1. The build
script checks out v3.0.1 in the submodule so both agree on pybind11 internals.
This alone does not fix problem 1, but leaving them mismatched is a hazard.

Also note `torchaudio` must be pinned to **2.10.0** — pip resolves 2.11.0 by
default, which needs torch 2.11 and fails to load with
`undefined symbol: torch_dtype_float4_e2m1fn_x2`.

## Setup

### Prerequisites

| | |
| --- | --- |
| CUDA toolkit | **13.0**, with `nvcc` (default `/usr/local/cuda-13.0`) |
| NVIDIA driver | one that supports CUDA 13 (tested on 580.126.09) |
| Python | 3.10-3.13 (built and tested on 3.12) |
| system packages | `libsndfile1-dev` |
| tooling | [`uv`](https://docs.astral.sh/uv/), `git`, a C++17 compiler |

The CUDA toolkit must be a real install, not just the pip `nvidia-*` wheels —
`fairseq2n` is compiled against its headers. If `nvcc` is not on `PATH`, set
`CUDA_HOME` (the build script also probes `/usr/local/cuda` and
`/usr/local/cuda-13.0`).

### Install

Run from the repository root. Every step matters in this order — in particular
`--no-deps` in step 5, which stops pip from replacing the cu130 torch with a
default-index build.

```bash
# 1. system dependency for fairseq2n's audio decoder
sudo apt-get install -y libsndfile1-dev
export CUDA_HOME=/usr/local/cuda-13.0

# 2. virtual environment
uv venv --python 3.12

# 3. torch + torchaudio for CUDA 13.
#    torchaudio MUST be pinned to 2.10.0: pip otherwise resolves 2.11.0, which
#    needs torch 2.11 and fails with `undefined symbol: torch_dtype_float4_e2m1fn_x2`.
uv pip install torch==2.10.0 torchaudio==2.10.0 \
    --index-url https://download.pytorch.org/whl/cu130

# 4. native build dependencies
uv pip install "cmake~=3.31" "ninja~=1.11" "tbb-devel==2021.8" \
    "setuptools~=80.9" "wheel~=0.45"

# 5. build and install fairseq2n + fairseq2 (clone, patch, cmake, install, verify)
./cuda13/build_fairseq2n_cu130.sh

# 6. omnilingual-asr itself
uv pip install --no-build-isolation --no-deps -e .

# 7. remaining runtime dependencies
uv pip install numba pandas soundfile librosa
```

Step 5 takes roughly 10 minutes: it clones fairseq2 v0.6, checks out pybind11
v3.0.1, applies `patches/`, and compiles ~190 targets. `CUDA_ARCHS` defaults to
`120-real;120-virtual` (Blackwell / sm_120) — override it for another GPU, e.g.
`CUDA_ARCHS="90-real;90-virtual" ./cuda13/build_fairseq2n_cu130.sh` for H100.
`VENV` and `JOBS` are also overridable. The build tree lands in
`cuda13/fairseq2/` and is gitignored (~282 MB), so a fresh clone rebuilds it.

Step 6 uses `--no-deps` deliberately. `pyproject.toml` lists an unpinned `torch`,
and a normal install would pull the default-index (CUDA 12) wheel over the cu130
one. Step 7 then adds the runtime packages that `--no-deps` skipped; `kenlm` is
listed upstream but is not needed for inference.

### Resulting versions

```
torch 2.10.0+cu130   torchaudio 2.10.0+cu130   triton 3.6.0
fairseq2 0.6         fairseq2n 0.6 (source)    numpy 1.26.4
```

### Using the fused decode path

Optional, and only applies to the LLM-ASR cards:

```python
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
from omnilingual_asr.fused import enable_fused_decoding

pipe = ASRInferencePipeline(model_card="omniASR_LLM_300M_v2")
enable_fused_decoding(pipe)
print(pipe.transcribe(["audio.wav"], lang=["eng_Latn"]))
```

It JIT-compiles a small CUDA extension on first use (a few seconds, then cached
in `~/.cache/torch_extensions`). If `nvcc` or `ninja` is unavailable it falls
back to `F.linear` and still works. See
[OPTIMIZATION.md](OPTIMIZATION.md).

## Verifying

```bash
CUDA_VISIBLE_DEVICES=3 .venv/bin/python cuda13/verify_cuda13.py
```

Checks the toolchain versions, the tensor round-trip that used to segfault, and
runs both an LLM and a CTC model against a reference transcription.

## Performance

Getting it running was the first half. See **[OPTIMIZATION.md](OPTIMIZATION.md)**
for the fused decode path in `src/omnilingual_asr/fused/`, which cuts end-to-end
latency 2.25× (300M) / 1.87× (7B) with byte-identical transcriptions, and
`bench/` for the benchmarks and equivalence tests behind those numbers.
