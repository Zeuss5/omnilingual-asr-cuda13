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

## Reproducing

```bash
apt-get install -y libsndfile1-dev
uv pip install torch==2.10.0 torchaudio==2.10.0 --index-url https://download.pytorch.org/whl/cu130
uv pip install "cmake~=3.31" "ninja~=1.11" "tbb-devel==2021.8"
./cuda13/build_fairseq2n_cu130.sh          # clones, patches, builds, installs
uv pip install --no-build-isolation --no-deps -e .
```

`CUDA_ARCHS` defaults to `120-real;120-virtual` (Blackwell); override for other
GPUs. The build tree lands in `cuda13/fairseq2/` and is gitignored.

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
