#!/usr/bin/env bash
# Build fairseq2n 0.6 from source against PyTorch 2.10 / CUDA 13.0.
#
# Why this exists: the fairseq2n 0.6 wheel on PyPI pins torch==2.8.0 and links
# libcudart.so.12, so there is no prebuilt path to CUDA 13. We build it against
# the cu130 torch instead, with two changes on top of the v0.6 tag:
#
#   1. pybind11 is bumped 2.11.1 -> 3.0.1 to match the copy torch 2.10 ships.
#   2. patches/0001-...patch fixes fairseq2n's hand-rolled THPVariable struct.
#      torch 2.10 changed `cdata` from at::MaybeOwned<at::Tensor> to a plain
#      at::Tensor, so the old layout segfaulted on every Python->C++ tensor
#      conversion (i.e. any tensor entering a fairseq2 data pipeline).
#
# Prerequisites: CUDA 13.0 toolkit, libsndfile1-dev, and a venv with the cu130
# torch already installed.
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV="${VENV:-$PROJECT_ROOT/.venv}"
CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
SRC="$PROJECT_ROOT/cuda13/fairseq2"
# RTX PRO 6000 Blackwell is sm_120. Override for other cards.
CUDA_ARCHS="${CUDA_ARCHS:-120-real;120-virtual}"
JOBS="${JOBS:-32}"

export PATH="$VENV/bin:$CUDA_HOME/bin:$PATH"
export CUDA_HOME

if [ ! -d "$SRC" ]; then
    echo "==> Cloning fairseq2 v0.6"
    git clone --depth 1 --branch v0.6.0 --recurse-submodules \
        https://github.com/facebookresearch/fairseq2.git "$SRC"

    echo "==> Bumping vendored pybind11 to v3.0.1 (matches torch 2.10)"
    git -C "$SRC/native/third-party/pybind11" \
        fetch --depth 1 origin refs/tags/v3.0.1:refs/tags/v3.0.1
    git -C "$SRC/native/third-party/pybind11" checkout -q v3.0.1

    echo "==> Applying patches"
    git -C "$SRC" apply "$PROJECT_ROOT/cuda13/patches/"*.patch
fi

echo "==> Configuring (CUDA $("$CUDA_HOME/bin/nvcc" --version | tail -1), arch $CUDA_ARCHS)"
cmake -GNinja \
    -DFAIRSEQ2N_USE_CUDA=ON \
    -DCMAKE_CUDA_ARCHITECTURES="$CUDA_ARCHS" \
    -DFAIRSEQ2N_PYTHON_DEVEL=OFF \
    -DFAIRSEQ2N_INSTALL_STANDALONE=ON \
    -DCMAKE_BUILD_TYPE=Release \
    -S "$SRC/native" -B "$SRC/native/build"

echo "==> Building"
cmake --build "$SRC/native/build" -j "$JOBS"

echo "==> Installing fairseq2n + fairseq2 into $VENV"
VIRTUAL_ENV="$VENV" uv pip install --no-build-isolation \
    --reinstall-package fairseq2n "$SRC/native/python"
VIRTUAL_ENV="$VENV" uv pip install --no-build-isolation "$SRC[arrow]"

echo "==> Verifying"
"$VENV/bin/python" - <<'PY'
import torch, fairseq2n
from fairseq2.data.data_pipeline import read_sequence
assert fairseq2n.torch_version().startswith("2.10"), fairseq2n.torch_version()
assert fairseq2n.supports_cuda() and fairseq2n.cuda_version()[0] == 13
# The regression the patch fixes: a tensor entering a native data pipeline.
(t,) = list(read_sequence([torch.randn(4)]).map(lambda x: x * 2).and_return())
print(f"OK  torch={torch.__version__}  fairseq2n built for "
      f"{fairseq2n.torch_version()} / CUDA {fairseq2n.cuda_version()}")
PY
