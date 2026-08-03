# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""JIT-compiled raw CUDA kernels for the decode step.

The bf16 GEMV in ``csrc/gemv.cu`` beats both cuBLAS and a tuned Triton
equivalent on every omniASR projection shape (see cuda13/OPTIMIZATION.md);
cuBLAS' ``gemvx`` path in particular only reaches 69-79% of peak bandwidth on
the smaller ``o``/``down`` matrices.

Compilation is lazy and best-effort: if nvcc or ninja is unavailable, or the
build fails, :func:`gemv` is ``None`` and the decoder falls back to
``F.linear``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

_ext: Any = None
_tried = False
_error: str | None = None


def _cuda_home() -> str | None:
    if os.environ.get("CUDA_HOME"):
        return os.environ["CUDA_HOME"]
    for c in ("/usr/local/cuda", "/usr/local/cuda-13.0"):
        if Path(c, "bin", "nvcc").exists():
            return c
    return None


def load_extension() -> Any:
    """Compile (once) and return the extension module, or ``None``."""
    global _ext, _tried, _error
    if _tried:
        return _ext
    _tried = True

    if not torch.cuda.is_available():
        _error = "no CUDA device"
        return None

    home = _cuda_home()
    if home is None:
        _error = "CUDA toolkit not found (set CUDA_HOME)"
        return None
    os.environ.setdefault("CUDA_HOME", home)
    os.environ["PATH"] = f"{home}/bin:" + os.environ.get("PATH", "")

    major, minor = torch.cuda.get_device_capability(0)
    arch = f"{major}{minor}"
    src = Path(__file__).parent / "csrc" / "gemv.cu"

    try:
        from torch.utils.cpp_extension import load

        _ext = load(
            name=f"omniasr_gemv_sm{arch}",
            sources=[str(src)],
            extra_cuda_cflags=[
                "-O3", "--use_fast_math",
                f"-gencode=arch=compute_{arch},code=sm_{arch}",
            ],
            verbose=False,
        )
    except Exception as e:  # pragma: no cover - depends on local toolchain
        _error = f"{type(e).__name__}: {e}"
        _ext = None
    return _ext


def gemv(x: torch.Tensor, w: torch.Tensor) -> torch.Tensor | None:
    """``x @ w.T`` for bf16 with a small leading dim, or ``None`` if unavailable."""
    ext = load_extension()
    if ext is None or x.shape[0] > 8 or x.dtype != torch.bfloat16:
        return None
    return ext.gemv_bf16(x, w)


def unavailable_reason() -> str | None:
    load_extension()
    return _error
