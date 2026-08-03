# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Triton kernels for the omniASR Llama decode step.

Every kernel here targets batch-1-ish autoregressive decoding, where the whole
step is bound by streaming weights and KV cache from HBM. They are written so
the entire step can be captured in a CUDA graph: no host-side reads of device
state, and the current KV length is passed as a *device* tensor rather than a
Python int.
"""

from __future__ import annotations

import math

import torch
import triton
import triton.language as tl
from torch import Tensor


# --------------------------------------------------------------------------
# RMSNorm, optionally fused with the preceding residual add.
#
# The decoder layer is pre-norm, so the pattern is always
#     x = x + delta ; h = rmsnorm(x) * w
# Fusing them means x is read once instead of three times.
# --------------------------------------------------------------------------
@triton.jit
def _add_rmsnorm_kernel(
    X,          # [R, D]  running residual stream, updated in place
    DELTA,      # [R, D]  sublayer output to add (ignored if HAS_DELTA=False)
    W,          # [D]     norm weight
    Y,          # [R, D]  normalized output
    stride_x,
    stride_d,
    stride_y,
    D: tl.constexpr,
    EPS: tl.constexpr,
    HAS_DELTA: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    cols = tl.arange(0, BLOCK)
    mask = cols < D

    x = tl.load(X + row * stride_x + cols, mask=mask, other=0.0).to(tl.float32)
    if HAS_DELTA:
        d = tl.load(DELTA + row * stride_d + cols, mask=mask, other=0.0).to(tl.float32)
        x = x + d
        tl.store(X + row * stride_x + cols, x.to(X.dtype.element_ty), mask=mask)

    # Matches fairseq2 / torch rms_norm: epsilon inside the sqrt, fp32 accum.
    var = tl.sum(x * x, axis=0) / D
    x = x * tl.rsqrt(var + EPS)

    w = tl.load(W + cols, mask=mask, other=0.0).to(tl.float32)
    tl.store(Y + row * stride_y + cols, (x * w).to(Y.dtype.element_ty), mask=mask)


def add_rmsnorm(
    x: Tensor, weight: Tensor, eps: float, delta: Tensor | None = None,
    out: Tensor | None = None,
) -> Tensor:
    """``x += delta`` (in place, if given) then return ``rmsnorm(x) * weight``."""
    r, d = x.shape
    out = torch.empty_like(x) if out is None else out
    block = triton.next_power_of_2(d)
    _add_rmsnorm_kernel[(r,)](
        x, delta if delta is not None else x, weight, out,
        x.stride(0), delta.stride(0) if delta is not None else 0, out.stride(0),
        D=d, EPS=eps, HAS_DELTA=delta is not None, BLOCK=block,
        num_warps=8 if block >= 4096 else 4,
    )
    return out


# --------------------------------------------------------------------------
# SwiGLU over a packed [gate | up] buffer produced by a single GEMM.
# --------------------------------------------------------------------------
@triton.jit
def _swiglu_kernel(
    GU,         # [R, 2*I]  gate in [:, :I], up in [:, I:]
    OUT,        # [R, I]
    stride_gu,
    stride_out,
    I,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    off = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
    mask = off < I

    g = tl.load(GU + row * stride_gu + off, mask=mask, other=0.0).to(tl.float32)
    u = tl.load(GU + row * stride_gu + I + off, mask=mask, other=0.0).to(tl.float32)

    tl.store(OUT + row * stride_out + off,
             (g * tl.sigmoid(g) * u).to(OUT.dtype.element_ty), mask=mask)


def swiglu(gate_up: Tensor, out: Tensor | None = None) -> Tensor:
    """SiLU(gate) * up for a packed ``[R, 2*I]`` tensor."""
    r, two_i = gate_up.shape
    i = two_i // 2
    out = gate_up.new_empty((r, i)) if out is None else out
    block = 1024
    _swiglu_kernel[(r, triton.cdiv(i, block))](
        gate_up, out, gate_up.stride(0), out.stride(0), i, BLOCK=block, num_warps=4,
    )
    return out


# --------------------------------------------------------------------------
# RoPE on q and k for a single decode step, fused with the KV cache write.
#
# fairseq2's RotaryEncoder is the interleaved ("GPT-J") variant: it views the
# head dim as complex pairs (x[2i], x[2i+1]) and multiplies by e^{i*p*freq_i}
# with freq_i = theta^(-2i/Dh). Position p is the state bag's step number.
# --------------------------------------------------------------------------
@triton.jit
def _rope_write_kv_kernel(
    QKV,        # [R, 3*H*Dh]  q | k | v, updated in place for q and k
    K_CACHE,    # [R, H, MAX, Dh]
    V_CACHE,    # [R, H, MAX, Dh]
    POS,        # [] int32, current position (device-side, graph safe)
    stride_qkv,
    stride_cr, stride_ch, stride_ct,
    H: tl.constexpr,
    DH: tl.constexpr,
    HALF: tl.constexpr,
    LOG_THETA: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)

    pos = tl.load(POS).to(tl.float32)

    i = tl.arange(0, HALF)
    # freq_i = theta ** (-2i / DH)
    freq = tl.exp(-(2.0 * i.to(tl.float32) / DH) * LOG_THETA)
    ang = pos * freq
    cos, sin = tl.cos(ang), tl.sin(ang)

    q_base = QKV + row * stride_qkv + head * DH
    k_base = q_base + H * DH
    v_base = q_base + 2 * H * DH

    qe = tl.load(q_base + 2 * i).to(tl.float32)
    qo = tl.load(q_base + 2 * i + 1).to(tl.float32)
    tl.store(q_base + 2 * i, (qe * cos - qo * sin).to(QKV.dtype.element_ty))
    tl.store(q_base + 2 * i + 1, (qe * sin + qo * cos).to(QKV.dtype.element_ty))

    ke = tl.load(k_base + 2 * i).to(tl.float32)
    ko = tl.load(k_base + 2 * i + 1).to(tl.float32)
    kre = (ke * cos - ko * sin).to(QKV.dtype.element_ty)
    kro = (ke * sin + ko * cos).to(QKV.dtype.element_ty)
    tl.store(k_base + 2 * i, kre)
    tl.store(k_base + 2 * i + 1, kro)

    # Append to the KV cache at `pos`.
    ipos = tl.load(POS)
    kc = K_CACHE + row * stride_cr + head * stride_ch + ipos * stride_ct
    vc = V_CACHE + row * stride_cr + head * stride_ch + ipos * stride_ct
    tl.store(kc + 2 * i, kre)
    tl.store(kc + 2 * i + 1, kro)

    v = tl.load(v_base + tl.arange(0, DH))
    tl.store(vc + tl.arange(0, DH), v)


def rope_write_kv(
    qkv: Tensor, k_cache: Tensor, v_cache: Tensor, pos: Tensor,
    n_heads: int, head_dim: int, theta: float,
) -> None:
    """Apply RoPE to q/k in ``qkv`` in place and append k/v to the cache."""
    r = qkv.shape[0]
    _rope_write_kv_kernel[(r, n_heads)](
        qkv, k_cache, v_cache, pos,
        qkv.stride(0),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        H=n_heads, DH=head_dim, HALF=head_dim // 2,
        LOG_THETA=math.log(theta), num_warps=4,
    )


# --------------------------------------------------------------------------
# Single-query attention over the KV cache (flash-decoding style).
#
# The number of keys is read from a device tensor so the kernel can live inside
# a CUDA graph. Work is split across the sequence to get enough parallelism:
# batch*heads alone is only 8-40 programs on a 188-SM GPU.
# --------------------------------------------------------------------------
@triton.jit
def _decode_attn_split_kernel(
    Q,          # [R, H*DH]
    K_CACHE, V_CACHE,   # [R, H, MAX, DH]
    PART_O,     # [R, H, SPLITS, DH]  fp32 partial outputs
    PART_M,     # [R, H, SPLITS]      fp32 running max
    PART_L,     # [R, H, SPLITS]      fp32 running sum
    LEN,        # [] int32, number of valid keys
    stride_q,
    stride_cr, stride_ch, stride_ct,
    stride_por, stride_poh, stride_pos,
    stride_pmr, stride_pmh,
    SCALE: tl.constexpr,
    H: tl.constexpr,
    DH: tl.constexpr,
    SPLITS: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)
    split = tl.program_id(2)

    n = tl.load(LEN)
    # Contiguous chunk of the sequence for this split.
    per = tl.cdiv(n, SPLITS)
    lo = split * per
    hi = tl.minimum(lo + per, n)

    d = tl.arange(0, DH)
    q = tl.load(Q + row * stride_q + head * DH + d).to(tl.float32) * SCALE

    m_i = float("-inf")
    l_i = 0.0
    acc = tl.zeros([DH], dtype=tl.float32)

    kv_base = row * stride_cr + head * stride_ch
    for start in range(lo, hi, BLOCK_N):
        offs = start + tl.arange(0, BLOCK_N)
        valid = offs < hi
        k = tl.load(
            K_CACHE + kv_base + offs[:, None] * stride_ct + d[None, :],
            mask=valid[:, None], other=0.0,
        ).to(tl.float32)
        # [BLOCK_N]
        s = tl.sum(k * q[None, :], axis=1)
        s = tl.where(valid, s, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(s, axis=0))
        alpha = tl.exp(m_i - m_new)
        p = tl.exp(s - m_new)

        v = tl.load(
            V_CACHE + kv_base + offs[:, None] * stride_ct + d[None, :],
            mask=valid[:, None], other=0.0,
        ).to(tl.float32)

        acc = acc * alpha + tl.sum(p[:, None] * v, axis=0)
        l_i = l_i * alpha + tl.sum(p, axis=0)
        m_i = m_new

    po = PART_O + row * stride_por + head * stride_poh + split * stride_pos
    tl.store(po + d, acc)
    pm = PART_M + row * stride_pmr + head * stride_pmh + split
    tl.store(pm, m_i)
    tl.store(PART_L + row * stride_pmr + head * stride_pmh + split, l_i)


@triton.jit
def _decode_attn_reduce_kernel(
    PART_O, PART_M, PART_L,
    OUT,        # [R, H*DH]
    stride_por, stride_poh, stride_pos,
    stride_pmr, stride_pmh,
    stride_o,
    H: tl.constexpr,
    DH: tl.constexpr,
    SPLITS: tl.constexpr,
):
    row = tl.program_id(0)
    head = tl.program_id(1)

    d = tl.arange(0, DH)
    s = tl.arange(0, SPLITS)

    m = tl.load(PART_M + row * stride_pmr + head * stride_pmh + s)
    l = tl.load(PART_L + row * stride_pmr + head * stride_pmh + s)
    # Splits past the end of the sequence contribute nothing.
    m = tl.where(l > 0.0, m, float("-inf"))

    m_max = tl.max(m, axis=0)
    scale = tl.where(l > 0.0, tl.exp(m - m_max), 0.0)
    l_tot = tl.sum(l * scale, axis=0)

    o = tl.load(
        PART_O + row * stride_por + head * stride_poh + s[:, None] * stride_pos + d[None, :]
    )
    acc = tl.sum(o * scale[:, None], axis=0) / l_tot

    tl.store(OUT + row * stride_o + head * DH + d, acc.to(OUT.dtype.element_ty))


def decode_attention(
    q: Tensor, k_cache: Tensor, v_cache: Tensor, seq_len: Tensor,
    n_heads: int, head_dim: int, workspace: dict, out: Tensor | None = None,
) -> Tensor:
    """Attention for a single query position against ``seq_len`` cached keys.

    ``seq_len`` is a device tensor, so this is safe to capture in a CUDA graph.
    """
    r = q.shape[0]
    splits = workspace["splits"]
    part_o, part_m, part_l = workspace["o"], workspace["m"], workspace["l"]
    out = torch.empty_like(q) if out is None else out

    _decode_attn_split_kernel[(r, n_heads, splits)](
        q, k_cache, v_cache, part_o, part_m, part_l, seq_len,
        q.stride(0),
        k_cache.stride(0), k_cache.stride(1), k_cache.stride(2),
        part_o.stride(0), part_o.stride(1), part_o.stride(2),
        part_m.stride(0), part_m.stride(1),
        SCALE=1.0 / math.sqrt(head_dim),
        H=n_heads, DH=head_dim, SPLITS=splits, BLOCK_N=workspace["block_n"],
        num_warps=workspace["num_warps"], num_stages=workspace["num_stages"],
    )
    _decode_attn_reduce_kernel[(r, n_heads)](
        part_o, part_m, part_l, out,
        part_o.stride(0), part_o.stride(1), part_o.stride(2),
        part_m.stride(0), part_m.stride(1),
        out.stride(0),
        H=n_heads, DH=head_dim, SPLITS=splits, num_warps=4,
    )
    return out


def make_attn_workspace(
    rows: int, n_heads: int, head_dim: int, device, splits: int = 8, block_n: int = 64,
    num_warps: int = 8, num_stages: int = 2,
) -> dict:
    return {
        "splits": splits,
        "block_n": block_n,
        "num_warps": num_warps,
        "num_stages": num_stages,
        "o": torch.zeros(rows, n_heads, splits, head_dim, device=device, dtype=torch.float32),
        "m": torch.zeros(rows, n_heads, splits, device=device, dtype=torch.float32),
        "l": torch.zeros(rows, n_heads, splits, device=device, dtype=torch.float32),
    }
