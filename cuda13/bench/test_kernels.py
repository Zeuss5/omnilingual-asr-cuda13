"""Correctness tests for the fused Triton kernels against torch references.

    CUDA_VISIBLE_DEVICES=3 .venv/bin/python cuda13/bench/test_kernels.py
"""

from __future__ import annotations

import math
import sys

import torch
import torch.nn.functional as F

from omnilingual_asr.fused import kernels as K

torch.manual_seed(0)
dev, dt = "cuda", torch.bfloat16
fails = []


def check(name, got, ref, tol=2e-2):
    got, ref = got.float(), ref.float()
    denom = ref.abs().max().clamp_min(1e-6)
    err = (got - ref).abs().max() / denom
    ok = err.item() < tol
    print(f"  [{'PASS' if ok else 'FAIL'}] {name:<34} rel_err={err.item():.2e}")
    if not ok:
        fails.append(name)


D, R = 4096, 5
EPS = 1e-5

print("== add_rmsnorm ==")
x = torch.randn(R, D, device=dev, dtype=dt)
w = torch.randn(D, device=dev, dtype=dt)
ref = F.rms_norm(x.clone(), (D,), w, EPS)
check("rmsnorm (no residual)", K.add_rmsnorm(x.clone(), w, EPS), ref)

x = torch.randn(R, D, device=dev, dtype=dt)
delta = torch.randn(R, D, device=dev, dtype=dt)
x_ref = (x.float() + delta.float()).to(dt)
ref = F.rms_norm(x_ref, (D,), w, EPS)
x_run = x.clone()
got = K.add_rmsnorm(x_run, w, EPS, delta=delta)
check("add_rmsnorm output", got, ref)
check("add_rmsnorm updates x in place", x_run, x_ref)

print("== swiglu ==")
I = 2816
gu = torch.randn(R, 2 * I, device=dev, dtype=dt)
ref = F.silu(gu[:, :I].float()) * gu[:, I:].float()
check("swiglu", K.swiglu(gu), ref)

print("== rope + kv cache write ==")
H, DH, MAX, THETA = 8, 512, 512, 10000.0
pos_i = 137
qkv = torch.randn(R, 3 * H * DH, device=dev, dtype=dt)
kc = torch.zeros(R, H, MAX, DH, device=dev, dtype=dt)
vc = torch.zeros(R, H, MAX, DH, device=dev, dtype=dt)
pos = torch.tensor(pos_i, device=dev, dtype=torch.int32)


def rope_ref(t, p):
    """fairseq2 RotaryEncoder semantics: interleaved complex pairs."""
    idx = torch.arange(0, DH, 2, device=dev, dtype=torch.float32)
    freqs = 1.0 / (THETA ** (idx / DH))
    ang = p * freqs
    cf = torch.polar(torch.ones_like(ang), ang)
    c = torch.view_as_complex(t.float().unflatten(-1, (-1, 2)))
    return torch.view_as_real(c * cf).flatten(-2).to(t.dtype)


q_ref = rope_ref(qkv[:, : H * DH].unflatten(-1, (H, DH)), pos_i)
k_ref = rope_ref(qkv[:, H * DH : 2 * H * DH].unflatten(-1, (H, DH)), pos_i)
v_ref = qkv[:, 2 * H * DH :].unflatten(-1, (H, DH))

qkv_run = qkv.clone()
K.rope_write_kv(qkv_run, kc, vc, pos, H, DH, THETA)
check("rope(q)", qkv_run[:, : H * DH].unflatten(-1, (H, DH)), q_ref)
check("rope(k)", qkv_run[:, H * DH : 2 * H * DH].unflatten(-1, (H, DH)), k_ref)
check("k written to cache at pos", kc[:, :, pos_i, :], k_ref)
check("v written to cache at pos", vc[:, :, pos_i, :], v_ref)

print("== decode attention ==")
for n_ctx in (1, 37, 200, 511):
    kc = torch.randn(R, H, MAX, DH, device=dev, dtype=dt)
    vc = torch.randn(R, H, MAX, DH, device=dev, dtype=dt)
    q = torch.randn(R, H * DH, device=dev, dtype=dt)
    ln = torch.tensor(n_ctx, device=dev, dtype=torch.int32)
    ws = K.make_attn_workspace(R, H, DH, dev)
    got = K.decode_attention(q, kc, vc, ln, H, DH, ws)

    qh = q.unflatten(-1, (H, DH)).unsqueeze(2)          # [R,H,1,DH]
    ref = F.scaled_dot_product_attention(
        qh.float(), kc[:, :, :n_ctx].float(), vc[:, :, :n_ctx].float()
    ).squeeze(2).flatten(-2)
    check(f"decode_attention n={n_ctx}", got, ref, tol=3e-2)

print()
if fails:
    print(f"FAILED: {fails}")
    sys.exit(1)
print("All kernel tests passed.")
