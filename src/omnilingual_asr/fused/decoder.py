# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A fused, CUDA-graph-captured replacement for the omniASR Llama decoder.

Motivation: at batch 1 the decode step issues ~940 kernel launches and the GPU
spends most of the step idle waiting on the host. Profiling showed CPU enqueue
time (2.93 ms) essentially equal to the GPU span (3.10 ms). This module attacks
both halves of that:

* **Fewer, larger kernels.** q/k/v are packed into one GEMM and gate/up into
  another (7 GEMMs per layer -> 4). RMSNorm absorbs the preceding residual add,
  RoPE is fused with the KV-cache write, and SwiGLU is a single pass.
* **No launch overhead.** The step runs entirely against preallocated buffers
  with the KV length carried in a *device* tensor, so it can be captured once as
  a CUDA graph and replayed with a single launch.

It is numerically equivalent to ``StandardTransformerLMDecoder`` for this
architecture: pre-norm, RMSNorm, interleaved RoPE over the full head dim,
SwiGLU, no grouped-query attention, no biases.

The module is inference-only and assumes decoding is sequential (one active
state bag at a time), which is how the ASR beam search drives it.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Module, Parameter, ParameterList

from fairseq2.nn import BatchLayout, IncrementalState, IncrementalStateBag

from omnilingual_asr.fused import cuda_ops, kernels as K


class FusedKVCache(IncrementalState):
    """Preallocated KV cache for every layer.

    Buffer addresses must stay fixed for the lifetime of a captured CUDA graph,
    so the cache is allocated once and reused across utterances; only ``cur_len``
    is reset. ``len_dev`` is the number of keys the attention kernel should read
    and is deliberately a device tensor so the kernel needs no host round trip.
    """

    def __init__(
        self, rows: int, n_layers: int, n_heads: int, head_dim: int,
        max_len: int, dtype: torch.dtype, device: torch.device,
    ) -> None:
        shape = (n_layers, rows, n_heads, max_len, head_dim)
        self.k = torch.zeros(shape, dtype=dtype, device=device)
        self.v = torch.zeros(shape, dtype=dtype, device=device)
        self.rows = rows
        self.max_len = max_len
        self.cur_len = 0
        # pos_dev: slot the next token writes to. len_dev: valid keys *after*
        # that write, i.e. cur_len + 1.
        self.pos_dev = torch.zeros((), dtype=torch.int32, device=device)
        self.len_dev = torch.ones((), dtype=torch.int32, device=device)
        # With nbest == 1 the beam search always passes the identity
        # permutation, so the reorder copy is pure waste.
        self.skip_reorder = False

    def reset(self) -> None:
        self.cur_len = 0
        self._sync()

    def _sync(self) -> None:
        self.pos_dev.fill_(self.cur_len)
        self.len_dev.fill_(self.cur_len + 1)

    def set_len(self, n: int) -> None:
        if n > self.max_len:
            raise ValueError(
                f"KV cache overflow: need {n} positions but only {self.max_len} "
                f"are reserved. Rebuild the fused decoder with a larger max_seq_len."
            )
        self.cur_len = n
        self._sync()

    def advance(self, n: int = 1) -> None:
        self.set_len(self.cur_len + n)

    def reorder(self, new_order: Tensor) -> None:
        if self.skip_reorder or self.cur_len == 0:
            return
        n = self.cur_len
        # Only the populated prefix matters; copying the whole reservation would
        # cost more than the decode step itself.
        self.k[:, :, :, :n].copy_(self.k[:, :, :, :n].index_select(1, new_order))
        self.v[:, :, :, :n].copy_(self.v[:, :, :, :n].index_select(1, new_order))

    def size_bytes(self) -> int:
        return 2 * self.k.element_size() * self.k[:, :, :, : self.cur_len].numel()

    def capacity_bytes(self) -> int:
        return 2 * self.k.element_size() * self.k.numel()


class FusedLlamaDecoder(Module):
    """Drop-in replacement for ``model.llama_decoder``.

    Keeps the ``(seqs, seqs_layout, state_bag=...)`` signature and the
    ``(N, S, M)`` output shape, so the existing beam search runs unchanged.
    """

    def __init__(
        self, ref: Module, *, max_seq_len: int = 4096, use_cuda_graph: bool = True,
        use_cuda_gemv: bool = True,
    ) -> None:
        super().__init__()

        layers = list(ref.layers)
        l0 = layers[0]
        attn0 = l0.self_attn

        _validate(ref, layers)

        self.n_layers = len(layers)
        self.n_heads = attn0.num_heads
        self.head_dim = attn0.head_dim
        self.model_dim = self.n_heads * self.head_dim
        self.theta = attn0.pos_encoder.theta
        self.eps = l0.self_attn_layer_norm.eps
        self.max_seq_len = max_seq_len
        self.use_cuda_graph = use_cuda_graph

        w0 = attn0.q_proj.weight
        self._device, self.dtype = w0.device, w0.dtype

        # --- pack weights: one GEMM for q|k|v, one for gate|up ---
        def plist(ws):
            return ParameterList([Parameter(w, requires_grad=False) for w in ws])

        self.w_qkv = plist([
            torch.cat([l.self_attn.q_proj.weight, l.self_attn.k_proj.weight,
                       l.self_attn.v_proj.weight], 0).contiguous() for l in layers])
        self.w_o = plist([l.self_attn.output_proj.weight.contiguous() for l in layers])
        self.w_gate_up = plist([
            torch.cat([l.ffn.gate_proj.weight, l.ffn.inner_proj.weight], 0).contiguous()
            for l in layers])
        self.w_down = plist([l.ffn.output_proj.weight.contiguous() for l in layers])
        self.attn_norm = plist([l.self_attn_layer_norm.weight for l in layers])
        self.ffn_norm = plist([l.ffn_layer_norm.weight for l in layers])
        self.final_norm = Parameter(ref.layer_norm.weight, requires_grad=False)

        self.inner_dim = self.w_gate_up[0].shape[0] // 2

        # Swept inside a CUDA graph (launch overhead hides the differences
        # otherwise). Small blocks + many splits win because batch*heads is only
        # 8 programs here, so parallelism has to come from the sequence axis.
        # 1.25-1.55x over SPLITS=8/BLOCK_N=64 across ctx 200-1500.
        self.attn_splits = 32
        self.attn_block_n = 16
        self.attn_warps = 2
        self.attn_stages = 3

        # Projections go through a raw-CUDA bf16 GEMV when it is available.
        # It reaches 1.41-1.47 TB/s against cuBLAS' 1.06-1.44 (cuBLAS' gemvx
        # path is only ~69% of peak on the smaller o/down matrices), and beats
        # a tuned Triton equivalent, which topped out at 1.32. Falls back to
        # F.linear if the toolchain cannot build the extension.
        self.use_cuda_gemv = use_cuda_gemv and cuda_ops.load_extension() is not None

        self._caches: dict[int, FusedKVCache] = {}
        self._graphs: dict[int, torch.cuda.CUDAGraph] = {}
        self._bufs: dict[int, dict] = {}
        # Set by the integration layer when the beam is 1 (identity reorder).
        self._skip_reorder = False

    # ------------------------------------------------------------------
    def _get_cache(self, bag: IncrementalStateBag, rows: int) -> FusedKVCache:
        cache = bag.maybe_get_state(self, FusedKVCache)
        if cache is not None:
            return cache

        cache = self._caches.get(rows)
        if cache is None:
            cache = FusedKVCache(
                rows, self.n_layers, self.n_heads, self.head_dim,
                self.max_seq_len, self.dtype, self._device,
            )
            self._caches[rows] = cache
        else:
            # Reuse the same buffers across utterances so the captured graph,
            # which baked in their addresses, stays valid.
            cache.reset()
        cache.skip_reorder = self._skip_reorder
        bag.set_state(self, cache)
        return cache

    def _get_bufs(self, rows: int) -> dict:
        b = self._bufs.get(rows)
        if b is None:
            dev, dt = self._device, self.dtype
            b = {
                "in": torch.zeros(rows, self.model_dim, device=dev, dtype=dt),
                "out": torch.zeros(rows, self.model_dim, device=dev, dtype=dt),
                "norm": torch.empty(rows, self.model_dim, device=dev, dtype=dt),
                "attn": torch.empty(rows, self.model_dim, device=dev, dtype=dt),
                "swiglu": torch.empty(rows, self.inner_dim, device=dev, dtype=dt),
                "ws": K.make_attn_workspace(
                    rows, self.n_heads, self.head_dim, dev,
                    splits=self.attn_splits, block_n=self.attn_block_n,
                    num_warps=self.attn_warps, num_stages=self.attn_stages,
                ),
            }
            self._bufs[rows] = b
        return b

    # ------------------------------------------------------------------
    # prefill: many tokens at once; dynamic shapes, plain torch ops
    # ------------------------------------------------------------------
    @torch.inference_mode()
    def _prefill(self, seqs: Tensor, cache: FusedKVCache) -> Tensor:
        _, s, _ = seqs.shape
        h, dh = self.n_heads, self.head_dim
        start = cache.cur_len
        if start + s > cache.max_len:
            raise ValueError(
                f"KV cache overflow during prefill: {start + s} > {cache.max_len}. "
                f"Rebuild the fused decoder with a larger max_seq_len.")

        pos = torch.arange(start, start + s, device=seqs.device, dtype=torch.float32)
        idx = torch.arange(0, dh, 2, device=seqs.device, dtype=torch.float32)
        cf = torch.polar(torch.ones(s, dh // 2, device=seqs.device),
                         torch.outer(pos, 1.0 / (self.theta ** (idx / dh))))

        def rope(t: Tensor) -> Tensor:                       # [N,S,H,Dh]
            c = torch.view_as_complex(t.float().unflatten(-1, (-1, 2)))
            return torch.view_as_real(c * cf[None, :, None, :]).flatten(-2).to(t.dtype)

        mask = None if start == 0 else _suffix_causal_mask(
            s, start + s, seqs.device, seqs.dtype)

        x = seqs
        for i in range(self.n_layers):
            res = x
            hn = F.rms_norm(x, (self.model_dim,), self.attn_norm[i], self.eps)
            q, k, v = F.linear(hn, self.w_qkv[i]).split(self.model_dim, dim=-1)
            q = rope(q.unflatten(-1, (h, dh))).transpose(1, 2)
            k = rope(k.unflatten(-1, (h, dh))).transpose(1, 2)
            v = v.unflatten(-1, (h, dh)).transpose(1, 2)

            cache.k[i, :, :, start : start + s] = k
            cache.v[i, :, :, start : start + s] = v

            attn = F.scaled_dot_product_attention(
                q, cache.k[i, :, :, : start + s], cache.v[i, :, :, : start + s],
                attn_mask=mask, is_causal=(mask is None and s > 1),
            )
            x = res + F.linear(attn.transpose(1, 2).flatten(-2), self.w_o[i])

            res = x
            hn = F.rms_norm(x, (self.model_dim,), self.ffn_norm[i], self.eps)
            g, u = F.linear(hn, self.w_gate_up[i]).split(self.inner_dim, dim=-1)
            x = res + F.linear(F.silu(g) * u, self.w_down[i])

        cache.set_len(start + s)
        return F.rms_norm(x, (self.model_dim,), self.final_norm, self.eps)

    # ------------------------------------------------------------------
    # decode: exactly one token, fully fused, CUDA-graph capturable
    # ------------------------------------------------------------------
    def _matmul(self, a: Tensor, w: Tensor) -> Tensor:
        if self.use_cuda_gemv:
            y = cuda_ops.gemv(a, w)
            if y is not None:
                return y
        return F.linear(a, w)

    def _decode_body(self, x: Tensor, cache: FusedKVCache, b: dict, out: Tensor) -> None:
        h, dh, D = self.n_heads, self.head_dim, self.model_dim
        norm, ws = b["norm"], b["ws"]
        mm = self._matmul
        # Each sublayer's residual add is deferred into the *next* RMSNorm, so
        # the running stream `x` is read and written once per sublayer.
        delta: Tensor | None = None

        for i in range(self.n_layers):
            K.add_rmsnorm(x, self.attn_norm[i], self.eps, delta=delta, out=norm)
            qkv = mm(norm, self.w_qkv[i])
            K.rope_write_kv(qkv, cache.k[i], cache.v[i], cache.pos_dev, h, dh, self.theta)
            attn = K.decode_attention(
                qkv[:, :D], cache.k[i], cache.v[i], cache.len_dev, h, dh, ws,
                out=b["attn"],
            )
            delta = mm(attn, self.w_o[i])

            K.add_rmsnorm(x, self.ffn_norm[i], self.eps, delta=delta, out=norm)
            gu = mm(norm, self.w_gate_up[i])
            K.swiglu(gu, out=b["swiglu"])
            delta = mm(b["swiglu"], self.w_down[i])

        K.add_rmsnorm(x, self.final_norm, self.eps, delta=delta, out=out)

    @torch.inference_mode()
    def _decode(self, seqs: Tensor, cache: FusedKVCache) -> Tensor:
        rows = seqs.shape[0]
        b = self._get_bufs(rows)
        x = seqs.squeeze(1)

        if not self.use_cuda_graph:
            run = x.clone()
            out = torch.empty_like(run)
            self._decode_body(run, cache, b, out)
            cache.advance()
            return out.unsqueeze(1)

        if rows not in self._graphs:
            self._capture(rows, cache, b)

        b["in"].copy_(x)
        self._graphs[rows].replay()
        cache.advance()
        # Clone: the static buffer is overwritten by the next replay.
        return b["out"].unsqueeze(1).clone()

    def _capture(self, rows: int, cache: FusedKVCache, b: dict) -> None:
        # Warm up on a side stream: capture cannot JIT Triton kernels or run
        # cuBLAS autotuning, so everything must already be resident.
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(3):
                self._decode_body(b["in"].clone(), cache, b, b["out"])
        torch.cuda.current_stream().wait_stream(s)
        torch.cuda.synchronize()

        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g), torch.inference_mode():
            self._decode_body(b["in"], cache, b, b["out"])
        self._graphs[rows] = g

    # ------------------------------------------------------------------
    def forward(
        self, seqs: Tensor, seqs_layout: BatchLayout | None = None, *,
        state_bag: IncrementalStateBag | None = None, **kw,
    ) -> Tensor:
        if state_bag is None:
            raise NotImplementedError("fused decoder is inference-only (needs a state bag)")

        cache = self._get_cache(state_bag, seqs.shape[0])
        if seqs.shape[1] > 1:
            return self._prefill(seqs, cache)
        return self._decode(seqs, cache)


def _validate(ref: Module, layers: list) -> None:
    """Fail loudly if the reference decoder is not the shape this fusion assumes.

    Every deviation here would silently produce wrong output rather than an
    error, so check them up front rather than trusting the model card.
    """
    from fairseq2.models.transformer import TransformerNormOrder
    from fairseq2.nn import RMSNorm

    if not isinstance(getattr(ref, "layer_norm", None), RMSNorm):
        raise NotImplementedError("fused decoder requires a final RMSNorm")

    eps = layers[0].self_attn_layer_norm.eps
    theta = layers[0].self_attn.pos_encoder.theta
    for i, lyr in enumerate(layers):
        a, f = lyr.self_attn, lyr.ffn
        why = None
        if lyr.norm_order != TransformerNormOrder.PRE:
            why = f"norm_order is {lyr.norm_order}, expected PRE"
        elif not isinstance(lyr.self_attn_layer_norm, RMSNorm) or not isinstance(
            lyr.ffn_layer_norm, RMSNorm
        ):
            why = "layer norms must be RMSNorm"
        elif a.num_query_groups != 1:
            why = "grouped-query attention is not supported"
        elif a.pos_encoder is None or type(a.pos_encoder).__name__ != "RotaryEncoder":
            why = "attention must use RotaryEncoder"
        elif a.pos_encoder.encoding_dim != a.head_dim:
            why = (f"RoPE covers {a.pos_encoder.encoding_dim} of {a.head_dim} dims; "
                   "partial rotation is not supported")
        elif a.q_norm is not None or a.k_norm is not None:
            why = "q/k norms are not supported"
        elif any(m.bias is not None for m in
                 (a.q_proj, a.k_proj, a.v_proj, a.output_proj,
                  f.gate_proj, f.inner_proj, f.output_proj)):
            why = "projection biases are not supported"
        elif type(f).__name__ != "GLUFeedForwardNetwork":
            why = f"ffn is {type(f).__name__}, expected GLUFeedForwardNetwork"
        elif type(f.gate_activation).__name__ != "SiLU":
            why = f"gate activation is {type(f.gate_activation).__name__}, expected SiLU"
        elif lyr.self_attn_layer_norm.eps != eps or lyr.ffn_layer_norm.eps != eps:
            why = "all layers must share one RMSNorm epsilon"
        elif a.pos_encoder.theta != theta:
            why = "all layers must share one RoPE theta"
        if why is not None:
            raise NotImplementedError(f"fused decoder: layer {i}: {why}")


def _suffix_causal_mask(q_len: int, k_len: int, device, dtype) -> Tensor:
    """Causal mask for ``q_len`` queries sitting at the end of ``k_len`` keys."""
    i = torch.arange(k_len - q_len, k_len, device=device).unsqueeze(1)
    j = torch.arange(k_len, device=device).unsqueeze(0)
    return torch.zeros(q_len, k_len, device=device, dtype=dtype).masked_fill(
        j > i, float("-inf"))
