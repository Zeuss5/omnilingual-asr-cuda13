"""Correctness + speed of the persistent cooperative megakernel vs the current
fused decode path."""
import os
import sys
import time

import torch
from torch.utils.cpp_extension import load

os.environ.setdefault("CUDA_HOME", "/usr/local/cuda-13.0")
here = os.path.dirname(os.path.abspath(__file__))
ext = load(
    name="mega_kernel",
    sources=[os.path.join(here, "mega.cu")],
    extra_cuda_cflags=["-O3", "--use_fast_math", "-gencode=arch=compute_120,code=sm_120"],
    verbose=False,
)

from fairseq2.nn import BatchLayout, IncrementalStateBag
from omnilingual_asr.fused.decoder import FusedLlamaDecoder
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

CTX = 200
dev, dt = "cuda", torch.bfloat16

pipe = ASRInferencePipeline(model_card="omniASR_LLM_300M_v2")
ref = pipe.model.llama_decoder
fused = FusedLlamaDecoder(ref, max_seq_len=CTX + 128).to(dev)
D, I, H, DH, L = fused.model_dim, fused.inner_dim, fused.n_heads, fused.head_dim, fused.n_layers
MAX = fused.max_seq_len
print(f"D={D} I={I} H={H} DH={DH} L={L} MAX={MAX}")

THREADS = 256
maxb = ext.mega_max_blocks(THREADS)
print(f"max co-resident blocks @{THREADS} threads: {maxb}")
BLOCKS = min(maxb, 1024)
print(f"using {BLOCKS} blocks ({BLOCKS*THREADS//32} warps)")

# --- prime the fused decoder's cache with a prefill, then mirror it ---
ctx = torch.randn(1, CTX, D, device=dev, dtype=dt) * 0.1
bag = IncrementalStateBag(max_num_steps=8192)
with torch.inference_mode():
    fused(seqs=ctx, seqs_layout=BatchLayout.of(ctx, [CTX]), state_bag=bag)
bag.increment_step_nr(CTX)
cache = bag.maybe_get_state(fused, type(next(iter(fused._caches.values()))))

# layer pointer table
nbytes = ext.layer_struct_bytes()
table = torch.zeros(L * nbytes // 8, dtype=torch.int64)
for i in range(L):
    ptrs = [
        fused.w_qkv[i].data_ptr(), fused.w_o[i].data_ptr(),
        fused.w_gate_up[i].data_ptr(), fused.w_down[i].data_ptr(),
        fused.attn_norm[i].data_ptr(), fused.ffn_norm[i].data_ptr(),
        cache.k[i].data_ptr(), cache.v[i].data_ptr(),
    ]
    for j, p in enumerate(ptrs):
        table[i * 8 + j] = p
table = table.cuda()

x = torch.zeros(D, device=dev, dtype=dt)
nrm = torch.zeros(D, device=dev, dtype=dt)
qkv = torch.zeros(3 * D, device=dev, dtype=dt)
attn = torch.zeros(D, device=dev, dtype=dt)
sw = torch.zeros(I, device=dev, dtype=dt)
part_o = torch.zeros(H, 16, DH, device=dev, dtype=torch.float32)
part_m = torch.zeros(H, 16, device=dev, dtype=torch.float32)
part_l = torch.zeros(H, 16, device=dev, dtype=torch.float32)
out = torch.zeros(D, device=dev, dtype=dt)


def mega_step(inp):
    x.copy_(inp)
    ext.mega_launch(table, L, x, nrm, qkv, attn, sw, part_o, part_m, part_l,
                    out, fused.final_norm.data, cache.pos_dev.int(), cache.len_dev.int(),
                    D, I, H, DH, MAX, fused.eps, fused.theta, BLOCKS, THREADS)
    return out


tok = torch.randn(1, 1, D, device=dev, dtype=dt) * 0.1

# reference: one fused decode step from the same cache state
import copy
k_save, v_save = cache.k.clone(), cache.v.clone()
len_save, pos_save = cache.cur_len, int(cache.pos_dev.item())

with torch.inference_mode():
    ref_out = fused(seqs=tok, seqs_layout=BatchLayout.of(tok, [1]), state_bag=bag)[:, 0]
torch.cuda.synchronize()

# restore cache and run the megakernel from the identical state
with torch.inference_mode():
    cache.k.copy_(k_save); cache.v.copy_(v_save)
    cache.set_len(len_save)
    got = mega_step(tok[0, 0])
torch.cuda.synchronize()

rel = ((got.float() - ref_out[0].float()).abs().max() / ref_out.float().abs().max()).item()
print(f"\nmegakernel vs fused decoder: max rel err = {rel:.3e}  "
      f"[{'PASS' if rel < 5e-2 else 'FAIL'}]")


def timed(fn, it=50):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    a, b = torch.cuda.Event(True), torch.cuda.Event(True)
    a.record()
    for _ in range(it):
        fn()
    b.record()
    torch.cuda.synchronize()
    return a.elapsed_time(b) / it


with torch.inference_mode():
    cache.set_len(len_save)
    t_mega = timed(lambda: mega_step(tok[0, 0]))
print(f"\nmegakernel          : {t_mega:.3f} ms/step")

# current path, graph-captured, same cache state
with torch.inference_mode():
    cache.set_len(len_save)
b = fused._get_bufs(1)
if 1 not in fused._graphs:
    fused._capture(1, cache, b)
def cur():
    b["in"].copy_(tok[0])
    fused._graphs[1].replay()
with torch.inference_mode():
    t_cur = timed(cur)
print(f"current fused+graph : {t_cur:.3f} ms/step")
print(f"\n-> megakernel is {t_cur/t_mega:.2f}x vs current  "
      f"({'FASTER' if t_mega < t_cur else 'SLOWER'})")
print(f"   SoL floor 1.62 ms: megakernel at {1.62/t_mega*100:.0f}%, current at {1.62/t_cur*100:.0f}%")

# ---- diagnostic: GEMV stages only, cooperative vs 48 separate launches ----
import omnilingual_asr.fused.cuda_ops as co
gext = co.load_extension()
gb = ext.gemv_only_max_blocks(THREADS)
GB = min(gb, 2048)
print(f"\n== GEMV-only diagnostic ==")
print(f"  gemv_only max co-resident blocks: {gb} -> using {GB} ({GB*THREADS//32} warps)")

with torch.inference_mode():
    t_coop = timed(lambda: ext.gemv_only_launch(table, L, x, nrm, qkv, attn, sw, D, I, GB, THREADS))

    xin = torch.randn(1, D, device=dev, dtype=dt)
    sin = torch.randn(1, I, device=dev, dtype=dt)
    def sep():
        for i in range(L):
            gext.gemv_bf16(xin, fused.w_qkv[i])
            gext.gemv_bf16(xin, fused.w_o[i])
            gext.gemv_bf16(xin, fused.w_gate_up[i])
            gext.gemv_bf16(sin, fused.w_down[i])
    t_sep = timed(sep)

print(f"  cooperative persistent grid : {t_coop:.3f} ms")
print(f"  48 separate kernel launches : {t_sep:.3f} ms")
print(f"  -> persistent grid is {t_sep/t_coop:.2f}x ({'faster' if t_coop<t_sep else 'SLOWER'})")
