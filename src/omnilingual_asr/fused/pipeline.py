# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Swap the fused decode path into an existing :class:`ASRInferencePipeline`."""

from __future__ import annotations

import gc

import torch

from omnilingual_asr.fused.decoder import FusedLlamaDecoder
from omnilingual_asr.models.wav2vec2_llama.model import Wav2Vec2LlamaModel


def keep_allocator_cache(pipeline) -> None:
    """Stop the pipeline calling ``torch.cuda.empty_cache()`` after every batch.

    Upstream ``_apply_model`` flushes the caching allocator once per batch. That
    returns every block to the driver, so the next utterance re-``cudaMalloc``s
    its working set — including the 67 MB decoder-input buffer. Measured on the
    300M model it costs more than it saves and makes latency wildly variable
    (133 ms steady-state vs. 140-490 ms with the flush). Peak memory is
    unchanged in steady state; only the willingness to hold freed blocks differs.
    """
    if getattr(pipeline, "_keeps_allocator_cache", False):
        return

    original = pipeline._apply_model

    def _apply_model(batch):
        real = torch.cuda.empty_cache
        torch.cuda.empty_cache = lambda: None
        try:
            return original(batch)
        finally:
            torch.cuda.empty_cache = real

    pipeline._apply_model = _apply_model
    pipeline._keeps_allocator_cache = True


def enable_fused_decoding(
    pipeline,
    *,
    max_seq_len: int = 4096,
    use_cuda_graph: bool = True,
    fast_beam_search: bool = True,
    keep_cache: bool = True,
) -> None:
    """Replace the pipeline's Llama decoder with the fused implementation.

    :param max_seq_len: KV cache reservation, in tokens. Must cover the encoder
        context plus the generated transcript. ~50 frames/s of audio, so 40 s of
        audio needs ~2000 plus the transcript.
    :param use_cuda_graph: Capture the decode step as a CUDA graph. Turn off to
        isolate the effect of fusion alone.
    :param fast_beam_search: Also apply the sync-free beam search step.
    :param keep_cache: Also apply :func:`keep_allocator_cache`.
    """
    model = pipeline.model
    if not isinstance(model, Wav2Vec2LlamaModel):
        raise TypeError("fused decoding only applies to Wav2Vec2LlamaModel (LLM cards)")
    if isinstance(model.llama_decoder, FusedLlamaDecoder):
        return

    fused = FusedLlamaDecoder(
        model.llama_decoder, max_seq_len=max_seq_len, use_cuda_graph=use_cuda_graph
    )
    model.llama_decoder = fused

    gen = pipeline.beam_search_generator
    if gen is not None:
        # The fused decoder's KV cache is registered in the state bag, so
        # beam reordering flows through it. With a beam of 1 the permutation is
        # always the identity and the copy is pure overhead.
        fused._skip_reorder = gen.config.nbest == 1
        if fast_beam_search:
            from omnilingual_asr.fused.beamsearch import patch_generator

            patch_generator(gen)

    if keep_cache:
        keep_allocator_cache(pipeline)

    # Drop the now-unreferenced per-projection weights.
    gc.collect()
    torch.cuda.empty_cache()
