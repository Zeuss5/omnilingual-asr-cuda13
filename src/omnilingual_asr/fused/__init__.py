# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Fused, CUDA-graph-captured decode path for omniASR LLM-ASR models.

See ``cuda13/OPTIMIZATION.md``. Typical use::

    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
    from omnilingual_asr.fused import enable_fused_decoding

    pipe = ASRInferencePipeline(model_card="omniASR_LLM_300M_v2")
    enable_fused_decoding(pipe)

Transcriptions are unchanged; only latency differs.
"""

from omnilingual_asr.fused.decoder import FusedKVCache as FusedKVCache
from omnilingual_asr.fused.decoder import FusedLlamaDecoder as FusedLlamaDecoder
from omnilingual_asr.fused.pipeline import enable_fused_decoding as enable_fused_decoding
from omnilingual_asr.fused.pipeline import keep_allocator_cache as keep_allocator_cache

__all__ = [
    "FusedKVCache",
    "FusedLlamaDecoder",
    "enable_fused_decoding",
    "keep_allocator_cache",
]
