# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""A sync-free rewrite of the LLM-ASR beam search step.

The upstream loop is correct but stalls the pipeline: boolean-mask assignments
like ``candidate_scores[eos_mask] = -inf`` and
``out_tokens[no_token_mask, t] = pad`` are implemented as
``nonzero()`` + index_put, and ``nonzero`` must copy its result size back to the
host. That is four device->host syncs per generated token, on top of the one the
``done`` check needs, and each one drains the launch queue that the fused
decoder is trying to keep full.

This version computes the same values with ``torch.where``/``masked_fill_``,
leaving exactly one sync per step (the termination check). It also skips the
beam-reordering gather when the beam is 1, where the permutation is by
construction the identity.

Semantics are unchanged; the emitted token sequence is identical.
"""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor

from fairseq2.nn import BatchLayout, IncrementalStateBag


@torch.inference_mode()
def _generate_one_segment_fast(
    self, decoder_context_inputs: Tensor, decoder_context_seq_lens: List[int],
) -> Tuple[Tensor, List[int]]:
    B = decoder_context_inputs.size(0)
    device = decoder_context_inputs.device
    dtype = decoder_context_inputs.dtype
    nbest = self.config.nbest
    ex_separator = torch.arange(B, device=device).unsqueeze(1) * nbest

    decoder_inputs = torch.zeros(
        [B * nbest, self.model.max_generation_length, self.model.model_dim],
        device=device, dtype=dtype,
    )
    decoder_inputs[:, : decoder_context_inputs.size(1)] = (
        decoder_context_inputs.repeat_interleave(nbest, dim=0)
    )
    context_lengths = torch.tensor(
        decoder_context_seq_lens, device=device
    ).repeat_interleave(nbest)

    assert self.pad_idx is not None, "`pad_idx` must be specified"
    assert self.eos_idx is not None, "`eos_idx` must be set"
    out_tokens = torch.full_like(
        decoder_inputs[:, :, 0], fill_value=self.pad_idx, dtype=torch.int
    )
    scores = torch.zeros_like(decoder_inputs[:, 0, 0], dtype=torch.float) - 1e6
    scores[::nbest] = 0.0

    state_bag = IncrementalStateBag(max_num_steps=self.model.max_generation_length)
    min_context_len = int(context_lengths.min()) - 1  # remove double BOS input
    prefill_seqs = decoder_inputs[:, :min_context_len]
    prefill_seq_lens = [min_context_len] * B * nbest
    self.model.llama_decoder(
        seqs=prefill_seqs,
        seqs_layout=BatchLayout.of(prefill_seqs, prefill_seq_lens),
        state_bag=state_bag,
    )
    state_bag.increment_step_nr(min_context_len)

    eos_mask = torch.zeros_like(context_lengths, dtype=torch.bool)
    pad_val = torch.tensor(self.pad_idx, device=device, dtype=torch.int)
    eos_val = torch.tensor(self.eos_idx, device=device, dtype=torch.int)
    zero = torch.zeros((), device=device, dtype=torch.float)

    done = False
    t = min_context_len  # == int(context_lengths.min()) - 1, but a host int
    step_lens = [1] * B * nbest
    while not done:
        iterative_seqs = decoder_inputs[:, t : t + 1]
        dec_out = self.model.llama_decoder(
            seqs=iterative_seqs,
            seqs_layout=BatchLayout.of(iterative_seqs, step_lens),
            state_bag=state_bag,
        )
        state_bag.increment_step_nr(1)
        logits = self.model.final_proj(dec_out).squeeze(1)      # [rows, V]
        log_probs = F.log_softmax(logits, dim=-1)

        if self.config.length_norm:
            n_tokens = torch.logical_and(
                out_tokens[:, :t] != self.pad_idx, out_tokens[:, :t] != self.eos_idx
            ).sum(dim=1, keepdim=True)
            if n_tokens[0, 0] > 0:
                candidate_scores = (scores.unsqueeze(1) * n_tokens + log_probs) / (n_tokens + 1)
            else:
                candidate_scores = scores.unsqueeze(1) + log_probs
        else:
            candidate_scores = scores.unsqueeze(1) + log_probs

        # Finished hypotheses may only continue with EOS, keeping their score.
        candidate_scores.masked_fill_(eos_mask.unsqueeze(1), -torch.inf)
        candidate_scores[:, self.eos_idx] = torch.where(
            eos_mask, scores, candidate_scores[:, self.eos_idx]
        )

        if nbest == 1:
            top_scores, top_idx_v = candidate_scores.max(dim=-1)
            top_scores = top_scores.view(-1)
            top_idx_v = top_idx_v.view(-1)
            # top_idx_b would be arange(B): every gather below is a no-op.
        else:
            top_scores, top_idx = candidate_scores.view(B, -1).topk(
                k=nbest, dim=-1, sorted=True
            )
            top_idx_nbest, top_idx_v = self.idx_1d_to_2d(top_idx, candidate_scores.size(-1))
            top_idx_b = (top_idx_nbest + ex_separator).view(-1)

            out_tokens = out_tokens[top_idx_b]
            eos_mask = eos_mask[top_idx_b]
            scores = scores[top_idx_b]
            state_bag.reorder(top_idx_b)
            top_scores = top_scores.view(-1)
            top_idx_v = top_idx_v.view(-1)

        scores = torch.where(eos_mask, scores, top_scores)

        # Rows whose context has not been consumed yet emit pad and score 0;
        # rows that already emitted EOS keep emitting EOS (that wins over pad).
        no_token_mask = t < context_lengths - 1
        col = torch.where(
            eos_mask, eos_val, torch.where(no_token_mask, pad_val, top_idx_v.int())
        )
        out_tokens[:, t] = col
        scores = torch.where(no_token_mask, zero, scores)

        new_tokens = out_tokens[:, t : t + 1]
        eos_mask = (new_tokens == self.eos_idx).squeeze(1)

        new_tokens_embedded = self.model.embed_text(new_tokens, dtype=dtype)
        decoder_inputs[:, t + 1] = torch.where(
            no_token_mask.unsqueeze(1),
            decoder_inputs[:, t + 1],
            new_tokens_embedded.squeeze(1).to(decoder_inputs.dtype),
        )

        if t % 250 == 0:
            cpu_tokens = out_tokens[:, t - self.config.compression_window : t].cpu().numpy()
            ratios_floats = [
                self.compression_ratio(np.array_str(cpu_tokens[i]).replace("\n", ""))
                for i in range(B * nbest)
            ]
            ratios = torch.tensor(ratios_floats, device=device)
            early_stopping_mask = torch.logical_and(
                ratios > self.config.compression_threshold,
                t > context_lengths + self.config.compression_window,
            )
            eos_mask = torch.logical_or(eos_mask, early_stopping_mask)

        # `t` is a host int here, so check it first and short-circuit; the
        # eos_mask reduction is the single remaining device->host sync.
        done = t == self.model.max_generation_length - 4 or bool(torch.all(eos_mask))
        t += 1

    out_tokens = out_tokens[::nbest]
    valid_tokens_mask = torch.logical_and(
        torch.logical_and(out_tokens != self.pad_idx, out_tokens != self.bos_idx),
        out_tokens != self.eos_idx,
    )
    valid_tokens_count = valid_tokens_mask.sum(dim=1)
    final_tokens = torch.full(
        [B, int(valid_tokens_count.max())], fill_value=self.pad_idx,
        dtype=torch.int64, device=device,
    )
    for i in range(B):
        final_tokens[i, : valid_tokens_count[i]] = out_tokens[i][valid_tokens_mask[i]]

    return final_tokens, valid_tokens_count.tolist()


def patch_generator(generator) -> None:
    """Bind the sync-free step onto a ``Wav2Vec2LlamaBeamSearchSeq2SeqGenerator``."""
    generator.generate_hypotheses_one_segment = _generate_one_segment_fast.__get__(
        generator, type(generator)
    )
