#
# Copyright (C) 2020-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This file is based in part on ctcdecode/__init__.py from https://github.com/parlance/ctcdecode,
# commit 431408f22d93ef5ebc4422995111bbb081b971a9 on Apr 4, 2020, 20:54:49 UTC+1.
#
from types import SimpleNamespace as namespace

import numpy as np

from . import impl


class SeqCtcLmDecoder:
    """
    Non-batched sequential CTC + n-gram LM beam search decoder
    """
    def __init__(self, alphabet, beam_size, max_candidates=None,
            cutoff_prob=1.0, cutoff_top_n=40,
            scorer_lm_fname=None, alpha=None, beta=None):
        self.alphabet = alphabet
        self.beam_size = beam_size
        self.max_candidates = max_candidates or 0
        self.cutoff_prob = cutoff_prob
        self.cutoff_top_n = cutoff_top_n

        if scorer_lm_fname is not None:
            assert alpha is not None and beta is not None, "alpha and beta arguments must be provided to use LM"
            self.lm_scorer = impl.ScorerYoklm(alpha, beta, scorer_lm_fname, alphabet)
        else:
            self.lm_scorer = None
        self.decoder_state = impl.CtcDecoderStateNumpy()
        self.decoder_state.init(
            alphabet + [''],
            blank_idx = len(alphabet),
            beam_size = beam_size,
            lm_scorer = self.lm_scorer,
        )
        self.decoder_state.set_config("cutoff_top_n", cutoff_top_n, required=True)
        self.decoder_state.set_config("cutoff_prob", cutoff_prob, required=True)

    def append_data(self, probs, log_probs):
        """
          Args:
        probs (numpy.ndarray), 2d array with symbol probabilities (frames x symbols)
        log_probs (bool), True to accept natural-logarithmic probabilities, False to accept probabilities directly.
        """
        assert len(probs.shape) == 2
        if self.decoder_state.is_finalized():
            self.decoder_state.new_sequence()
        self.decoder_state.append_numpy(probs, log_probs)

    def decode(self, probs=None, log_probs=None, finalize=True):
        if probs is not None:
            assert log_probs is not None, "When 'probs' argument is provided, 'log_probs' argument must be provided as well"
            self.append_data(probs, log_probs)
        symbols, timesteps, scores, cand_lengths = self.decoder_state.decode_numpy(limit_candidates=self.max_candidates, finalize=finalize)
        cand_starts = np.empty(cand_lengths.shape[0] + 1, dtype=cand_lengths.dtype)
        cand_starts[0] = 0
        cand_lengths.cumsum(out=cand_starts[1:])

        def text_decoder(symbol_idxs):
            return ''.join(self.alphabet[idx] for idx in symbol_idxs)
        candidates = [
            namespace(
                conf=scores[res_idx],
                text=text_decoder(symbols[cand_starts[res_idx]:cand_starts[res_idx+1]]),
                ts=list(timesteps[cand_starts[res_idx]:cand_starts[res_idx+1]]),
            )
            for res_idx in range(cand_lengths.shape[0])
        ]
        return candidates


class BatchedCtcLmDecoder:
    """
    Batched CTC + n-gram LM beam search decoder
    """
    def __init__(self, alphabet, model_path=None, alpha=0, beta=0, cutoff_top_n=40, cutoff_prob=1.0, beam_width=100,
                 max_candidates_per_batch=None, num_processes=4, blank_id=0, log_probs_input=False, loader='yoklm'):
        self.cutoff_top_n = cutoff_top_n
        self._beam_width = beam_width
        self._max_candidates_per_batch = max_candidates_per_batch
        self._scorer = None
        self._num_processes = num_processes
        self._alphabet = list(alphabet)
        self._blank_id = blank_id
        self._log_probs = bool(log_probs_input)
        if model_path is not None:
            if loader == 'yoklm':
                self._scorer = impl.ScorerYoklm(alpha, beta, model_path, self._alphabet)
            else:
                raise ValueError("Unknown loader type: \"%s\"" % loader)
        self._cutoff_prob = cutoff_prob

    def decode(self, probs, seq_lens=None):
        # We expect probs as batch x seq x label_size
        batch_size, max_seq_len = probs.shape[0], probs.shape[1]
        if seq_lens is None:
            seq_lens = np.full(batch_size, max_seq_len, dtype=np.int32)
        max_candidates_per_batch = self._max_candidates_per_batch
        if max_candidates_per_batch is None or max_candidates_per_batch > self._beam_width:
            max_candidates_per_batch = self._beam_width

        output, timesteps, scores, out_seq_len = impl.batched_ctc_lm_decoder(
            probs,  # batch_size x max_seq_lens x vocab_size
            seq_lens,  # batch_size
            self._alphabet,  # list(str)
            self._beam_width,
            max_candidates_per_batch,
            self._num_processes,
            self._cutoff_prob,
            self.cutoff_top_n,
            self._blank_id,
            self._log_probs,
            self._scorer,
        )

        output.shape =      (batch_size, max_candidates_per_batch, -1)
        timesteps.shape =   (batch_size, max_candidates_per_batch, -1)
        scores.shape =      (batch_size, max_candidates_per_batch)
        out_seq_len.shape = (batch_size, max_candidates_per_batch)

        return output, scores, timesteps, out_seq_len

    def character_based(self):
        return self._scorer.is_character_based() if self._scorer else None

    def max_order(self):
        return self._scorer.get_max_order() if self._scorer else None

    def dict_size(self):
        return self._scorer.get_dict_size() if self._scorer else None

    def reset_params(self, alpha, beta):
        if self._scorer is not None:
            self._scorer.reset_params(alpha, beta)
