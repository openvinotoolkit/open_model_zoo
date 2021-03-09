#
# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
import numpy as np

import ctcdecode_numpy
from utils.alphabet import CtcdecoderAlphabet


class CtcnumpyBeamSearchDecoder:
    def __init__(self, alphabet, beam_size, max_candidates=None,
            cutoff_prob=1.0, cutoff_top_n=40,
            scorer_lm_fname=None, alpha=0.75, beta=1.85):
        if isinstance(alphabet, list):
            alphabet = CtcdecoderAlphabet(alphabet)
        self.alphabet = alphabet

        self.beam_size = beam_size
        self.max_candidates = max_candidates
        self.cutoff_prob = cutoff_prob
        self.cutoff_top_n = cutoff_top_n

        self.decoder_state = ctcdecode_numpy.CTCBeamDecoder(
            alphabet.characters + [''],  # labels
            model_path = scorer_lm_fname,
            alpha = alpha,
            beta = beta,
            cutoff_top_n = cutoff_top_n,
            cutoff_prob = cutoff_prob,
            beam_width = beam_size,
            max_candidates_per_batch = max_candidates,
            blank_id = len(alphabet.characters),
        )

    def decode(self, probs):
        output, scores, timesteps, out_seq_len = self.decoder_state.decode(probs[np.newaxis])
        assert out_seq_len.shape[0] == 1
        beam_results = [
            dict(conf=scores[0,res_idx], text=self.alphabet.decode(output[0,res_idx,:out_seq_len[0,res_idx]]), ts=list(timesteps[0,res_idx]))
            for res_idx in range(out_seq_len.shape[1])
        ]
        return beam_results
