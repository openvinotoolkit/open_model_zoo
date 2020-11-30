#
# Copyright (C) 2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This file is based in part on ctcdecode/__init__.py from https://github.com/parlance/ctcdecode,
# commit 431408f22d93ef5ebc4422995111bbb081b971a9 on Apr 4, 2020, 20:54:49 UTC+1.
#
import numpy as np

from . import impl as ctc_decode


class CTCBeamDecoder(object):
    def __init__(self, labels, model_path=None, alpha=0, beta=0, cutoff_top_n=40, cutoff_prob=1.0, beam_width=100,
                 max_candidates_per_batch=None, num_processes=4, blank_id=0, log_probs_input=False, loader='yoklm'):
        self.cutoff_top_n = cutoff_top_n
        self._beam_width = beam_width
        self._max_candidates_per_batch = max_candidates_per_batch
        self._scorer = None
        self._num_processes = num_processes
        self._labels = list(labels)
        self._blank_id = blank_id
        self._log_probs = bool(log_probs_input)
        if model_path is not None:
            if loader == 'yoklm':
                self._scorer = ctc_decode.create_scorer_yoklm(alpha, beta, model_path, self._labels)
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
        if self._scorer is not None:
            output, timesteps, scores, out_seq_len = ctc_decode.numpy_beam_decode(
                probs,  # batch_size x max_seq_lens x vocab_size
                seq_lens,  # batch_size
                self._labels,  # list(str)
                self._beam_width,
                max_candidates_per_batch,
                self._num_processes,
                self._cutoff_prob,
                self.cutoff_top_n,
                self._blank_id,
                self._log_probs,  # log_input, bool
                self._scorer,
            )
        else:
            output, timesteps, scores, out_seq_len = ctc_decode.numpy_beam_decode_no_lm(
                probs,  # batch_size x max_seq_lens x vocab_size
                seq_lens,  # batch_size
                self._labels,  # list(str)
                self._beam_width,
                max_candidates_per_batch,
                self._num_processes,
                self._cutoff_prob,
                self.cutoff_top_n,
                self._blank_id,
                self._log_probs,  # log_input, bool
            )
        output.shape =      (batch_size, max_candidates_per_batch, -1)
        timesteps.shape =   (batch_size, max_candidates_per_batch, -1)
        scores.shape =      (batch_size, max_candidates_per_batch)
        out_seq_len.shape = (batch_size, max_candidates_per_batch)

        return output, scores, timesteps, out_seq_len

    def character_based(self):
        return ctc_decode.is_character_based(self._scorer) if self._scorer else None

    def max_order(self):
        return ctc_decode.get_max_order(self._scorer) if self._scorer else None

    def dict_size(self):
        return ctc_decode.get_dict_size(self._scorer) if self._scorer else None

    def reset_params(self, alpha, beta):
        if self._scorer is not None:
            ctc_decode.reset_params(self._scorer, alpha, beta)

    def __del__(self):
        if self._scorer is not None:
            ctc_decode.delete_scorer(self._scorer)
