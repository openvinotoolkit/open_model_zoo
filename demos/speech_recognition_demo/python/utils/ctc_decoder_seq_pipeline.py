#
# Copyright (C) 2019-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from copy import deepcopy

from ctcdecode_numpy import SeqCtcLmDecoder

from utils.alphabet import get_default_alphabet, load_alphabet, CtcdecoderAlphabet
from utils.pipelines import BlockedSeqPipelineStage, SeqPipeline


class CtcDecoderSeqPipelineStage(BlockedSeqPipelineStage):
    def __init__(self, lm=None, profile=None, beam_width=500, max_candidates=None,
            online=False):
        assert profile is not None, "profile argument must be provided"
        self.p = deepcopy(profile)
        self.beam_width = beam_width
        self.max_candidates = max_candidates
        self.online = online

        alphabet = self.p['alphabet']
        if alphabet is None:
            self.alphabet = get_default_alphabet()
        elif isinstance(alphabet, str):
            self.alphabet = load_alphabet(alphabet)  # shall not include <blank> token
        else:
            self.alphabet = alphabet  # list-like

        alphabet_decoder = CtcdecoderAlphabet(self.alphabet)
        self._decoder = SeqCtcLmDecoder(self.alphabet, self.beam_width, max_candidates=max_candidates,
            scorer_lm_fname=lm, alpha=self.p['alpha'], beta=self.p['beta'],
            text_decoder=lambda symbols:alphabet_decoder.decode(symbols))

        super().__init__(
            block_len=1, context_len=0,
            left_padding_len=0, right_padding_len=0,
            padding_shape=(len(self.alphabet) + 1,), cut_alignment=False)

    def _finalize_and_reset_state(self):
        self._reset_state()
        return self._decoder.decode(finalize=True)

    def process_data(self, data, finish=False):
        if data is not None:
            assert len(data.shape) == 2
        return super().process_data(data, finish=finish)

    def _process_blocks(self, probs, finish=False):
        assert len(probs.shape) == 2
        self._decoder.append_data(probs, log_probs=self.p['log_probs'])
        if self.online or finish:
            return [self._decoder.decode(finalize=finish)], probs.shape[0]
        else:
            return [], probs.shape[0]

    def _combine_output(self, processed_list):
        # Return the newest non-empty item
        processed_list = [out for out in processed_list if out is not None]
        if len(processed_list) == 0:
            return None
        return processed_list[-1]
