#
# Copyright (C) 2019-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
from copy import deepcopy
import codecs

from ctcdecode_numpy import SeqCtcLmDecoder

from asr_utils.pipelines import BlockedSeqPipelineStage


def load_alphabet(filename):
    characters = []
    with codecs.open(filename, 'r', 'utf-8') as f:
        for line in f:
            line = line.rstrip('\r\n')
            if line == '':  # empty line ends the alphabet
                break
            if line[0] == '#':  # comment
                continue
            if line.startswith('\\s'):  # "\s" for space as the first character
                line = ' ' + line[2:]
            elif line[0] == '\\':  # escaping, to enter "#" or "\" as the first character
                line = line[1:]
            characters.append(line)
    return characters


class CtcDecoderSeqPipelineStage(BlockedSeqPipelineStage):
    def __init__(self, profile, lm=None, beam_width=500, max_candidates=None,
            online=False):
        self.p = deepcopy(profile)
        self.online = online

        if isinstance(self.p['alphabet'], str):
            alphabet = load_alphabet(self.p['alphabet'])  # shall not include <blank> token
        else:
            alphabet = self.p['alphabet']  # list-like

        self._decoder = SeqCtcLmDecoder(alphabet, beam_width, max_candidates=max_candidates,
            scorer_lm_fname=lm, alpha=self.p['alpha'], beta=self.p['beta'])

        super().__init__(
            block_len=1, context_len=0,
            left_padding_len=0, right_padding_len=0,
            padding_shape=(len(alphabet) + 1,), cut_alignment=False)

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
