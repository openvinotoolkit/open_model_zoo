#
# Copyright (C) 2019-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This file is based in part on deepspeech_openvino_0.5.py by Feng Yen-Chang at
# https://github.com/openvinotoolkit/open_model_zoo/pull/419, commit 529805d011d9b405f142b2b40f4d202bd403a4f1 on Sep 19, 2019.
#
from copy import deepcopy

import numpy as np

from asr_utils.pipelines import BlockedSeqPipelineStage


class RnnSeqPipelineStage(BlockedSeqPipelineStage):
    def __init__(self, profile, ie, model, device='CPU'):
        """
        Load/compile to the target device the IE IR file with the network and initialize the pipeline stage.

        profile (dict), a dict with pre/post-processing parameters, see profiles.py
        ie (IECore), IECore object for model loading/compilation/inference
        model (str), filename of .xml IR file
        device (str), inferemnce device
        """
        self.p = deepcopy(profile)
        assert self.p['num_context_frames'] % 2 == 1, "num_context_frames must be odd"

        padding_len = self.p['num_context_frames'] // 2
        super().__init__(
            block_len=16, context_len=self.p['num_context_frames'] - 1,
            left_padding_len=padding_len, right_padding_len=padding_len,
            padding_shape=(self.p['num_mfcc_dct_coefs'],), cut_alignment=True)

        net = ie.read_network(model=model)
        self.exec_net = ie.load_network(network=net, device_name=device)

    def _reset_state(self):
        super()._reset_state()
        self._rnn_state = None

    def process_data(self, data, finish=False):
        if data is not None:
            assert len(data.shape) == 2
        return super().process_data(data, finish=finish)

    def _process_blocks(self, buffer, finish=False):
        assert buffer.shape[0] >= self._block_len + self._context_len
        processed = []
        for start_pos in range(self._context_len, buffer.shape[0] - self._block_len + 1, self._block_len):
            block = buffer[start_pos - self._context_len:start_pos + self._block_len]
            processed.append(self._process_block(block, finish=finish and start_pos + self._block_len >= buffer.shape[0]))
        assert not self._cut_alignment or processed[-1].shape[0] == self._block_len, "Networks with stride != 1 are not supported"
        # Here start_pos is its value on the last iteration of the loop
        buffer_skip_len = start_pos + self._block_len - self._context_len
        return processed, buffer_skip_len

    def _process_block(self, mfcc_features, finish=False):
        assert mfcc_features.shape[0] == self._block_len + self._context_len, "Wrong data length: _process_block() accepts a single block of data"

        # Create a view into the array with overlapping strides to simulate convolution with FC.
        # NB: Replacing this and the first FC layer with conv1d may improve speed a little.
        mfcc_features = np.lib.stride_tricks.as_strided(
            mfcc_features,
            (self._block_len, self._context_len + 1, self.p['num_mfcc_dct_coefs']),
            (mfcc_features.strides[0], mfcc_features.strides[0], mfcc_features.strides[1]),
            writeable = False,
        )

        if self._rnn_state is None:
            state_h = np.zeros(self.exec_net.input_info[self.p['in_state_h']].input_data.shape)
            state_c = np.zeros(self.exec_net.input_info[self.p['in_state_c']].input_data.shape)
        else:
            state_h, state_c = self._rnn_state

        infer_res = self.exec_net.infer(inputs={
            self.p['in_state_c']: state_c,
            self.p['in_state_h']: state_h,
            self.p['in_data']: [mfcc_features],
        })

        state_c = infer_res[self.p['out_state_c']]
        state_h = infer_res[self.p['out_state_h']]
        self._rnn_state = (state_h, state_c)

        probs = infer_res[self.p['out_data']].squeeze(1)
        return probs
