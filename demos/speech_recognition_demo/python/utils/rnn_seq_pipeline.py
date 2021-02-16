#
# Copyright (C) 2019-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This file is based in part on deepspeech_openvino_0.5.py by Feng Yen-Chang at
# https://github.com/openvinotoolkit/open_model_zoo/pull/419, commit 529805d011d9b405f142b2b40f4d202bd403a4f1 on Sep 19, 2019.
#
import os.path
from copy import deepcopy

import numpy as np
from openvino.inference_engine import IECore

from utils.pipelines import BlockedSeqPipelineStage


class RnnSeqPipelineStage(BlockedSeqPipelineStage):
    def __init__(self, model, model_bin=None, profile=None, ie=None, device='CPU', ie_extensions=[]):
        assert profile is not None, "profile argument must be provided"
        self.p = deepcopy(profile)
        assert self.p['num_context_frames'] % 2 == 1, "num_context_frames must be odd"

        padding_len = self.p['num_context_frames'] // 2
        super().__init__(
            block_len=16, context_len=self.p['num_context_frames'] - 1,
            left_padding_len=padding_len, right_padding_len=padding_len,
            padding_shape=(self.p['num_mfcc_dct_coefs'],), cut_alignment=True)

        self.net = self.exec_net = None
        self.default_device = device

        self.ie = ie if ie is not None else IECore()
        self._load_net(model, model_bin_fname=model_bin, device=device, ie_extensions=ie_extensions)

        if device is not None:
            self.activate_model(device)

    def _load_net(self, model_xml_fname, model_bin_fname=None, ie_extensions=[], device='CPU', device_config=None):
        """
        Load IE IR of the network,  and optionally check it for supported node types by the target device.

        model_xml_fname (str)
        model_bin_fname (str or None)
        ie_extensions (list of tuple(str,str)), list of plugins to load, each element is a pair
            (device_name, plugin_filename) (default [])
        device (str or None), check supported node types with this device; None = do not check (default 'CPU')
        device_config
        """
        if model_bin_fname is None:
            model_bin_fname = os.path.basename(model_xml_fname).rsplit('.', 1)[0] + '.bin'
            model_bin_fname = os.path.join(os.path.dirname(model_xml_fname), model_bin_fname)

        # Plugin initialization for specified device and load extensions library if specified
        for extension_device, extension_fname in ie_extensions:
            if extension_fname is None:
                continue
            self.ie.add_extension(extension_path=extension_fname, device_name=extension_device)

        # Read IR
        self.net = self.ie.read_network(model=model_xml_fname, weights=model_bin_fname)

    def activate_model(self, device):
        if self.exec_net is not None:
            return  # Assuming self.net didn't change
        # Loading model to the plugin
        self.exec_net = self.ie.load_network(network=self.net, device_name=device)

    def _reset_state(self):
        super()._reset_state()
        self._rnn_state = None

    def process_data(self, data, finish=False):
        assert self.exec_net is not None, "Need to call activate_model(device) method before process_data(...)"
        if data is not None:
            assert len(data.shape) == 2
        return super().process_data(data, finish=finish)

    def _process_block(self, mfcc_features, finish=False):
        assert mfcc_features.shape[0] == self._block_len + self._context_len, "Wrong data length: _process_block() accepts a single block of data"

        # Create a view into the array with overlapping strides to simulate convolution with FC
        mfcc_features = np.lib.stride_tricks.as_strided(  # TODO: replacing with conv1d may improve speed a little
            mfcc_features,
            (self._block_len, self._context_len + 1, self.p['num_mfcc_dct_coefs']),
            (mfcc_features.strides[0], mfcc_features.strides[0], mfcc_features.strides[1]),
            writeable = False,
        )

        if self._rnn_state is None:
            state_h = np.zeros((1, 2048))
            state_c = np.zeros((1, 2048))
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

        probs = infer_res[self.p['out_data']].squeeze(1)  # despite its name, this is actually probabilities after softmax
        return probs
