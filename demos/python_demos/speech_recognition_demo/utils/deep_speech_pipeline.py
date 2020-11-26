#
# Copyright (C) 2019-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This file is based in part on deepspeech_openvino_0.5.py by Feng Yen-Chang at
# https://github.com/openvinotoolkit/open_model_zoo/pull/419, commit 529805d011d9b405f142b2b40f4d202bd403a4f1 on Sep 19, 2019.
#
import os.path
from copy import deepcopy

import numpy as np
from openvino.inference_engine import IECore

import utils.alphabet as alphabet_module
from utils.audio_features import samples_to_melspectrum, melspectrum_to_mfcc
from utils.ctcnumpy_beam_search_decoder import CtcnumpyBeamSearchDecoder


PROFILES = {
    'mds06x_en': dict(
        alphabet = None,  # the default alphabet
        # alpha: Language model weight
        alpha = 0.75,
        # beta: Word insertion bonus (ignored without LM)
        beta = 1.85,
        model_sampling_rate = 16000,
        frame_window_size_seconds = 32e-3,
        frame_stride_seconds = 20e-3,
        mel_num = 40,
        mel_fmin = 20.,
        mel_fmax = 4000.,
        num_mfcc_dct_coefs = 26,
        num_context_frames = 19,
    ),
    'mds07x_en': dict(
        alphabet = None,  # the default alphabet
        alpha = 0.93128901720047,
        beta = 1.1834137439727783,
        model_sampling_rate = 16000,
        frame_window_size_seconds = 32e-3,
        frame_stride_seconds = 20e-3,
        mel_num = 40,
        mel_fmin = 20.,
        mel_fmax = 8000.,
        num_mfcc_dct_coefs = 26,
        num_context_frames = 19,
    ),
}
PROFILES['mds08x_en'] = PROFILES['mds07x_en']


class DeepSpeechPipeline:
    def __init__(self, model, model_bin=None, lm=None, beam_width=500, max_candidates=None,
            profile=PROFILES['mds08x_en'], ie=None, device='CPU', ie_extensions=[]):
        """
            Args:
        model (str), filename of IE IR .xml file of the network
        model_bin (str), filename of IE IR .xml file of the network (default (None) is the same as :model:, but
            with extension replaced with .bin)
        lm (str), filename of LM (language model)
        beam_width (int), the number of prefix candidates to retain during decoding in beam search (default 500)
        max_candidates (int), limit the number of returned candidates; None = do not limit (default None)
        profile (dict): a dict with pre/post-processing parameters
            alphabet (None or str or list(str)), alphabet matching the model (default None):
                None = [' ', 26 English letters, apostrophe];
                str = filename of a text file with the alphabet (excluding separator=blank symbol)
                list(str) = the alphabet itself (expluding separator=blank symbol)
            alpha (float), LM weight relative to audio model (default 0.75)
            beta (float), word insertion bonus to counteract LM's tendency to prefer fewer words (default 1.85)
            model_sampling_rate (float, in Hz)
            frame_window_size_seconds (float, in seconds)
            frame_stride_seconds (float, in seconds)
            mel_num (int)
            mel_fmin (float, in Hz)
            mel_fmax (float, in Hz)
            num_mfcc_dct_coefs (int)
            num_context_frames (int)
         ie (IECore or None), IECore object to run NN inference with.  Default is to use ie_core_singleton module.
            (default None)
        device (str), inference device for IE, passed here to 1. set default device, and 2. check supported node types
            in the model load; None = do not check (default 'CPU')
        ie_extensions (list(tuple(str,str))), list of IE extensions to load, each extension is defined by a pair
            (device, filename). Records with filename=None are ignored.  (default [])
        """
        self.p = deepcopy(profile)
        # model parameters
        self.num_batch_frames = 16

        self.beam_width = beam_width
        self.max_candidates = max_candidates
        alphabet = self.p['alphabet']
        if alphabet is None:
            self.alphabet = alphabet_module.get_default_alphabet()
        elif isinstance(alphabet, str):
            self.alphabet = alphabet_module.load_alphabet(alphabet)  # shall not include <blank> token
        else:
            self.alphabet = alphabet

        self.net = self.exec_net = None
        self.default_device = device

        self.ie = ie if ie is not None else IECore()
        self._load_net(model, model_bin_fname=model_bin, device=device, ie_extensions=ie_extensions)

        self.decoder = CtcnumpyBeamSearchDecoder(self.alphabet, self.beam_width, max_candidates=max_candidates,
            scorer_lm_fname=lm, alpha=self.p['alpha'], beta=self.p['beta'])

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

    def recognize_audio(self, audio, sampling_rate):
        mfcc_features = self.extract_mfcc(audio, sampling_rate)
        probs = self.extract_per_frame_probs(mfcc_features)
        del mfcc_features
        transcription = self.decode_probs(probs)
        return transcription

    def extract_mfcc(self, audio, sampling_rate):
        # Audio feature extraction
        if abs(sampling_rate - self.p['model_sampling_rate']) > self.p['model_sampling_rate'] * 0.1  or  (audio.shape + (1,))[1] != 1:
            raise ValueError("Input audio file should be {} kHz mono".format(self.p['model_sampling_rate']/1e3))
        if np.issubdtype(audio.dtype, np.integer):
            audio = audio/np.float32(32768) # normalize to -1 to 1, int16 to float32
        melspectrum = samples_to_melspectrum(
            audio.flatten(),
            sampling_rate,
            sampling_rate * self.p['frame_window_size_seconds'],
            sampling_rate * self.p['frame_stride_seconds'],
            n_mels = self.p['mel_num'],
            fmin = self.p['mel_fmin'],
            fmax = self.p['mel_fmax'],
        )
        features = melspectrum_to_mfcc(melspectrum, self.p['num_mfcc_dct_coefs'])
        return features

    def extract_per_frame_probs(self, mfcc_features, state=None, return_state=False, wrap_iterator=lambda x:x):
        assert self.exec_net is not None, "Need to call mds.activate(device) method before mds.stt(...)"

        padding = np.zeros((self.p['num_context_frames'] // 2, self.p['num_mfcc_dct_coefs']), dtype=mfcc_features.dtype)
        mfcc_features = np.concatenate((padding, mfcc_features, padding))  # TODO: replace with np.pad

        num_strides = len(mfcc_features) - self.p['num_context_frames'] + 1
        # Create a view into the array with overlapping strides to simulate convolution with FC
        mfcc_features = np.lib.stride_tricks.as_strided(  # TODO: replace with conv1d
            mfcc_features,
            (num_strides, self.p['num_context_frames'], self.p['num_mfcc_dct_coefs']),
            (mfcc_features.strides[0], mfcc_features.strides[0], mfcc_features.strides[1]),
            writeable = False,
        )

        if state is None:
            state_h = np.zeros((1, 2048))
            state_c = np.zeros((1, 2048))
        else:
            state_h, state_c = state

        probs = []
        for i in wrap_iterator(range(0, mfcc_features.shape[0], self.num_batch_frames)):
            chunk = mfcc_features[i:i + self.num_batch_frames]

            if len(chunk) < self.num_batch_frames:
                chunk = np.pad(
                    chunk,
                    (
                        (0, self.num_batch_frames - len(chunk)),
                        (0, 0),
                        (0, 0),
                    ),
                    mode = 'constant',
                    constant_values = 0,
                )

            res = self.exec_net.infer(inputs={
                'previous_state_c': state_c,
                'previous_state_h': state_h,
                'input_node': [chunk],
            })
            probs.append(res['logits'].squeeze(1))  # they are actually probabilities after softmax, not logits
            state_h = res['cudnn_lstm/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell/BlockLSTM/TensorIterator.1']
            state_c = res['cudnn_lstm/rnn/multi_rnn_cell/cell_0/cudnn_compatible_lstm_cell/BlockLSTM/TensorIterator.2']
        probs = np.concatenate(probs)

        if not return_state:
            return probs
        else:
            return probs, (state_h, state_c)

    def decode_probs(self, probs):
        """
        Return list of pairs (-log_score, text) in order of decreasing (audio+LM) score
        """
        return self.decoder.decode(probs)
