#
# Copyright (C) 2019-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This file is based in part on deepspeech_openvino_0.5.py by Feng Yen-Chang at
# https://github.com/openvinotoolkit/open_model_zoo/pull/419, commit 529805d011d9b405f142b2b40f4d202bd403a4f1 on Sep 19, 2019.
#
from copy import deepcopy

from utils.pipelines import SeqPipeline
from utils.audio_features import AudioFeaturesSeqPipelineStage
from utils.rnn_seq_pipeline import RnnSeqPipelineStage
from utils.ctc_decoder_seq_pipeline import CtcDecoderSeqPipelineStage


class DeepSpeechSeqPipeline(SeqPipeline):
    def __init__(self, model, model_bin=None, lm=None, beam_width=500, max_candidates=None,
            profile=None, ie=None, device='CPU', ie_extensions=[], online_decoding=False):
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
            in_state_c, in_state_h, out_state_c, out_state_h, in_data, out_data (str), IR node names
            log_probs (bool), True is input data contains base e log(probabilities), False if simply probabilities.
        ie (IECore or None), IECore object to run NN inference with.  Default is to use ie_core_singleton module.
            (default None)
        device (str), inference device for IE, passed here to 1. set default device, and 2. check supported node types
            in the model load; None = do not check (default 'CPU')
        ie_extensions (list(tuple(str,str))), list of IE extensions to load, each extension is defined by a pair
            (device, filename). Records with filename=None are ignored.  (default [])
        online_decoding (bool), set to True to return partial decoded text after every input data piece (default False)
        """
        assert profile is not None, "profile argument must be provided"
        self.p = deepcopy(profile)
        self.mfcc_stage = AudioFeaturesSeqPipelineStage(profile)
        self.rnn_stage = RnnSeqPipelineStage(model, model_bin=model_bin, profile=profile,
                ie=ie, device=device, ie_extensions=ie_extensions)
        self.ctc_stage = CtcDecoderSeqPipelineStage(lm=lm, profile=profile, beam_width=beam_width,
                max_candidates=max_candidates, online=online_decoding)

        super().__init__([
            self.mfcc_stage,
            self.rnn_stage,
            self.ctc_stage,
        ])

    def activate_model(self, device):
        self.rnn_stage.activate_model(device)

    def recognize_audio(self, audio, sampling_rate, finish=True):
        """
        Run a segment of audio through ASR pipeline.
        Use finish=True (default) to run recognition once for a whole utterance.
        Use finish=False for online recognition, provinging data cut into segments and getting updated recognition
        result after each segment. Set finish=True for the last segment to get the final result and reset pipeline state.
        """
        if audio is not None:
            if abs(sampling_rate - self.p['model_sampling_rate']) > self.p['model_sampling_rate'] * 0.1  or  (audio.shape + (1,))[1] != 1:
                raise ValueError("Input audio file should be {} kHz mono".format(self.p['model_sampling_rate']/1e3))
        return self.process_data(audio, finish=finish)

    def extract_mfcc(self, audio, sampling_rate, finish=True):
        # Audio feature extraction
        if audio is not None:
            if abs(sampling_rate - self.p['model_sampling_rate']) > self.p['model_sampling_rate'] * 0.1  or  (audio.shape + (1,))[1] != 1:
                raise ValueError("Input audio file should be {} kHz mono".format(self.p['model_sampling_rate']/1e3))
        return self.mfcc_stage.process_data(audio, finish=finish)

    def extract_per_frame_probs(self, mfcc_features, finish=True):
        return self.rnn_stage.process_data(mfcc_features, finish=finish)

    def decode_probs(self, probs, finish=True):
        """
        If finish==True or online_decoding==True:
          Return list of pairs (-log_score, text) in order of decreasing (audio+LM) score
        If finish==False and online_decoding==False:
          Return None
        """
        return self.ctc_stage.process_data(probs, finish=finish)
