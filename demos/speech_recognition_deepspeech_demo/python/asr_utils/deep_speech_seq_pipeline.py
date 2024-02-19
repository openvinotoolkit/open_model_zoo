#
# Copyright (C) 2019-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This file is based in part on deepspeech_openvino_0.5.py by Feng Yen-Chang at
# https://github.com/openvinotoolkit/open_model_zoo/pull/419, commit 529805d011d9b405f142b2b40f4d202bd403a4f1 on Sep 19, 2019.
#
from copy import deepcopy

from asr_utils.audio_features import AudioFeaturesSeqPipelineStage
from asr_utils.rnn_seq_pipeline import RnnSeqPipelineStage
from asr_utils.ctc_decoder_seq_pipeline import CtcDecoderSeqPipelineStage


class DeepSpeechSeqPipeline:
    def __init__(self, profile, core, model, lm=None, beam_width=500, max_candidates=None,
            device='CPU', online_decoding=False):
        """
            Args:
        profile (dict), a dict with pre/post-processing parameters, see profiles.py
        core (Core or None), Core object for model loading/compilation/inference
        model (str), filename of IR .xml model file
        lm (str), filename of LM (language model)
        beam_width (int), the number of prefix candidates to retain during decoding in beam search (default 500)
        max_candidates (int), limit the number of returned candidates; None = do not limit (default None)
        device (str), inference device
        online_decoding (bool), set to True to return partial decoded text after every input data piece (default False)
        """
        self.p = deepcopy(profile)
        self.mfcc_stage = AudioFeaturesSeqPipelineStage(profile)
        self.rnn_stage = RnnSeqPipelineStage(profile, core, model, device=device)
        self.ctc_stage = CtcDecoderSeqPipelineStage(profile, lm=lm, beam_width=beam_width,
                max_candidates=max_candidates, online=online_decoding)

    def recognize_audio(self, audio, sampling_rate, finish=True):
        """
        Run a segment of audio through ASR pipeline.
        Use finish=True (default) to run recognition once for a whole utterance.
        Use finish=False for online recognition, provinging data cut into segments and getting updated recognition
        result after each segment. Set finish=True for the last segment to get the final result and reset pipeline state.
        """
        if audio is not None:
            # Double-check sampling rate to avoid a possibly hard-to-debug error
            if abs(sampling_rate / self.p['model_sampling_rate'] - 1) > 0.1  or  (audio.shape + (1,))[1] != 1:
                raise ValueError("Input audio should be {} kHz mono".format(self.p['model_sampling_rate']/1e3))
        audio_features = self.mfcc_stage.process_data(audio, finish=finish)
        probs = self.rnn_stage.process_data(audio_features, finish=finish)
        return self.ctc_stage.process_data(probs, finish=finish)
