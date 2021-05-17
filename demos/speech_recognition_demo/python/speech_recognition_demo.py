#!/usr/bin/env python3
#
# Copyright (C) 2019-2021 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This file is based in part on deepspeech_openvino_0.5.py by Feng Yen-Chang at
# https://github.com/openvinotoolkit/open_model_zoo/pull/419, commit 529805d011d9b405f142b2b40f4d202bd403a4f1 on Sep 19, 2019.
#
import time
import wave
import timeit
import argparse

import yaml
import numpy as np
from tqdm import tqdm
from openvino.inference_engine import IECore

from utils.profiles import PROFILES
from utils.deep_speech_seq_pipeline import DeepSpeechSeqPipeline


def build_argparser():
    parser = argparse.ArgumentParser(description="Speech recognition demo")
    parser.add_argument('-i', '--input', type=str, metavar="FILENAME", required=True,
                        help="Path to an audio file in WAV PCM 16 kHz mono format")
    parser.add_argument('-d', '--device', default='CPU', type=str,
                        help="Optional. Specify the target device to infer on, for example: CPU, GPU, FPGA, HDDL, MYRIAD or HETERO. "
                             "The demo will look for a suitable IE plugin for this device. (default is CPU)")
    parser.add_argument('-m', '--model', type=str, metavar="FILENAME", required=True,
                        help="Path to an .xml file with a trained model (required)")
    parser.add_argument('-L', '--lm', type=str, metavar="FILENAME",
                        help="path to language model file (optional)")
    parser.add_argument('-p', '--profile', type=str, metavar="NAME", required=True,
                        help="Choose pre/post-processing profile: "
                             "mds06x_en for Mozilla DeepSpeech v0.6.x, "
                             "mds07x_en/mds08x_en/mds09x_en for Mozilla DeepSpeech v0.7.x/v0.8.x/v0.9.x(English), "
                             "other: filename of a YAML file (required)")
    parser.add_argument('-b', '--beam-width', type=int, default=500, metavar="N",
                        help="Beam width for beam search in CTC decoder (default 500)")
    parser.add_argument('-c', '--max-candidates', type=int, default=1, metavar="N",
                        help="Show top N (or less) candidates (default 1)")

    parser.add_argument('--online', action='store_true',
                        help="Switch to realtime online ASR mode")
    parser.add_argument('--block-size', type=int, default=None,
                        help="Block size in audio samples for streaming into ASR pipeline "
                        "(defaults to samples in 10 sec for offline; samples in 16 frame strides for online)")
    parser.add_argument('--online-window', type=int, default=79,
                        help="In online mode, show this many characters on screen (default 79)")
    return parser


def get_profile(profile_name):
    if profile_name in PROFILES:
        return PROFILES[profile_name]
    with open(profile_name, 'rt') as f:
        profile = yaml.safe_load(f)
    return profile


def main():
    args = build_argparser().parse_args()
    profile = get_profile(args.profile)
    if args.block_size is None:
        sr = profile['model_sampling_rate']
        args.block_size = round(sr * 10) if not args.online else round(sr * profile['frame_stride_seconds'] * 16)

    start_load_time = timeit.default_timer()
    stt = DeepSpeechSeqPipeline(
        ie = IECore(),
        model = args.model,
        lm = args.lm,
        beam_width = args.beam_width,
        max_candidates = args.max_candidates,
        profile = profile,
        device = args.device,
        online_decoding = args.online,
    )
    print("Loading, including network weights, IE initialization, LM, building LM vocabulary trie: {} s".format(timeit.default_timer() - start_load_time))

    start_proc_time = timeit.default_timer()
    with wave.open(args.input, 'rb') as wave_read:
        channel_num, sample_width, sampling_rate, pcm_length, compression_type, _ = wave_read.getparams()
        assert sample_width == 2, "Only 16-bit WAV PCM supported"
        assert compression_type == 'NONE', "Only linear PCM WAV files supported"
        assert channel_num == 1, "Only mono WAV PCM supported"
        print("Audio file length: {} s".format(pcm_length / sampling_rate))
        print("")

        audio_pos = 0
        play_start_time = timeit.default_timer()
        iter_wrapper = tqdm if not args.online else (lambda x: x)
        for audio_iter in iter_wrapper(range(0, pcm_length, args.block_size)):
            audio_block = np.frombuffer(wave_read.readframes(args.block_size * channel_num), dtype=np.int16).reshape((-1, channel_num))
            if audio_block.shape[0] == 0:
                break
            audio_pos += audio_block.shape[0]
            #
            # It is possible to call stt.recognize_audio(): 1) for either whole audio files or
            # by splitting files into blocks, and 2) to reuse stt object for multiple files like this:
            #   transcription1 = stt.recognize_audio(whole_audio1, sampling_rate)
            #   transcription2 = stt.recognize_audio(whole_audio2, sampling_rate)
            #   stt.recognize_audio(whole_audio3_block1, sampling_rate, finish=False)
            #   transcription3 = stt.recognize_audio(whole_audio3_block2, sampling_rate, finish=True)
            # If you need intermediate features, you can call pipeline stage by stage: see
            # the implementation of DeepSpeechSeqPipeline.recognize_audio() method.
            #
            partial_transcr = stt.recognize_audio(audio_block, sampling_rate, finish=False)
            if args.online:
                if partial_transcr is not None and len(partial_transcr) > 0:
                    print('\r' + partial_transcr[0].text[-args.online_window:], end='')
                to_wait = play_start_time + audio_pos/sampling_rate - timeit.default_timer()
                if to_wait > 0:
                    time.sleep(to_wait)

    transcription = stt.recognize_audio(None, sampling_rate, finish=True)
    if args.online:
        # Replace the transcription with its finalized version for online mode
        if transcription is not None and len(transcription) > 0:
            print('\r' + transcription[0].text[-args.online_window:])
    else:  #  not args.online
        # Only show processing time in offline mode because online mode is being slowed down to real time by time.sleep()
        print("Processing time (incl. loading audio, MFCC, RNN and beam search): {} s".format(timeit.default_timer() - start_proc_time))

    print("\nTranscription(s) and confidence score(s):")
    for candidate in transcription:
        print("{}\t{}".format(candidate.conf, candidate.text))


if __name__ == '__main__':
    main()
