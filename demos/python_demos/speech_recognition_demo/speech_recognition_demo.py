#!/usr/bin/env python
#
# Copyright (C) 2019-2020 Intel Corporation
# SPDX-License-Identifier: Apache-2.0
#
# This file is based in part on deepspeech_openvino_0.5.py by Feng Yen-Chang at
# https://github.com/opencv/open_model_zoo/pull/419, commit 529805d011d9b405f142b2b40f4d202bd403a4f1 on Sep 19, 2019.
#
import time
import wave
import numpy as np
import argparse

from tqdm import tqdm
from contexttimer import Timer

from utils.deep_speech_pipeline import DeepSpeechPipeline


def build_argparser():
    parser = argparse.ArgumentParser(description="Speech recognition example")
    parser.add_argument('-i', '--input', help="Path to an audio file in WAV PCM 16 kHz mono format",
                        type=str, metavar="FILENAME", required=True)
    parser.add_argument('-d', '--device', default='CPU', type=str,
                        help="Optional. Specify the target device to infer on, for example: CPU, GPU, FPGA, HDDL, MYRIAD or HETERO. "
                             "The sample will look for a suitable IE plugin for this device. (default is CPU)")
    parser.add_argument('-m', '--model', type=str, metavar="FILENAME",
                        help="Path to an .xml file with a trained model (required)", required=True)

    parser.add_argument('-b', '--beam-width', type=int, default=500, metavar="N",
                        help="Beam width for beam search in CTC decoder (default 500)")
    parser.add_argument('-L', '--lm', type=str, metavar="FILENAME",
                        help="path to language model file (optional)")
    parser.add_argument('-a', '--alphabet', type=str, metavar="FILENAME",
                        help="path to alphabet file matching the model (defaults to the 28-symbol alphabet with English letters)")
    parser.add_argument('--alpha', type=float, default=0.75, metavar="X",
                        help="Language model weight (default 0.75)")
    parser.add_argument('--beta', type=float, default=1.85, metavar="X",
                        help="Word insertion bonus, ignored without LM (default 1.85)")

    parser.add_argument('-l', '--cpu_extension', type=str, metavar="FILENAME",
                        help="Optional. Required for CPU custom layers. "
                             "MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the"
                             " kernels implementations.")
    return parser


def main():
    start_time = time.time()
    with Timer() as timer:
        args = build_argparser().parse_args()

        stt = DeepSpeechPipeline(
            model = args.model,
            lm = args.lm,
            alphabet = args.alphabet,
            beam_width = args.beam_width,
            alpha = args.alpha,
            beta = args.beta,
            device = args.device,
            ie_extensions = [(args.device, args.cpu_extension)] if args.device=='CPU' else [],
        )

        wave_read = wave.open(args.input, 'rb')
        channel_num, sample_width, sampling_rate, pcm_length, compression_type, _ = wave_read.getparams()
        assert sample_width == 2, "Only 16-bit WAV PCM supported"
        assert compression_type == 'NONE', "Only linear PCM WAV files supported"
        assert channel_num == 1, "Only mono WAV PCM supported"
        audio = np.frombuffer(wave_read.readframes(pcm_length * channel_num), dtype=np.int16).reshape((pcm_length, channel_num))
        wave_read.close()
    print("Loading, including network weights, IE initialization, LM, building LM vocabulary trie, loading audio: {} s".format(timer.elapsed))
    print("Audio file length: {} s".format(audio.shape[0] / sampling_rate))

    # Now it is enough to call:
    #   transcription = stt.recognize_audio(audio, sampling_rate)
    # if you don't need to access intermediate features like character probabilities or audio features.

    with Timer() as timer:
        audio_features = stt.extract_mfcc(audio, sampling_rate=sampling_rate)
    print("MFCC time: {} s".format(timer.elapsed))

    with Timer() as timer:
        character_probs = stt.extract_per_frame_probs(audio_features, wrap_iterator=tqdm)
    print("RNN time: {} s".format(timer.elapsed))

    with Timer() as timer:
        transcription = stt.decode_probs(character_probs)
    print("Beam search time: {} s".format(timer.elapsed))
    print("Overall time: {} s".format(time.time() - start_time))

    print("\nTranscription and confidence score:")
    max_candidates = 1
    for candidate in transcription[:max_candidates]:
        print(
            "{}\t{}".format(
                candidate['conf'],
                candidate['text'],
            )
        )


if __name__ == '__main__':
    main()
