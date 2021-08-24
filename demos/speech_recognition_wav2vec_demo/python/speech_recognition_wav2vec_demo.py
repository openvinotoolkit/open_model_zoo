#!/usr/bin/env python3
"""
 Copyright (C) 2021 Intel Corporation

 Licensed under the Apache License, Version 2.0 (the "License");
 you may not use this file except in compliance with the License.
 You may obtain a copy of the License at

      http://www.apache.org/licenses/LICENSE-2.0

 Unless required by applicable law or agreed to in writing, software
 distributed under the License is distributed on an "AS IS" BASIS,
 WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 See the License for the specific language governing permissions and
 limitations under the License.
"""

from argparse import ArgumentParser, SUPPRESS
from itertools import groupby
import logging as log
from time import perf_counter
import sys

import numpy as np
import wave

from openvino.inference_engine import IECore, get_version

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)

def build_argparser():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    parser.add_argument('-m', '--model', help='Required. Path to an .xml file with a trained model.', required=True)
    parser.add_argument('-i', '--input', help="Required. Path to an audio file in WAV PCM 16 kHz mono format", required=True)
    parser.add_argument('-d', '--device', default='CPU',
                        help="Optional. Specify the target device to infer on, for example: "
                             "CPU, GPU, HDDL, MYRIAD or HETERO. "
                             "The demo will look for a suitable IE plugin for this device. Default value is CPU.")
    return parser


class Wav2Vec:
    alphabet = [
        "[pad]", "[s]", "[s]", "[unk]", "|",
        "e", "t", "a", "o", "n", "i", "h", "s", "r", "d", "l", "u",
        "m", "w", "c", "f", "g", "y", "p", "b", "v", "k", "'", "x", "j", "q", "z"]
    words_delimiter = '|'
    pad_token = '<pad>'

    def __init__(self, ie, model_path, input_shape, device):
        self.ie = ie
        log.info('Reading model {}'.format(model_path))
        network = self.ie.read_network(model_path)
        if len(network.input_info) != 1:
            raise RuntimeError('Wav2Vec must have one input')
        model_input_shape = next(iter(network.input_info.values())).input_data.shape
        if len(model_input_shape) != 2:
            raise RuntimeError('Wav2Vec input must be 2-dimensional')
        if len(network.outputs) != 1:
            raise RuntimeError('Wav2Vec must have one output')
        model_output_shape = next(iter(network.outputs.values())).shape
        if len(model_output_shape) != 3:
            raise RuntimeError('Wav2Vec output must be 3-dimensional')
        if model_output_shape[2] != len(self.alphabet):
            raise RuntimeError(f'Wav2Vec output third dimension size must be {len(self.alphabet)}')
        network.reshape({next(iter(network.input_info)): input_shape})
        self.exec_net = self.ie.load_network(network, device)
        log.info('The model {} is loaded to {}'.format(model_path, device))

    @staticmethod
    def preprocess(sound):
        return (sound - np.mean(sound)) / (np.std(sound) + 1e-15)

    def infer(self, audio):
        return next(iter(self.exec_net.infer({next(iter(self.exec_net.input_info)): audio}).values()))

    def decode(self, logits):
        token_ids = np.squeeze(np.argmax(logits, -1))
        tokens = [self.alphabet[idx] for idx in token_ids]
        tokens = [token_group[0] for token_group in groupby(tokens)]
        tokens = [t for t in tokens if t != self.pad_token]
        res_string = ''.join([t if t != self.words_delimiter else ' ' for t in tokens]).strip()
        res_string = ' '.join(res_string.split(' '))
        res_string = res_string.lower()
        return res_string


def main():
    args = build_argparser().parse_args()

    start_time = perf_counter()
    with wave.open(args.input, 'rb') as wave_read:
        channel_num, sample_width, sampling_rate, pcm_length, compression_type, _ = wave_read.getparams()
        assert sample_width == 2, "Only 16-bit WAV PCM supported"
        assert compression_type == 'NONE', "Only linear PCM WAV files supported"
        assert channel_num == 1, "Only mono WAV PCM supported"
        assert sampling_rate == 16000, "Only 16 KHz audio supported"
        audio = np.frombuffer(wave_read.readframes(pcm_length * channel_num), dtype=np.int16).reshape((1, pcm_length))
        audio =  audio.astype(float) / np.iinfo(np.int16).max

    log.info('OpenVINO Inference Engine')
    log.info('\tbuild: {}'.format(get_version()))
    ie = IECore()

    net = Wav2Vec(ie, args.model, audio.shape, args.device)
    normalized_audio = net.preprocess(audio)
    character_probs = net.infer(normalized_audio)
    transcription = net.decode(character_probs)
    total_latency = (perf_counter() - start_time) * 1e3
    log.info("Metrics report:")
    log.info("\tLatency: {:.1f} ms".format(total_latency))
    print(transcription)

if __name__ == '__main__':
    main()
