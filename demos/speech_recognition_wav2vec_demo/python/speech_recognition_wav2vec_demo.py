#!/usr/bin/env python3
"""
 Copyright (C) 2021-2024 Intel Corporation

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
import json
import logging as log
from pathlib import Path
from time import perf_counter
import sys

import numpy as np
import wave

from openvino import Core, get_version, PartialShape

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)

def build_argparser():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    parser.add_argument('-m', '--model', help='Required. Path to an .xml file with a trained model.', required=True)
    parser.add_argument('-i', '--input', help="Required. Path to an audio file in WAV PCM 16 kHz mono format.", required=True)
    parser.add_argument('-d', '--device', default='CPU',
                        help="Optional. Specify a device to infer on (the list of available devices is shown below). Use "
                        "'-d HETERO:<comma-separated_devices_list>' format to specify HETERO plugin. Use "
                        "'-d MULTI:<comma-separated_devices_list>' format to specify MULTI plugin. Default is CPU")
    parser.add_argument('--vocab', help='Optional. Path to an .json file with encoding vocabulary.')
    parser.add_argument('--dynamic_shape', action='store_true',
                        help='Optional. Using dynamic shapes for inputs of model.')
    return parser


class Wav2Vec:
    alphabet = [
        "<pad>", "<s>", "</s>", "<unk>", "|",
        "e", "t", "a", "o", "n", "i", "h", "s", "r", "d", "l", "u",
        "m", "w", "c", "f", "g", "y", "p", "b", "v", "k", "'", "x", "j", "q", "z"]
    words_delimiter = '|'
    pad_token = '<pad>'

    def __init__(self, core, model_path, input_shape, device, vocab_file, dynamic_flag):
        log.info('Reading model {}'.format(model_path))
        model = core.read_model(model_path)
        if len(model.inputs) != 1:
            raise RuntimeError('Wav2Vec must have one input')
        self.input_tensor_name = model.inputs[0].get_any_name()
        model_input_shape = model.inputs[0].partial_shape
        if len(model_input_shape) != 2:
            raise RuntimeError('Wav2Vec input must be 2-dimensional')
        if len(model.outputs) != 1:
            raise RuntimeError('Wav2Vec must have one output')
        model_output_shape = model.outputs[0].partial_shape
        if len(model_output_shape) != 3:
            raise RuntimeError('Wav2Vec output must be 3-dimensional')
        if model_output_shape[2] != len(self.alphabet):
            raise RuntimeError(f'Wav2Vec output third dimension size must be {len(self.alphabet)}')
        if not dynamic_flag:
            model.reshape({self.input_tensor_name: PartialShape(input_shape)})
        elif not model.is_dynamic():
            model.reshape({self.input_tensor_name: PartialShape((-1, -1))})
        compiled_model = core.compile_model(model, device)
        self.output_tensor = compiled_model.outputs[0]
        self.infer_request = compiled_model.create_infer_request()
        log.info('The model {} is loaded to {}'.format(model_path, device))
        self._init_vocab(vocab_file)

    def _init_vocab(self, vocab_file):
        if vocab_file is not None:
            vocab_file = Path(vocab_file)
            if not vocab_file.exists():
                raise RuntimeError(f'vocab file: {vocab_file} does not exist')
            if vocab_file.suffix != '.json':
                raise RuntimeError('Wav2Vec demo support only vocabulary stored in json')
            with vocab_file.open('r') as vf:
                encoding_vocab = json.load(vf)
                self.decoding_vocab = {int(v): k for k, v in encoding_vocab.items()}
                return
        self.decoding_vocab = dict(enumerate(self.alphabet))


    @staticmethod
    def preprocess(sound):
        return (sound - np.mean(sound)) / (np.std(sound) + 1e-15)

    def infer(self, audio):
        input_data = {self.input_tensor_name: audio}
        return self.infer_request.infer(input_data)[self.output_tensor]

    def decode(self, logits):
        token_ids = np.squeeze(np.argmax(logits, -1))
        tokens = [self.decoding_vocab[idx] for idx in token_ids]
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

    log.info('OpenVINO Runtime')
    log.info('\tbuild: {}'.format(get_version()))
    core = Core()

    model = Wav2Vec(core, args.model, audio.shape, args.device, args.vocab, args.dynamic_shape)
    normalized_audio = model.preprocess(audio)
    character_probs = model.infer(normalized_audio)
    transcription = model.decode(character_probs)
    total_latency = (perf_counter() - start_time) * 1e3
    log.info("Metrics report:")
    log.info("\tLatency: {:.1f} ms".format(total_latency))
    print(transcription)

if __name__ == '__main__':
    sys.exit(main() or 0)
