#!/usr/bin/env python3
"""
 Copyright (C) 2018-2022 Intel Corporation

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
import logging as log
from time import perf_counter
import sys
import json

# Workaround to import librosa on Linux without installed libsndfile.so
try:
    import librosa
except OSError:
    import types
    sys.modules['soundfile'] = types.ModuleType('fake_soundfile')
    import librosa

import numpy as np
import scipy
import wave

from openvino.runtime import Core, get_version, PartialShape

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)


class Vocab:
    def __init__(
            self,
            vocab = " abcdefghijklmnopqrstuvwxyz'-",
            space_symbol = " ",
            pad_id = 29,
            bos_id = 29,
            eos_id = 29,
            unk_id = 29
    ):
        self.vocab = vocab
        self.space_symbol = space_symbol
        self.pad_id = pad_id
        self.bos_id = bos_id
        self.eos_id = eos_id
        self.unk_id = unk_id

    def __len__(self):
        return len(self.vocab)

    def __getitem__(self, idx):
        assert idx < len(self.vocab)
        return self.vocab[idx]

    def remove_special_symbols(self, s):
        for t in [self.pad_id, self.bos_id, self.eos_id, self.unk_id]:
            s = s.replace(self.vocab[t], "")
        s = s.replace(self.space_symbol, " ")
        if s[0] == " ":
            s = s[1:]
        return s


class QuartzNet:
    pad_to = 16

    def __init__(self, core, model_path, input_shape, vocab, device):
        self.vocab = vocab
        assert not input_shape[2] % self.pad_to, f"{self.pad_to} must be a divisor of input_shape's third dimension"
        log.info('Reading model {}'.format(model_path))
        model = core.read_model(model_path)
        if len(model.inputs) != 1:
            raise RuntimeError('QuartzNet must have one input')
        self.input_tensor_name = model.inputs[0].get_any_name()
        model_input_shape = model.inputs[0].shape
        if len(model_input_shape) != 3:
            raise RuntimeError('QuartzNet input must be 3-dimensional')
        if model_input_shape[1] != input_shape[1]:
            raise RuntimeError("QuartzNet input second dimension can't be reshaped")
        if model_input_shape[2] % self.pad_to:
            raise RuntimeError(f'{self.pad_to} must be a divisor of QuartzNet input third dimension')
        if len(model.outputs) != 1:
            raise RuntimeError('QuartzNet must have one output')
        model_output_shape = model.outputs[0].shape
        if len(model_output_shape) != 3:
            raise RuntimeError('QuartzNet output must be 3-dimensional')
        if model_output_shape[2] != len(self.vocab):
            raise RuntimeError(f'QuartzNet output third dimension size must be {len(self.vocab)}')
        model.reshape({self.input_tensor_name: PartialShape(input_shape)})
        compiled_model = core.compile_model(model, device)
        self.infer_request = compiled_model.create_infer_request()
        log.info('The model {} is loaded to {}'.format(model_path, device))

    def infer(self, melspectrogram):
        input_data = {self.input_tensor_name: melspectrogram}
        return next(iter(self.infer_request.infer(input_data).values()))

    def ctc_greedy_decode(self, pred, remove_special_symbols=False):
        prev_id = blank_id = self.vocab.pad_id
        transcription = []
        for idx in pred[0].argmax(1):
            if prev_id != idx != blank_id:
                transcription.append(self.vocab[idx])
            prev_id = idx
        out = ''.join(transcription)
        if remove_special_symbols:
            out = self.vocab.remove_special_symbols(out)
        return out

    @classmethod
    def audio_to_melspectrum(cls, audio, sampling_rate):
        assert sampling_rate == 16000, "Only 16 KHz audio supported"
        preemph = 0.97
        preemphased = np.concatenate([audio[:1], audio[1:] - preemph * audio[:-1].astype(np.float32)])

        win_length = round(sampling_rate * 0.02)
        spec = np.abs(librosa.core.spectrum.stft(preemphased, n_fft=512, hop_length=round(sampling_rate * 0.01),
            win_length=win_length, center=True, window=scipy.signal.windows.hann(win_length), pad_mode='reflect'))
        mel_basis = librosa.filters.mel(sampling_rate, 512, n_mels=64, fmin=0.0, fmax=8000.0, norm='slaney', htk=False)
        log_melspectrum = np.log(np.dot(mel_basis, np.power(spec, 2)) + 2 ** -24)

        normalized = (log_melspectrum - log_melspectrum.mean(1)[:, None]) / (log_melspectrum.std(1)[:, None] + 1e-5)
        remainder = normalized.shape[1] % cls.pad_to
        if remainder != 0:
            return np.pad(normalized, ((0, 0), (0, cls.pad_to - remainder)))[None]
        return normalized[None]


def build_argparser():
    parser = ArgumentParser(add_help=False)
    parser.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    parser.add_argument('-m', '--model', help='Required. Path to an .xml file with a trained model.', required=True)
    parser.add_argument('-i', '--input', help="Required. Path to an audio file in WAV PCM 16 kHz mono format", required=True)
    parser.add_argument('-v', '--vocab', help="Optional. Path to vocabulary file in .json format", default=None)
    parser.add_argument('-d', '--device', default='CPU',
                        help="Optional. Specify the target device to infer on, for example: "
                             "CPU, GPU, HDDL, MYRIAD or HETERO. "
                             "The demo will look for a suitable IE plugin for this device. Default value is CPU.")
    return parser


def main():
    args = build_argparser().parse_args()

    start_time = perf_counter()
    with wave.open(args.input, 'rb') as wave_read:
        channel_num, sample_width, sampling_rate, pcm_length, compression_type, _ = wave_read.getparams()
        assert sample_width == 2, "Only 16-bit WAV PCM supported"
        assert compression_type == 'NONE', "Only linear PCM WAV files supported"
        assert channel_num == 1, "Only mono WAV PCM supported"
        assert sampling_rate == 16000, "Only 16 KHz audio supported"
        audio = np.frombuffer(wave_read.readframes(pcm_length * channel_num), dtype=np.int16).reshape((pcm_length, channel_num))

    log_melspectrum = QuartzNet.audio_to_melspectrum(audio.flatten(), sampling_rate)

    log.info('OpenVINO Inference Engine')
    log.info('\tbuild: {}'.format(get_version()))
    core = Core()

    if args.vocab is None:
        vocab = Vocab()
    else:
        vocab = Vocab(**json.load(open(args.vocab)))
    quartz_net = QuartzNet(core, args.model, log_melspectrum.shape, vocab, args.device)
    character_probs = quartz_net.infer(log_melspectrum)
    transcription = quartz_net.ctc_greedy_decode(character_probs, remove_special_symbols=False if args.vocab is None else True)
    total_latency = (perf_counter() - start_time) * 1e3
    log.info("Metrics report:")
    log.info("\tLatency: {:.1f} ms".format(total_latency))
    print(transcription)

if __name__ == '__main__':
    sys.exit(main() or 0)
