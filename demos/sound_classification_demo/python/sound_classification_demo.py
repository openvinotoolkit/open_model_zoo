#!/usr/bin/env python3
"""
 Copyright (C) 2020-2024 Intel Corporation

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
import sys
from time import perf_counter
import wave

import numpy as np
from openvino import Core, get_version

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)


def type_overlap(arg):
    if arg.endswith('%'):
        res = float(arg[:-1]) / 100
    else:
        res = int(arg)
    return res


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')

    args.add_argument('-i', '--input', type=str, required=True,
                      help="Required. Input to process")
    args.add_argument('-m', "--model", type=str, required=True,
                      help="Required. Path to an .xml file with a trained model.")
    args.add_argument("-d", "--device", type=str, default="CPU",
                      help="Optional. Specify the target device to infer on; CPU or GPU is"
                           " acceptable. The demo will look for a suitable plugin for device specified. "
                           "Default value is CPU")
    args.add_argument('--labels', type=str, default=None,
                      help="Optional. Labels mapping file")
    args.add_argument('-sr', '--sample_rate', type=int,
                      help="Optional. Set sample rate for audio input")
    args.add_argument('-ol', '--overlap', type=type_overlap, default=0,
                      help='Optional. Set the overlapping between audio clip in samples or percent')

    return parser.parse_args()


class AudioSource:
    def __init__(self, source, channels=2, samplerate=None):
        self.samplerate = samplerate
        samplerate, audio = read_wav(source, as_float=True)
        audio = audio.T
        if audio.shape[0] != channels:
            raise RuntimeError("Audio has unsupported number of channels - {} (expected {})"
                               .format(audio.shape[0], channels))
        if self.samplerate:
            if self.samplerate != samplerate:
                audio = resample(audio, samplerate, self.samplerate)
        else:
            self.samplerate = samplerate

        self.audio = audio

    def duration(self):
        return self.audio.shape[1] / self.samplerate

    def chunks(self, size, hop=None, num_chunks=1):
        def get_clip(pos, size):
            if pos > self.audio.shape[1]:
                return np.zeros((self.audio.shape[0], size), dtype=self.audio.dtype)
            if pos + size > self.audio.shape[1]:
                clip = np.zeros((self.audio.shape[0], size), dtype=self.audio.dtype)
                clip[:, :self.audio.shape[1]-pos] = self.audio[:, pos:]
                return clip
            else:
                return self.audio[:, pos:pos+size]
        if not hop:
            hop = size
        pos = 0

        while pos < self.audio.shape[1]:
            chunk = np.zeros((num_chunks, self.audio.shape[0], size), dtype=self.audio.dtype)
            for n in range(num_chunks):
                chunk[n, :, :] = get_clip(pos, size)
                pos += hop
            yield chunk


def resample(audio, sample_rate, new_sample_rate):
    duration = audio.shape[1] / float(sample_rate)
    x_old = np.linspace(0, duration, audio.shape[1])
    x_new = np.linspace(0, duration, int(duration*new_sample_rate))
    data = np.array([np.interp(x_new, x_old, channel) for channel in audio])

    return data


def read_wav(file, as_float=False):
    sampwidth_types = {
        1: np.uint8,
        2: np.int16,
        4: np.int32
    }

    with wave.open(file, "rb") as wav:
        params = wav.getparams()
        data = wav.readframes(params.nframes)
        if params.sampwidth in sampwidth_types:
            data = np.frombuffer(data, dtype=sampwidth_types[params.sampwidth])
        else:
            raise RuntimeError("Couldn't process file {}: unsupported sample width {}"
                               .format(file, params.sampwidth))
        data = np.reshape(data, (params.nframes, params.nchannels))
        if as_float:
            data = (data - np.mean(data)) / (np.std(data) + 1e-15)

    return params.framerate, data


def main():
    args = build_argparser()

    log.info('OpenVINO Runtime')
    log.info('\tbuild: {}'.format(get_version()))
    core = Core()

    log.info('Reading model {}'.format(args.model))
    model = core.read_model(args.model)

    if len(model.inputs) != 1:
        log.error("Demo supports only models with 1 input layer")
        sys.exit(1)
    input_tensor_name = model.inputs[0].get_any_name()
    if len(model.outputs) != 1:
        log.error("Demo supports only models with 1 output layer")
        sys.exit(1)

    batch_size, channels, one, length = model.inputs[0].shape
    if one != 1:
        raise RuntimeError("Wrong third dimension size of model input shape - {} (expected 1)".format(one))

    hop = length - args.overlap if isinstance(args.overlap, int) else int(length * (1.0 - args.overlap))
    if hop < 0:
        log.error("Wrong value for '-ol/--overlap' argument - overlapping more than clip length")
        sys.exit(1)

    compiled_model = core.compile_model(model, args.device)
    output_tensor = compiled_model.outputs[0]
    infer_request = compiled_model.create_infer_request()
    log.info('The model {} is loaded to {}'.format(args.model, args.device))

    labels = []
    if args.labels:
        with open(args.labels, "r") as file:
            labels = [line.rstrip() for line in file.readlines()]

    start_time = perf_counter()
    audio = AudioSource(args.input, channels=channels, samplerate=args.sample_rate)

    outputs = []
    clips = 0
    for idx, chunk in enumerate(audio.chunks(length, hop, num_chunks=batch_size)):
        chunk = np.reshape(chunk, model.inputs[0].shape)
        output = infer_request.infer({input_tensor_name: chunk})[output_tensor]
        clips += batch_size
        for batch, data in enumerate(output):
            chunk_start_time = (idx*batch_size + batch)*hop / audio.samplerate
            chunk_end_time = ((idx*batch_size + batch)*hop + length) / audio.samplerate
            outputs.append(data)
            label = np.argmax(data)
            if chunk_start_time < audio.duration():
                log.info("[{:.2f}-{:.2f}] - {:6.2%} {:s}".format(chunk_start_time, chunk_end_time, data[label],
                                                                 labels[label] if labels else "Class {}".format(label)))
    total_latency = (perf_counter() - start_time) * 1e3
    log.info("Metrics report:")
    log.info("\tLatency: {:.1f} ms".format(total_latency))
    sys.exit(0)

if __name__ == '__main__':
    main()
