#!/usr/bin/env python3
"""
 Copyright (C) 2020 Intel Corporation

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
import logging
import sys
import time
import wave

import numpy as np
from openvino.inference_engine import IECore


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
    args.add_argument("-l", "--cpu_extension", type=str, default=None,
                      help="Optional. Required for CPU custom layers. Absolute path to a shared library with "
                           "the kernels implementations.")
    args.add_argument("-d", "--device", type=str, default="CPU",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is"
                           " acceptable. The sample will look for a suitable plugin for device specified. "
                           "Default value is CPU")
    args.add_argument('--labels', type=str, default=None,
                      help="Optional. Labels mapping file")
    args.add_argument('-sr', '--samplerate', type=int,
                      help="Optional. Set sample rate for audio input")
    args.add_argument('-ol', '--overlap', type=type_overlap, default=0,
                      help='Optional. Set the overlapping between audio clip in samples or percent')

    return parser.parse_args()


class AudioSource:
    def __init__(self, source, channels=2, samplerate=None):
        self.source = source
        self.samplerate = samplerate
        self.channels = channels

    def load(self):
        samplerate, audio = read_wav(self.source, as_float=True)
        audio = audio.T
        if audio.shape[0] != self.channels:
            raise RuntimeError("Audio has unsupported number of channels - {} (expected {})"
                               .format(audio.shape[0], self.channels))
        if self.samplerate:
            if self.samplerate != samplerate:
                audio = resample(audio, samplerate, self.samplerate)
        else:
            self.samplerate = samplerate

        self.audio = audio

    def chunks(self, size, hop=None, num_chunks=1):
        if not hop:
            hop = size
        pos = 0

        while pos + (num_chunks-1)*hop + size <= self.audio.shape[1]:
            yield np.array([self.audio[:, pos+n*hop: pos+n*hop+size] for n in range(num_chunks)])
            pos += hop*num_chunks


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
        3: np.int32,
        4: np.int32
    }
    sampwidth_max = {
        1: 255,
        2: 2**15,
        3: 2*23,
        4: 2**31
    }
    with wave.open(file, "rb") as wav:
        params = wav.getparams()
        data = wav.readframes(params.nframes)
        if sampwidth_types.get(params.sampwidth):
            data = np.frombuffer(data, dtype=sampwidth_types[params.sampwidth])
        else:
            raise RuntimeError("Couldn't process file {}: unsupported sample width {}"
                               .format(str(file), params.sampwidth))
        data = np.reshape(data, (params.nframes, params.nchannels))
        if as_float:
            data = data / sampwidth_max[params.sampwidth]
            if params.sampwidth == 1:
                data -= 0.5
                data *= 2

    return params.framerate, data


def main():
    args = build_argparser()

    logging.basicConfig(format="[ %(levelname)s ] %(message)s", level=logging.INFO, stream=sys.stdout)
    log = logging.getLogger()

    log.info("Creating Inference Engine")
    ie = IECore()

    if args.device == "CPU" and args.cpu_extension:
        ie.add_extension(args.cpu_extension, 'CPU')

    log.info("Loading model {}".format(args.model))
    net = ie.read_network(args.model, args.model[:-4] + ".bin")

    if args.device == "CPU":
        supported_layers = ie.query_network(net, args.device)
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) > 0:
            log.error("Following layers are not supported by the {} plugin:\n {}"
                      .format(args.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using "
                      "-l/--cpu_extension command line argument")
            sys.exit(1)

    if len(net.inputs) != 1:
        log.error("Demo supports only models with 1 input layer")
        sys.exit(1)
    input_blob = next(iter(net.inputs))
    input_shape = net.inputs[input_blob].shape
    if len(net.outputs) != 1:
        log.error("Demo supports only models with 1 output layer")
        sys.exit(1)
    output_blob = next(iter(net.outputs))

    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=args.device)

    log.info("Preparing input")

    labels = []
    if args.labels:
        with open(args.labels, "r") as file:
            labels = [l.rstrip() for l in file.readlines()]

    batch_size, channels, one, length = input_shape
    if one != 1:
        raise RuntimeError("Wrong third dimension size of model input shape - {} (expected 1)".format(one))

    audio = AudioSource(args.input, channels=channels, samplerate=args.samplerate)
    audio.load()

    hop = length - args.overlap if isinstance(args.overlap, int) else int(length * (1.0 - args.overlap))
    if hop < 0:
        log.error("Wrong value for '-ol/--overlap' argument - overlapping more than clip length")
        sys.exit(1)

    log.info("Starting inference")
    outputs = []
    clips = 0
    infer_time = 0
    for idx, chunk in enumerate(audio.chunks(length, hop, num_chunks=batch_size)):
        if len(chunk.shape) != len(input_shape):
            chunk = np.reshape(chunk, newshape=input_shape)
        infer_start_time = time.perf_counter()
        output = exec_net.infer(inputs={input_blob: chunk})
        infer_time += time.perf_counter() - infer_start_time
        clips += batch_size
        output = output[output_blob]
        for batch, data in enumerate(output):
            start_time = (idx*batch_size + batch)*hop / audio.samplerate
            end_time = ((idx*batch_size + batch)*hop + length) / audio.samplerate
            outputs.append(data)
            label = np.argmax(data)
            log.info("[{:.2f}-{:.2f}] - {:6.2%} {:s}".format(start_time, end_time, data[label],
                                                             labels[label] if labels else "Class {}".format(label)))

    if clips == 0:
        log.error("Audio too short for inference by that model")
        sys.exit(1)
    total = np.mean(outputs, axis=0)
    label = np.argmax(total)
    log.info("Average over the audio prediction - {:6.2%} {:s}"
             .format(total[label], labels[label] if labels else "Class {}".format(label)))
    logging.info("Average infer time - {:.1f} ms per clip".format(infer_time / clips * 1000))


if __name__ == '__main__':
    main()
