#!/usr/bin/env python3

"""
 Copyright (c) 2021 Intel Corporation

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
from json import encoder
import logging as log
import sys
import copy
from time import perf_counter
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path

import numpy as np
import wave

from openvino.runtime import Core, get_version

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model_path", help="Required. Path to an .xml file with a trained model",
                      required=True, type=Path)
    args.add_argument("-i", "--input", help="Required. Path to a 16kHz wav file with speech+noise",
                      required=True, type=str)
    args.add_argument("-o", "--output", help="Optional. Path to output wav file for cleaned speech",
                      required=False, type=str, default="noise_suppression_demo_out.wav")
    args.add_argument("-d", "--device",
                      help="Optional. Target device to perform inference on. "
                           "Default value is CPU",
                      default="CPU", type=str)
    return parser

def wav_read(wav_name):
    with wave.open(wav_name, "rb") as wav:
        if wav.getsampwidth() != 2:
            raise RuntimeError("wav file {} does not have int16 format".format(wav_name))
        if wav.getframerate() != 16000:
            raise RuntimeError("wav file {} does not have 16kHz sampling rate".format(wav_name))

        data = wav.readframes( wav.getnframes() )
        x = np.frombuffer(data, dtype=np.int16)
        x = x.astype(np.float32) * (1.0 / np.iinfo(np.int16).max)
        if wav.getnchannels() > 1:
            x = x.reshape(-1, wav.getnchannels())
            x = x.mean(1)
    return x

def wav_write(wav_name, x):
    x = (x*np.iinfo(np.int16).max).astype(np.int16)
    with wave.open(wav_name, "wb") as wav:
        wav.setnchannels(1)
        wav.setframerate(16000)
        wav.setsampwidth(2)
        wav.writeframes(x.tobytes())

def main():
    args = build_argparser().parse_args()

    log.info('OpenVINO Inference Engine')
    log.info('\tbuild: {}'.format(get_version()))
    core = Core()

    # read IR
    log.info("Reading model {}".format(args.model))
    ov_encoder = core.read_model(args.model)

    # check input and output names
    if len(ov_encoder.inputs) == len(ov_encoder.outputs):
        input_shapes = {}
        output_names = []

        for const_obj in ov_encoder.inputs:
            input_shapes[const_obj.get_names().pop()] = const_obj.shape

        for const_obj in ov_encoder.outputs:
            output_names.append(const_obj.get_names().pop())
    else:
        log.error("Number of inputs of the model ({}) is not equal to number of outputs({})".format(len(ov_encoder.inputs), len(ov_encoder.outputs)))


    assert "input" in input_shapes.keys(), "'input' is not presented in model"
    assert "output" in output_names, "'output' is not presented in model"
    state_inp_names = [n for n in input_shapes.keys() if "state" in n]
    state_param_num = sum(np.prod(input_shapes[n]) for n in state_inp_names)
    log.debug("State_param_num = {} ({:.1f}Mb)".format(state_param_num, state_param_num*4e-6))

    # load model to the device
    compiled_model = core.compile_model(ov_encoder, args.device)

    infer_request = compiled_model.create_infer_request()

    log.info('The model {} is loaded to {}'.format(args.model, args.device))

    start_time = perf_counter()
    sample_inp = wav_read(args.input)

    input_size = input_shapes["input"][1]
    res = None

    samples_out = []
    while sample_inp is not None and sample_inp.shape[0] > 0:
        if sample_inp.shape[0] > input_size:
            input = sample_inp[:input_size]
            sample_inp = sample_inp[input_size:]
        else:
            input = np.pad(sample_inp, ((0, input_size - sample_inp.shape[0]), ), mode='constant')
            sample_inp = None

        #forms input
        inputs = {"input": input[None, :]}

        #add states to input
        for n in state_inp_names:
            if res:
                inputs[n] = infer_request.get_tensor(n.replace('inp', 'out')).data
            else:
                #on the first iteration fill states by zeros
                inputs[n] = np.zeros(input_shapes[n], dtype=np.float32)

        infer_request.infer(inputs)
        res = infer_request.get_tensor("output")
        samples_out.append(copy.deepcopy(res.data).squeeze(0))
    total_latency = (perf_counter() - start_time) * 1e3
    log.info("Metrics report:")
    log.info("\tLatency: {:.1f} ms".format(total_latency))
    log.info("\tSample length: {:.1f} ms".format(len(samples_out)*input_size/16.0))
    sample_out = np.concatenate(samples_out, 0)
    wav_write(args.output, sample_out)


if __name__ == '__main__':
    sys.exit(main() or 0)
