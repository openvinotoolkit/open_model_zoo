#!/usr/bin/env python3

"""
Copyright (C) 2022 Intel Corporation
SPDX-License-Identifier: Apache-2.0
"""
import copy
import logging as log
import sys
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter

import numpy as np
import wave

from openvino.runtime import Core, get_version

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)


def parse():
    def print_version():
        log.info('OpenVINO Runtime')
        print('\tbuild: {}'.format(get_version()))

    class DevicePrinter(ArgumentParser):
        def exit(self, status=0, message=None):
            if 0 == status and message is None:
                print('Available devices:', *Core().available_devices)
                print_version()
            else:
                print(message, file=sys.stderr)
            exit(status)

    parser = DevicePrinter(add_help=False)
    parser.add_argument('-h', '--help', action='help', help='show this help message and exit')

    parser.add_argument("-m", "--model", required=True, type=Path, metavar="<MODEL FILE>",
        help="path to an .xml file with a trained model")

    parser.add_argument("-i", "--input", required=True, type=Path, metavar="<WAV>",
        help="path to an input 16kHz WAV file")

    parser.add_argument("-d", "--device", default="CPU", metavar="<DEVICE>",
        help="specify a device to infer on (the list of available devices is shown below). Default is CPU")

    parser.add_argument("-o", "--output", default="noise_suppression_demo_out.wav", metavar="<WAV>",
        help="path to an output WAV file. Default is noise_suppression_demo_out.wav")

    args = parser.parse_args()
    print_version()
    return args

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
    args = parse()
    core = Core()
    log.info("Reading model {}".format(args.model))
    ov_encoder = core.read_model(args.model)

    # check input and output names
    if len(ov_encoder.inputs) == len(ov_encoder.outputs):
        input_shapes = {}

        for const_obj in ov_encoder.inputs:
            for name in const_obj.get_names():
                if ("state" in name) or ("input" in name):
                    input_shapes[name] = const_obj.shape
                else:
                    raise RuntimeError("The model expected input tensor with name 'input' or 'inp_state_*'")

        for const_obj in ov_encoder.outputs:
            for name in const_obj.get_names():
                if ("state" not in name) and ("output" not in name):
                    raise RuntimeError("The model expected output tensor with names 'output' and 'out_state_*'")

    else:
        raise RuntimeError("Number of inputs of the model ({}) is not equal to number of outputs({})".format(len(ov_encoder.inputs), len(ov_encoder.outputs)))

    state_inp_names = [n for n in input_shapes.keys() if "state" in n]
    state_param_num = sum(np.prod(input_shapes[n]) for n in state_inp_names)
    log.debug("State_param_num = {} ({:.1f}Mb)".format(state_param_num, state_param_num*4e-6))

    # load model to the device
    compiled_model = core.compile_model(ov_encoder, args.device)

    infer_request = compiled_model.create_infer_request()

    log.info('The model {} is loaded to {}'.format(args.model, args.device))

    start_time = perf_counter()
    sample_inp = wav_read(str(args.input))

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
    main()
