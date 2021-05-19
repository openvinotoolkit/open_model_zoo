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
import logging as log
import sys
import time
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path

import numpy as np
import wave

from openvino.inference_engine import IECore, Blob

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model",
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
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    log.info("Initializing Inference Engine")
    ie = IECore()
    version = ie.get_versions(args.device)[args.device]
    version_str = "{}.{}.{}".format(version.major, version.minor, version.build_number)
    log.info("Plugin version is {}".format(version_str))

    # read IR
    model_xml = args.model
    model_bin = model_xml.with_suffix(".bin")
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    ie_encoder = ie.read_network(model=model_xml, weights=model_bin)

    # check input and output names
    input_shapes = {k: v.input_data.shape for k, v in ie_encoder.input_info.items()}
    input_names = list(ie_encoder.input_info.keys())
    output_names = list(ie_encoder.outputs.keys())

    assert "input" in input_names, "'input' is not presented in model"
    assert "output" in output_names, "'output' is not presented in model"
    state_inp_names = [n for n in input_names if "state" in n]
    state_param_num = sum(np.prod(input_shapes[n]) for n in state_inp_names)
    log.info("state_param_num = {} ({:.1f}Mb)".format(state_param_num, state_param_num*4e-6))

    # load model to the device
    log.info("Loading model to the {}".format(args.device))
    ie_encoder_exec = ie.load_network(network=ie_encoder, device_name=args.device)

    sample_inp = wav_read(args.input)

    input_size = input_shapes["input"][1]
    res = None

    samples_out = []
    samples_times = []
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
                inputs[n] = res[n.replace('inp', 'out')].buffer
            else:
                #on the first iteration fill states by zeros
                inputs[n] = np.zeros(input_shapes[n], dtype=np.float32)

        t0 = time.perf_counter()
        # Set inputs manually through InferRequest functionality to speedup
        infer_request_ptr = ie_encoder_exec.requests[0]
        for n, data in inputs.items():
            info_ptr = ie_encoder.input_info[n]
            blob = Blob(info_ptr.tensor_desc, data)
            infer_request_ptr.set_blob(n, blob, info_ptr.preprocess_info)

        # infer by IE
        infer_request_ptr.infer()
        res = infer_request_ptr.output_blobs

        t1 = time.perf_counter()

        samples_times.append(t1-t0)
        samples_out.append(res["output"].buffer.squeeze(0))

    log.info("Sequence of length {:0.2f}s is processed by {:0.2f}s".format(
        sum(s.shape[0] for s in samples_out)/16000,
        sum(samples_times),

    ))
    sample_out = np.concatenate(samples_out, 0)
    wav_write(args.output, sample_out)


if __name__ == '__main__':
    sys.exit(main() or 0)
