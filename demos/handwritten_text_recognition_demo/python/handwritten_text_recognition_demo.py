#!/usr/bin/env python3

"""
 Copyright (c) 2020-2024 Intel Corporation
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

import sys
from time import perf_counter
import logging as log
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path

import cv2
import numpy as np

from openvino import Core, get_version
from utils.codec import CTCCodec

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument("-m", "--model", type=str, required=True,
                      help="Required. Path to an .xml file with a trained model.")
    args.add_argument("-i", "--input", type=str, required=True,
                      help="Required. Path to an image to infer")
    args.add_argument("-d", "--device", type=str, default="CPU",
                      help="Optional. Specify the target device to infer on; CPU, GPU or HETERO is "
                           "acceptable. The demo will look for a suitable plugin for device specified. Default "
                           "value is CPU")
    args.add_argument("-ni", "--number_iter", type=int, default=1,
                      help="Optional. Number of inference iterations")
    args.add_argument("-cl", "--charlist", type=str, default=str(Path(__file__).resolve().parents[3] / "data/dataset_classes/kondate_nakayosi.txt"),
                      help="Path to the decoding char list file. Default is for Japanese")
    args.add_argument("-dc", "--designated_characters", type=str, default=None, help="Optional. Path to the designated character file")
    args.add_argument("-tk", "--top_k", type=int, default=20, help="Optional. Top k steps in looking up the decoded character, until a designated one is found")
    args.add_argument("-ob", "--output_blob", type=str, default=None, help="Optional. Name of the output layer of the model. Default is None, in which case the demo will read the output name from the model, assuming there is only 1 output layer")
    return parser


def get_characters(args):
    '''Get characters'''
    with open(args.charlist, 'r', encoding='utf-8') as f:
        return ''.join(line.strip('\n') for line in f)


def preprocess_input(image_name, height, width):
    src = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
    if src is None:
        raise RuntimeError(f"Failed to imread {image_name}")
    ratio = float(src.shape[1]) / float(src.shape[0])
    tw = int(height * ratio)
    rsz = cv2.resize(src, (tw, height), interpolation=cv2.INTER_AREA).astype(np.float32)
    # [h,w] -> [c,h,w]
    img = rsz[None, :, :]
    _, h, w = img.shape
    # right edge padding
    pad_img = np.pad(img, ((0, 0), (0, height - h), (0, width - w)), mode='edge')
    return pad_img


def main():
    args = build_argparser().parse_args()

    # Plugin initialization
    log.info('OpenVINO Runtime')
    log.info('\tbuild: {}'.format(get_version()))
    core = Core()

    if 'GPU' in args.device:
        core.set_property("GPU", {"GPU_ENABLE_LOOP_UNROLLING": "NO", "CACHE_DIR": "./"})

    # Read IR
    log.info('Reading model {}'.format(args.model))
    model = core.read_model(args.model)

    if len(model.inputs) != 1:
        raise RuntimeError("Demo supports only single input topologies")
    input_tensor_name = model.inputs[0].get_any_name()

    if args.output_blob is not None:
        output_tensor_name = args.output_blob
    else:
        if len(model.outputs) != 1:
            raise RuntimeError("Demo supports only single output topologies")
        output_tensor_name = model.outputs[0].get_any_name()

    characters = get_characters(args)
    codec = CTCCodec(characters, args.designated_characters, args.top_k)
    if len(codec.characters) != model.output(output_tensor_name).shape[2]:
        raise RuntimeError("The text recognition model does not correspond to decoding character list")

    input_batch_size, input_channel, input_height, input_width = model.inputs[0].shape

    # Read and pre-process input image (NOTE: one image only)
    preprocessing_start_time = perf_counter()
    input_image = preprocess_input(args.input, height=input_height, width=input_width)[None, :, :, :]
    preprocessing_total_time = perf_counter() - preprocessing_start_time
    if input_batch_size != input_image.shape[0]:
        raise RuntimeError("The model's input batch size should equal the input image's batch size")
    if input_channel != input_image.shape[1]:
        raise RuntimeError("The model's input channel should equal the input image's channel")

    # Loading model to the plugin
    compiled_model = core.compile_model(model, args.device)
    infer_request = compiled_model.create_infer_request()
    log.info('The model {} is loaded to {}'.format(args.model, args.device))

    # Start sync inference
    start_time = perf_counter()
    for _ in range(args.number_iter):
        infer_request.infer(inputs={input_tensor_name: input_image})
        preds = infer_request.get_tensor(output_tensor_name).data[:]
        result = codec.decode(preds)
        print(result)
    total_latency = ((perf_counter() - start_time) / args.number_iter + preprocessing_total_time) * 1e3
    log.info("Metrics report:")
    log.info("\tLatency: {:.1f} ms".format(total_latency))

    sys.exit()


if __name__ == '__main__':
    main()
