#!/usr/bin/env python3

"""
 Copyright (c) 2020 Intel Corporation
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

import os
import sys
import time
import logging as log
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path

import cv2
import numpy as np

from openvino.inference_engine import IECore, get_version
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
                      help="Optional. Specify the target device to infer on; CPU, GPU, HDDL, MYRIAD or HETERO is "
                           "acceptable. The demo will look for a suitable plugin for device specified. Default "
                           "value is CPU")
    args.add_argument("-ni", "--number_iter", type=int, default=1,
                      help="Optional. Number of inference iterations")
    args.add_argument("-cl", "--charlist", type=str, default=str(Path(__file__).resolve().parents[3] / "data/dataset_classes/kondate_nakayosi.txt"),
                      help="Path to the decoding char list file. Default is for Japanese")
    args.add_argument("-dc", "--designated_characters", type=str, default=None, help="Optional. Path to the designated character file")
    args.add_argument("-tk", "--top_k", type=int, default=20, help="Optional. Top k steps in looking up the decoded character, until a designated one is found")
    return parser


def get_characters(args):
    '''Get characters'''
    with open(args.charlist, 'r', encoding='utf-8') as f:
        return ''.join(line.strip('\n') for line in f)


def preprocess_input(image_name, height, width):
    src = cv2.imread(image_name, cv2.IMREAD_GRAYSCALE)
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
    log.info('OpenVINO Inference Engine')
    log.info('\tbuild: {}'.format(get_version()))
    ie = IECore()

    # Read IR
    log.info('Reading model {}'.format(args.model))
    net = ie.read_network(args.model, os.path.splitext(args.model)[0] + ".bin")

    assert len(net.input_info) == 1, "Demo supports only single input topologies"
    assert len(net.outputs) == 1, "Demo supports only single output topologies"

    input_blob = next(iter(net.input_info))
    out_blob = next(iter(net.outputs))

    characters = get_characters(args)
    codec = CTCCodec(characters, args.designated_characters, args.top_k)
    assert len(codec.characters) == net.outputs[out_blob].shape[2], "The text recognition model does not correspond to decoding character list"

    input_batch_size, input_channel, input_height, input_width= net.input_info[input_blob].input_data.shape

    # Read and pre-process input image (NOTE: one image only)
    input_image = preprocess_input(args.input, height=input_height, width=input_width)[None, :, :, :]
    assert input_batch_size == input_image.shape[0], "The net's input batch size should equal the input image's batch size "
    assert input_channel == input_image.shape[1], "The net's input channel should equal the input image's channel"

    # Loading model to the plugin
    exec_net = ie.load_network(network=net, device_name=args.device)
    log.info('The model {} is loaded to {}'.format(args.model, args.device))

    # Start sync inference
    infer_time = []
    for i in range(args.number_iter):
        t0 = time.time()
        preds = exec_net.infer(inputs={input_blob: input_image})
        preds = preds[out_blob]
        result = codec.decode(preds)
        print(result)
        infer_time.append((time.time() - t0) * 1000)
    log.info("Average throughput: {} ms".format(np.average(np.asarray(infer_time))))

    sys.exit()


if __name__ == '__main__':
    main()
