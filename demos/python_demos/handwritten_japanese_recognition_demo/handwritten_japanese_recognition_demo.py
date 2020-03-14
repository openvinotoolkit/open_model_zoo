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

from __future__ import print_function
import os
import sys
import time
import logging as log
from argparse import ArgumentParser, SUPPRESS

import cv2
import numpy as np

from openvino.inference_engine import IENetwork, IECore
from utils.codec import CTCCodec


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument("-m", "--model", type=str, required=True,
                      help="Path to an .xml file with a trained model.")
    args.add_argument("-i", "--input", type=str, required=True,
                      help="Required. Path to an image to infer")
    args.add_argument("-d", "--device", type=str, default="CPU",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: is "
                           "acceptable. The sample will look for a suitable plugin for device specified. Default "
                           "value is CPU")
    args.add_argument("-ni", "--number_iter", type=int, default=1,
                      help="Optional. Number of inference iterations")
    args.add_argument("-cl", "--charlist", type=str, default=os.path.join(os.path.dirname(__file__), "data/kondate_nakayosi_char_list.txt"), help="Path to the decoding char list file")
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
    pad_img = np.pad(img, ((0, 0), (0, height - h), (0, width -  w)), mode='edge')
    return pad_img


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    # Plugin initialization
    ie = IECore()
    # Read IR
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)

    assert len(net.inputs) == 1, "Demo supports only single input topologies"
    assert len(net.outputs) == 1, "Demo supports only single output topologies"

    log.info("Preparing input/output blobs")
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))

    characters = get_characters(args)
    codec = CTCCodec(characters)
    assert len(codec.characters) == net.outputs[out_blob].shape[2], "The text recognition model does not correspond to decoding character list"

    input_batch_size, input_channel, input_height, input_width= net.inputs[input_blob].shape

    # Read and pre-process input image (NOTE: one image only)
    input_image = preprocess_input(args.input, height=input_height, width=input_width)[None,:,:,:]
    assert input_batch_size == input_image.shape[0], "The net's input batch size should equal the input image's batch size "
    assert input_channel == input_image.shape[1], "The net's input channel should equal the input image's channel"

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=args.device)

    # Start sync inference
    log.info("Starting inference ({} iterations)".format(args.number_iter))
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
