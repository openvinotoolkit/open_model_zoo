"""
 Copyright (c) 2019 Intel Corporation
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
    args.add_argument("-i", "--input", type=str, nargs="+", required=False,
                      help="Required. Path to an image to infer")
    args.add_argument('-ih', '--input-height', default=96, type=int,
                      help='IR input image height')
    args.add_argument('-iw', '--input-width', default=2000, type=int,
                      help='IR input image width')
    args.add_argument("-d", "--device", type=str, default="CPU",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: is "
                           "acceptable. The sample will look for a suitable plugin for device specified. Default "
                           "value is CPU")
    args.add_argument("-ni", "--number_iter", type=int, default=1,
                      help="Optional. Number of inference iterations")
    args.add_argument("-lang", "--lang", type=str, default='jp',
                      help="Language supported, default value is Japanese")
    args.add_argument("-b", "--batchsize", default=1, type=int, help="Batch size for inference")
    args.add_argument("-cl", "--charlist", type=str, default="./kondate_char_list.txt", help="Path to the decoding char list file")
    return parser


def get_characters(args):
    '''Get characters'''
    characters = ''
    chrs_list_file = ''
    if args.lang == 'jp':
        chrs_list_file = args.charlist
        with open(chrs_list_file, 'r') as f:
            for line in f.readlines():
                line = line.strip('\n')
                characters += line
    else:
        print("Error! %s language is not supported"% args.lang)
        sys.exit(0)
    return characters


def preprocess_input(image_name, height=96, width=2000):
    src = cv2.imread(image_name)
    if len(src.shape) == 3:
        src = np.expand_dims(cv2.cvtColor(src, cv2.COLOR_BGR2GRAY).astype(np.float32), -1)
    ratio = float(src.shape[1]) / float(src.shape[0])
    th = height
    tw = int(th * ratio)
    rsz = cv2.resize(src, (tw, th), interpolation=cv2.INTER_AREA).astype(np.float32)

    # [h,w] -> [c,h,w]
    img = rsz[None, :, :]
    c, h, w = img.shape
    # right edge padding
    pad_num = [0, 0, height - h, width -  w]
    pad_img = np.pad(img, ((0, 0), (pad_num[0], pad_num[2]), (pad_num[1], pad_num[3])), mode='edge')
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

    assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
    #assert len(net.outputs) == 1, "Sample supports only single output topologies"

    log.info("Preparing input/output blobs")
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))

    characters = get_characters(args)
    codec = CTCCodec(characters) # in main function

    # Read and pre-process input images (NOTE: one image only)
    input_image = preprocess_input(args.input[0], height=args.input_height, width=args.input_width)[None,:,:,:]

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    t0 = time.time()
    exec_net = ie.load_network(network=net, device_name=args.device)
    log.info("Plugin loading time: {} minutes".format((time.time()-t0)/60))

    # Start sync inference
    log.info("Starting inference ({} iterations)".format(args.number_iter))
    infer_time = []
    for i in range(args.number_iter):
        t0 = time.time()
        preds = exec_net.infer(inputs={input_blob: input_image})
        preds = preds["output"]
        result = codec.decode(preds)
        print(result)
        infer_time.append((time.time() - t0) * 1000)
    log.info("Average throughput: {} ms".format(np.average(np.asarray(infer_time))/args.batchsize))

    sys.exit()


if __name__ == '__main__':
    main()

