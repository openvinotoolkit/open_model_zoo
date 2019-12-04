#!/usr/bin/env python
"""
 Copyright (C) 2018-2019 Intel Corporation

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
import sys
import os
from argparse import ArgumentParser, SUPPRESS
import cv2
import numpy as np
import logging as log
from time import time
from openvino.inference_engine import IENetwork, IECore

from input_feature import get_input_feature


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.", required=True,
                      type=str)
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. "
                           "MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the"
                           " kernels implementations.", type=str, default=None)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL, MYRIAD or HETERO: is "
                           "acceptable. The sample will look for a suitable plugin for device specified. Default "
                           "value is CPU",
                      default="CPU", type=str)

    return parser


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    # Plugin initialization for specified device and load extensions library if specified
    log.info("Creating Inference Engine")
    ie = IECore()
    if args.cpu_extension and 'CPU' in args.device:
        ie.add_extension(args.cpu_extension, "CPU")
    # Read IR
    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    net = IENetwork(model=model_xml, weights=model_bin)

    if "CPU" in args.device:
        supported_layers = ie.query_network(net, "CPU")
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error("Following layers are not supported by the plugin for specified device {}:\n {}".
                      format(args.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)


    input_ids = 'IteratorGetNext/placeholder_out_port_0'
    input_mask = 'IteratorGetNext/placeholder_out_port_1'
    input_segment_ids = 'IteratorGetNext/placeholder_out_port_3'
   
    out_blob = 'bert/pooler/dense/Tanh'

    #net.batch_size = len(args.input)
    net.batch_size = 1

    #--------------------------------------- Loading model to the plugin
    time1 = time()
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=args.device)
    log.info("--------loading time is: {:.3f}s".format(time()-time1))

    #-------------------load weight and bias for fc(fully-connected) layer
    fc_weight = np.load('weight.npy')
    fc_bias = np.load('bias.npy')


    sentences = [
      "That movie was absolutely awful",
      "The acting was a bit lacking",
      "The film was creative and surprising",
      "Absolutely fantastic!"
    ]


    probabilities = []
    feature  = get_input_feature(sentences)
    for i in range(len(feature)):
      input_ids_blob = np.array(feature[i].input_ids).reshape((1,128))
      input_mask_blob = np.array(feature[i].input_mask).reshape((1,128))
      input_segment_ids_blob = np.array(feature[i].segment_ids).reshape((1,128))

      #--------------------------------------- Start sync inference
      res = exec_net.infer(inputs={input_ids: input_ids_blob,input_mask:input_mask_blob,input_segment_ids:input_segment_ids_blob})

      #---------------------------------------- Processing bert output,  add fc layer and softmax 
      res = res[out_blob]
      logits = np.matmul(res,fc_weight.T)
      logits = logits + fc_bias.reshape((1,2))
      probability = np.exp(logits)/np.sum(np.exp(logits),axis=1,keepdims = True)
      probabilities.append(probability)


    labels = ["Negative", "Positive"]
    prediction = [(sentence, prob, labels[np.argmax(prob)]) for sentence, prob in zip(sentences, probabilities)]
    print('prediction:',prediction)


if __name__ == '__main__':
    sys.exit(main() or 0)
