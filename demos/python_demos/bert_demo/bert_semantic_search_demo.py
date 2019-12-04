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
from termcolor import colored


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


    input_ids = 'Placeholder'
    input_mask = 'Placeholder_1'
    input_segment_ids = 'Placeholder_2'
   
    out_blob = 'bert/pooler/dense/Tanh'

    net.batch_size = 1

    #---------------------------------------read questions in stored txt file
    with open('questions.txt') as f:
      questions = [ v.strip() for v in f]
      log.info('questions number:{}'.format(len(questions)))

    feature  = get_input_feature(questions)


    #--------------------------------------- Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=args.device)

    #--------------------------------------- Start sync inference
    log.info("Starting inference for questions to get vectors ")

    vectors = []
    for i in range(len(feature)):
      input_ids_blob = np.array(feature[i].input_ids).reshape((1,128))
      input_mask_blob = np.array(feature[i].input_mask).reshape((1,128))
      input_segment_ids_blob = np.array(feature[i].segment_ids).reshape((1,128))
      res = exec_net.infer(inputs={input_ids: input_ids_blob,input_mask:input_mask_blob,input_segment_ids:input_segment_ids_blob})
      vectors.append(res[out_blob])

    vectors = np.array(vectors) 
    vectors = vectors.reshape((vectors.shape[0],vectors.shape[-1]))

    #--------------------------------------- type your question and search it with 5 most similar in stored question file.
    topk = 5
    while True:
      query_sentence = input('your questions: ')
      query_list = []
      query_list.append(query_sentence.strip())
      feature  = get_input_feature(query_list)
      input_ids_blob = np.array(feature[0].input_ids).reshape((1,128))
      input_mask_blob = np.array(feature[0].input_mask).reshape((1,128))
      input_segment_ids_blob = np.array(feature[0].segment_ids).reshape((1,128))
      res = exec_net.infer(inputs={input_ids: input_ids_blob,input_mask:input_mask_blob,input_segment_ids:input_segment_ids_blob})
      query_vec = res[out_blob]

      # compute normalized dot product as score
      score = np.sum(query_vec * vectors, axis=1) / np.linalg.norm(vectors, axis=1)
      topk_idx = np.argsort(score)[::-1][:topk]
      print('top %d questions similar to "%s"' % (topk, colored(query_sentence, 'green')))
      for idx in topk_idx:
        print('> %s\t%s' % (colored('%.1f' % score[idx], 'cyan'), colored(questions[idx], 'yellow')))


if __name__ == '__main__':
    sys.exit(main() or 0)
