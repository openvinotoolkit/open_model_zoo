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
import data_utils
from six.moves import xrange
import tensorflow as tf

BATCH_SIZE = 1
NUM_TIMESTEPS = 1
MAX_WORD_LEN = 50

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.", required=True,
                      type=str)
    args.add_argument("-i", "--input", help="Required. The prefix used for word predictions",
                      required=True,
                      type=str)
    args.add_argument("-v", "--vocab", help="Required. The vocab file",
                      required=True,
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
    args.add_argument("-n", "--number_samples",
                      help="Optional. Set number of samples. "
                           "number of samples", type=int, default=1)
    return parser

def _SampleSoftmax(softmax):
  return min(np.sum(np.cumsum(softmax) < np.random.rand()), len(softmax) - 1)

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + ".bin"
    prefix_words = args.input
    number_samples = args.number_samples
    
    print("open vocab file: {}".format(args.vocab))
    vocab = data_utils.CharsVocabulary(args.vocab, MAX_WORD_LEN)
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

    #assert len(net.inputs.keys()) == 1, "Sample supports only single input topologies"
    #assert len(net.outputs) == 1, "Sample supports only single output topologies"
    print(net.inputs)
    print(net.inputs['Variable_1/read/placeholder_port_0'].shape)
    print(net.inputs['Variable/read/placeholder_port_0'].shape)
    print(net.inputs['char_embedding/Reshape/placeholder_port_0'].shape)  ####
    print("outputs: {}".format(net.outputs))
 
    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=args.device)

   
    log.info("Preparing input blobs")
    #inter_inputs = iter(net.inputs)
    #input_blobs = {}
    #for i in range(len(net.inputs)):
    #    input_blobs[i] = next(inter_inputs)
    #    print(input_blobs[i])
    var = np.zeros([1, 9216], np.float32)
    var1 = np.zeros([1, 9216], np.float32)
  
    print("prefix_words:{}".format(prefix_words))
    if prefix_words.find('<S>') != 0:
        prefix_words = '<S> ' + prefix_words

    prefix = [vocab.word_to_id(w) for w in prefix_words.split()]
    print("prefix:{}".format(prefix))
    prefix_char_ids = [vocab.word_to_char_ids(w) for w in prefix_words.split()]
    print("prefix_char_ids:{}".format(prefix_char_ids))
    for _ in xrange(number_samples):
        inputs = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
        char_ids_inputs = np.zeros([BATCH_SIZE, NUM_TIMESTEPS, vocab.max_word_length], np.int32)
        
        samples = prefix[:]
        print("samples: {}".format(prefix))
        char_ids_samples = prefix_char_ids[:]
        print("char_ids_samples: {}".format(char_ids_samples))
        sent = ''
        while True:
            inputs[0, 0] = samples[0]
            print("inputs: {}".format(inputs))
            char_ids_inputs[0, 0, :] = char_ids_samples[0]  #input source
            print("char_id_inputs: {}".format(char_ids_inputs))
            samples = samples[1:]
            char_ids_samples = char_ids_samples[1:]  
            
            res = exec_net.infer(inputs={'char_embedding/Reshape/placeholder_port_0':char_ids_inputs,
                                         'Variable/read/placeholder_port_0':var,
                                         'Variable_1/read/placeholder_port_0':var1})  

            # Processing output blob
            print("Processing output blob")
            softmax_out = res["softmax_out"]
            print("softmax_out[0]: {}".format(softmax_out[0]))
            var = res["lstm/lstm_0/concat_2"][0]
            print("var: {}".format(var))
            var1 = res["lstm/lstm_1/concat_2"][0]
            print("var1: {}".format(var1))

            sample = _SampleSoftmax(softmax_out[0])
            print("sample: {}".format(sample))
            sample_char_ids = vocab.word_to_char_ids(vocab.id_to_word(sample))
            print("sample_char_ids: {}".format(sample_char_ids))
            if not samples:
                samples = [sample]
                char_ids_samples = [sample_char_ids]
            sent += vocab.id_to_word(samples[0]) + ' '
            sys.stderr.write('%s\n' % sent)

            if (vocab.id_to_word(samples[0]) == '</S>' or len(sent) > 100):
                break

 
'''
    log.info("Preparing input blobs")
    input_blob = next(iter(net.inputs))
    out_blob = next(iter(net.outputs))
    net.batch_size = len(args.input)

    # Read and pre-process input images
    n, c, h, w = net.inputs[input_blob].shape
    images = np.ndarray(shape=(n, c, h, w))
    for i in range(n):
        image = cv2.imread(args.input[i])
        if image.shape[:-1] != (h, w):
            log.warning("Image {} is resized from {} to {}".format(args.input[i], image.shape[:-1], (h, w)))
            image = cv2.resize(image, (w, h))
        image = image.transpose((2, 0, 1))  # Change data layout from HWC to CHW
        images[i] = image
    log.info("Batch size is {}".format(n))

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=args.device)

    # Start sync inference
    log.info("Starting inference in synchronous mode")
    res = exec_net.infer(inputs={input_blob: images})

    # Processing output blob
    log.info("Processing output blob")
    res = res[out_blob]
    log.info("Top {} results: ".format(args.number_top))
    if args.labels:
        with open(args.labels, 'r') as f:
            labels_map = [x.split(sep=' ', maxsplit=1)[-1].strip() for x in f]
    else:
        labels_map = None
    classid_str = "classid"
    probability_str = "probability"
    for i, probs in enumerate(res):
        probs = np.squeeze(probs)
        top_ind = np.argsort(probs)[-args.number_top:][::-1]
        print("Image {}\n".format(args.input[i]))
        print(classid_str, probability_str)
        print("{} {}".format('-' * len(classid_str), '-' * len(probability_str)))
        for id in top_ind:
            det_label = labels_map[id] if labels_map else "{}".format(id)
            label_length = len(det_label)
            space_num_before = (len(classid_str) - label_length) // 2
            space_num_after = len(classid_str) - (space_num_before + label_length) + 2
            space_num_before_prob = (len(probability_str) - len(str(probs[id]))) // 2
            print("{}{}{}{}{:.7f}".format(' ' * space_num_before, det_label,
                                          ' ' * space_num_after, ' ' * space_num_before_prob,
                                          probs[id]))
        print("\n")
    log.info("This sample is an API example, for any performance measurements please use the dedicated benchmark_app tool\n")
'''
if __name__ == '__main__':
    sys.exit(main() or 0)
