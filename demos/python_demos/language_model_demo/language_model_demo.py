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
from perplexity import Calculate_Perplexity
from unique import Calculate_Unique

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
    args.add_argument("-p", "--perplexity",
                      help="Optional. Calculate perplexity. ",
                      type=int, default=0)
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
    show_perplexity = bool(int(args.perplexity))
    
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
 
    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=args.device)

   
    log.info("Preparing input blobs")
    var = np.zeros([1, 9216], np.float32)
    var1 = np.zeros([1, 9216], np.float32)
  
    if prefix_words.find('<S>') != 0:
        prefix_words = '<S> ' + prefix_words

    prefix = [vocab.word_to_id(w) for w in prefix_words.split()]
    prefix_char_ids = [vocab.word_to_char_ids(w) for w in prefix_words.split()]

    if show_perplexity is True:
        sum_num = 0.0
        sum_den = 0.0
        sentence_perplexity = 0.0
        target_weights_in = np.ones([1, 1], np.float32)
    for _ in xrange(number_samples):
        inputs = np.zeros([BATCH_SIZE, NUM_TIMESTEPS], np.int32)
        char_ids_inputs = np.zeros([BATCH_SIZE, NUM_TIMESTEPS, vocab.max_word_length], np.int32)
        
        samples = prefix[:]
        char_ids_samples = prefix_char_ids[:]
        sent = ''
        while True:
            inputs[0, 0] = samples[0]
            char_ids_inputs[0, 0, :] = char_ids_samples[0]  #input source
            samples = samples[1:]
            char_ids_samples = char_ids_samples[1:]  

            cal_unique = Calculate_Unique(char_ids_inputs)
            y, idx = cal_unique._Unique()
            res = exec_net.infer(inputs={'char_embedding/EmbeddingLookupUnique/Unique/placeholder_out_port_0':y,
                                         'char_embedding/EmbeddingLookupUnique/Unique/placeholder_out_port_1':idx,
                                         'Variable/read/placeholder_port_0':var,
                                         'Variable_1/read/placeholder_port_0':var1})

            # Processing output blob
            softmax_out = res["softmax_out"]
            var = res["lstm/lstm_0/concat_2"][0]
            var1 = res["lstm/lstm_1/concat_2"][0]

            sample = _SampleSoftmax(softmax_out[0])
            sample_char_ids = vocab.word_to_char_ids(vocab.id_to_word(sample))
            if not samples:
                samples = [sample]
                char_ids_samples = [sample_char_ids]
            sent += vocab.id_to_word(samples[0]) + ' '
            sys.stderr.write('%s\n' % sent)

            if show_perplexity is True:
                tgts = np.array([[samples[0]]])
                cal_log_perp = Calculate_Perplexity(tgts, _, softmax_out[0])
                log_perp = cal_log_perp._log_perplexity_out()
                sum_num += log_perp * target_weights_in.mean()
                sum_den += target_weights_in.mean()
                if sum_den > 0:
                    sentence_perplexity = np.exp(sum_num / sum_den)

            if (vocab.id_to_word(samples[0]) == '</S>' or len(sent) > 100):
                break

    if show_perplexity is True:
        sys.stderr.write("Eval sentence perplexity: %f.\n" % sentence_perplexity)

if __name__ == '__main__':
    sys.exit(main() or 0)
