#!/usr/bin/env python
"""
 Copyright (C) 2019 Intel Corporation

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
import numpy as np
import logging as log
from time import time
from openvino.inference_engine import IENetwork, IECore

# For Speech Feature
import scipy.io.wavfile as wav
import codecs

from speech_features import audio_spectrogram, mfcc 
from ctc_beamsearch_decoder import CTCBeamSearchDecoder


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.", required=True,
                      type=str)
    args.add_argument("-i", "--input", help="Required. Path to a folder with images or path to an image files",
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
    args.add_argument("-a", "--alphabet", help="Path to a alphabet file", required=True, default="alphabet_b.txt", type=str)

    return parser

class Alphabet(object):
    def __init__(self, config_file):
        self._label_to_str = []
        self._str_to_label = {}
        self._size = 0
        with codecs.open(config_file, 'r', 'utf-8') as fin:
            for line in fin:
                if line[0:2] == '\\#':
                    line = '#\n'
                elif line[0] == '#':
                    continue
                self._label_to_str += line[:-1] # remove the line ending
                self._str_to_label[line[:-1]] = self._size
                self._size += 1

    def string_from_label(self, label):
        return self._label_to_str[label]

    def label_from_string(self, string):
        return self._str_to_label[string]

    def size(self):
        return self._size

n_input    = 26
n_context  = 9
n_steps    = 16
numcep     = n_input
numcontext = n_context
beamwidth  = 10

def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s", level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    
    alphabet = Alphabet(os.path.abspath(args.alphabet))
    
    # Speech feature extration
    fs, audio = wav.read(args.input)
    audio = audio/np.float32(32768) # normalize to -1 to 1, int 16 to float32
 
    audio = audio.reshape(-1, 1)
    spectrogram = audio_spectrogram(audio, (16000 * 32 / 1000), (16000 * 20 / 1000), True)
    features = mfcc(spectrogram.reshape(1, spectrogram.shape[0], -1), fs, 26)

    empty_context = np.zeros((numcontext, numcep), dtype=features.dtype)
    features = np.concatenate((empty_context, features, empty_context))    

    num_strides = len(features) - (n_context * 2)
    # Create a view into the array with overlapping strides of size
    # numcontext (past) + 1 (present) + numcontext (future)
    window_size = 2*n_context+1
    features = np.lib.stride_tricks.as_strided(
        features,
        (num_strides, window_size, n_input),
        (features.strides[0], features.strides[0], features.strides[1]),
        writeable=False)

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

    assert len(net.inputs.keys()) == 3, "Sample supports only three input topologies"
    assert len(net.outputs) == 3, "Sample supports only three output topologies"

    log.info("Preparing input blobs")
    input_iter = iter(net.inputs)
    input_blob1 = next(input_iter)
    input_blob2 = next(input_iter)
    input_blob3 = next(input_iter)
    
    log.info("Preparing output blobs")
    output_iter = iter(net.outputs)
    output_blob1 = next(output_iter)
    output_blob2 = next(output_iter)
    output_blob3 = next(output_iter)

    # Loading model to the plugin
    log.info("Loading model to the plugin")
    exec_net = ie.load_network(network=net, device_name=args.device)

    state_h = np.zeros((1, 2048))
    state_c = np.zeros((1, 2048))
    logits = np.empty([0, 1, alphabet.size()])    

    for i in range(0, len(features), n_steps):
        chunk = features[i:i+n_steps]
        
        if len(chunk) < n_steps:
            chunk = np.pad(chunk,
                           (
                            (0, n_steps - len(chunk)),
                            (0, 0),
                            (0, 0)
                           ),
                           mode='constant',
                           constant_values=0)

        res = exec_net.infer(inputs={'previous_state_c/read/placeholder_port_0': state_c,
                                     'previous_state_h/read/placeholder_port_0': state_h,
                                      'input_node': [chunk]})
                                      
        # Processing output blob
        #log.info("Processing output blob")
        logits = np.concatenate((logits, res['Softmax']))
        state_h = res['lstm_fused_cell/BlockLSTM/TensorIterator.1']
        state_c = res['lstm_fused_cell/BlockLSTM/TensorIterator.2']
        
    print("\n>>>{}\n".format(CTCBeamSearchDecoder(logits, alphabet._label_to_str, alphabet.string_from_label(-1), beamwidth)))

if __name__ == '__main__':
    sys.exit(main() or 0)

