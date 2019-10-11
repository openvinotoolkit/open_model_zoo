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


from openvino.inference_engine import IENetwork, IECore
import cv2
import numpy as np
import sys
import os
import logging as log
from argparse import ArgumentParser, SUPPRESS

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument('-i',
                      dest='input_source',
                      help='Required. Path to the input image.',
                      required=True, type=str, metavar='"<path>"')
    args.add_argument('-m', '--model',
                      help='Required. Path to an .xml file with a trained model.',
                      required=True, type=str, metavar='"<path>"')
    args.add_argument('-d', '--device',
                      help='Optional. Specify the target device to infer on: CPU, GPU, FPGA, HDDL or MYRIAD. '
                           'The demo will look for a suitable plugin for device specified '
                           '(by default, it is CPU).',
                           default='CPU', type=str, metavar='"<device>"')
    args.add_argument('-l', '--cpu_extension',
                      help='Required for CPU custom layers. '
                           'Absolute path to a shared library with the kernels implementation.',
                      default=None, type=str, metavar='"<absolute_path>"')
    return parser

def ctc_beam_search_decoder(inference_result):
    res=np.argmax(inference_result, 2)
    #delete repetitions
    j=''
    unique=[]
    for i in np.squeeze(res):
        if i != j:
            unique.append(i)
            j=i
    #delete a delimiter (36)
    decoded=[]
    for i in unique:
        if i != 36:
            decoded.append(i)
    return decoded
    
def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    
    model_xml = args.model
    model_bin = os.path.splitext(model_xml)[0] + '.bin'
    net = IENetwork(model_xml, model_bin)
    image_path = args.input_source
    input_image = cv2.imread(image_path)
    
    log.info('Creating Inference Engine...')
    ie = IECore() 
    
    if args.cpu_extension and 'CPU' in args.device:
        ie.add_extension(args.cpu_extension, 'CPU')
        
    if 'CPU' in args.device:
        supported_layers = ie.query_network(net, 'CPU')
        not_supported_layers = [l for l in net.layers.keys() if l not in supported_layers]
        if len(not_supported_layers) != 0:
            log.error('Following layers are not supported by the plugin for specified device {}:\n {}'.
                      format(args.device, ', '.join(not_supported_layers)))
            log.error("Please try to specify cpu extensions library path in sample's command line parameters using -l "
                      "or --cpu_extension command line argument")
            sys.exit(1)

    #load model to the plugin
    log.info('Loading model to the plugin')
    exec_net = ie.load_network(network=net, device_name=args.device)
    
    #prepare input
    log.info("Preparing input blobs...")
    input_blob = next(iter(net.inputs))
    n, c, h, w = net.inputs[input_blob].shape
    image = cv2.resize(input_image, (w, h))
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image,axis=0)

    #run inference
    log.info('Starting inference...')
    res = exec_net.infer(inputs={input_blob: image})
    out_blob = next(iter(net.outputs))
    res = res[out_blob]
    
    #decode the inference result
    log.info('Decoding the inference result...')
    decoded=ctc_beam_search_decoder(res)

    #match decoded array to the alphabet
    alphabet="abcdefghijklmnopqrstuvwxyz0123456789"
    word=[]
    for i in decoded:
        word.append(alphabet[int(i)])
    msg='Detected text is "{}"'.format("".join(word))
    log.info(msg)
    cv2.imshow(msg,input_image)
    cv2.waitKey()
    log.info('Demo finished successfully')
    log.info("This demo is an API example, for any performance measurements please use the dedicated benchmark_app tool "
             "from the openVINO toolkit\n")

if __name__ == '__main__':
    sys.exit(main() or 0)
