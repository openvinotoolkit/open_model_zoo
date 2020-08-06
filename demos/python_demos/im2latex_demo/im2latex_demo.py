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

import logging as log
import os
import sys
from argparse import SUPPRESS, ArgumentParser

import cv2 as cv
import numpy as np
from tqdm import tqdm

from utils import END_TOKEN, START_TOKEN, Vocab
from openvino.inference_engine import IECore

def crop(img, target_shape):
    target_height, target_width = target_shape
    img_h, img_w = img.shape[0:2]
    new_w = min(target_width, img_w)
    new_h = min(target_height, img_h)
    img = img[:new_h, :new_w, :]
    return img


def resize(img, target_shape):
    target_height, target_width = target_shape
    img_h, img_w = img.shape[0:2]
    scale = min(target_height / img_h, target_width / img_w)
    img = cv.resize(img, None, fx=scale, fy=scale)
    return img


PREPROCESSING = {
    'crop': crop,
    'resize': resize
}

COLOR_WHITE = (255, 255, 255)


def print_stats(module):
    perf_counts = module.requests[0].get_perf_counts()
    print('{:<70} {:<15} {:<15} {:<15} {:<10}'.format('name', 'layer_type', 'exet_type', 'status',
                                                      'real_time, us'))
    for layer, stats in perf_counts.items():
        print('{:<70} {:<15} {:<15} {:<15} {:<10}'.format(layer, stats['layer_type'], stats['exec_type'],
                                                          stats['status'], stats['real_time']))


def preprocess_image(preprocess, image_raw, tgt_shape):
    target_height, target_width = tgt_shape
    image_raw = preprocess(image_raw, tgt_shape)
    img_h, img_w = image_raw.shape[0:2]
    image_raw = cv.copyMakeBorder(image_raw, 0, target_height - img_h,
                                  0, target_width - img_w, cv.BORDER_CONSTANT,
                                  None, COLOR_WHITE)
    image = image_raw.transpose((2, 0, 1))
    return image


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help',
                      default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m_encoder", help="Required. Path to an .xml file with a trained encoder part of the model",
                      required=True, type=str)
    args.add_argument("-m_decoder", help="Required. Path to an .xml file with a trained decoder part of the model",
                      required=True, type=str)
    args.add_argument("-i", "--input", help="Required. Path to a folder with images or path to an image files",
                      required=True, type=str)
    args.add_argument("-o", "--output_file",
                      help="Optional. Path to file where to store output. If not mentioned, result will be stored"
                      "in the console.",
                      type=str)
    args.add_argument("--vocab_path", help="Required. Path to vocab file to construct meaningful phrase",
                      type=str, required=True)
    args.add_argument("--max_formula_len",
                      help="Optional. Defines maximum length of the formula (number of tokens to decode)",
                      default="128", type=int)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is "
                           "acceptable. Sample will look for a suitable plugin for device specified. Default value is CPU",
                      default="CPU", type=str)
    args.add_argument('--preprocessing_type', choices=PREPROCESSING.keys(),
                      help="Required. Type of the preprocessing", required=True, default='Crop')
    args.add_argument('-pc', '--perf_counts',
                      action='store_true', default=False)
    return parser


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    log.info("Creating Inference Engine")
    ie = IECore()
    ie.set_config(
        {"PERF_COUNT": "YES" if args.perf_counts else "NO"}, args.device)

    # encoder part
    encoder_model_xml = args.m_encoder
    encoder_model_bin = os.path.splitext(encoder_model_xml)[0] + ".bin"
    log.info("Loading encoder files:\n\t{}\n\t{}".format(
        encoder_model_xml, encoder_model_bin))
    encoder = ie.read_network(encoder_model_xml, encoder_model_bin)

    # check if all layers are supported
    supported_layers = ie.query_network(encoder, args.device)
    not_supported_layers = [
        l for l in encoder.layers.keys() if l not in supported_layers]
    if len(not_supported_layers) != 0:
        log.error("Following layers are not supported by the plugin for specified device {}:\n{}".
                  format(args.device, ', '.join(not_supported_layers)))

    # decoder part:
    dec_step_model_xml = args.m_decoder
    dec_step_model_bin = os.path.splitext(dec_step_model_xml)[0] + ".bin"

    log.info("Loading decoder files:\n\t{}\n\t{}".format(
        dec_step_model_xml, dec_step_model_bin))
    dec_step = ie.read_network(dec_step_model_xml, dec_step_model_bin)

    # check if all layers are supported
    supported_layers = ie.query_network(dec_step, args.device)
    not_supported_layers = [
        l for l in dec_step.layers.keys() if l not in supported_layers]
    if len(not_supported_layers) != 0:
        log.error("Following layers are not supported by the plugin for specified device {}:\n{}".
                  format(args.device, ', '.join(not_supported_layers)))
    _, _, height, width = encoder.input_info['imgs'].input_data.shape
    target_shape = (height, width)
    images_list = []
    if os.path.isdir(args.input):
        inputs = [os.path.join(args.input, inp)
                  for inp in os.listdir(args.input)]
    else:
        inputs = [args.input]
    log.info("Loading vocab file")
    vocab = Vocab(args.vocab_path)

    log.info("Loading and preprocessing images")
    for filenm in tqdm(inputs):
        image_raw = cv.imread(filenm)
        assert image_raw is not None, "Error reading image {}".format(filenm)
        image = preprocess_image(
            PREPROCESSING[args.preprocessing_type], image_raw, target_shape)
        record = dict(img_name=filenm, img=image, formula=None)
        images_list.append(record)

    log.info("Loading networks")
    exec_net_encoder = ie.load_network(
        network=encoder, device_name=args.device)
    exec_net_decoder = ie.load_network(
        network=dec_step, device_name=args.device)

    log.info("Starting inference")

    for rec in tqdm(images_list):
        image = rec['img']

        enc_res = exec_net_encoder.infer(
            inputs={'imgs': np.expand_dims(image, axis=0)})
        # get results
        row_enc_out = enc_res['row_enc_out']
        dec_states_h = enc_res['hidden']
        dec_states_c = enc_res['context']
        O_t = enc_res['init_0']

        tgt = np.array([[START_TOKEN]])
        logits = []
        for _ in range(args.max_formula_len):
            dec_res = exec_net_decoder.infer(inputs={'row_enc_out': row_enc_out,
                                                     'dec_st_c': dec_states_c, 'dec_st_h': dec_states_h,
                                                     'O_t_minus_1': O_t, 'tgt': tgt
                                                     })

            dec_states_h = dec_res['dec_st_h_t']
            dec_states_c = dec_res['dec_st_c_t']
            O_t = dec_res['O_t']
            logit = dec_res['logit']
            logits.append(logit)
            tgt = np.array([[np.argmax(logit, axis=1)]])

            if tgt[0][0][0] == END_TOKEN:
                break
        if args.perf_counts:
            log.info("Encoder perfomane statistics")
            print_stats(exec_net_encoder)
            log.info("Decoder perfomane statistics")
            print_stats(exec_net_decoder)

        logits = np.array(logits)
        logits = logits.squeeze(axis=1)
        targets = np.argmax(logits, axis=1)
        rec["formula"] = vocab.construct_phrase(targets)
    if args.output_file:
        log.info("Writing results to the file {}".format(args.output_file))
        with open(args.output_file, 'w') as output_file:
            for rec in images_list:
                output_file.write(rec['img_name'] +
                                  '\t' + rec['formula'] + '\n')
    else:
        for rec in images_list:
            print("Image name: {}\nformula: {} \n".format(
                rec['img_name'], rec['formula']))

    log.info("This demo is an API example, for any performance measurements please use the dedicated benchmark_app tool "
             "from the openVINO toolkit\n")


if __name__ == '__main__':
    sys.exit(main() or 0)
