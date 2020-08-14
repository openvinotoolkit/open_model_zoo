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

from openvino.inference_engine import IECore

from utils import END_TOKEN, START_TOKEN, Vocab


def crop(img, target_shape):
    target_height, target_width = target_shape
    img_h, img_w = img.shape[0:2]
    new_w = min(target_width, img_w)
    new_h = min(target_height, img_h)
    return img[:new_h, :new_w, :]


def resize(img, target_shape):
    target_height, target_width = target_shape
    img_h, img_w = img.shape[0:2]
    scale = min(target_height / img_h, target_width / img_w)
    return cv.resize(img, None, fx=scale, fy=scale)


PREPROCESSING = {
    'crop': crop,
    'resize': resize
}

COLOR_WHITE = (255, 255, 255)


def read_net(model_xml, ie, device):
    model_bin = os.path.splitext(model_xml)[0] + ".bin"

    log.info("Loading network files:\n\t{}\n\t{}".format(model_xml, model_bin))
    model = ie.read_network(model_xml, model_bin)

    supported_layers = ie.query_network(model, device)
    not_supported_layers = [l for l in model.layers.keys() if l not in supported_layers]
    if len(not_supported_layers) != 0:
        log.error("Following layers are not supported by the plugin for specified device {}:\n{}".
                  format(device, ', '.join(not_supported_layers)))
    return model


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
    return np.expand_dims(image, axis=0)


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
                      help="Optional. Type of the preprocessing", default='crop')
    args.add_argument('-pc', '--perf_counts',
                      action='store_true', default=False)
    args.add_argument('--imgs_layer', help='Optional. Encoder input name for images. See README for details.',
                      default='imgs')
    args.add_argument('--row_enc_out_layer', help='Optional. Encoder output key for row_enc_out. See README for details.',
                      default='row_enc_out')
    args.add_argument('--hidden_layer', help='Optional. Encoder output key for hidden. See README for details.',
                      default='hidden')
    args.add_argument('--context_layer', help='Optional. Encoder output key for context. See README for details.',
                      default='context')
    args.add_argument('--init_0_layer', help='Optional. Encoder output key for init_0. See README for details.',
                      default='init_0')
    args.add_argument('--dec_st_c_layer', help='Optional. Decoder input key for dec_st_c. See README for details.',
                      default='dec_st_c')
    args.add_argument('--dec_st_h_layer', help='Optional. Decoder input key for dec_st_h. See README for details.',
                      default='dec_st_h')
    args.add_argument('--dec_st_c_t_layer', help='Optional. Decoder output key for dec_st_c_t. See README for details.',
                      default='dec_st_c_t')
    args.add_argument('--dec_st_h_t_layer', help='Optional. Decoder output key for dec_st_h_t. See README for details.',
                      default='dec_st_h_t')
    args.add_argument('--output_layer', help='Optional. Decoder output key for output. See README for details.',
                      default='output')
    args.add_argument('--output_prev_layer', help='Optional. Decoder input key for output_prev. See README for details.',
                      default='output_prev')
    args.add_argument('--logit_layer', help='Optional. Decoder output key for logit. See README for details.',
                      default='logit')
    args.add_argument('--tgt_layer', help='Optional. Decoder input key for tgt. See README for details.',
                      default='tgt')
    return parser


def main():
    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    log.info("Creating Inference Engine")
    ie = IECore()
    ie.set_config(
        {"PERF_COUNT": "YES" if args.perf_counts else "NO"}, args.device)

    encoder = read_net(args.m_encoder, ie, args.device)
    dec_step = read_net(args.m_decoder, ie, args.device)

    batch_dim, channels, height, width = encoder.input_info['imgs'].input_data.shape
    assert batch_dim == 1, "Demo only works with batch size 1."
    assert channels in (1, 3), "Input image is not 1 or 3 channeled image."
    target_shape = (height, width)
    images_list = []
    if os.path.isdir(args.input):
        inputs = sorted(os.path.join(args.input, inp)
                        for inp in os.listdir(args.input))
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
    exec_net_encoder = ie.load_network(network=encoder, device_name=args.device)
    exec_net_decoder = ie.load_network(network=dec_step, device_name=args.device)

    log.info("Starting inference")
    for rec in tqdm(images_list):
        image = rec['img']

        enc_res = exec_net_encoder.infer(inputs={args.imgs_layer: image})
        # get results
        row_enc_out = enc_res[args.row_enc_out_layer]
        dec_states_h = enc_res[args.hidden_layer]
        dec_states_c = enc_res[args.context_layer]
        output = enc_res[args.init_0_layer]

        tgt = np.array([[START_TOKEN]])
        logits = []
        for _ in range(args.max_formula_len):
            dec_res = exec_net_decoder.infer(inputs={args.row_enc_out_layer: row_enc_out,
                                                     args.dec_st_c_layer: dec_states_c,
                                                     args.dec_st_h_layer: dec_states_h,
                                                     args.output_prev_layer: output,
                                                     args.tgt_layer: tgt
                                                     }
                                             )

            dec_states_h = dec_res[args.dec_st_h_t_layer]
            dec_states_c = dec_res[args.dec_st_c_t_layer]
            output = dec_res[args.output_layer]
            logit = dec_res[args.logit_layer]
            logits.append(logit)
            tgt = np.array([[np.argmax(logit, axis=1)]])

            if tgt[0][0][0] == END_TOKEN:
                break
        if args.perf_counts:
            log.info("Encoder perfomance statistics")
            print_stats(exec_net_encoder)
            log.info("Decoder perfomanсe statistics")
            print_stats(exec_net_decoder)

        logits = np.array(logits)
        logits = logits.squeeze(axis=1)
        targets = np.argmax(logits, axis=1)
        if args.output_file:
            with open(args.output_file, 'a') as output_file:
                output_file.write(rec['img_name'] + '\t' +  vocab.construct_phrase(targets) + '\n')
        else:
            print("Image name: {}\nFormula: {}\n".format(rec['img_name'], vocab.construct_phrase(targets)))

    log.info("This demo is an API example, for any performance measurements please use the dedicated benchmark_app tool "
             "from the openVINO toolkit\n")


if __name__ == '__main__':
    sys.exit(main() or 0)
