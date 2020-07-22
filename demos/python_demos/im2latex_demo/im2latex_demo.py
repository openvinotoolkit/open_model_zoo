from __future__ import print_function

import datetime
import logging as log
import os
import sys
from argparse import SUPPRESS, ArgumentParser
from collections import namedtuple
from copy import deepcopy
from time import time

import cv2
import numpy as np
from tqdm import tqdm

import os,sys,inspect

from utils import BatchCropPadToTGTShape as preprocess
from utils import END_TOKEN, START_TOKEN, read_vocab
from openvino.inference_engine import IECore, IENetwork

COLOR_WHITE = 255


def print_stats(module):
    perf_counts = module.requests[0].get_perf_counts()
    print('{:<70} {:<15} {:<15} {:<15} {:<10}'.format('name', 'layer_type', 'exet_type', 'status',
                                                      'real_time, us'))
    for layer, stats in perf_counts.items():
        print('{:<70} {:<15} {:<15} {:<15} {:<10}'.format(layer, stats['layer_type'], stats['exec_type'],
                                                          stats['status'], stats['real_time']))


def preprocess_image(image_raw, tgt_shape):
    target_height, target_width = tgt_shape
    print(image_raw.shape)
    image_raw = preprocess(tgt_shape)(image_raw)[0]
    cv2.imwrite("debug_preproc.png", image_raw)
    assert image_raw.shape[0] == target_height and image_raw.shape[1] == target_width, image_raw.shape
    image = image_raw / float(COLOR_WHITE)
    image = image.transpose((2, 0, 1))
    return image


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help',
                      default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("--encoder", help="Required. Path to an .xml file with a trained encoder part of the model",
                      required=True, type=str)
    args.add_argument("--dec_step", help="Required. Path to an .xml file with a trained decoder step part of the model",
                      required=True, type=str)
    args.add_argument("-i", "--input", help="Required. Path to a folder with images or path to an image files",
                      required=True, type=str, nargs="+")
    args.add_argument("-o", "--output_file",
                      help="Optional. Path to file where to store output. If not mentioned, result will be stored"
                      "in the console.",
                      type=str)
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. "
                           "Absolute MKLDNN (CPU)-targeted custom layers. Absolute path to a shared library with the "
                           "kernels implementations", type=str)
    args.add_argument("--vocab_path", help="Path to vocab file to construct meaningful phrase",
                      default='../print_dataset/vocab.pkl', type=str)
    args.add_argument("--target_shape", help="Required. Target image shape (height, width). "
                      "Example: 100 500",
                      required=True, type=int, nargs="+")
    args.add_argument("--max_formula_len",
                      help="Optional. Defines maximum length of the formula (number of tokens to decode)",
                      default="128", type=int)
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is "
                           "acceptable. Sample will look for a suitable plugin for device specified. Default value is CPU",
                      default="CPU", type=str)
    args.add_argument('-pf', '--perf_stats',
                      action='store_true', default=False)

    return parser


def main():

    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()
    target_shape = tuple(args.target_shape)
    log.info("Creating Inference Engine")
    ie = IECore()
    if args.cpu_extension:
        ie.add_extension(args.cpu_extension, 'CPU')
    ie.set_config(
        {"PERF_COUNT": "YES" if args.perf_stats else "NO"}, args.device)

    # encoder part
    encoder_model_xml = args.encoder
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
        # sys.exit(1)

    # add outputs because MO cuts them
    # encoder.add_outputs(['row_enc_out/sink_port_0', 'hidden/sink_port_0', 'context/sink_port_0'])

    # Loading model to the plugin
    # log.info("Loading encoder and decoder to the plugin")
    # exec_net_encoder = ie.load_network(network=encoder, device_name=args.device)

    # decoder part:
    dec_step_model_xml = args.dec_step
    dec_step_model_bin = os.path.splitext(dec_step_model_xml)[0] + ".bin"

    log.info("Loading decoder files:\n\t{}\n\t{}".format(
        dec_step_model_xml, dec_step_model_bin))
    dec_step = ie.read_network(dec_step_model_xml, dec_step_model_bin)
    #ie.read_network(model=dec_step_model_xml, weights=dec_step_model_bin)

    # add decoder outputs
    dec_step.add_outputs(["dec_st_c_t", "dec_st_h_t", "O_t"])

    # exec_net_decoder = ie.load_network(network=dec_step, device_name=args.device)

    # check if all layers are supported
    supported_layers = ie.query_network(dec_step, args.device)
    not_supported_layers = [
        l for l in dec_step.layers.keys() if l not in supported_layers]
    if len(not_supported_layers) != 0:
        log.error("Following layers are not supported by the plugin for specified device {}:\n{}".
                  format(args.device, ', '.join(not_supported_layers)))
        # sys.exit(1)

    # read and preprocess images
    # Images = namedtuple("Images", "img_name, img, formula")

    images_list = []
    inputs = list(args.input)
    log.info("Loading vocab file")
    vocab = read_vocab(args.vocab_path)

    log.info("Loading and preprocessing images")
    for filenm in tqdm(inputs):
        image_raw = cv2.imread(filenm)
        assert image_raw is not None, "Error reading image {}".format(filenm)
        #
        image = image_raw
        image = preprocess_image(image_raw, target_shape)
        record = dict(img_name=filenm, img=image, formula=None)
        images_list.append(record)

    # inference
    log.info("Starting inference")

    for rec_idx, rec in enumerate(tqdm(images_list)):
        image = rec['img']

        exec_net_encoder = ie.load_network(
            network=encoder, device_name=args.device)
        exec_net_decoder = ie.load_network(
            network=dec_step, device_name=args.device)

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
            tgt = np.array([[np.argmax(np.array(logit), axis=1)]])

            if tgt[0][0][0] == END_TOKEN:
                break
        if args.perf_stats:
            log.info("Encoder perfomane statistics")
            print_stats(exec_net_encoder)
            log.info("Decoder perfomane statistics")
            print_stats(exec_net_decoder)

        logits = np.array(logits)
        logits = logits.squeeze(axis=1)
        targets = np.argmax(logits, axis=1)
        result_phrase = vocab.construct_phrase(targets)

        rec["formula"] = result_phrase
    if args.output_file:
        log.info("Writing results to file {}".format(args.output_file))
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
