#!/usr/bin/env python
"""
 Copyright (c) 2018 Intel Corporation

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
import cv2 as cv
import numpy as np
import os
from argparse import ArgumentParser, SUPPRESS
import xml.etree.ElementTree as ET
import re
import logging as log
import sys


def build_arg():
    parser = ArgumentParser(add_help=False)
    in_args = parser.add_argument_group('Options')
    in_args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Help with the script.')
    in_args.add_argument("-m", "--model", help="Required. Path to .xml file with pre-trained model",
                         required=True, type=str)
    in_args.add_argument("-d", "--device",
                         help="Optional. Specify target device for infer: CPU, GPU, FPGA, HDDL or MYRIAD. "
                         "Default: CPU",
                         default="CPU", type=str)
    in_args.add_argument("-i", "--input", help="Optional. Path to a test video file.", type=str)
    in_args.add_argument("-n", "--no_show", help="Optional. Disable display of results on screen.",
                         action='store_true', default=False)
    in_args.add_argument("-v", "--verbose", help="Optional. Enable display of processing logs on screen.",
                         action='store_true', default=False)
    return parser


def get_parameters(xml_path):
    root = ET.parse(xml_path).getroot()
    pattern = r'data\[([0-9]*\.*[0-9]*)\]'
    mean_data = root.findall("./meta_data/cli_parameters/mean_values")[0]
    mean_const = float(re.match(pattern, mean_data.attrib['value']).group(1))
    return mean_const


def get_model_files(config_xml_in):
    model_path = os.path.splitext(config_xml_in)[0]
    weights_bin_in = model_path + ".bin"
    level_up = os.path.split(os.path.split(model_path)[0])[0]
    coeffs_in = os.path.join(level_up, os.path.split(level_up)[1]) + ".npy"
    return config_xml_in, weights_bin_in, coeffs_in


if __name__ == '__main__':
    args = build_arg().parse_args()
    config_xml, weights_bin, coeffs = get_model_files(args.model)

    # mean are stored in the source caffe model and passed to IR
    mean = get_parameters(config_xml)

    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.INFO if not args.verbose else log.DEBUG, stream=sys.stdout)

    log.debug("Load network")
    load_net = IENetwork(model=config_xml, weights=weights_bin)
    exec_net = IECore().load_network(network=load_net, device_name=args.device)

    assert len(load_net.inputs) == 1, "Number of inputs does not match network inputs"
    input_blob = next(iter(load_net.inputs))
    input_shape = load_net.inputs[input_blob].shape
    assert input_shape[1] == 1, "Expected model output shape with 1 channel"

    assert len(load_net.outputs) == 1, "Number of outputs does not match network outputs"
    output_blob = next(iter(load_net.outputs))
    output_shape = load_net.outputs[output_blob].shape
    assert output_shape == [1, 313, 56, 56], "Shape of outputs does not match network shape outputs"

    _, _, h_in, w_in = input_shape

    try:
        input_source = args.input
    except TypeError:
        input_source = 0
    cap = cv.VideoCapture(input_source)

    color_coeff = np.load(coeffs).astype(np.float32)
    while True:
        log.debug("#############################")
        hasFrame, original_frame = cap.read()
        if not hasFrame:
            break
        (h_orig, w_orig) = original_frame.shape[:2]

        log.debug("Preprocessing frame")
        if original_frame.shape[2] > 1:
            frame = cv.cvtColor(cv.cvtColor(original_frame, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2RGB)
        else:
            frame = cv.cvtColor(original_frame, cv.COLOR_GRAY2RGB)

        img_rgb = frame.astype(np.float32) / 255.
        img_lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2Lab)
        img_l = img_lab[:, :, 0]

        img_rs = cv.resize(img_rgb, (w_in, h_in))
        img_lab_rs = cv.cvtColor(img_rs, cv.COLOR_RGB2Lab)
        img_l_rs = img_lab_rs[:, :, 0]
        img_l_rs -= mean

        log.debug("Network inference")
        res = exec_net.infer(inputs={input_blob: [img_l_rs]})

        (n_out, c_out, h_out, w_out) = res[output_blob].shape
        update_res = np.zeros((n_out, 2, h_out, w_out)).astype(np.float32)
        assert color_coeff.shape == (313, 2), "Current shape of color coefficients does not match required shape"

        update_res[0, :, :, :] = (res[output_blob] * color_coeff.transpose()[:, :, np.newaxis, np.newaxis]).sum(1)

        log.debug("Get results")
        out = update_res[0, :, :, :].transpose((1, 2, 0))
        out = cv.resize(out, (w_orig, h_orig))
        img_lab_out = np.concatenate((img_l[:, :, np.newaxis], out), axis=2)
        img_bgr_out = np.clip(cv.cvtColor(img_lab_out, cv.COLOR_Lab2BGR), 0, 1)

        if not args.no_show:
            log.debug("Show results")
            imshowSize = (640, 480)
            cv.imshow('origin', cv.resize(original_frame, imshowSize))
            cv.imshow('gray', cv.resize(frame, imshowSize))
            cv.imshow('colorized', cv.resize(img_bgr_out, imshowSize))

        if not cv.waitKey(1) < 0:
            break
