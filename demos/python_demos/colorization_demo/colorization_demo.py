#!/usr/bin/env python3
"""
 Copyright (c) 2018-2020 Intel Corporation

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

from openvino.inference_engine import IECore
import cv2 as cv
import numpy as np
import os
from argparse import ArgumentParser, SUPPRESS
import logging as log
import sys

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'common'))
import monitors


def build_arg():
    parser = ArgumentParser(add_help=False)
    in_args = parser.add_argument_group('Options')
    in_args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Help with the script.')
    in_args.add_argument("-m", "--model", help="Required. Path to .xml file with pre-trained model.",
                         required=True, type=str)
    in_args.add_argument("-d", "--device",
                         help="Optional. Specify target device for infer: CPU, GPU, FPGA, HDDL or MYRIAD. "
                              "Default: CPU",
                         default="CPU", type=str)
    in_args.add_argument('-i', "--input",
                         help='Required. Input to process.',
                         required=True, type=str, metavar='"<path>"')
    in_args.add_argument("--no_show", help="Optional. Disable display of results on screen.",
                         action='store_true', default=False)
    in_args.add_argument("-v", "--verbose", help="Optional. Enable display of processing logs on screen.",
                         action='store_true', default=False)
    in_args.add_argument("-u", "--utilization_monitors", default="", type=str,
                      help="Optional. List of monitors to show initially.")
    return parser


if __name__ == '__main__':
    args = build_arg().parse_args()

    log.basicConfig(format="[ %(levelname)s ] %(message)s",
                    level=log.INFO if not args.verbose else log.DEBUG, stream=sys.stdout)

    log.debug("Load network")
    ie = IECore()
    load_net = ie.read_network(args.model, os.path.splitext(args.model)[0] + ".bin")
    load_net.batch_size = 1
    exec_net = ie.load_network(network=load_net, device_name=args.device)

    input_blob = next(iter(load_net.input_info))
    input_shape = load_net.input_info[input_blob].input_data.shape
    assert input_shape[1] == 1, "Expected model input shape with 1 channel"

    assert len(load_net.outputs) == 1, "Expected number of outputs is equal 1"
    output_blob = next(iter(load_net.outputs))
    output_shape = load_net.outputs[output_blob].shape

    _, _, h_in, w_in = input_shape

    try:
        input_source = int(args.input)
    except ValueError:
        input_source = args.input

    cap = cv.VideoCapture(input_source)
    if not cap.isOpened():
        assert "{} not exist".format(input_source)

    imshow_size = (640, 480)
    graph_size = (imshow_size[0] // 2, imshow_size[1] // 4)
    presenter = monitors.Presenter(args.utilization_monitors, imshow_size[1] * 2 - graph_size[1], graph_size)

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

        img_rgb = frame.astype(np.float32) / 255
        img_lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2Lab)
        img_l_rs = cv.resize(img_lab.copy(), (w_in, h_in))[:, :, 0]

        log.debug("Network inference")
        res = exec_net.infer(inputs={input_blob: [img_l_rs]})

        update_res = np.squeeze(res[output_blob])

        log.debug("Get results")
        out = update_res.transpose((1, 2, 0))
        out = cv.resize(out, (w_orig, h_orig))
        img_lab_out = np.concatenate((img_lab[:, :, 0][:, :, np.newaxis], out), axis=2)
        img_bgr_out = np.clip(cv.cvtColor(img_lab_out, cv.COLOR_Lab2BGR), 0, 1)

        original_image = cv.resize(original_frame, imshow_size)
        grayscale_image = cv.resize(frame, imshow_size)
        colorize_image = (cv.resize(img_bgr_out, imshow_size) * 255).astype(np.uint8)
        lab_image = cv.resize(img_lab_out, imshow_size).astype(np.uint8)

        original_image = cv.putText(original_image, 'Original', (25, 50),
                                    cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
        grayscale_image = cv.putText(grayscale_image, 'Grayscale', (25, 50),
                                     cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
        colorize_image = cv.putText(colorize_image, 'Colorize', (25, 50),
                                    cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)
        lab_image = cv.putText(lab_image, 'LAB interpretation', (25, 50),
                               cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv.LINE_AA)

        ir_image = [cv.hconcat([original_image, grayscale_image]),
                    cv.hconcat([lab_image, colorize_image])]
        final_image = cv.vconcat(ir_image)
        presenter.drawGraphs(final_image)
        if not args.no_show:
            log.debug("Show results")
            cv.imshow('Colorization Demo', final_image)
            key = cv.waitKey(1)
            if key in {ord("q"), ord("Q"), 27}:
                break
            presenter.handleKey(key)
    print(presenter.reportMeans())
