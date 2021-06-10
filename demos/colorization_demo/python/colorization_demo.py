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
import logging as log
import sys
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
import monitors
from images_capture import open_images_capture


def build_arg():
    parser = ArgumentParser(add_help=False)
    in_args = parser.add_argument_group('Options')
    in_args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Help with the script.')
    in_args.add_argument("-m", "--model", help="Required. Path to .xml file with pre-trained model.",
                         required=True, type=Path)
    in_args.add_argument("-d", "--device",
                         help="Optional. Specify target device for infer: CPU, GPU, HDDL or MYRIAD. "
                              "Default: CPU",
                         default="CPU", type=str)
    in_args.add_argument('-i', "--input", required=True,
                         help='Required. An input to process. The input must be a single image, '
                              'a folder of images, video file or camera id.')
    in_args.add_argument('--loop', default=False, action='store_true',
                         help='Optional. Enable reading the input in a loop.')
    in_args.add_argument('-o', '--output', required=False,
                         help='Optional. Name of the output file(s) to save.')
    in_args.add_argument('-limit', '--output_limit', required=False, default=1000, type=int,
                         help='Optional. Number of frames to store in output. '
                              'If 0 is set, all frames are stored.')
    in_args.add_argument("--no_show", help="Optional. Don't show output.",
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
    load_net = ie.read_network(args.model, args.model.with_suffix(".bin"))
    load_net.batch_size = 1
    exec_net = ie.load_network(network=load_net, device_name=args.device)

    input_blob = next(iter(load_net.input_info))
    input_shape = load_net.input_info[input_blob].input_data.shape
    assert input_shape[1] == 1, "Expected model input shape with 1 channel"

    inputs = {}
    for input_name in load_net.input_info:
        inputs[input_name] = np.zeros(load_net.input_info[input_name].input_data.shape)

    assert len(load_net.outputs) == 1, "Expected number of outputs is equal 1"
    output_blob = next(iter(load_net.outputs))
    output_shape = load_net.outputs[output_blob].shape

    _, _, h_in, w_in = input_shape

    cap = open_images_capture(args.input, args.loop)
    original_frame = cap.read()
    if original_frame is None:
        raise RuntimeError("Can't read an image from the input")

    frames_processed = 0
    imshow_size = (640, 480)
    graph_size = (imshow_size[0] // 2, imshow_size[1] // 4)
    presenter = monitors.Presenter(args.utilization_monitors, imshow_size[1] * 2 - graph_size[1], graph_size)

    video_writer = cv.VideoWriter()
    if args.output and not video_writer.open(args.output, cv.VideoWriter_fourcc(*'MJPG'),
                                             cap.fps(), (imshow_size[0] * 2, imshow_size[1] * 2)):
        raise RuntimeError("Can't open video writer")

    while original_frame is not None:
        log.debug("#############################")
        (h_orig, w_orig) = original_frame.shape[:2]

        log.debug("Preprocessing frame")
        if original_frame.shape[2] > 1:
            frame = cv.cvtColor(cv.cvtColor(original_frame, cv.COLOR_BGR2GRAY), cv.COLOR_GRAY2RGB)
        else:
            frame = cv.cvtColor(original_frame, cv.COLOR_GRAY2RGB)

        img_rgb = frame.astype(np.float32) / 255
        img_lab = cv.cvtColor(img_rgb, cv.COLOR_RGB2Lab)
        img_l_rs = cv.resize(img_lab.copy(), (w_in, h_in))[:, :, 0]
        inputs[input_blob] = img_l_rs

        log.debug("Network inference")

        res = exec_net.infer(inputs=inputs)

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

        frames_processed += 1
        if video_writer.isOpened() and (args.output_limit <= 0 or frames_processed <= args.output_limit):
            video_writer.write(final_image)

        presenter.drawGraphs(final_image)
        if not args.no_show:
            log.debug("Show results")
            cv.imshow('Colorization Demo', final_image)
            key = cv.waitKey(1)
            if key in {ord("q"), ord("Q"), 27}:
                break
            presenter.handleKey(key)
        original_frame = cap.read()
    print(presenter.reportMeans())
