#!/usr/bin/env python3
"""
 Copyright (C) 2021-2022 Intel Corporation

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
import sys
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from time import perf_counter


import cv2
import numpy as np
from openvino.runtime import Core, get_version, PartialShape

import mtcnn_utils as utils

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))

import monitors
from images_capture import open_images_capture
from openvino.model_zoo.model_api.performance_metrics import PerformanceMetrics

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)


score_threshold = [0.6, 0.7, 0.7]
iou_threshold = [0.5, 0.7, 0.7, 0.7]


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument("-i", "--input",
                      help="Required. Path to a test image file.",
                      required=True, type=str)
    args.add_argument("-m_p", "--model_pnet",
                      help="Required. Path to an .xml file with a pnet model.",
                      required=True, type=Path, metavar='"<path>"')
    args.add_argument("-m_r", "--model_rnet",
                      help="Required. Path to an .xml file with a rnet model.",
                      required=True, type=Path, metavar='"<path>"')
    args.add_argument("-m_o", "--model_onet",
                      help="Required. Path to an .xml file with a onet model.",
                      required=True, type=Path, metavar='"<path>"')
    args.add_argument("-th", "--threshold",
                      help="Optional. The threshold to define the face is recognized or not.",
                      type=float, default=0.6, metavar='"<num>"')
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, HDDL, MYRIAD or HETERO is "
                           "acceptable. The demo will look for a suitable plugin for device specified. Default "
                           "value is CPU",
                      default="CPU", type=str, metavar='"<device>"')
    args.add_argument('--loop', default=False, action='store_true',
                       help='Optional. Enable reading the input in a loop.')
    args.add_argument("--no_show",
                      help="Optional. Don't show output",
                      action='store_true')
    args.add_argument('-o', '--output', required=False,
                      help='Optional. Name of the output file(s) to save.')
    args.add_argument('-limit', '--output_limit', required=False, default=1000, type=int,
                      help='Optional. Number of frames to store in output. '
                            'If 0 is set, all frames are stored.')
    args.add_argument('-u', '--utilization_monitors', default='', type=str,
                      help='Optional. List of monitors to show initially.')

    return parser


def preprocess_image(image, w, h):
    image = cv2.resize(image, (w, h))
    # Change input shape to [B,C,W,H] for MTCNN
    image = image.transpose((2, 1, 0))
    image = np.expand_dims(image, axis=0)
    return image


def main():
    args = build_argparser().parse_args()

    cap = open_images_capture(args.input, args.loop)

    # Plugin initialization for specified device and load extensions library if specified
    log.info('OpenVINO Runtime')
    log.info('\tbuild: {}'.format(get_version()))
    core = Core()

    # Read IR
    log.info('Reading Proposal model {}'.format(args.model_pnet))
    p_net = core.read_model(args.model_pnet)
    if len(p_net.inputs) != 1:
        raise RuntimeError("Pnet supports only single input topologies")
    if len(p_net.outputs) != 2:
        raise RuntimeError("Pnet supports two output topologies")

    log.info('Reading Refine model {}'.format(args.model_rnet))
    r_net = core.read_model(args.model_rnet)
    if len(r_net.inputs) != 1:
        raise RuntimeError("Rnet supports only single input topologies")
    if len(r_net.outputs) != 2:
        raise RuntimeError("Rnet supports two output topologies")

    log.info('Reading Output model {}'.format(args.model_onet))
    o_net = core.read_model(args.model_onet)
    if len(o_net.inputs) != 1:
        raise RuntimeError("Onet supports only single input topologies")
    if len(o_net.outputs) != 3:
        raise RuntimeError("Onet supports three output topologies")

    pnet_input_tensor_name = p_net.inputs[0].get_any_name()
    rnet_input_tensor_name = r_net.inputs[0].get_any_name()
    onet_input_tensor_name = o_net.inputs[0].get_any_name()

    for node in p_net.outputs:
        if node.shape[1] == 2:
            pnet_cls_name = node.get_any_name()
        elif node.shape[1] == 4:
            pnet_roi_name = node.get_any_name()
        else:
            raise RuntimeError("Unsupported output layer for Pnet")

    for node in r_net.outputs:
        if node.shape[1] == 2:
            rnet_cls_name = node.get_any_name()
        elif node.shape[1] == 4:
            rnet_roi_name = node.get_any_name()
        else:
            raise RuntimeError("Unsupported output layer for Rnet")

    for node in o_net.outputs:
        if node.shape[1] == 2:
            onet_cls_name = node.get_any_name()
        elif node.shape[1] == 4:
            onet_roi_name = node.get_any_name()
        elif node.shape[1] == 10:
            onet_pts_name = node.get_any_name()
        else:
            raise RuntimeError("Unsupported output layer for Onet")

    next_frame_id = 0

    metrics = PerformanceMetrics()
    presenter = None
    video_writer = cv2.VideoWriter()
    is_loaded_before = False

    while True:
        start_time = perf_counter()
        origin_image = cap.read()
        if origin_image is None:
            if next_frame_id == 0:
                raise ValueError("Can't read an image from the input")
            break
        if next_frame_id == 0:
            presenter = monitors.Presenter(args.utilization_monitors, 55,
                                           (round(origin_image.shape[1] / 4), round(origin_image.shape[0] / 8)))
            if args.output and not video_writer.open(args.output, cv2.VideoWriter_fourcc(*'MJPG'),
                                                     cap.fps(), (origin_image.shape[1], origin_image.shape[0])):
                raise RuntimeError("Can't open video writer")
        next_frame_id += 1

        rgb_image = cv2.cvtColor(origin_image, cv2.COLOR_BGR2RGB)
        oh, ow, _ = rgb_image.shape

        scales = utils.calculate_scales(rgb_image)

        # *************************************
        # Pnet stage
        # *************************************

        pnet_res = []
        for i, scale in enumerate(scales):
            hs = int(oh*scale)
            ws = int(ow*scale)
            image = preprocess_image(rgb_image, ws, hs)

            p_net.reshape({pnet_input_tensor_name: PartialShape([1, 3, ws, hs])})  # Change weidth and height of input blob
            compiled_pnet = core.compile_model(p_net, args.device)
            infer_request_pnet = compiled_pnet.create_infer_request()
            if i == 0 and not is_loaded_before:
                log.info("The Proposal model {} is loaded to {}".format(args.model_pnet, args.device))

            infer_request_pnet.infer(inputs={pnet_input_tensor_name: image})
            p_res = {name: infer_request_pnet.get_tensor(name).data[:] for name in {pnet_roi_name, pnet_cls_name}}
            pnet_res.append(p_res)

        image_num = len(scales)
        rectangles = []
        for i in range(image_num):
            roi = pnet_res[i][pnet_roi_name]
            cls = pnet_res[i][pnet_cls_name]
            _, _, out_h, out_w = cls.shape
            out_side = max(out_h, out_w)
            rectangle = utils.detect_face_12net(cls[0][1], roi[0], out_side, 1/scales[i], ow, oh,
                                                score_threshold[0], iou_threshold[0])
            rectangles.extend(rectangle)
        rectangles = utils.NMS(rectangles, iou_threshold[1], 'iou')

        # Rnet stage
        if len(rectangles) > 0:

            r_net.reshape({rnet_input_tensor_name: PartialShape([len(rectangles), 3, 24, 24])})  # Change batch size of input blob
            compiled_rnet = core.compile_model(r_net, args.device)
            infer_request_rnet = compiled_rnet.create_infer_request()
            if not is_loaded_before:
                log.info("The Refine model {} is loaded to {}".format(args.model_rnet, args.device))

            rnet_input = []
            for rectangle in rectangles:
                crop_img = rgb_image[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
                crop_img = preprocess_image(crop_img, 24, 24)
                rnet_input.extend(crop_img)

            infer_request_rnet.infer(inputs={rnet_input_tensor_name: rnet_input})
            rnet_res = {name: infer_request_rnet.get_tensor(name).data[:] for name in {rnet_roi_name, rnet_cls_name}}

            roi = rnet_res[rnet_roi_name]
            cls = rnet_res[rnet_cls_name]
            rectangles = utils.filter_face_24net(cls, roi, rectangles, ow, oh, score_threshold[1], iou_threshold[2])

        # Onet stage
        if len(rectangles) > 0:

            o_net.reshape({onet_input_tensor_name: PartialShape([len(rectangles), 3, 48, 48])})  # Change batch size of input blob
            compiled_onet = core.compile_model(o_net, args.device)
            infer_request_onet = compiled_onet.create_infer_request()
            if not is_loaded_before:
                log.info("The Output model {} is loaded to {}".format(args.model_onet, args.device))
                is_loaded_before = True

            onet_input = []
            for rectangle in rectangles:
                crop_img = rgb_image[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
                crop_img = preprocess_image(crop_img, 48, 48)
                onet_input.extend(crop_img)

            infer_request_onet.infer(inputs={onet_input_tensor_name: onet_input})
            onet_res = {name: infer_request_onet.get_tensor(name).data[:] for name in {onet_roi_name, onet_cls_name, onet_pts_name}}

            roi = onet_res[onet_roi_name]
            cls = onet_res[onet_cls_name]
            pts = onet_res[onet_pts_name]
            rectangles = utils.filter_face_48net(cls, roi, pts, rectangles, ow, oh,
                                                 score_threshold[2], iou_threshold[3])

        # display results
        for rectangle in rectangles:
            # Draw detected boxes
            cv2.putText(origin_image, 'confidence: {:.2f}'.format(rectangle[4]),
                        (int(rectangle[0]), int(rectangle[1])), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0))
            cv2.rectangle(origin_image, (int(rectangle[0]), int(rectangle[1])), (int(rectangle[2]), int(rectangle[3])),
                          (255, 0, 0), 1)
            # Draw landmarks
            for i in range(5, 15, 2):
                cv2.circle(origin_image, (int(rectangle[i+0]), int(rectangle[i+1])), 2, (0, 255, 0))

        metrics.update(start_time, origin_image)

        if video_writer.isOpened() and (args.output_limit <= 0 or next_frame_id <= args.output_limit):
            video_writer.write(origin_image)

        if not args.no_show:
            cv2.imshow('MTCNN Results', origin_image)
            key = cv2.waitKey(1)
            if key in {ord('q'), ord('Q'), 27}:
                break
            presenter.handleKey(key)

    metrics.log_total()


if __name__ == '__main__':
    sys.exit(main() or 0)
