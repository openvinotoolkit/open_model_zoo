#!/usr/bin/env python3
"""
 Copyright (C) 2021 Intel Corporation

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

import logging
import sys
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from time import perf_counter


import cv2
import numpy as np
from openvino.inference_engine import IECore

import mtcnn_utils as utils

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))

import monitors
from images_capture import open_images_capture
from performance_metrics import PerformanceMetrics

logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()


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
    metrics = PerformanceMetrics()

    args = build_argparser().parse_args()

    # Plugin initialization for specified device and load extensions library if specified
    log.info("Creating Inference Engine")

    ie = IECore()

    # Read IR
    log.info("Loading network files:\n\t{}".format(args.model_pnet))
    p_net = ie.read_network(args.model_pnet)
    assert len(p_net.input_info.keys()) == 1, "Pnet supports only single input topologies"
    assert len(p_net.outputs) == 2, "Pnet supports two output topologies"

    log.info("Loading network files:\n\t{}".format(args.model_rnet))
    r_net = ie.read_network(args.model_rnet)
    assert len(r_net.input_info.keys()) == 1, "Rnet supports only single input topologies"
    assert len(r_net.outputs) == 2, "Rnet supports two output topologies"

    log.info("Loading network files:\n\t{}".format(args.model_onet))
    o_net = ie.read_network(args.model_onet)
    assert len(o_net.input_info.keys()) == 1, "Onet supports only single input topologies"
    assert len(o_net.outputs) == 3, "Onet supports three output topologies"

    log.info("Preparing input blobs")
    pnet_input_blob = next(iter(p_net.input_info))
    rnet_input_blob = next(iter(r_net.input_info))
    onet_input_blob = next(iter(o_net.input_info))

    log.info("Preparing output blobs")
    for name, blob in p_net.outputs.items():
        if blob.shape[1] == 2:
            pnet_cls_name = name
        elif blob.shape[1] == 4:
            pnet_roi_name = name
        else:
            raise RuntimeError("Unsupported output layer for Pnet")

    for name, blob in r_net.outputs.items():
        if blob.shape[1] == 2:
            rnet_cls_name = name
        elif blob.shape[1] == 4:
            rnet_roi_name = name
        else:
            raise RuntimeError("Unsupported output layer for Rnet")

    for name, blob in o_net.outputs.items():
        if blob.shape[1] == 2:
            onet_cls_name = name
        elif blob.shape[1] == 4:
            onet_roi_name = name
        elif blob.shape[1] == 10:
            onet_pts_name = name
        else:
            raise RuntimeError("Unsupported output layer for Onet")

    cap = open_images_capture(args.input, args.loop)

    next_frame_id = 0

    log.info('Starting inference...')
    print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")

    presenter = None
    video_writer = cv2.VideoWriter()

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
        log.info("Loading Pnet model to the plugin")

        t0 = cv2.getTickCount()
        pnet_res = []
        for scale in scales:
            hs = int(oh*scale)
            ws = int(ow*scale)
            image = preprocess_image(rgb_image, ws, hs)

            p_net.reshape({pnet_input_blob: [1, 3, ws, hs]})  # Change weidth and height of input blob
            exec_pnet = ie.load_network(network=p_net, device_name=args.device)

            p_res = exec_pnet.infer(inputs={pnet_input_blob: image})
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
            log.info("Loading Rnet model to the plugin")

            r_net.reshape({rnet_input_blob: [len(rectangles), 3, 24, 24]})  # Change batch size of input blob
            exec_rnet = ie.load_network(network=r_net, device_name=args.device)

            rnet_input = []
            for rectangle in rectangles:
                crop_img = rgb_image[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
                crop_img = preprocess_image(crop_img, 24, 24)
                rnet_input.extend(crop_img)

            rnet_res = exec_rnet.infer(inputs={rnet_input_blob: rnet_input})

            roi = rnet_res[rnet_roi_name]
            cls = rnet_res[rnet_cls_name]
            rectangles = utils.filter_face_24net(cls, roi, rectangles, ow, oh, score_threshold[1], iou_threshold[2])

        # Onet stage
        if len(rectangles) > 0:
            log.info("Loading Onet model to the plugin")

            o_net.reshape({onet_input_blob: [len(rectangles), 3, 48, 48]})  # Change batch size of input blob
            exec_onet = ie.load_network(network=o_net, device_name=args.device)

            onet_input = []
            for rectangle in rectangles:
                crop_img = rgb_image[int(rectangle[1]):int(rectangle[3]), int(rectangle[0]):int(rectangle[2])]
                crop_img = preprocess_image(crop_img, 48, 48)
                onet_input.extend(crop_img)

            onet_res = exec_onet.infer(inputs={onet_input_blob: onet_input})

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

        infer_time = (cv2.getTickCount() - t0) / cv2.getTickFrequency()  # Record infer time
        cv2.putText(origin_image, 'summary: {:.1f} FPS'.format(1.0 / infer_time),
                    (5, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 200))

        if video_writer.isOpened() and (args.output_limit <= 0 or next_frame_id <= args.output_limit):
            video_writer.write(origin_image)

        if not args.no_show:
            cv2.imshow('MTCNN Results', origin_image)
            key = cv2.waitKey(1)
            if key in {ord('q'), ord('Q'), 27}:
                break
            presenter.handleKey(key)

        metrics.update(start_time, origin_image)

    metrics.print_total()


if __name__ == '__main__':
    sys.exit(main() or 0)
