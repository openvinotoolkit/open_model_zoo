#!/usr/bin/env python3
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

import logging as log
import sys
import time
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path

import cv2
import numpy as np
from openvino.inference_engine import IECore

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))

from instance_segmentation_demo.model_utils import check_model
from instance_segmentation_demo.tracker import StaticIOUTracker
from instance_segmentation_demo.visualizer import Visualizer

import monitors
from images_capture import open_images_capture


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument('-m', '--model',
                      help='Required. Path to an .xml file with a trained model.',
                      required=True, type=Path, metavar='"<path>"')
    args.add_argument('--labels',
                      help='Required. Path to a text file with class labels.',
                      required=True, type=str, metavar='"<path>"')
    args.add_argument('-i', '--input', required=True,
                      help='Required. An input to process. The input must be a single image, '
                           'a folder of images, video file or camera id.')
    args.add_argument('--loop', default=False, action='store_true',
                      help='Optional. Enable reading the input in a loop.')
    args.add_argument('-o', '--output', required=False,
                      help='Optional. Name of the output file(s) to save.')
    args.add_argument('-limit', '--output_limit', required=False, default=1000, type=int,
                      help='Optional. Number of frames to store in output. '
                           'If 0 is set, all frames are stored.')
    args.add_argument('-d', '--device',
                      help='Optional. Specify the target device to infer on: CPU, GPU, HDDL or MYRIAD. '
                           'The demo will look for a suitable plugin for device specified '
                           '(by default, it is CPU).',
                      default='CPU', type=str, metavar='"<device>"')
    args.add_argument('-l', '--cpu_extension',
                      help='Required for CPU custom layers. '
                           'Absolute path to a shared library with the kernels implementation.',
                      default=None, type=str, metavar='"<absolute_path>"')
    args.add_argument('--delay',
                      help='Optional. Interval in milliseconds of waiting for a key to be pressed.',
                      default=0, type=int, metavar='"<num>"')
    args.add_argument('-pt', '--prob_threshold',
                      help='Optional. Probability threshold for detections filtering.',
                      default=0.5, type=float, metavar='"<num>"')
    args.add_argument('--no_keep_aspect_ratio',
                      help='Optional. Force image resize not to keep aspect ratio.',
                      action='store_true')
    args.add_argument('--no_track',
                      help='Optional. Disable tracking.',
                      action='store_true')
    args.add_argument('--show_scores',
                      help='Optional. Show detection scores.',
                      action='store_true')
    args.add_argument('--show_boxes',
                      help='Optional. Show bounding boxes.',
                      action='store_true')
    args.add_argument('-pc', '--perf_counts',
                      help='Optional. Report performance counters.',
                      action='store_true')
    args.add_argument('-r', '--raw_output_message',
                      help='Optional. Output inference results raw values.',
                      action='store_true')
    args.add_argument("--no_show",
                      help="Optional. Don't show output.",
                      action='store_true')
    args.add_argument('-u', '--utilization_monitors', default='', type=str,
                      help='Optional. List of monitors to show initially.')
    return parser


def main():
    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    # Plugin initialization for specified device and load extensions library if specified.
    log.info('Creating Inference Engine...')
    ie = IECore()
    if args.cpu_extension and 'CPU' in args.device:
        ie.add_extension(args.cpu_extension, 'CPU')
    # Read IR
    log.info('Loading network')
    net = ie.read_network(args.model, args.model.with_suffix('.bin'))
    image_input, image_info_input, (n, c, h, w), model_type, postprocessor = check_model(net)
    args.no_keep_aspect_ratio = model_type == 'yolact' or args.no_keep_aspect_ratio

    log.info('Loading IR to the plugin...')
    exec_net = ie.load_network(network=net, device_name=args.device, num_requests=2)

    cap = open_images_capture(args.input, args.loop)
    frame = cap.read()
    if frame is None:
        raise RuntimeError("Can't read an image from the input")

    if args.no_track:
        tracker = None
    else:
        tracker = StaticIOUTracker()

    with open(args.labels, 'rt') as labels_file:
        class_labels = labels_file.read().splitlines()

    if args.delay:
        delay = args.delay
    else:
        delay = int(cap.get_type() in ('VIDEO', 'CAMERA'))

    frames_processed = 0
    out_frame_size = (frame.shape[1], frame.shape[0])
    presenter = monitors.Presenter(args.utilization_monitors, 45,
                (round(out_frame_size[0] / 4), round(out_frame_size[1] / 8)))
    visualizer = Visualizer(class_labels, show_boxes=args.show_boxes, show_scores=args.show_scores)
    video_writer = cv2.VideoWriter()
    if args.output and not video_writer.open(args.output, cv2.VideoWriter_fourcc(*'MJPG'),
                                             cap.fps(), out_frame_size):
        raise RuntimeError("Can't open video writer")

    render_time = 0

    log.info('Starting inference...')
    print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")

    while frame is not None:
        if args.no_keep_aspect_ratio:
            # Resize the image to a target size.
            scale_x = w / frame.shape[1]
            scale_y = h / frame.shape[0]
            input_image = cv2.resize(frame, (w, h))
        else:
            # Resize the image to keep the same aspect ratio and to fit it to a window of a target size.
            scale_x = scale_y = min(h / frame.shape[0], w / frame.shape[1])
            input_image = cv2.resize(frame, None, fx=scale_x, fy=scale_y)

        input_image_size = input_image.shape[:2]
        input_image = np.pad(input_image, ((0, h - input_image_size[0]),
                                           (0, w - input_image_size[1]),
                                           (0, 0)),
                             mode='constant', constant_values=0)
        # Change data layout from HWC to CHW.
        input_image = input_image.transpose((2, 0, 1))
        input_image = input_image.reshape((n, c, h, w)).astype(np.float32)
        input_image_info = np.asarray([[input_image_size[0], input_image_size[1], 1]], dtype=np.float32)

        # Run the net.
        inf_start = time.time()
        feed_dict = {image_input: input_image}
        if image_info_input:
            feed_dict[image_info_input] = input_image_info
        outputs = exec_net.infer(feed_dict)
        inf_end = time.time()
        det_time = inf_end - inf_start

        # Parse detection results of the current request
        scores, classes, boxes, masks = postprocessor(
            outputs, scale_x, scale_y, *frame.shape[:2], h, w, args.prob_threshold)

        render_start = time.time()

        if len(boxes) and args.raw_output_message:
            log.info('Detected boxes:')
            log.info('  Class ID | Confidence |     XMIN |     YMIN |     XMAX |     YMAX ')
            for box, cls, score, mask in zip(boxes, classes, scores, masks):
                log.info('{:>10} | {:>10f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} '.format(cls, score, *box))

        # Get instance track IDs.
        masks_tracks_ids = None
        if tracker is not None:
            masks_tracks_ids = tracker(masks, classes)

        # Visualize masks.
        frame = visualizer(frame, boxes, classes, scores, presenter, masks, masks_tracks_ids)

        # Draw performance stats.
        inf_time_message = 'Inference time: {:.3f} ms'.format(det_time * 1000)
        render_time_message = 'OpenCV rendering time: {:.3f} ms'.format(render_time * 1000)
        cv2.putText(frame, inf_time_message, (15, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (200, 10, 10), 1)
        cv2.putText(frame, render_time_message, (15, 30), cv2.FONT_HERSHEY_COMPLEX, 0.5, (10, 10, 200), 1)

        # Print performance counters.
        if args.perf_counts:
            perf_counts = exec_net.requests[0].get_perf_counts()
            log.info('Performance counters:')
            print('{:<70} {:<15} {:<15} {:<15} {:<10}'.format('name', 'layer_type', 'exet_type', 'status',
                                                              'real_time, us'))
            for layer, stats in perf_counts.items():
                print('{:<70} {:<15} {:<15} {:<15} {:<10}'.format(layer, stats['layer_type'], stats['exec_type'],
                                                                  stats['status'], stats['real_time']))
        frames_processed += 1
        if video_writer.isOpened() and (args.output_limit <= 0 or frames_processed <= args.output_limit):
            video_writer.write(frame)

        if not args.no_show:
            # Show resulting image.
            cv2.imshow('Results', frame)
        render_end = time.time()
        render_time = render_end - render_start

        if not args.no_show:
            key = cv2.waitKey(delay)
            esc_code = 27
            if key == esc_code:
                break
            presenter.handleKey(key)
        frame = cap.read()

    print(presenter.reportMeans())
    cv2.destroyAllWindows()


if __name__ == '__main__':
    sys.exit(main() or 0)
