#!/usr/bin/env python
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
import json
from os.path import exists
from argparse import ArgumentParser, SUPPRESS

import cv2
import numpy as np

from asl_recognition_demo.common import load_ie_core
from asl_recognition_demo.video_stream import VideoStream
from asl_recognition_demo.video_library import VideoLibrary
from asl_recognition_demo.person_detector import PersonDetector
from asl_recognition_demo.person_tracker import PersonTracker
from asl_recognition_demo.action_recognizer import ActionRecognizer

DETECTOR_OUTPUT_SHAPE = -1, 5
TRACKER_SCORE_THRESHOLD = 0.5
TRACKER_IOU_THRESHOLD = 0.5
ACTION_NET_INPUT_FPS = 15
ACTION_NUM_CLASSES = 100
ACTION_IMAGE_SCALE = 256
ACTION_SCORE_THRESHOLD = 0.8
SAMPLES_WINDOW_SIZE = 640, 480


def build_argparser():
    """ Returns argument parser. """

    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument('-m_a', '--action_model',
                      help='Required. Path to an .xml file with a trained asl recognition model.',
                      required=True, type=str)
    args.add_argument('-m_d', '--detection_model',
                      help='Required. Path to an .xml file with a trained person detector model.',
                      required=True, type=str)
    args.add_argument('-i', '--input',
                      help='Required. Path to a video file or a device node of a web-camera.',
                      required=True, type=str)
    args.add_argument('-c', '--class_map',
                      help='Required. Path to a file with ASL classes.',
                      required=True, type=str)
    args.add_argument('-s', '--samples_dir',
                      help='Optional. Path to a directory with video samples of gestures.',
                      default=None, type=str)
    args.add_argument('-d', '--device',
                      help='Optional. Specify the target device to infer on: CPU, GPU, FPGA, HDDL '
                           'or MYRIAD. The demo will look for a suitable plugin for device '
                           'specified (by default, it is CPU).',
                      default='CPU', type=str)
    args.add_argument("-l", "--cpu_extension",
                      help="Optional. Required for CPU custom layers. Absolute path to "
                           "a shared library with the kernels implementations.", type=str,
                      default=None)
    args.add_argument('--no_show', action='store_true',
                      help='Optional. Do not visualize inference results.')

    return parser


def load_class_map(file_path):
    """ Returns class names map. """

    if file_path is not None and exists(file_path):
        with open(file_path, 'r') as input_stream:
            data = json.load(input_stream)
            class_map = dict(enumerate(data))
    else:
        class_map = None

    return class_map


def main():
    """ Main function. """

    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    ie_core = load_ie_core(args.device, args.cpu_extension)

    person_detector = PersonDetector(args.detection_model, args.device, ie_core,
                                     num_requests=2, output_shape=DETECTOR_OUTPUT_SHAPE)
    action_recognizer = ActionRecognizer(args.action_model, args.device, ie_core,
                                         num_requests=2, img_scale=ACTION_IMAGE_SCALE,
                                         num_classes=ACTION_NUM_CLASSES)

    video_stream = VideoStream(args.input, ACTION_NET_INPUT_FPS, action_recognizer.input_length)
    video_stream.start()

    person_tracker = PersonTracker(person_detector, TRACKER_SCORE_THRESHOLD, TRACKER_IOU_THRESHOLD)

    class_map = load_class_map(args.class_map)
    assert class_map is not None

    samples_library = None
    if args.samples_dir is not None and exists(args.samples_dir):
        samples_library = VideoLibrary(args.samples_dir, SAMPLES_WINDOW_SIZE, list(class_map.values()))

    last_caption = None
    person_roi = None

    start_time = time.perf_counter()
    while True:
        frame = video_stream.get_live_frame()
        batch = video_stream.get_batch()
        if frame is None or batch is None:
            break

        person_roi = person_tracker.get_roi(frame)
        if person_roi is not None:
            recognizer_result = action_recognizer(batch, person_roi)
            if recognizer_result is not None:
                action_class_id = np.argmax(recognizer_result)
                action_class_label = \
                    class_map[action_class_id] if class_map is not None else action_class_id

                action_class_score = np.max(recognizer_result)
                if action_class_score > ACTION_SCORE_THRESHOLD:
                    last_caption = 'Last gesture: {} '.format(action_class_label)

        end_time = time.perf_counter()
        elapsed_time = end_time - start_time
        start_time = end_time
        current_fps = 1.0 / elapsed_time
        cv2.putText(frame, 'FPS: {:.2f}'.format(current_fps), (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        if last_caption is not None:
            cv2.putText(frame, last_caption, (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        if person_roi is not None:
            cv2.rectangle(frame, (person_roi[0], person_roi[1]),
                          (person_roi[2], person_roi[3]), (128, 128, 128), 1)

        if args.no_show:
            continue

        cv2.imshow('Demo', frame)

        key = cv2.waitKey(1)
        if key == 27:
            break

        if samples_library is not None:
            sample_frame = samples_library.get_frame()
            if sample_frame is not None:
                cv2.imshow('Sample', sample_frame)

            if key == ord('n'):
                samples_library.next()
            elif key == ord('p'):
                samples_library.prev()

    cv2.destroyAllWindows()
    video_stream.release()
    if samples_library is not None:
        samples_library.release()


if __name__ == '__main__':
    sys.exit(main() or 0)
