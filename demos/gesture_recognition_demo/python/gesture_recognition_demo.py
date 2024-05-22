#!/usr/bin/env python3
"""
 Copyright (c) 2019-2024 Intel Corporation

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
from time import perf_counter
import json
import os
from pathlib import Path
import multiprocessing
from argparse import ArgumentParser, SUPPRESS

import cv2
import numpy as np

from gesture_recognition_demo.common import load_core
from gesture_recognition_demo.video_stream import VideoStream
from gesture_recognition_demo.video_library import VideoLibrary
from gesture_recognition_demo.person_detector import PersonDetector
from gesture_recognition_demo.tracker import Tracker
from gesture_recognition_demo.action_recognizer import ActionRecognizer
from gesture_recognition_demo.visualizer import Visualizer

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python/model_zoo'))

import monitors
from model_api.performance_metrics import PerformanceMetrics

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)

DETECTOR_OUTPUT_SHAPE = -1, 5
TRACKER_SCORE_THRESHOLD = 0.4
TRACKER_IOU_THRESHOLD = 0.3
ACTION_NET_INPUT_FPS = 15
ACTION_IMAGE_SCALE = 256
SAMPLES_MAX_WINDOW_SIZE = 1000
SAMPLES_TRG_FPS = 20
VISUALIZER_TRG_FPS = 60
OBJECT_IDS = [ord(str(n)) for n in range(10)]


def build_argparser():
    """ Returns argument parser. """

    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument('-m_a', '--action_model',
                      help='Required. Path to an .xml file with a trained gesture recognition model.',
                      required=True, type=str)
    args.add_argument('-m_d', '--detection_model',
                      help='Required. Path to an .xml file with a trained person detector model.',
                      required=True, type=str)
    args.add_argument('-i', '--input', required=True,
                      help='Required. Path to a video file or a device node of a web-camera.')
    args.add_argument('-o', '--output', required=False,
                      help='Optional. Name of the output file(s) to save. Frames of odd width or height can be truncated. See https://github.com/opencv/opencv/pull/24086')
    args.add_argument('-limit', '--output_limit', required=False, default=1000, type=int,
                      help='Optional. Number of frames to store in output. '
                           'If 0 is set, all frames are stored.')
    args.add_argument('-c', '--class_map',
                      help='Required. Path to a file with gesture classes.',
                      required=True, type=str)
    args.add_argument('-s', '--samples_dir',
                      help='Optional. Path to a directory with video samples of gestures.',
                      default=None, type=str)
    args.add_argument('-t', '--action_threshold',
                      help='Optional. Threshold for the predicted score of an action.',
                      default=0.8, type=float)
    args.add_argument('-d', '--device',
                      help="Optional. Specify a device to infer on (the list of available devices is shown below). Use "
                           "'-d HETERO:<comma-separated_devices_list>' format to specify HETERO plugin. Use "
                           "'-d MULTI:<comma-separated_devices_list>' format to specify MULTI plugin. Default is CPU",
                      default='CPU', type=str)
    args.add_argument('--no_show', action='store_true',
                      help='Optional. Do not visualize inference results.')

    args.add_argument('-u', '--utilization_monitors', default='', type=str,
                      help='Optional. List of monitors to show initially.')
    return parser


def load_class_map(file_path):
    """ Returns class names map. """

    if file_path is not None and os.path.exists(file_path):
        with open(file_path, 'r') as input_stream:
            data = json.load(input_stream)
            class_map = dict(enumerate(data))
    else:
        class_map = None

    return class_map


def main():
    args = build_argparser().parse_args()

    class_map = load_class_map(args.class_map)
    if class_map is None:
        raise RuntimeError("Can't read {}".format(args.class_map))

    core = load_core()

    person_detector = PersonDetector(args.detection_model, args.device, core,
                                     num_requests=2, output_shape=DETECTOR_OUTPUT_SHAPE)
    action_recognizer = ActionRecognizer(args.action_model, args.device, core,
                                         num_requests=2, img_scale=ACTION_IMAGE_SCALE,
                                         num_classes=len(class_map))

    person_tracker = Tracker(person_detector, TRACKER_SCORE_THRESHOLD, TRACKER_IOU_THRESHOLD)

    video_stream = VideoStream(args.input, ACTION_NET_INPUT_FPS, action_recognizer.input_length)
    video_stream.start()

    metrics = PerformanceMetrics()
    visualizer = Visualizer(VISUALIZER_TRG_FPS)
    visualizer.register_window('Demo')
    presenter = monitors.Presenter(args.utilization_monitors)

    samples_library = None
    if args.samples_dir is not None and os.path.exists(args.samples_dir):
        visualizer.register_window('Gesture library')
        visualizer.start()

        library_queue = visualizer.get_queue('Gesture library')
        samples_library = VideoLibrary(args.samples_dir, SAMPLES_MAX_WINDOW_SIZE,
                                       list(class_map.values()), library_queue, SAMPLES_TRG_FPS)
        samples_library.start()
    else:
        visualizer.start()

    last_caption = None
    active_object_id = -1
    tracker_labels_map = {}
    tracker_labels = set()

    frames_processed = 0

    while True:
        start_time = perf_counter()
        frame = video_stream.get_live_frame()
        batch = video_stream.get_batch()
        if frame is None or batch is None:
            break
        if frames_processed == 0:
            video_writer = cv2.VideoWriter()
            if args.output and not video_writer.open(args.output, cv2.VideoWriter_fourcc(*'MJPG'),
                                                     video_stream.fps(), (frame.shape[1], frame.shape[0])):
                raise RuntimeError("Can't open video writer")

        detections, tracker_labels_map = person_tracker.add_frame(
            frame, len(OBJECT_IDS), tracker_labels_map)
        if detections is None:
            active_object_id = -1
            last_caption = None

        if len(detections) == 1:
            active_object_id = 0

        if active_object_id >= 0:
            cur_det = [det for det in detections if det.id == active_object_id]
            if len(cur_det) != 1:
                active_object_id = -1
                last_caption = None
                continue

            recognizer_result = action_recognizer(batch, cur_det[0].roi.reshape(-1))
            if recognizer_result is not None:
                action_class_id = np.argmax(recognizer_result)
                action_class_label = \
                    class_map[action_class_id] if class_map is not None else action_class_id

                action_class_score = np.max(recognizer_result)
                if action_class_score > args.action_threshold:
                    last_caption = 'Last gesture: {} '.format(action_class_label)

        presenter.drawGraphs(frame)

        if detections is not None:
            tracker_labels = {det.id for det in detections}

            for det in detections:
                roi_color = (0, 255, 0) if active_object_id == det.id else (128, 128, 128)
                border_width = 2 if active_object_id == det.id else 1
                person_roi = det.roi[0]
                cv2.rectangle(frame, (person_roi[0], person_roi[1]),
                              (person_roi[2], person_roi[3]), roi_color, border_width)
                cv2.putText(frame, str(det.id), (person_roi[0] + 10, person_roi[1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, roi_color, 2)

        if last_caption is not None:
            cv2.putText(frame, last_caption, (10, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        metrics.update(start_time, frame)

        frames_processed += 1
        if video_writer.isOpened() and (args.output_limit <= 0 or frames_processed <= args.output_limit):
            video_writer.write(frame)

        if args.no_show:
            continue

        visualizer.put_queue(frame, 'Demo')
        key = visualizer.get_key()

        if key == 27:  # esc
            break
        elif key == ord(' '):  # space
            active_object_id = -1
            last_caption = None
        elif key == 13:  # enter
            last_caption = None
        elif key in OBJECT_IDS:  # 0-9
            local_bbox_id = int(chr(key))
            if local_bbox_id in tracker_labels:
                active_object_id = local_bbox_id
        else:
            presenter.handleKey(key)

        if samples_library is not None:
            if key == ord('f'):  # forward
                samples_library.next()
            elif key == ord('b'):  # backward
                samples_library.prev()

    if samples_library is not None:
        samples_library.release()
    visualizer.release()
    video_stream.release()

    metrics.log_total()
    for rep in presenter.reportMeans():
        log.info(rep)


if __name__ == '__main__':
    # https://github.com/opencv/opencv/issues/5150#issuecomment-197413178
    # https://github.com/opencv/opencv/issues/5150#issuecomment-197540235
    multiprocessing.set_start_method('spawn')
    sys.exit(main() or 0)
