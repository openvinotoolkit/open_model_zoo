#!/usr/bin/env python3
"""
 Copyright (c) 2021-2024 Intel Corporation

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
from pathlib import Path
import sys
from time import perf_counter
from argparse import ArgumentParser, SUPPRESS

import cv2
import numpy as np

from place_recognition_demo.place_recognition import PlaceRecognition
from place_recognition_demo.visualizer import visualize

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python/model_zoo'))

import monitors
from images_capture import open_images_capture
from model_api.performance_metrics import PerformanceMetrics

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)


def build_argparser():
    """ Returns argument parser. """

    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument('-m', '--model',
                      help='Required. Path to an .xml file with a trained model.',
                      required=True, type=Path)
    args.add_argument('-i', '--input', required=True,
                      help='Required. An input to process. The input must be a single image, '
                           'a folder of images, video file or camera id.')
    args.add_argument('-gf', '--gallery_folder',
                      help='Required. Path to a folder with images in the gallery.',
                      required=True, type=Path)
    args.add_argument('--gallery_size', required=False, type=int,
                      help='Optional. Number of images from the gallery used for processing')
    args.add_argument('--loop', default=False, action='store_true',
                      help='Optional. Enable reading the input in a loop.')
    args.add_argument('-o', '--output', required=False,
                      help='Optional. Name of the output file(s) to save. Frames of odd width or height can be truncated. See https://github.com/opencv/opencv/pull/24086')
    args.add_argument('-limit', '--output_limit', required=False, default=1000, type=int,
                      help='Optional. Number of frames to store in output. '
                           'If 0 is set, all frames are stored.')
    args.add_argument('-d', '--device',
                      help='Optional. Specify the target device to infer on: CPU or GPU.'
                           'The demo will look for a suitable plugin for device '
                           'specified (by default, it is CPU).',
                      default='CPU', type=str)
    args.add_argument('--no_show', action='store_true',
                      help='Optional. Do not visualize inference results.')
    args.add_argument('-u', '--utilization_monitors', default='', type=str,
                      help='Optional. List of monitors to show initially.')
    return parser


def time_elapsed(func, *args):
    """ Auxiliary function that helps to measure elapsed time. """

    start_time = perf_counter()
    res = func(*args)
    elapsed = perf_counter() - start_time
    return elapsed, res


def main():
    args = build_argparser().parse_args()

    cap = open_images_capture(args.input, args.loop)

    place_recognition = PlaceRecognition(args.model, args.device, args.gallery_folder, args.gallery_size)

    compute_embeddings_times = []
    search_in_gallery_times = []

    frames_processed = 0
    presenter = monitors.Presenter(args.utilization_monitors, 0)
    video_writer = cv2.VideoWriter()
    metrics = PerformanceMetrics()

    while True:
        start_time = perf_counter()
        frame = cap.read()

        if frame is None:
            if frames_processed == 0:
                raise ValueError("Can't read an image from the input")
            break

        elapsed, probe_embedding = time_elapsed(place_recognition.compute_embedding, frame)
        compute_embeddings_times.append(elapsed)

        elapsed, (sorted_indexes, distances) = time_elapsed(place_recognition.search_in_gallery, probe_embedding)
        search_in_gallery_times.append(elapsed)

        image, key = visualize(frame, [str(place_recognition.impaths[i]) for i in sorted_indexes],
                               distances[sorted_indexes], place_recognition.input_size,
                               np.mean(compute_embeddings_times), np.mean(search_in_gallery_times),
                               imshow_delay=3, presenter=presenter, no_show=args.no_show)

        metrics.update(start_time)
        if frames_processed == 0:
            if args.output and not video_writer.open(args.output, cv2.VideoWriter_fourcc(*'MJPG'), cap.fps(),
                                                     (image.shape[1], image.shape[0])):
                raise RuntimeError("Can't open video writer")

        frames_processed += 1
        if video_writer.isOpened() and (args.output_limit <= 0 or frames_processed <= args.output_limit):
            video_writer.write(image)

        if key == 27:
            break

    metrics.log_total()
    for rep in presenter.reportMeans():
        log.info(rep)


if __name__ == '__main__':
    sys.exit(main() or 0)
