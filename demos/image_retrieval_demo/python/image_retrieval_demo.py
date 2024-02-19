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
from pathlib import Path
import sys
from time import perf_counter
from argparse import ArgumentParser, SUPPRESS

import cv2
import numpy as np

from image_retrieval_demo.image_retrieval import ImageRetrieval
from image_retrieval_demo.common import central_crop
from image_retrieval_demo.visualizer import visualize
from image_retrieval_demo.roi_detector_on_video import RoiDetectorOnVideo

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python/model_zoo'))

import monitors
from images_capture import open_images_capture
from model_api.performance_metrics import PerformanceMetrics

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)

INPUT_SIZE = 224


def build_argparser():
    """ Returns argument parser. """

    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS,
                      help='Show this help message and exit.')
    args.add_argument('-m', '--model',
                      help='Required. Path to an .xml file with a trained model.',
                      required=True, type=str)
    args.add_argument('-i', '--input', required=True,
                      help='Required. Path to a video file or a device node of a web-camera.')
    args.add_argument('--loop', default=False, action='store_true',
                      help='Optional. Enable reading the input in a loop.')
    args.add_argument('-o', '--output', required=False,
                      help='Optional. Name of the output file(s) to save. Frames of odd width or height can be truncated. See https://github.com/opencv/opencv/pull/24086')
    args.add_argument('-limit', '--output_limit', required=False, default=1000, type=int,
                      help='Optional. Number of frames to store in output. '
                           'If 0 is set, all frames are stored.')
    args.add_argument('-g', '--gallery',
                      help='Required. Path to a file listing gallery images.',
                      required=True, type=str)
    args.add_argument('-gt', '--ground_truth',
                      help='Optional. Ground truth class.',
                      type=str)
    args.add_argument('-d', '--device',
                      help='Optional. Specify the target device to infer on: CPU or GPU. '
                           'The demo will look for a suitable plugin for device '
                           'specified (by default, it is CPU).',
                      default='CPU', type=str)
    args.add_argument('--no_show', action='store_true',
                      help='Optional. Do not visualize inference results.')
    args.add_argument('-u', '--utilization_monitors', default='', type=str,
                      help='Optional. List of monitors to show initially.')
    return parser


def compute_metrics(positions):
    ''' Computes top-N metrics. '''

    top_1_acc = 0
    top_5_acc = 0
    top_10_acc = 0

    for position in positions:
        if position < 1:
            top_1_acc += 1
        if position < 5:
            top_5_acc += 1
        if position < 10:
            top_10_acc += 1

    mean_pos = np.mean(positions)

    if positions:
        log.info("result: top1 {0:.2f} top5 {1:.2f} top10 {2:.2f} mean_pos {3:.2f}".format(
            top_1_acc / len(positions), top_5_acc / len(positions), top_10_acc / len(positions),
            mean_pos))

    return top_1_acc, top_5_acc, top_10_acc, mean_pos


def time_elapsed(func, *args):
    """ Auxiliary function that helps measure elapsed time. """

    start_time = perf_counter()
    res = func(*args)
    elapsed = perf_counter() - start_time
    return elapsed, res


def main():
    args = build_argparser().parse_args()

    cap = open_images_capture(args.input, args.loop)
    if cap.get_type() not in ('VIDEO', 'CAMERA'):
        raise RuntimeError("The input should be a video file or a numeric camera ID")
    frames = RoiDetectorOnVideo(cap)

    img_retrieval = ImageRetrieval(args.model, args.device, args.gallery, INPUT_SIZE)

    compute_embeddings_times = []
    search_in_gallery_times = []

    positions = []

    frames_processed = 0
    presenter = monitors.Presenter(args.utilization_monitors, 0)
    video_writer = cv2.VideoWriter()
    metrics = PerformanceMetrics()

    for image, view_frame in frames:
        start_time = perf_counter()
        position = None
        sorted_indexes = []

        if image is not None:
            image = central_crop(image, divide_by=5, shift=1)

            elapsed, probe_embedding = time_elapsed(img_retrieval.compute_embedding, image)
            compute_embeddings_times.append(elapsed)

            elapsed, (sorted_indexes, distances) = time_elapsed(img_retrieval.search_in_gallery,
                                                                probe_embedding)
            search_in_gallery_times.append(elapsed)

            sorted_classes = [img_retrieval.gallery_classes[i] for i in sorted_indexes]

            if args.ground_truth is not None:
                position = sorted_classes.index(
                    img_retrieval.text_label_to_class_id[args.ground_truth])
                positions.append(position)
                log.info("ROI detected, found: %d, position of target: %d",
                         sorted_classes[0], position)
            else:
                log.info("ROI detected, found: %s", sorted_classes[0])

        image, key = visualize(view_frame, position,
                        [img_retrieval.impaths[i] for i in sorted_indexes],
                        distances[sorted_indexes] if position is not None else None,
                        img_retrieval.input_size, np.mean(compute_embeddings_times),
                        np.mean(search_in_gallery_times), imshow_delay=3, presenter=presenter, no_show=args.no_show)

        metrics.update(start_time)
        if frames_processed == 0:
            if args.output and not video_writer.open(args.output, cv2.VideoWriter_fourcc(*'MJPG'),
                                                     cap.fps(), (image.shape[1], image.shape[0])):
                raise RuntimeError("Can't open video writer")
        frames_processed += 1
        if video_writer.isOpened() and (args.output_limit <= 0 or frames_processed <= args.output_limit):
            video_writer.write(image)

        if key == 27:
            break

    metrics.log_total()
    for rep in presenter.reportMeans():
        log.info(rep)

    if positions:
        compute_metrics(positions)


if __name__ == '__main__':
    sys.exit(main() or 0)
