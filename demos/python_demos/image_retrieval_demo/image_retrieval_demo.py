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
from argparse import ArgumentParser, SUPPRESS

import numpy as np

from image_retrieval_demo.image_retrieval import ImageRetrieval
from image_retrieval_demo.common import central_crop
from image_retrieval_demo.visualizer import visualize
from image_retrieval_demo.roi_detector_on_video import RoiDetectorOnVideo

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
    args.add_argument('-i',
                      help='Required. Path to a video file or a device node of a web-camera.',
                      required=True, type=str)
    args.add_argument('-g', '--gallery',
                      help='Required. Path to a file listing gallery images.',
                      required=True, type=str)
    args.add_argument('-gt', '--ground_truth',
                      help='Optional. Ground truth class.',
                      type=str)
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

    start_time = time.perf_counter()
    res = func(*args)
    elapsed = time.perf_counter() - start_time
    return elapsed, res


def main():
    """ Main function. """

    log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.INFO, stream=sys.stdout)
    args = build_argparser().parse_args()

    img_retrieval = ImageRetrieval(args.model, args.device, args.gallery, INPUT_SIZE,
                                   args.cpu_extension)

    frames = RoiDetectorOnVideo(args.i)

    compute_embeddings_times = []
    search_in_gallery_times = []

    positions = []

    for image, view_frame in frames:
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
                log.info("ROI detected, found: %d, postion of target: %d",
                         sorted_classes[0], position)
            else:
                log.info("ROI detected, found: %s", sorted_classes[0])

        key = visualize(view_frame, position,
                        [img_retrieval.impaths[i] for i in sorted_indexes],
                        distances[sorted_indexes] if position is not None else None,
                        img_retrieval.input_size, np.mean(compute_embeddings_times),
                        np.mean(search_in_gallery_times), imshow_delay=3, no_show=args.no_show)

        if key == 27:
            break

    if positions:
        compute_metrics(positions)


if __name__ == '__main__':
    sys.exit(main() or 0)
