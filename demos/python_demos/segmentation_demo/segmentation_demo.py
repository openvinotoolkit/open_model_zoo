#!/usr/bin/env python3
"""
 Copyright (C) 2018-2020 Intel Corporation
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
import random
import sys
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from time import perf_counter
import os

import cv2
import numpy as np
from openvino.inference_engine import IECore

sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'common'))

from models import *
import monitors
from pipelines import AsyncPipeline
from performance_metrics import PerformanceMetrics

logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()


CITYSCAPES_COLORS = [
    (128, 64, 128),
    (232, 35, 244),
    (70, 70, 70),
    (156, 102, 102),
    (153, 153, 190),
    (153, 153, 153),
    (30, 170, 250),
    (0, 220, 220),
    (35, 142, 107),
    (152, 251, 152),
    (180, 130, 70),
    (60, 20, 220),
    (0, 0, 255),
    (142, 0, 0),
    (70, 0, 0),
    (100, 60, 0),
    (90, 0, 0),
    (230, 0, 0),
    (32, 11, 119),
    (0, 74, 111),
    (81, 0, 81)
]


def apply_color_map(input):
    ### Initializing colors array if needed
    colors = []
    rng = random.Random(0xACE)
    if not colors:
        colors = np.zeros(size=(256, 1), dtype=np.uint8)
        for i in range(len(CITYSCAPES_COLORS)):
            colors[i, 0] = (CITYSCAPES_COLORS[i][0], CITYSCAPES_COLORS[i][1], CITYSCAPES_COLORS[i][2])
        for i in range(len(colors)):
            colors[i, 0] = (rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255))

    # Converting class to color
    out = cv2.applyColorMap(input, colors)
    return out


def render_segmentation_data(frame, objects):
    # Visualizing result data over source image
    return frame / 2 + apply_color_map(objects) / 2


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument('-m', '--model', help='Required. Path to an .xml file with a trained model.',
                      required=True, type=Path)
    args.add_argument('-i', '--input', required=True, type=str,
                      help='Required. Path to an image, folder with images, video file or a numeric camera ID.')
    args.add_argument('-d', '--device', default='CPU', type=str,
                      help='Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is '
                           'acceptable. The sample will look for a suitable plugin for device specified. '
                           'Default value is CPU.')

    common_model_args = parser.add_argument_group('Common model options')
    common_model_args.add_argument('--labels', help='Optional. Labels mapping file.', default=None, type=str)
    common_model_args.add_argument('-t', '--prob_threshold', default=0.5, type=float,
                                   help='Optional. Probability threshold for detections filtering.')
    common_model_args.add_argument('--keep_aspect_ratio', action='store_true', default=False,
                                   help='Optional. Keeps aspect ratio on resize.')

    infer_args = parser.add_argument_group('Inference options')
    infer_args.add_argument('-nireq', '--num_infer_requests', help='Optional. Number of infer requests',
                            default=1, type=int)
    infer_args.add_argument('-nstreams', '--num_streams',
                            help='Optional. Number of streams to use for inference on the CPU or/and GPU in throughput '
                                 'mode (for HETERO and MULTI device cases use format '
                                 '<device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>).',
                            default='', type=str)
    infer_args.add_argument('-nthreads', '--num_threads', default=None, type=int,
                            help='Optional. Number of threads to use for inference on CPU (including HETERO cases).')

    io_args = parser.add_argument_group('Input/output options')
    io_args.add_argument('-loop', '--loop', help='Optional. Loops input data.', action='store_true', default=False)
    io_args.add_argument('-no_show', '--no_show', help="Optional. Don't show output.", action='store_true')
    io_args.add_argument('-u', '--utilization_monitors', default='', type=str,
                         help='Optional. List of monitors to show initially.')

    debug_args = parser.add_argument_group('Debug options')
    debug_args.add_argument('-r', '--raw_output_message', help='Optional. Output inference results raw values showing.',
                            default=False, action='store_true')
    return parser

def get_plugin_configs(device, num_streams, num_threads):
    config_user_specified = {}

    devices_nstreams = {}
    if num_streams:
        devices_nstreams = {device: num_streams for device in ['CPU', 'GPU'] if device in device} \
            if num_streams.isdigit() \
            else dict(device.split(':', 1) for device in num_streams.split(','))

    if 'CPU' in device:
        if num_threads is not None:
            config_user_specified['CPU_THREADS_NUM'] = str(num_threads)
        if 'CPU' in devices_nstreams:
            config_user_specified['CPU_THROUGHPUT_STREAMS'] = devices_nstreams['CPU'] \
                if int(devices_nstreams['CPU']) > 0 \
                else 'CPU_THROUGHPUT_AUTO'

    if 'GPU' in device:
        if 'GPU' in devices_nstreams:
            config_user_specified['GPU_THROUGHPUT_STREAMS'] = devices_nstreams['GPU'] \
                if int(devices_nstreams['GPU']) > 0 \
                else 'GPU_THROUGHPUT_AUTO'

    return config_user_specified


def main():
    metrics = PerformanceMetrics()
    args = build_argparser().parse_args()

    log.info('Initializing Inference Engine...')
    ie = IECore()

    plugin_config = get_plugin_configs(args.device, args.num_streams, args.num_threads)

    log.info('Loading network...')

    model = SegmentationModel(ie, args.model)

    pipeline = AsyncPipeline(ie, model, plugin_config, device=args.device, max_num_requests=args.num_infer_requests)

    try:
        input_stream = int(args.input)
    except ValueError:
        input_stream = args.input
    cap = cv2.VideoCapture(input_stream)
    if not cap.isOpened():
        log.error('OpenCV: Failed to open capture: ' + str(input_stream))
        sys.exit(1)

    next_frame_id = 0
    next_frame_id_to_show = 0

    log.info('Starting inference...')
    print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")

    presenter = monitors.Presenter(args.utilization_monitors, 55,
                                   (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 4),
                                    round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 8)))

    while cap.isOpened():
        if pipeline.callback_exceptions:
            raise pipeline.callback_exceptions[0]
        # Process all completed requests
        results = pipeline.get_result(next_frame_id_to_show)
        if results:
            objects, frame_meta = results
            frame = frame_meta['frame']
            start_time = frame_meta['start_time']

            presenter.drawGraphs(frame)
            #frame = render_segmentation_data(frame, objects)
            metrics.update(start_time, frame)
            if not args.no_show:
                cv2.imshow('Detection Results', frame)
                key = cv2.waitKey(1)

                ESC_KEY = 27
                # Quit.
                if key in {ord('q'), ord('Q'), ESC_KEY}:
                    break
                presenter.handleKey(key)
            next_frame_id_to_show += 1
            continue

        if pipeline.is_ready():
            # Get new image/frame
            start_time = perf_counter()
            ret, frame = cap.read()
            if not ret:
                if args.loop:
                    cap.open(input_stream)
                else:
                    cap.release()
                continue

            # Submit for inference
            pipeline.submit_data(frame, next_frame_id, {'frame': frame, 'start_time': start_time})
            next_frame_id += 1
        else:
            # Wait for empty request
            pipeline.await_any()

    pipeline.await_all()
    # Process completed requests
    while pipeline.has_completed_request():
        results = pipeline.get_result(next_frame_id_to_show)
        #print(results)
        if results:
            objects, frame_meta = results
            frame = frame_meta['frame']
            start_time = frame_meta['start_time']
            presenter.drawGraphs(frame)
            #frame = render_segmentation_data(frame, objects)
            metrics.update(start_time, frame)
            if not args.no_show:
                cv2.imshow('Segmentation Results', frame)
                key = cv2.waitKey(1)

                ESC_KEY = 27
                # Quit.
                if key in {ord('q'), ord('Q'), ESC_KEY}:
                    break
                presenter.handleKey(key)
            next_frame_id_to_show += 1
        else:
            break

    metrics.print_total()
    print(presenter.reportMeans())


if __name__ == '__main__':
    sys.exit(main() or 0)
