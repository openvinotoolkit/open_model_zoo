#!/usr/bin/env python3
"""
 Copyright (C) 2018-2021 Intel Corporation

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
from openvino.inference_engine import IECore, get_version

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python/openvino/model_zoo'))

from model_api import models
from model_api.performance_metrics import PerformanceMetrics, put_highlighted_text
from model_api.pipelines import get_user_config, parse_devices, AsyncPipeline

import monitors
from images_capture import open_images_capture
from helpers import resolution, log_blobs_info, log_runtime_settings, log_latency_per_stage

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument('-m', '--model', help='Required. Path to an .xml file with a trained model.',
                      required=True, type=Path)
    args.add_argument('-i', '--input', required=True,
                      help='Required. An input to process. The input must be a single image, '
                           'a folder of images, video file or camera id.')
    args.add_argument('-d', '--device', default='CPU', type=str,
                      help='Optional. Specify the target device to infer on; CPU, GPU, HDDL or MYRIAD is '
                           'acceptable. The demo will look for a suitable plugin for device specified. '
                           'Default value is CPU.')

    common_model_args = parser.add_argument_group('Common model options')
    common_model_args.add_argument('--labels', help='Required. Labels mapping file.', default=None,
                                   required=True, type=Path)
    common_model_args.add_argument('-ntop', help='Optional. Number of top results. Default value is 5. Must be from 1 to 10.', default=5,
                                   required=False, type=int, choices=range(1, 11))

    infer_args = parser.add_argument_group('Inference options')
    infer_args.add_argument('-nireq', '--num_infer_requests', help='Optional. Number of infer requests',
                            default=0, type=int)
    infer_args.add_argument('-nstreams', '--num_streams',
                            help='Optional. Number of streams to use for inference on the CPU or/and GPU in throughput '
                                 'mode (for HETERO and MULTI device cases use format '
                                 '<device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>).',
                            default='', type=str)
    infer_args.add_argument('-nthreads', '--num_threads', default=None, type=int,
                            help='Optional. Number of threads to use for inference on CPU (including HETERO cases).')

    io_args = parser.add_argument_group('Input/output options')
    io_args.add_argument('--loop', default=False, action='store_true',
                         help='Optional. Enable reading the input in a loop.')
    io_args.add_argument('-o', '--output', required=False,
                         help='Optional. Name of the output file(s) to save.')
    io_args.add_argument('--pause', required=False, default=1, type=int,
                         help='Optional. Pause in ms between frames to show.')
    io_args.add_argument('-limit', '--output_limit', required=False, default=1000, type=int,
                         help='Optional. Number of frames to store in output. '
                              'If 0 is set, all frames are stored.')
    io_args.add_argument('--no_show', help="Optional. Don't show output.", action='store_true')
    io_args.add_argument('--output_resolution', default=None, type=resolution,
                         help='Optional. Specify the maximum output window resolution '
                              'in (width x height) format. Example: 1280x720. '
                              'Input frame size used by default.')
    io_args.add_argument('-u', '--utilization_monitors', default='', type=str,
                         help='Optional. List of monitors to show initially.')

    debug_args = parser.add_argument_group('Debug options')
    debug_args.add_argument('-r', '--raw_output_message', help='Optional. Output inference results raw values showing.',
                            default=False, action='store_true')
    return parser


def draw_labels(frame, classifications, output_transform):
    frame = output_transform.resize(frame)
    label_height = cv2.getTextSize(classifications[0][1], cv2.FONT_HERSHEY_COMPLEX, 0.75, 2)[0][1]
    labels_pos =  frame.shape[0] - label_height * (2 * len(classifications) + 1)
    if (labels_pos < 0):
        labels_pos = label_height
        log.warning("Too much labels to display on this frame, some will be omitted ")

    offset_y = labels_pos
    for classification in classifications:
        label = '{} {:.1%}'.format(classification[1], classification[2])
        label_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, 0.75, 2)[0][0]
        offset_y += label_height * 2
        put_highlighted_text(frame, label, (frame.shape[1] - label_width, offset_y),
            cv2.FONT_HERSHEY_COMPLEX, 0.75, (10, 10, 210), 2)
    return frame


def print_raw_results(classifications, frame_id):
    label_max_len = len(max(classifications, key=lambda item:len(item[1]))[1])
    log.debug(' ------------------- Frame # {} ------------------ '.format(frame_id))
    log.debug(' Class ID | {:^{width}s}| Confidence '.format("Label", width = label_max_len))
    for classification in classifications:
        class_id = classification[0]
        label = classification[1]
        conf = classification[2]
        log.debug('{:^9} | {:^{width}}| {:^10f} '.format(class_id, label, conf, width = label_max_len))


def main():
    args = build_argparser().parse_args()

    cap = open_images_capture(args.input, args.loop)

    log.info('OpenVINO Inference Engine')
    log.info('\tbuild: {}'.format(get_version()))
    ie = IECore()

    plugin_config = get_user_config(args.device, args.num_streams, args.num_threads)

    log.info('Reading model {}'.format(args.model))
    model = models.Classification(ie, args.model, ntop=args.ntop, labels=args.labels, log = log)
    log_blobs_info(model)

    async_pipeline = AsyncPipeline(ie, model, plugin_config,
                                      device=args.device, max_num_requests=args.num_infer_requests)

    log.info('The model {} is loaded to {}'.format(args.model, args.device))
    log_runtime_settings(async_pipeline.exec_net, set(parse_devices(args.device)))

    next_frame_id = 0
    next_frame_id_to_show = 0

    metrics = PerformanceMetrics()
    render_metrics = PerformanceMetrics()
    presenter = None
    output_transform = None
    video_writer = cv2.VideoWriter()

    while True:
        if async_pipeline.callback_exceptions:
            raise async_pipeline.callback_exceptions[0]
        # Process all completed requests
        results = async_pipeline.get_result(next_frame_id_to_show)
        if results:
            classifications, frame_meta = results
            frame = frame_meta['frame']
            start_time = frame_meta['start_time']

            if len(classifications) and args.raw_output_message:
                print_raw_results(classifications, next_frame_id_to_show)

            presenter.drawGraphs(frame)
            rendering_start_time = perf_counter()
            frame = draw_labels(frame, classifications, output_transform)
            render_metrics.update(rendering_start_time)
            metrics.update(start_time, frame)

            if video_writer.isOpened() and (args.output_limit <= 0 or next_frame_id_to_show <= args.output_limit-1):
                video_writer.write(frame)
            next_frame_id_to_show += 1

            if not args.no_show:
                cv2.imshow('Classification Results', frame)
                key = cv2.waitKey(args.pause)

                ESC_KEY = 27
                # Quit.
                if key in {ord('q'), ord('Q'), ESC_KEY}:
                    break
                presenter.handleKey(key)
            continue

        if async_pipeline.is_ready():
            # Get new image/frame
            start_time = perf_counter()
            frame = cap.read()
            if frame is None:
                if next_frame_id == 0:
                    raise ValueError("Can't read an image from the input")
                break
            if next_frame_id == 0:
                output_transform = models.OutputTransform(frame.shape[:2], args.output_resolution)
                if args.output_resolution:
                    output_resolution = output_transform.new_resolution
                else:
                    output_resolution = (frame.shape[1], frame.shape[0])
                presenter = monitors.Presenter(args.utilization_monitors, 55,
                                               (round(output_resolution[0] / 4), round(output_resolution[1] / 8)))
                if args.output and not video_writer.open(args.output, cv2.VideoWriter_fourcc(*'MJPG'),
                                                         cap.fps(), output_resolution):
                    raise RuntimeError("Can't open video writer")
            # Submit for inference
            async_pipeline.submit_data(frame, next_frame_id, {'frame': frame, 'start_time': start_time})
            next_frame_id += 1

        else:
            # Wait for empty request
            async_pipeline.await_any()

    async_pipeline.await_all()
    # Process completed requests
    for next_frame_id_to_show in range(next_frame_id_to_show, next_frame_id):
        results = async_pipeline.get_result(next_frame_id_to_show)
        while results is None:
            results = async_pipeline.get_result(next_frame_id_to_show)
        classifications, frame_meta = results
        frame = frame_meta['frame']
        start_time = frame_meta['start_time']

        if len(classifications) and args.raw_output_message:
            print_raw_results(classifications, next_frame_id_to_show)

        presenter.drawGraphs(frame)
        rendering_start_time = perf_counter()
        frame = draw_labels(frame, classifications, output_transform)
        render_metrics.update(rendering_start_time)
        metrics.update(start_time, frame)

        if video_writer.isOpened() and (args.output_limit <= 0 or next_frame_id_to_show <= args.output_limit-1):
            video_writer.write(frame)

        if not args.no_show:
            cv2.imshow('Classification Results', frame)
            key = cv2.waitKey(args.pause)

            ESC_KEY = 27
            # Quit.
            if key in {ord('q'), ord('Q'), ESC_KEY}:
                break
            presenter.handleKey(key)

    metrics.log_total()
    log_latency_per_stage(cap.reader_metrics.get_latency(),
                          async_pipeline.preprocess_metrics.get_latency(),
                          async_pipeline.inference_metrics.get_latency(),
                          async_pipeline.postprocess_metrics.get_latency(),
                          render_metrics.get_latency())
    for rep in presenter.reportMeans():
        log.info(rep)


if __name__ == '__main__':
    sys.exit(main() or 0)
