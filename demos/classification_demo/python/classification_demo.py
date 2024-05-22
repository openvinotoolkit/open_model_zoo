#!/usr/bin/env python3
"""
 Copyright (C) 2018-2024 Intel Corporation

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

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python/model_zoo'))

from model_api.models import Classification, OutputTransform
from model_api.performance_metrics import put_highlighted_text, PerformanceMetrics
from model_api.pipelines import get_user_config, AsyncPipeline
from model_api.adapters import create_core, OpenvinoAdapter, OVMSAdapter

import monitors
from images_capture import open_images_capture
from helpers import resolution, log_latency_per_stage

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument('-m', '--model', required=True,
                      help='Required. Path to an .xml file with a trained model '
                           'or address of model inference service if using OVMS adapter.')
    args.add_argument('--adapter', help='Optional. Specify the model adapter. Default is openvino.',
                      default='openvino', type=str, choices=('openvino', 'ovms'))
    args.add_argument('-i', '--input', required=True,
                      help='Required. An input to process. The input must be a single image, '
                           'a folder of images, video file or camera id.')
    args.add_argument('-d', '--device', default='CPU', type=str,
                      help="Optional. Specify a device to infer on (the list of available devices is shown below). Use "
                           "'-d HETERO:<comma-separated_devices_list>' format to specify HETERO plugin. Use "
                           "'-d MULTI:<comma-separated_devices_list>' format to specify MULTI plugin. Default is CPU")

    common_model_args = parser.add_argument_group('Common model options')
    common_model_args.add_argument('--labels', help='Optional. Labels mapping file.', default=None, type=str)
    common_model_args.add_argument('-topk', help='Optional. Number of top results. Default value is 5. Must be from 1 to 10.', default=5,
                                   type=int, choices=range(1, 11))
    common_model_args.add_argument('--layout', type=str, default=None,
                                   help='Optional. Model inputs layouts. '
                                        'Ex. NCHW or input0:NCHW,input1:NC in case of more than one input.')

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
                         help='Optional. Name of the output file(s) to save. Frames of odd width or height can be truncated. See https://github.com/opencv/opencv/pull/24086')
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

    input_transform_args = parser.add_argument_group('Input transform options')
    input_transform_args.add_argument('--reverse_input_channels', default=False, action='store_true',
                                      help='Optional. Switch the input channels order from '
                                           'BGR to RGB.')
    input_transform_args.add_argument('--mean_values', default=None, type=float, nargs=3,
                                      help='Optional. Normalize input by subtracting the mean '
                                           'values per channel. Example: 255.0 255.0 255.0')
    input_transform_args.add_argument('--scale_values', default=None, type=float, nargs=3,
                                      help='Optional. Divide input by scale values per channel. '
                                           'Division is applied after mean values subtraction. '
                                           'Example: 255.0 255.0 255.0')

    debug_args = parser.add_argument_group('Debug options')
    debug_args.add_argument('-r', '--raw_output_message', help='Optional. Output inference results raw values showing.',
                            default=False, action='store_true')
    return parser


def draw_labels(frame, classifications, output_transform):
    frame = output_transform.resize(frame)
    class_label = ""
    if classifications:
        class_label = classifications[0][1]
    font_scale = 0.7
    label_height = cv2.getTextSize(class_label, cv2.FONT_HERSHEY_COMPLEX, font_scale, 2)[0][1]
    initial_labels_pos =  frame.shape[0] - label_height * (int(1.5 * len(classifications)) + 1)

    if (initial_labels_pos < 0):
        initial_labels_pos = label_height
        log.warning('Too much labels to display on this frame, some will be omitted')
    offset_y = initial_labels_pos

    header = "Label:     Score:"
    label_width = cv2.getTextSize(header, cv2.FONT_HERSHEY_COMPLEX, font_scale, 2)[0][0]
    put_highlighted_text(frame, header, (frame.shape[1] - label_width, offset_y),
        cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 0, 0), 2)

    for idx, class_label, score in classifications:
        label = '{}. {}    {:.2f}'.format(idx, class_label, score)
        label_width = cv2.getTextSize(label, cv2.FONT_HERSHEY_COMPLEX, font_scale, 2)[0][0]
        offset_y += int(label_height * 1.5)
        put_highlighted_text(frame, label, (frame.shape[1] - label_width, offset_y),
            cv2.FONT_HERSHEY_COMPLEX, font_scale, (255, 0, 0), 2)
    return frame


def print_raw_results(classifications, frame_id):
    label_max_len = 0
    if classifications:
        label_max_len = len(max([cl[1] for cl in classifications], key=len))

    log.debug(' ------------------- Frame # {} ------------------ '.format(frame_id))

    if label_max_len != 0:
        log.debug(' Class ID | {:^{width}s}| Confidence '.format('Label', width=label_max_len))
    else:
        log.debug(' Class ID | Confidence ')

    for class_id, class_label, score in classifications:
        if class_label != "":
            log.debug('{:^9} | {:^{width}s}| {:^10f} '.format(class_id, class_label, score, width=label_max_len))
        else:
            log.debug('{:^9} | {:^10f} '.format(class_id, score))


def main():
    args = build_argparser().parse_args()

    cap = open_images_capture(args.input, args.loop)
    delay = int(cap.get_type() in {'VIDEO', 'CAMERA'})

    if args.adapter == 'openvino':
        plugin_config = get_user_config(args.device, args.num_streams, args.num_threads)
        model_adapter = OpenvinoAdapter(create_core(), args.model, device=args.device, plugin_config=plugin_config,
                                        max_num_requests=args.num_infer_requests, model_parameters = {'input_layouts': args.layout})
    elif args.adapter == 'ovms':
        model_adapter = OVMSAdapter(args.model)

    config = {
        'mean_values':  args.mean_values,
        'scale_values': args.scale_values,
        'reverse_input_channels': args.reverse_input_channels,
        'topk': args.topk,
        'path_to_labels': args.labels
    }
    model = Classification(model_adapter, config)
    model.log_layers_info()

    async_pipeline = AsyncPipeline(model)

    next_frame_id = 0
    next_frame_id_to_show = 0

    metrics = PerformanceMetrics()
    render_metrics = PerformanceMetrics()
    presenter = None
    output_transform = None
    video_writer = cv2.VideoWriter()
    ESC_KEY = 27
    key = -1
    while True:
        if async_pipeline.callback_exceptions:
            raise async_pipeline.callback_exceptions[0]
        # Process all completed requests
        results = async_pipeline.get_result(next_frame_id_to_show)
        if results:
            classifications, frame_meta = results
            frame = frame_meta['frame']
            start_time = frame_meta['start_time']
            if args.raw_output_message:
                print_raw_results(classifications, next_frame_id_to_show)

            presenter.drawGraphs(frame)
            rendering_start_time = perf_counter()
            frame = draw_labels(frame, classifications, output_transform)
            if delay or args.no_show:
                render_metrics.update(rendering_start_time)
                metrics.update(start_time, frame)

            if video_writer.isOpened() and (args.output_limit <= 0 or next_frame_id_to_show <= args.output_limit-1):
                video_writer.write(frame)
            next_frame_id_to_show += 1

            if not args.no_show:
                cv2.imshow('Classification Results', frame)
                key = cv2.waitKey(delay)
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
                output_transform = OutputTransform(frame.shape[:2], args.output_resolution)
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
    if async_pipeline.callback_exceptions:
        raise async_pipeline.callback_exceptions[0]
    if key not in {ord('q'), ord('Q'), ESC_KEY}:
        # Process completed requests
        for next_frame_id_to_show in range(next_frame_id_to_show, next_frame_id):
            results = async_pipeline.get_result(next_frame_id_to_show)
            classifications, frame_meta = results
            frame = frame_meta['frame']
            start_time = frame_meta['start_time']

            if args.raw_output_message:
                print_raw_results(classifications, next_frame_id_to_show)

            presenter.drawGraphs(frame)
            rendering_start_time = perf_counter()
            frame = draw_labels(frame, classifications, output_transform)
            if delay or args.no_show:
                render_metrics.update(rendering_start_time)
                metrics.update(start_time, frame)

            if video_writer.isOpened() and (args.output_limit <= 0 or next_frame_id_to_show <= args.output_limit-1):
                video_writer.write(frame)

            if not args.no_show:
                cv2.imshow('Classification Results', frame)
                key = cv2.waitKey(delay)

                # Quit.
                if key in {ord('q'), ord('Q'), ESC_KEY}:
                    break
                presenter.handleKey(key)

    if delay or args.no_show:
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
