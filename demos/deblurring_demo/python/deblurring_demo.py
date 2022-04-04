"""
 Copyright (C) 2018-2022 Intel Corporation

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

from openvino.model_zoo.model_api.models import Deblurring
from openvino.model_zoo.model_api.performance_metrics import PerformanceMetrics
from openvino.model_zoo.model_api.pipelines import get_user_config, AsyncPipeline
from openvino.model_zoo.model_api.adapters import create_core, OpenvinoAdapter, OVMSAdapter

import monitors
from images_capture import open_images_capture
from helpers import log_latency_per_stage

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument('-m', '--model', required=True,
                      help='Required. Path to an .xml file with a trained model '
                           'or address of model inference service if using OVMS adapter.')
    args.add_argument('-i', '--input', required=True,
                      help='Required. An input to process. The input must be a single image, '
                           'a folder of images or anything that cv2.VideoCapture can process.')
    args.add_argument('--adapter', help='Optional. Specify the model adapter. Default is openvino.',
                      default='openvino', type=str, choices=('openvino', 'ovms'))
    args.add_argument('-d', '--device', default='CPU', type=str,
                      help='Optional. Specify the target device to infer on; CPU, GPU, HDDL or MYRIAD is '
                           'acceptable. The demo will look for a suitable plugin for device specified. '
                           'Default value is CPU.')
    args.add_argument('--layout', type=str, default=None,
                      help='Optional. Model inputs layouts. '
                           'Ex. NCHW or input0:NCHW,input1:NC in case of more than one input.')

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
    io_args.add_argument('--loop', default=False, action='store_true',
                         help='Optional. Enable reading the input in a loop.')
    io_args.add_argument('-o', '--output', required=False,
                         help='Optional. Name of the output file(s) to save.')
    io_args.add_argument('-limit', '--output_limit', required=False, default=1000, type=int,
                         help='Optional. Number of frames to store in output. '
                              'If 0 is set, all frames are stored.')
    io_args.add_argument('--no_show', help="Optional. Don't show output.", action='store_true')
    io_args.add_argument('-u', '--utilization_monitors', default='', type=str,
                         help='Optional. List of monitors to show initially.')
    return parser


def main():
    args = build_argparser().parse_args()

    cap = open_images_capture(args.input, args.loop)
    next_frame_id = 1
    next_frame_id_to_show = 0

    metrics = PerformanceMetrics()
    render_metrics = PerformanceMetrics()
    video_writer = cv2.VideoWriter()

    if args.adapter == 'openvino':
        plugin_config = get_user_config(args.device, args.num_streams, args.num_threads)
        model_adapter = OpenvinoAdapter(create_core(), args.model, device=args.device, plugin_config=plugin_config,
                                        max_num_requests=args.num_infer_requests, model_parameters = {'input_layouts': args.layout})
    elif args.adapter == 'ovms':
        model_adapter = OVMSAdapter(args.model)

    start_time = perf_counter()
    frame = cap.read()
    if frame is None:
        raise RuntimeError("Can't read an image from the input")

    model = Deblurring(model_adapter, preload=False)
    model.reshape(frame.shape)
    model.log_layers_info()

    pipeline = AsyncPipeline(model)

    pipeline.submit_data(frame, 0, {'frame': frame, 'start_time': start_time})

    presenter = monitors.Presenter(args.utilization_monitors, 55,
                                   (round(frame.shape[1] / 4), round(frame.shape[0] / 8)))
    if args.output and not video_writer.open(args.output, cv2.VideoWriter_fourcc(*'MJPG'),
                                             cap.fps(), (2 * frame.shape[1], frame.shape[0])):
        raise RuntimeError("Can't open video writer")

    while True:
        if pipeline.is_ready():
            # Get new image/frame
            start_time = perf_counter()
            frame = cap.read()
            if frame is None:
                break

            # Submit for inference
            pipeline.submit_data(frame, next_frame_id, {'frame': frame, 'start_time': start_time})
            next_frame_id += 1
        else:
            # Wait for empty request
            pipeline.await_any()

        if pipeline.callback_exceptions:
            raise pipeline.callback_exceptions[0]
        # Process all completed requests
        results = pipeline.get_result(next_frame_id_to_show)
        if results:
            result_frame, frame_meta = results
            input_frame = frame_meta['frame']
            start_time = frame_meta['start_time']

            rendering_start_time = perf_counter()
            if input_frame.shape != result_frame.shape:
                input_frame = cv2.resize(input_frame, (result_frame.shape[1], result_frame.shape[0]))
            final_image = cv2.hconcat([input_frame, result_frame])
            render_metrics.update(rendering_start_time)

            presenter.drawGraphs(final_image)
            metrics.update(start_time, final_image)

            if video_writer.isOpened() and (args.output_limit <= 0 or next_frame_id_to_show <= args.output_limit-1):
                video_writer.write(final_image)
            next_frame_id_to_show += 1

            if not args.no_show:
                cv2.imshow('Deblurring Results', final_image)
                key = cv2.waitKey(1)
                if key == 27 or key == 'q' or key == 'Q':
                    break
                presenter.handleKey(key)

    pipeline.await_all()
    if pipeline.callback_exceptions:
        raise pipeline.callback_exceptions[0]
    # Process completed requests
    for next_frame_id_to_show in range(next_frame_id_to_show, next_frame_id):
        results = pipeline.get_result(next_frame_id_to_show)
        result_frame, frame_meta = results
        input_frame = frame_meta['frame']
        start_time = frame_meta['start_time']

        rendering_start_time = perf_counter()
        if input_frame.shape != result_frame.shape:
            input_frame = cv2.resize(input_frame, (result_frame.shape[1], result_frame.shape[0]))
        final_image = cv2.hconcat([input_frame, result_frame])
        render_metrics.update(rendering_start_time)

        presenter.drawGraphs(final_image)
        metrics.update(start_time, final_image)

        if video_writer.isOpened() and (args.output_limit <= 0 or next_frame_id_to_show <= args.output_limit-1):
            video_writer.write(final_image)

        if not args.no_show:
            cv2.imshow('Deblurring Results', final_image)
            key = cv2.waitKey(1)

    metrics.log_total()
    log_latency_per_stage(cap.reader_metrics.get_latency(),
                          pipeline.preprocess_metrics.get_latency(),
                          pipeline.inference_metrics.get_latency(),
                          pipeline.postprocess_metrics.get_latency(),
                          render_metrics.get_latency())
    for rep in presenter.reportMeans():
        log.info(rep)


if __name__ == '__main__':
    sys.exit(main() or 0)
