#!/usr/bin/env python3
"""
 Copyright (C) 2021 Intel Corporation

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
import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))
sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python/openvino/model_zoo'))

from model_api.models import OutputTransform, get_instance_segmentation_model
from model_api.performance_metrics import PerformanceMetrics
from model_api.pipelines import get_user_config, AsyncPipeline
from model_api.adapters import create_core, OpenvinoAdapter, RemoteAdapter

import monitors
from images_capture import open_images_capture
from helpers import resolution, log_latency_per_stage

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument('-m', '--model', help='Required. Path to an .xml file with a trained model.',
                      required=True)
    args.add_argument('--adapter', help='Optional. Specify the model adapter. Default is openvino.',
                      default='openvino', type=str, choices=('openvino', 'remote'))
    args.add_argument('-i', '--input', required=True,
                      help='Required. An input to process. The input must be a single image, '
                           'a folder of images, video file or camera id.')
    args.add_argument('-d', '--device', default='CPU', type=str,
                      help='Optional. Specify the target device to infer on; CPU, GPU, HDDL or MYRIAD is '
                           'acceptable. The demo will look for a suitable plugin for device specified. '
                           'Default value is CPU.')
    args.add_argument('-pt', '--prob_threshold',
                      help='Optional. Probability threshold for detections filtering.',
                      default=0.5, type=float, metavar='"<num>"')
    args.add_argument('--keep_aspect_ratio',
                      help='Optional. Force image resize to keep aspect ratio.',
                      action='store_true')
    args.add_argument('--labels', help='Optional. Labels mapping file.', default=None, type=str)
    args.add_argument('--target_bgr', default=None, type=str,
                      help='Optional. Background onto which to composite the output (by default to green field).')
    args.add_argument('--blur_bgr', default=False, action='store_true',
                      help='Optional. Blur background.')

    infer_args = parser.add_argument_group('Inference options')
    infer_args.add_argument('-nireq', '--num_infer_requests', help='Optional. Number of infer requests.',
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
    io_args.add_argument('--show_with_original_frame',
                         help="Optional. Merge the result frame with the original one.", action='store_true')
    io_args.add_argument('--output_resolution', default=None, type=resolution,
                         help='Optional. Specify the maximum output window resolution '
                              'in (width x height) format. Example: 1280x720. '
                              'Input frame size used by default.')
    io_args.add_argument('-u', '--utilization_monitors', default='', type=str,
                         help='Optional. List of monitors to show initially.')

    debug_args = parser.add_argument_group('Debug options')
    debug_args.add_argument('-r', '--raw_output_message', help='Optional. Output inference results as mask histogram.',
                            default=False, action='store_true')
    return parser


def print_raw_results(outputs, frame_id):
    scores, classes, boxes, masks = outputs
    log.debug('  -------------------------- Frame # {} --------------------------  '.format(frame_id))
    log.debug('  Class ID | Confidence |     XMIN |     YMIN |     XMAX |     YMAX ')
    for box, cls, score, mask in zip(boxes, classes, scores, masks):
        log.debug('{:>10} | {:>10f} | {:>8.2f} | {:>8.2f} | {:>8.2f} | {:>8.2f} '.format(cls, score, *box))


def fit_to_window(input_img, output_resolution):
    h, w = input_img.shape[:2]
    target_shape = np.append(output_resolution[::-1], [3])
    output = np.zeros(target_shape, dtype=np.uint8)
    scale = min(output_resolution[1] / h, output_resolution[0] / w)
    input_img = cv2.resize(input_img, None, fx=scale, fy=scale)
    dw = output_resolution[0] - input_img.shape[1]
    dh = output_resolution[1] - input_img.shape[0]
    x0 = dw // 2
    x1 = dw // 2 if dw % 2 == 0 else dw // 2 + 1
    y0 = dh // 2
    y1 = dh // 2 if dh % 2 == 0 else dh // 2 + 1
    output[y0: output.shape[0] - y1, x0: output.shape[1] - x1, :] = input_img
    return output


def render_results(frame, objects, output_resolution, target_bgr, person_id, blur_bgr=False, show_with_original_frame=False):
    if target_bgr is None:
        target_bgr = np.full(frame.shape, [155, 255, 120], dtype=np.uint8)
    else:
        target_bgr = cv2.resize(target_bgr, (frame.shape[1], frame.shape[0]))
    classes, masks = objects[1], objects[3]
    # Choose masks only for person class
    valid_inds = classes == person_id
    masks = [mask for mask, is_valid in zip(masks, valid_inds) if is_valid]
    if not len(masks):
        output = frame
    else:
        composed_mask = masks[0]
        for i in range(1, len(masks)):
            composed_mask = np.logical_or(composed_mask, masks[i])
        # Smooth contours of the predicted mask
        composed_mask = cv2.medianBlur(composed_mask.astype(np.uint8), 11)
        composed_mask = np.repeat(np.expand_dims(composed_mask, axis=-1), 3, axis=2)
        if target_bgr is not None and blur_bgr:
            target_bgr = cv2.blur(cv2.resize(target_bgr, (target_bgr.shape[1], target_bgr.shape[0])), (7, 7))
        output = np.where(composed_mask == 1, frame, target_bgr)
    if show_with_original_frame:
        output = cv2.hconcat([frame, output])
    h, w = output.shape[:2]
    if (w, h) != tuple(output_resolution):
        output = fit_to_window(output, output_resolution)
    return output


def main():
    args = build_argparser().parse_args()

    cap = open_images_capture(args.input, args.loop)

    target_bgr = open_images_capture(args.target_bgr, loop=True) if args.target_bgr else None

    if args.adapter == 'openvino':
        plugin_config = get_user_config(args.device, args.num_streams, args.num_threads)
        model_adapter = OpenvinoAdapter(create_core(), args.model, device=args.device, plugin_config=plugin_config,
                                        max_num_requests=args.num_infer_requests)
    elif args.adapter == 'remote':
        log.info('Reading model {}'.format(args.model))
        serving_config = {"address": "localhost", "port": 9000}
        model_adapter = RemoteAdapter(args.model, serving_config)

    labels = ['__background__', 'person'] if args.labels is None else args.labels

    model = get_instance_segmentation_model(model_adapter, prob_threshold=args.prob_threshold,
                                            labels=labels, keep_aspect_ratio=args.keep_aspect_ratio)
    person_id = -1
    for i, label in enumerate(labels):
        if label == 'person':
            person_id = i
            break
    assert person_id >= 0, 'Person class did not find in labels list.'

    model.log_layers_info()

    pipeline = AsyncPipeline(model)

    next_frame_id = 0
    next_frame_id_to_show = 0

    metrics = PerformanceMetrics()
    render_metrics = PerformanceMetrics()
    presenter = None
    output_transform = None
    video_writer = cv2.VideoWriter()
    while True:
        if pipeline.is_ready():
            # Get new image/frame
            start_time = perf_counter()
            frame = cap.read()
            bgr = target_bgr.read() if target_bgr is not None else None
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
                                                         cap.fps(), tuple(output_resolution)):
                    raise RuntimeError("Can't open video writer")
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
            objects, frame_meta = results
            if args.raw_output_message:
                print_raw_results(objects, next_frame_id_to_show)
            frame = frame_meta['frame']
            start_time = frame_meta['start_time']
            rendering_start_time = perf_counter()
            frame = render_results(frame, objects, output_resolution, bgr, person_id,
                                   args.blur_bgr, args.show_with_original_frame)
            render_metrics.update(rendering_start_time)
            presenter.drawGraphs(frame)
            metrics.update(start_time, frame)

            if video_writer.isOpened() and (args.output_limit <= 0 or next_frame_id_to_show <= args.output_limit-1):
                video_writer.write(frame)
            next_frame_id_to_show += 1

            if not args.no_show:
                cv2.imshow('Background subtraction results', frame)
                key = cv2.waitKey(1)
                if key == 27 or key == 'q' or key == 'Q':
                    break
                presenter.handleKey(key)

    pipeline.await_all()
    # Process completed requests
    for next_frame_id_to_show in range(next_frame_id_to_show, next_frame_id):
        results = pipeline.get_result(next_frame_id_to_show)
        while results is None:
            results = pipeline.get_result(next_frame_id_to_show)
        objects, frame_meta = results
        if args.raw_output_message:
            print_raw_results(objects, next_frame_id_to_show, model.labels)
        frame = frame_meta['frame']
        start_time = frame_meta['start_time']

        rendering_start_time = perf_counter()
        frame = render_results(frame, objects, output_resolution, bgr, person_id,
                               args.blur_bgr, args.show_with_original_frame)
        render_metrics.update(rendering_start_time)
        presenter.drawGraphs(frame)
        metrics.update(start_time, frame)

        if video_writer.isOpened() and (args.output_limit <= 0 or next_frame_id_to_show <= args.output_limit-1):
            video_writer.write(frame)

        if not args.no_show:
            cv2.imshow('Background subtraction results', frame)
            cv2.waitKey(1)

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
