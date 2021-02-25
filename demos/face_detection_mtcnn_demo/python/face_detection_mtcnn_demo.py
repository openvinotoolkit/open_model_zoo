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

import colorsys
import logging
import random
import sys
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from time import perf_counter

import cv2
import numpy as np
from openvino.inference_engine import IECore

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))

import models
import monitors
from pipelines import MtcnnPipeline
from images_capture import open_images_capture
from performance_metrics import PerformanceMetrics

logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument('-i', '--input', required=True,
                      help='Required. An input to process. The input must be a single image, '
                           'a folder of images, video file or camera id.')
    args.add_argument('-m_p', '--model_proposal', help='Required. Path to an .xml file with a trained model.',
                      required=True, type=Path)
    args.add_argument('-m_r', '--model_refine', help='Required. Path to an .xml file with a trained model.',
                      required=True, type=Path)
    args.add_argument('-m_o', '--model_output', help='Required. Path to an .xml file with a trained model.',
                      required=True, type=Path)
    args.add_argument('-d_p', '--device_proposal', type=str, default='CPU',
                      help='Optional. Specify the target device to infer on MTCNN Proposal stage;'
                           ' CPU, GPU, FPGA, HDDL or MYRIAD is acceptable. '
                           'The sample will look for a suitable plugin for device specified. '
                           'Default value is CPU.')
    args.add_argument('-d_r', '--device_refine', type=str, default='CPU',
                      help='Optional. Specify the target device to infer on MTCNN Refine stage;'
                           ' CPU, GPU, FPGA, HDDL or MYRIAD is acceptable. '
                           'The sample will look for a suitable plugin for device specified. '
                           'Default value is CPU.')
    args.add_argument('-d_o', '--device_output', type=str, default='CPU',
                      help='Optional. Specify the target device to infer on MTCNN Output stage;'
                           ' CPU, GPU, FPGA, HDDL or MYRIAD is acceptable. '
                           'The sample will look for a suitable plugin for device specified. '
                           'Default value is CPU.')

    infer_args = parser.add_argument_group('Inference options')
    infer_args.add_argument('--proposal_async', default=False, action='store_true')
    infer_args.add_argument('--refine_batch_size', type=int, default=0,
                            help='Optional. Sets fixed batch size for MTCNN Refine stage, disallowing dynamic reshape.'
                                 'Forces asynchronous mode for MTCNN Refine stage. Dynamic reshape is set by default.')
    infer_args.add_argument('--refine_requests', type=int, default=1,
                            help='Optional. Number of infer requests for MTCNN Refine stage. '
                                 'Works only with --refine_batch_size > 0.')
    infer_args.add_argument('--refine_nstreams', type=str, default='',
                            help='Optional. Number of streams to use for inference on the CPU or/and GPU in throughput '
                                 'mode (for HETERO and MULTI device cases use format <device1>:<nstreams1>, '
                                 '<device2>:<nstreams2> or just <nstreams>) for MTCNN Refine stage.'
                                 'Works only with --refine_batch_size > 0.')
    infer_args.add_argument('--refine_nthreads', type=int, default=1,
                            help='Optional. Number of threads to use for inference on CPU (including HETERO cases) for'
                                 'MTCNN Refine stage. Works only with --refine_batch_size > 0.')

    infer_args.add_argument('--output_batch_size', type=int, default=0,
                            help='Optional. Sets fixed batch size for MTCNN Output stage, disallowing dynamic reshape.'
                                 'Forces asynchronous mode for MTCNN Output stage. Dynamic reshape is set by default.')
    infer_args.add_argument('--output_requests', type=int, default=1,
                            help='Optional. Number of infer requests for MTCNN Output stage. '
                                 'Works only with --output_batch_size > 0.')
    infer_args.add_argument('--output_nstreams', type=str, default='',
                            help='Optional. Number of streams to use for inference on the CPU or/and GPU in throughput '
                                 'mode (for HETERO and MULTI device cases use format <device1>:<nstreams1>, '
                                 '<device2>:<nstreams2> or just <nstreams>) for MTCNN Output stage.'
                                 'Works only with --output_batch_size > 0.')
    infer_args.add_argument('--output_nthreads', type=int, default=1,
                            help='Optional. Number of threads to use for inference on CPU (including HETERO cases) for'
                                 'MTCNN Output stage. Works only with --output_batch_size > 0.')

    io_args = parser.add_argument_group('Input/output options')
    io_args.add_argument('--loop', default=False, action='store_true',
                         help='Optional. Enable reading the input in a loop.')
    io_args.add_argument('-o', '--output', required=False,
                         help='Optional. Name of output to save.')
    io_args.add_argument('-limit', '--output_limit', required=False, default=1000, type=int,
                         help='Optional. Number of frames to store in output. '
                              'If -1 is set, all frames are stored.')
    io_args.add_argument('--no_show', help="Optional. Don't show output.", action='store_true')
    io_args.add_argument('-u', '--utilization_monitors', default='', type=str,
                         help='Optional. List of monitors to show initially.')

    debug_args = parser.add_argument_group('Debug options')
    debug_args.add_argument('-r', '--raw_output_message', help='Optional. Output inference results raw values showing.',
                            default=False, action='store_true')
    return parser


class ColorPalette:
    def __init__(self, n, rng=None):
        assert n > 0

        if rng is None:
            rng = random.Random(0xACE)

        candidates_num = 100
        hsv_colors = [(1.0, 1.0, 1.0)]
        for _ in range(1, n):
            colors_candidates = [(rng.random(), rng.uniform(0.8, 1.0), rng.uniform(0.5, 1.0))
                                 for _ in range(candidates_num)]
            min_distances = [self.min_distance(hsv_colors, c) for c in colors_candidates]
            arg_max = np.argmax(min_distances)
            hsv_colors.append(colors_candidates[arg_max])

        self.palette = [self.hsv2rgb(*hsv) for hsv in hsv_colors]

    @staticmethod
    def dist(c1, c2):
        dh = min(abs(c1[0] - c2[0]), 1 - abs(c1[0] - c2[0])) * 2
        ds = abs(c1[1] - c2[1])
        dv = abs(c1[2] - c2[2])
        return dh * dh + ds * ds + dv * dv

    @classmethod
    def min_distance(cls, colors_set, color_candidate):
        distances = [cls.dist(o, color_candidate) for o in colors_set]
        return np.min(distances)

    @staticmethod
    def hsv2rgb(h, s, v):
        return tuple(round(c * 255) for c in colorsys.hsv_to_rgb(h, s, v))

    def __getitem__(self, n):
        return self.palette[n % len(self.palette)]

    def __len__(self):
        return len(self.palette)


def put_highlighted_text(frame, message, position, font_face, font_scale, color, thickness):
    cv2.putText(frame, message, position, font_face, font_scale, (255, 255, 255), thickness + 1)  # white border
    cv2.putText(frame, message, position, font_face, font_scale, color, thickness)


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


def draw_detections(frame, detections, palette, labels, threshold, draw_landmark=True, draw_confidence=True):
    size = frame.shape[:2]
    for detection in detections:
        if detection.score > threshold:
            xmin = max(int(detection.xmin), 0)
            ymin = max(int(detection.ymin), 0)
            xmax = min(int(detection.xmax), size[1])
            ymax = min(int(detection.ymax), size[0])
            class_id = int(detection.id)
            color = palette[class_id]
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            if draw_confidence:
                det_label = labels[class_id] if labels and len(labels) >= class_id else '#{}'.format(class_id)
                cv2.putText(frame, '{} {:.1%}'.format(det_label, detection.score),
                            (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
            if isinstance(detection, models.DetectionWithLandmarks) and draw_landmark:
                for landmark in detection.landmarks:
                    landmark = (int(landmark[0]), int(landmark[1]))
                    cv2.circle(frame, landmark, 2, (0, 255, 255), 2)
    return frame


def print_raw_results(size, detections, labels, threshold):
    log.info(' Class ID | Confidence | XMIN | YMIN | XMAX | YMAX ')
    for detection in detections:
        if detection.score > threshold:
            xmin = max(int(detection.xmin), 0)
            ymin = max(int(detection.ymin), 0)
            xmax = min(int(detection.xmax), size[1])
            ymax = min(int(detection.ymax), size[0])
            class_id = int(detection.id)
            det_label = labels[class_id] if labels and len(labels) >= class_id else '#{}'.format(class_id)
            log.info('{:^9} | {:10f} | {:4} | {:4} | {:4} | {:4} '
                     .format(det_label, detection.score, xmin, ymin, xmax, ymax))


def main():
    args = build_argparser().parse_args()

    log.info('Initializing Inference Engine...')
    ie = IECore()

    refine_config_plugin = get_plugin_configs(args.device_refine, args.refine_nstreams, args.refine_nthreads)
    output_config_plugin = get_plugin_configs(args.device_output, args.output_nstreams, args.output_nthreads)

    log.info('Loading network...')

    model_proposal = models.ProposalModel(ie, args.model_proposal)
    model_refine = models.RefineModel(ie, args.model_refine)
    model_output = models.OutputModel(ie, args.model_output)

    detector_pipeline = MtcnnPipeline(ie, model_proposal, model_refine, model_output,
                                      pm_sync=not args.proposal_async,
                                      pm_device=args.device_proposal,
                                      rm_batch_size=args.refine_batch_size,
                                      rm_config=refine_config_plugin,
                                      rm_num_requests=args.refine_requests,
                                      rm_device=args.device_refine,
                                      om_batch_size=args.output_batch_size,
                                      om_config=output_config_plugin,
                                      om_num_requests=args.output_requests,
                                      om_device=args.device_output)

    cap = open_images_capture(args.input, args.loop)

    log.info('Starting inference...')
    print("Use 's' key to disable/enable scores drawing, 'l' to disable/enable landmarks drawing")
    print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")

    palette = ColorPalette(1)
    metrics = PerformanceMetrics()
    video_writer = cv2.VideoWriter()

    draw_lanmdmark = True
    draw_confidence = True
    total_frames = 0
    while True:
        start_time = perf_counter()
        frame = cap.read()
        if frame is None:
            break
        total_frames += 1
        if total_frames == 1:
            presenter = monitors.Presenter(args.utilization_monitors, 55,
                                           (round(frame.shape[1] / 4), round(frame.shape[0] / 8)))
            if args.output:
                video_writer.open(args.output, cv2.VideoWriter_fourcc(*'MJPG'), cap.fps(),
                                  (frame.shape[1], frame.shape[0]))
                if not video_writer.isOpened():
                    raise RuntimeError("Can't open video writer")
        detections = detector_pipeline.infer(frame)

        presenter.drawGraphs(frame)
        draw_detections(frame, detections, palette, None, 0.5, draw_lanmdmark, draw_confidence)
        metrics.update(start_time, frame)

        if video_writer.isOpened() and (args.output_limit <= 0 or total_frames <= args.output_limit - 1):
            video_writer.write(frame)

        if not args.no_show:
            cv2.imshow('Detection Results', frame)
            key = cv2.waitKey(1)
            ESC_KEY = 27
            # Quit.
            if key in {ord('q'), ord('Q'), ESC_KEY}:
                break
            elif key in {ord('l'), ord('L')}:
                draw_lanmdmark = not draw_lanmdmark
            elif  key in {ord('s'), ord('S')}:
                draw_confidence = not draw_confidence
            presenter.handleKey(key)

    metrics.print_total()


if __name__ == '__main__':
    sys.exit(main() or 0)
