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

import colorsys
import logging
import os.path as osp
import random
import sys
from argparse import ArgumentParser, SUPPRESS
from itertools import cycle, islice
from enum import Enum
from time import perf_counter

import cv2
import numpy as np
from openvino.inference_engine import IECore

sys.path.append(osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'common'))

from models import *
from pipelines import AsyncPipeline, SyncPipeline
import monitors

logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument('-m', '--model', help='Required. Path to an .xml file with a trained model.',
                      required=True, type=str)
    args.add_argument('--type', help='Required. Specify model type', type=str, required=True,
                      choices=('ssd', 'yolo', 'faceboxes', 'centernet', 'retina'))
    args.add_argument('-i', '--input', required=True, type=str,
                      help='Required. Path to an image, folder with images, video file or a numeric camera ID.')
    args.add_argument('-d', '--device',
                      help='Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is '
                           'acceptable. The sample will look for a suitable plugin for device specified. '
                           'Default value is CPU.', default='CPU', type=str)

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
                                 '<device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>)',
                            default='', type=str)
    infer_args.add_argument('-nthreads', '--num_threads', default=None, type=int,
                            help='Optional. Number of threads to use for inference on CPU (including HETERO cases)')
    infer_args.add_argument('-loop', '--loop', help='Optional. Number of times to repeat the input.',
                            type=int, default=0)
    infer_args.add_argument('-no_show', '--no_show', help="Optional. Don't show output", action='store_true')
    infer_args.add_argument('-u', '--utilization_monitors', default='', type=str,
                            help='Optional. List of monitors to show initially.')
    infer_args.add_argument('--sync', action='store_true')

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

    @staticmethod
    def min_distance(colors_set, color_candidate):
        distances = [__class__.dist(o, color_candidate) for o in colors_set]
        return np.min(distances)

    @staticmethod
    def hsv2rgb(h, s, v):
        return tuple(round(c * 255) for c in colorsys.hsv_to_rgb(h, s, v))

    def __getitem__(self, n):
        return self.palette[n % len(self.palette)]

    def __len__(self):
        return len(self.palette)


class Modes(Enum):
    USER_SPECIFIED = 0
    MIN_LATENCY = 1
    SYNC = 2


class ModeInfo:
    def __init__(self):
        self.last_start_time = perf_counter()
        self.last_end_time = None
        self.frames_count = 0
        self.latency_sum = 0


def get_model(model_name, ie, args):
    if model_name == 'ssd':
        return SSD(ie, args.model, log, labels=args.labels, keep_aspect_ratio_resize=args.keep_aspect_ratio)
    elif model_name == 'yolo':
        return YOLO(ie, args.model, log, labels=args.labels,
                    threshold=args.prob_threshold, keep_aspect_ratio=args.keep_aspect_ratio)
    elif model_name == 'faceboxes':
        return FaceBoxes(ie, args.model, log, threshold=args.prob_threshold)
    elif model_name == 'centernet':
        return CenterNet(ie, args.model, log, labels=args.labels, threshold=args.prob_threshold)
    elif model_name == 'retina':
        return RetinaFace(ie, args.model, log, threshold=args.prob_threshold)

    log.error('No such model type as "{}". See "--type" option for details.'.format(model_name))
    sys.exit(1)


def put_highlighted_text(frame, message, position, font_face, font_scale, color, thickness):
    cv2.putText(frame, message, position, font_face, font_scale, (255, 255, 255), thickness + 1)  # white border
    cv2.putText(frame, message, position, font_face, font_scale, color, thickness)


def get_plugin_configs(device, num_streams, num_threads):
    config_user_specified = {}
    config_min_latency = {}

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

        config_min_latency['CPU_THROUGHPUT_STREAMS'] = '1'

    if 'GPU' in device:
        if 'GPU' in devices_nstreams:
            config_user_specified['GPU_THROUGHPUT_STREAMS'] = devices_nstreams['GPU'] \
                if int(devices_nstreams['GPU']) > 0 \
                else 'GPU_THROUGHPUT_AUTO'

        config_min_latency['GPU_THROUGHPUT_STREAMS'] = '1'

    return config_user_specified, config_min_latency


def draw_detections(frame, detections, palette, labels, threshold, draw_landmarks=False):
    size = frame.shape[:2]
    for detection in detections:
        if detection.score > threshold:
            xmin = max(int(detection.xmin), 0)
            ymin = max(int(detection.ymin), 0)
            xmax = min(int(detection.xmax), size[1])
            ymax = min(int(detection.ymax), size[0])
            class_id = int(detection.id)
            color = palette[class_id]
            det_label = labels[class_id] if labels and len(labels) >= class_id else '#{}'.format(class_id)
            cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(frame, '{} {:.1%}'.format(det_label, detection.score),
                        (xmin, ymin - 7), cv2.FONT_HERSHEY_COMPLEX, 0.6, color, 1)
            if draw_landmarks:
                for landmark in detection.landmarks:
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

    config_user_specified, config_min_latency = get_plugin_configs(args.device, args.num_streams, args.num_threads)

    log.info('Loading network...')

    model = get_model(args.type, ie, args)
    has_landmarks = args.type == 'retina'

    if args.sync:
        mode = Modes.SYNC
        mode_info = {mode: ModeInfo()} # For backward compatibility with statistics gatherer
        detector = SyncPipeline(ie, model, device=args.device)
    else:
        completed_request_results = {}
        modes = cycle(islice(Modes, 2))
        prev_mode = mode = next(modes)

        mode_info = {mode: ModeInfo()}
        exceptions = []
        detectors = {
            Modes.USER_SPECIFIED:
                AsyncPipeline(ie, model, device=args.device, plugin_config=config_user_specified,
                              caught_exceptions=exceptions, completed_requests=completed_request_results,
                              max_num_requests=args.num_infer_requests),
            Modes.MIN_LATENCY:
                AsyncPipeline(ie, model, device=args.device, plugin_config=config_min_latency,
                              caught_exceptions=exceptions, completed_requests=completed_request_results,
                              max_num_requests=args.num_infer_requests)
        }

    log.info('Using {} mode'.format(mode.name))


    try:
        input_stream = int(args.input)
    except ValueError:
        input_stream = args.input
    try:
        cap = cv2.VideoCapture(input_stream)
        if not cap.isOpened():
            raise Exception('OpenCV: Failed to open capture: ' + str(input_stream))
    except Exception as e:
        log.error(e)
        sys.exit(1)

    next_frame_id = 0
    next_frame_id_to_show = 0
    input_repeats = 0

    log.info('Starting inference...')
    print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")
    print("To switch between min_latency/user_specified modes, press TAB key in the output window")

    palette = ColorPalette(len(model.labels) if model.labels else 100)
    presenter = monitors.Presenter(args.utilization_monitors, 55,
                                   (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 4),
                                    round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 8)))

    if args.sync:
        while cap.isOpened():
            start_time = perf_counter()
            ret, frame = cap.read()
            if not ret:
                if input_repeats < args.loop or args.loop < 0:
                    cap.open(input_stream)
                    input_repeats += 1
                else:
                    cap.release()
                continue

            detections, _ = detector.submit_data(frame)

            if len(detections) and args.raw_output_message:
                print_raw_results(frame.shape[:2], detections, model.labels, args.prob_threshold)

            origin_im_size = frame.shape[:-1]
            presenter.drawGraphs(frame)
            frame = draw_detections(frame, detections, palette, model.labels, args.prob_threshold, has_landmarks)
            mode_message = '{} mode'.format(mode.name)
            put_highlighted_text(frame, mode_message, (10, int(origin_im_size[0] - 20)),
                                 cv2.FONT_HERSHEY_COMPLEX, 0.75, (10, 10, 200), 2)

            mode_info[mode].frames_count += 1
            mode_info[mode].last_end_time = perf_counter()

            # Frames count is always zero if mode has just been switched (i.e. prev_mode != mode).
            if mode_info[mode].frames_count != 0:
                fps_message = 'FPS: {:.1f}'.format(mode_info[mode].frames_count / \
                                                   (perf_counter() - mode_info[mode].last_start_time))
                mode_info[mode].latency_sum += perf_counter() - start_time
                latency_message = 'Latency: {:.1f} ms'.format((mode_info[mode].latency_sum / \
                                                               mode_info[mode].frames_count) * 1e3)
                # Draw performance stats over frame.
                put_highlighted_text(frame, fps_message, (15, 20), cv2.FONT_HERSHEY_COMPLEX, 0.75, (200, 10, 10), 2)
                put_highlighted_text(frame, latency_message, (15, 50), cv2.FONT_HERSHEY_COMPLEX, 0.75, (200, 10, 10), 2)

            if not args.no_show:
                cv2.imshow('Detection Results', frame)
                key = cv2.waitKey(1)

                ESC_KEY = 27
                # Quit.
                if key in {ord('q'), ord('Q'), ESC_KEY}:
                    break
    else:
        while cap.isOpened():
            start_time = perf_counter()
            ret, frame = cap.read()
            if not ret:
                if input_repeats < args.loop or args.loop < 0:
                    cap.open(input_stream)
                    input_repeats += 1
                else:
                    cap.release()
                continue

            detectors[mode].submit_data(frame, next_frame_id, {'frame': frame, 'start_time': start_time})
            next_frame_id += 1

            _, results = detectors[mode].get_result(next_frame_id_to_show)
            if results:
                objects, frame_meta = results
                frame = frame_meta['frame']
                start_time = frame_meta['start_time']

                if len(objects) and args.raw_output_message:
                    print_raw_results(frame.shape[:2], objects, model.labels, args.prob_threshold)

                origin_im_size = frame.shape[:-1]
                presenter.drawGraphs(frame)
                frame = draw_detections(frame, objects, palette, model.labels, args.prob_threshold, has_landmarks)
                mode_message = '{} mode'.format(mode.name)
                put_highlighted_text(frame, mode_message, (10, int(origin_im_size[0] - 20)),
                                     cv2.FONT_HERSHEY_COMPLEX, 0.75, (10, 10, 200), 2)

                next_frame_id_to_show += 1
                if prev_mode == mode:
                    mode_info[mode].frames_count += 1
                elif len(completed_request_results) == 0:
                    mode_info[prev_mode].last_end_time = perf_counter()
                    prev_mode = mode

                # Frames count is always zero if mode has just been switched (i.e. prev_mode != mode).
                if mode_info[mode].frames_count != 0:
                    fps_message = 'FPS: {:.1f}'.format(mode_info[mode].frames_count / \
                                                       (perf_counter() - mode_info[mode].last_start_time))
                    mode_info[mode].latency_sum += perf_counter() - start_time
                    latency_message = 'Latency: {:.1f} ms'.format((mode_info[mode].latency_sum / \
                                                                   mode_info[mode].frames_count) * 1e3)
                    # Draw performance stats over frame.
                    put_highlighted_text(frame, fps_message, (15, 20), cv2.FONT_HERSHEY_COMPLEX, 0.75, (200, 10, 10), 2)
                    put_highlighted_text(frame, latency_message, (15, 50),
                                         cv2.FONT_HERSHEY_COMPLEX, 0.75, (200, 10, 10), 2)

                if not args.no_show:
                    cv2.imshow('Detection Results', frame)
                    key = cv2.waitKey(1)

                    ESC_KEY = 27
                    TAB_KEY = 9
                    # Quit.
                    if key in {ord('q'), ord('Q'), ESC_KEY}:
                        break
                    # Switch mode.
                    # Disable mode switch if the previous switch has not been finished yet.
                    if key == TAB_KEY and mode_info[mode].frames_count > 0:
                        mode = next(modes)
                        detectors[prev_mode].await_all()
                        mode_info[prev_mode].last_end_time = perf_counter()
                        mode_info[mode] = ModeInfo()
                        log.info('Using {} mode'.format(mode.name))
                    else:
                        presenter.handleKey(key)
                next_frame_id_to_show += 1

            detectors[mode].await_any()

        if exceptions:
            raise exceptions[0]

        for exec_net in detectors.values():
            exec_net.await_all()

    for mode_value, mode_stats in mode_info.items():
        log.info('')
        log.info('Mode: {}'.format(mode_value.name))

        end_time = mode_stats.last_end_time if mode_stats.last_end_time is not None \
            else perf_counter()
        log.info('FPS: {:.1f}'.format(mode_stats.frames_count / \
                                      (end_time - mode_stats.last_start_time)))
        log.info('Latency: {:.1f} ms'.format((mode_stats.latency_sum / \
                                              mode_stats.frames_count) * 1e3))
    print(presenter.reportMeans())


if __name__ == '__main__':
    sys.exit(main() or 0)
