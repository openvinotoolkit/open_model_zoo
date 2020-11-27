#!/usr/bin/env python3
"""
 Copyright (C) 2020 Intel Corporation

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
import os.path as osp
import sys
from argparse import ArgumentParser, SUPPRESS
from itertools import cycle
from enum import Enum
from time import perf_counter

import cv2
import numpy as np
from openvino.inference_engine import IECore

from human_pose_estimation_demo.model import HPEAssociativeEmbedding, HPEOpenPose
from human_pose_estimation_demo.visualization import show_poses

sys.path.append(osp.join(osp.dirname(osp.dirname(osp.abspath(__file__))), 'common'))
import monitors
from helpers import put_highlighted_text


logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument('-i', '--input', help='Required. Path to an image, video file or a numeric camera ID.',
                      required=True, type=str)
    args.add_argument('-m', '--model', help='Required. Path to an .xml file with a trained model.',
                      required=True, type=str)
    args.add_argument('-at', '--architecture_type', choices=('ae', 'openpose'), required=True, type=str,
                      help='Required. Type of the network, either "ae" for Associative Embedding '
                           'or "openpose" for OpenPose.')
    args.add_argument('--tsize', default=None, type=int,
                      help='Optional. Target input size. This demo implements image pre-processing pipeline '
                           'that is common to human pose estimation approaches. Image is resize first to some '
                           'target size and then the network is reshaped to fit the input image shape. '
                           'By default target image size is determined based on the input shape from IR. '
                           'Alternatively it can be manually set via this parameter. Note that for OpenPose-like '
                           'nets image is resized to a predefined height, which is the target size in this case. '
                           'For Associative Embedding-like nets target size is the length of a short image side.')
    args.add_argument('-t', '--prob_threshold', help='Optional. Probability threshold for poses filtering.',
                      default=0.1, type=float)
    args.add_argument('-r', '--raw_output_message', help='Optional. Output inference results raw values showing.',
                      default=False, action='store_true')
    args.add_argument('-d', '--device',
                      help='Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is '
                           'acceptable. The sample will look for a suitable plugin for device specified. '
                           'Default value is CPU.', default='CPU', type=str)
    args.add_argument('-nireq', '--num_infer_requests', help='Optional. Number of infer requests',
                      default=1, type=int)
    args.add_argument('-nstreams', '--num_streams',
                      help='Optional. Number of streams to use for inference on the CPU or/and GPU in throughput mode '
                           '(for HETERO and MULTI device cases use format <device1>:<nstreams1>,<device2>:<nstreams2> '
                           'or just <nstreams>)',
                      default='', type=str)
    args.add_argument('-nthreads', '--num_threads',
                      help='Optional. Number of threads to use for inference on CPU (including HETERO cases)',
                      default=None, type=int)
    args.add_argument('-loop', '--loop', help='Optional. Number of times to repeat the input.', type=int, default=0)
    args.add_argument('-no_show', '--no_show', help="Optional. Don't show output", action='store_true')
    args.add_argument('-u', '--utilization_monitors', default='', type=str,
                      help='Optional. List of monitors to show initially.')
    return parser


class Modes(Enum):
    USER_SPECIFIED = 0
    MIN_LATENCY = 1


class ModeInfo:
    def __init__(self):
        self.last_start_time = perf_counter()
        self.last_end_time = None
        self.frames_count = 0
        self.latency_sum = 0


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


def main():
    args = build_argparser().parse_args()

    log.info('Initializing Inference Engine...')
    ie = IECore()

    config_user_specified, config_min_latency = get_plugin_configs(args.device, args.num_streams, args.num_threads)

    log.info('Loading network...')
    completed_request_results = {}
    modes = cycle(Modes)
    prev_mode = mode = next(modes)
    log.info('Using {} mode'.format(mode.name))
    mode_info = {mode: ModeInfo()}
    exceptions = []

    if args.architecture_type == 'ae':
        HPE = HPEAssociativeEmbedding
    else:
        HPE = HPEOpenPose

    hpes = {
        Modes.USER_SPECIFIED:
            HPE(ie, args.model, target_size=args.tsize, device=args.device, plugin_config=config_user_specified,
                results=completed_request_results, max_num_requests=args.num_infer_requests,
                caught_exceptions=exceptions),
        Modes.MIN_LATENCY:
            HPE(ie, args.model, target_size=args.tsize, device=args.device.split(':')[-1].split(',')[0],
                plugin_config=config_min_latency, results=completed_request_results, max_num_requests=1,
                caught_exceptions=exceptions)
    }

    try:
        input_stream = int(args.input)
    except ValueError:
        input_stream = args.input
    cap = cv2.VideoCapture(input_stream)
    wait_key_time = 1

    next_frame_id = 0
    next_frame_id_to_show = 0
    input_repeats = 0

    log.info('Starting inference...')
    print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")
    print("To switch between min_latency/user_specified modes, press TAB key in the output window")

    presenter = monitors.Presenter(args.utilization_monitors, 55,
        (round(cap.get(cv2.CAP_PROP_FRAME_WIDTH) / 4), round(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) / 8)))

    while (cap.isOpened() \
           or completed_request_results \
           or len(hpes[mode].empty_requests) < len(hpes[mode].requests)) \
          and not exceptions:
        if next_frame_id_to_show in completed_request_results:
            frame_meta, raw_outputs = completed_request_results.pop(next_frame_id_to_show)
            poses, scores = hpes[mode].postprocess(raw_outputs, frame_meta)
            valid_poses = scores > args.prob_threshold
            poses = poses[valid_poses]
            scores = scores[valid_poses]

            frame = frame_meta['frame']
            start_time = frame_meta['start_time']

            if len(poses) and args.raw_output_message:
                log.info('Poses:')

            origin_im_size = frame.shape[:-1]
            presenter.drawGraphs(frame)
            show_poses(frame, poses, scores, pose_score_threshold=args.prob_threshold,
                point_score_threshold=args.prob_threshold)

            if args.raw_output_message:
                for pose, pose_score in zip(poses, scores):
                    pose_str = ' '.join('({:.2f}, {:.2f}, {:.2f})'.format(p[0], p[1], p[2]) for p in pose)
                    log.info('{} | {:.2f}'.format(pose_str, pose_score))

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
                put_highlighted_text(frame, latency_message, (15, 50), cv2.FONT_HERSHEY_COMPLEX, 0.75, (200, 10, 10), 2)

            if not args.no_show:
                cv2.imshow('Pose estimation results', frame)
                key = cv2.waitKey(wait_key_time)

                ESC_KEY = 27
                TAB_KEY = 9
                # Quit.
                if key in {ord('q'), ord('Q'), ESC_KEY}:
                    break
                # Switch mode.
                # Disable mode switch if the previous switch has not been finished yet.
                if key == TAB_KEY and mode_info[mode].frames_count > 0:
                    mode = next(modes)
                    hpes[prev_mode].await_all()
                    mode_info[prev_mode].last_end_time = perf_counter()
                    mode_info[mode] = ModeInfo()
                    log.info('Using {} mode'.format(mode.name))
                else:
                    presenter.handleKey(key)

        elif hpes[mode].empty_requests and cap.isOpened():
            start_time = perf_counter()
            ret, frame = cap.read()
            if not ret:
                if input_repeats < args.loop or args.loop < 0:
                    cap.open(input_stream)
                    input_repeats += 1
                else:
                    cap.release()
                continue

            hpes[mode](frame, next_frame_id, {'frame': frame, 'start_time': start_time})
            next_frame_id += 1

        else:
            hpes[mode].await_any()

    if exceptions:
        raise exceptions[0]

    for exec_net in hpes.values():
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
