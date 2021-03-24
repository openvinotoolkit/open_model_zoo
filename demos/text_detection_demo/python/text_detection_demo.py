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

import logging
import sys
from argparse import ArgumentParser, SUPPRESS
from pathlib import Path
from time import perf_counter

import cv2
from openvino.inference_engine import IECore

sys.path.append(str(Path(__file__).resolve().parents[2] / 'common/python'))


import models
import monitors
from images_capture import open_images_capture
from performance_metrics import PerformanceMetrics

from pipelines import TwoStagePipeline

logging.basicConfig(format='[ %(levelname)s ] %(message)s', level=logging.INFO, stream=sys.stdout)
log = logging.getLogger()


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument('-m_td', required=True, type=Path,
                      help='Required. Path to an .xml file with a Text Detection model.')
    args.add_argument('-m_tr', required=True, type=Path,
                      help='Required. Path to an .xml file with a Text Recognition model.')
    args.add_argument('-i', '--input', required=True, type=str,
                      help='Required. An input to process. The input must be a single image, '
                           'a folder of images, video file or camera id.')
    args.add_argument('-d_td', '--device_td', default='CPU', type=str,
                      help='Optional. Specify the target device to infer on Detection stage; CPU, GPU, FPGA, HDDL or MYRIAD is '
                           'acceptable. The sample will look for a suitable plugin for device specified. '
                           'Default value is CPU.')
    args.add_argument('-d_tr', '--device_tr', default='CPU', type=str,
                      help='Optional. Specify the target device to infer on Recognition stage; CPU, GPU, FPGA, HDDL or MYRIAD is '
                           'acceptable. The sample will look for a suitable plugin for device specified. '
                           'Default value is CPU.')

    common_model_args = parser.add_argument_group('Common model options')
    common_model_args.add_argument('-t', '--prob_threshold', default=0.9, type=float,
                                   help='Optional. Probability threshold for text detections filtering.')
    common_model_args.add_argument('-a', '--alphabet',
                                   help='Optional. Alphabet used for text decoding.',
                                   default='0123456789abcdefghijklmnopqrstuvwxyz')
    common_model_args.add_argument('--tr_pt_first', default=False, action='store_true',
                                   help='Optional. Specifies if pad token is the first symbol in the alphabet.')
    common_model_args.add_argument('-b', '--bandwidth', default=0, type=int,
                                   help='Optional. Bandwidth for CTC beam search decoder. Default value is 0, '
                                        'in this case CTC greedy decoder will be used.')

    infer_args = parser.add_argument_group('Inference options')
    infer_args.add_argument('-nireq_td', '--num_requests_td', default=1, type=int,
                            help='Optional. Number of infer requests for Detection stage.')
    infer_args.add_argument('-nstreams_td', '--num_streams_td',
                            help='Optional. Number of streams to use for inference on the CPU or/and GPU in throughput '
                                 'mode (for HETERO and MULTI device cases use format '
                                 '<device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>) for Detection stage.',
                            default='', type=str)
    infer_args.add_argument('-nthreads_td', '--num_threads_td', default=None, type=int,
                            help='Optional. Number of threads to use for inference on CPU (including HETERO cases) '
                                 'for Detection stage.')
    infer_args.add_argument('-nireq_tr', '--num_requests_tr', default=1, type=int,
                            help='Optional. Number of infer requests for Recognition stage.')
    infer_args.add_argument('-nstreams_tr', '--num_streams_tr',
                            help='Optional. Number of streams to use for inference on the CPU or/and GPU in throughput '
                                 'mode (for HETERO and MULTI device cases use format '
                                 '<device1>:<nstreams1>,<device2>:<nstreams2> or just <nstreams>) for Recognition stage.',
                            default='', type=str)
    infer_args.add_argument('-nthreads_tr', '--num_threads_tr', default=None, type=int,
                            help='Optional. Number of threads to use for inference on CPU (including HETERO cases) '
                                 'for Recognition stage.')

    io_args = parser.add_argument_group('Input/output options')
    io_args.add_argument('--loop', default=False, action='store_true',
                         help='Optional. Enable reading the input in a loop.')
    io_args.add_argument('-o', '--output',
                         help='Optional. Name of output to save.')
    io_args.add_argument('-limit', '--output_limit', default=1000, type=int,
                         help='Optional. Number of frames to store in output. '
                              'If 0 is set, all frames are stored.')
    io_args.add_argument('--no_show', help="Optional. Don't show output.", action='store_true')
    io_args.add_argument('-u', '--utilization_monitors', default='', type=str,
                         help='Optional. List of monitors to show initially.')

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


def draw_detections(frame, detections, texts):
    color = (50, 205, 50)
    for detection in detections:
        xmin = max(int(detection.xmin), 0)
        ymin = max(int(detection.ymin), 0)
        xmax = min(int(detection.xmax), frame.shape[1])
        ymax = min(int(detection.ymax), frame.shape[0])
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), color, 2)

    for id, detection in enumerate(detections):
        xmin = max(int(detection.xmin), 0)
        ymin = max(int(detection.ymin), 0)
        text = texts[id]
        textsize = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 1)[0]
        cv2.rectangle(frame, (xmin, ymin), (xmin + textsize[0], ymin - textsize[1]), color, cv2.FILLED)
        cv2.putText(frame, text, (xmin, ymin), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

    return frame


def main():
    args = build_argparser().parse_args()

    log.info('Initializing Inference Engine...')
    ie = IECore()

    plugin_config_td = get_plugin_configs(args.device_td, args.num_streams_td, args.num_threads_td)
    plugin_config_tr = get_plugin_configs(args.device_tr, args.num_streams_tr, args.num_threads_tr)

    cap = open_images_capture(args.input, args.loop)

    start_time = perf_counter()
    frame = cap.read()
    if frame is None:
        raise RuntimeError("Can't read an image from the input")

    frame_size = frame.shape[:2]
    text_detection = models.CTPN(ie, args.m_td, input_size=frame_size, threshold=args.prob_threshold)
    if args.tr_pt_first:
        alphabet = '#' + args.alphabet
    else:
        alphabet = args.alphabet + '#'
    text_recognition = models.TextRecognition(ie, args.m_tr, alphabet=alphabet, bandwidth=args.bandwidth)

    pipeline = TwoStagePipeline(ie, text_detection, text_recognition,
                                plugin_config_td, plugin_config_tr,
                                args.device_td, args.device_tr,
                                args.num_requests_td, args.num_requests_tr)

    log.info('Starting inference...')
    print("To close the application, press 'CTRL+C' here or switch to the output window and press ESC key")

    pipeline.submit_data(frame, 0, {'frame': frame, 'start_time': start_time})

    next_frame_id = 1
    next_frame_id_to_show = 0

    metrics = PerformanceMetrics()
    video_writer = cv2.VideoWriter()
    presenter = monitors.Presenter(args.utilization_monitors, 55,
                                   (round(frame_size[1] / 4), round(frame_size[0] / 8)))
    if args.output and not video_writer.open(args.output, cv2.VideoWriter_fourcc(*'MJPG'),
                                             cap.fps(), (frame_size[1], frame_size[0])):
        raise RuntimeError("Can't open video writer")

    while True:
        results = pipeline.get_result()
        if results:
            # Get detections and recognitions
            (detections, frame_meta), texts = results
            frame = frame_meta['frame']
            start_time = frame_meta['start_time']

            presenter.drawGraphs(frame)
            frame = draw_detections(frame, detections, texts)
            metrics.update(start_time, frame)

            next_frame_id_to_show += 1
            if video_writer.isOpened() and (args.output_limit <= 0 or next_frame_id_to_show <= args.output_limit):
                video_writer.write(frame)

            if not args.no_show:
                cv2.imshow('Text Detection Results', frame)
                key = cv2.waitKey(1)
                if key in {ord('q'), ord('Q'), 27}:
                    break
                presenter.handleKey(key)
            continue

        if pipeline.is_ready() and next_frame_id_to_show - next_frame_id <= 1:
            # Get new frame
            start_time = perf_counter()
            frame = cap.read()
            if frame is None:
                break
            # Submit to text detection model
            pipeline.submit_data(frame, next_frame_id, {'frame': frame, 'start_time': start_time})
            next_frame_id += 1
        else:
            # Wait for empty request
            pipeline.await_any()

    pipeline.await_all()

    # Process completed requests
    while pipeline.has_completed_request():
        results = pipeline.get_result()
        if results:
            (detections, frame_meta), texts = results
            frame = frame_meta['frame']
            start_time = frame_meta['start_time']

            presenter.drawGraphs(frame)
            frame = draw_detections(frame, detections, texts)
            metrics.update(start_time, frame)

            next_frame_id_to_show += 1
            if video_writer.isOpened() and (args.output_limit <= 0 or next_frame_id_to_show <= args.output_limit):
                video_writer.write(frame)

            if not args.no_show:
                cv2.imshow('Text Detection Results', frame)
                key = cv2.waitKey(1)
                if key in {ord('q'), ord('Q'), 27}:
                    break
                presenter.handleKey(key)

    metrics.print_total()
    print(presenter.reportMeans())


if __name__ == '__main__':
    sys.exit(main() or 0)
