#!/usr/bin/env python3
"""
 Copyright (c) 2020-2024 Intel Corporation

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

import sys
import logging as log
from argparse import ArgumentParser, SUPPRESS
from os import path

from openvino import Core, get_version

from action_recognition_demo.models import IEModel, DummyDecoder
from action_recognition_demo.result_renderer import ResultRenderer
from action_recognition_demo.steps import run_pipeline

sys.path.append(path.join(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))), 'common/python'))
import monitors
from images_capture import open_images_capture

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument('-i', '--input', required=True,
                      help='Required. An input to process. The input must be a single image, '
                           'a folder of images, video file or camera id.')
    args.add_argument('--loop', default=False, action='store_true',
                      help='Optional. Enable reading the input in a loop.')
    args.add_argument('-o', '--output', required=False,
                      help='Optional. Name of the output file(s) to save. Frames of odd width or height can be truncated. See https://github.com/opencv/opencv/pull/24086')
    args.add_argument('-limit', '--output_limit', required=False, default=1000, type=int,
                      help='Optional. Number of frames to store in output. '
                           'If 0 is set, all frames are stored.')
    args.add_argument('-at', '--architecture_type', help='Required. Specify architecture type.',
                      type=str, required=True, choices=('en-de', 'en-mean', 'i3d-rgb'))
    args.add_argument('-m_en', '--m_encoder', help='Required. Path to encoder model.', required=True, type=str)
    decoder_args = args.add_mutually_exclusive_group()
    decoder_args.add_argument('-m_de', '--m_decoder',
                              help="Optional. Path to decoder model. Only for -at en-de.",
                              default=None, type=str)
    decoder_args.add_argument('--seq', dest='decoder_seq_size',
                              help='Optional. Length of sequence that decoder takes as input.',
                              default=16, type=int)
    args.add_argument('-d', '--device',
                      help="Optional. Specify a device to infer on (the list of available devices is shown below). Use "
                           "'-d HETERO:<comma-separated_devices_list>' format to specify HETERO plugin. Use "
                           "'-d MULTI:<comma-separated_devices_list>' format to specify MULTI plugin. Default is CPU",
                      default='CPU', type=str)
    args.add_argument('-lb', '--labels', help='Optional. Path to file with label names.', type=str)
    args.add_argument('--no_show', action='store_true', help="Optional. Don't show output.")
    args.add_argument('-s', '--smooth', dest='label_smoothing', help='Optional. Number of frames used for output label smoothing.',
                      default=30, type=int)
    args.add_argument('-u', '--utilization-monitors', default='', type=str,
                      help='Optional. List of monitors to show initially.')
    args.add_argument('-r', '--raw_output_message', help='Optional. Output inference results raw values showing.',
                      default=False, action='store_true')

    return parser


def main():
    args = build_argparser().parse_args()

    if args.labels:
        with open(args.labels) as f:
            labels = [line.strip() for line in f]
    else:
        labels = None

    log.info('OpenVINO Runtime')
    log.info('\tbuild: {}'.format(get_version()))
    core = Core()

    decoder_target_device = 'CPU'
    if args.device != 'CPU':
        encoder_target_device = args.device
        decoder_target_device = args.device
    else:
        encoder_target_device = decoder_target_device

    models = [IEModel(args.m_encoder, core, encoder_target_device, model_type='Action Recognition Encoder',
                      num_requests=1)]

    if args.architecture_type == 'en-de':
        if args.m_decoder is None:
            raise RuntimeError('No decoder for encoder-decoder model type (-m_de) provided')
        models.append(IEModel(args.m_decoder, core, decoder_target_device, model_type='Action Recognition Decoder', num_requests=2))
        seq_size = models[1].input_shape[1]
    elif args.architecture_type == 'en-mean':
        models.append(DummyDecoder(num_requests=2))
        seq_size = args.decoder_seq_size
    elif args.architecture_type == 'i3d-rgb':
        seq_size = models[0].input_shape[1]

    presenter = monitors.Presenter(args.utilization_monitors, 70)
    result_presenter = ResultRenderer(no_show=args.no_show, presenter=presenter, output=args.output, limit=args.output_limit, labels=labels,
                                      label_smoothing_window=args.label_smoothing)
    cap = open_images_capture(args.input, args.loop)
    run_pipeline(cap, args.architecture_type, models, result_presenter.render_frame, args.raw_output_message,
                 seq_size=seq_size, fps=cap.fps())

    for rep in presenter.reportMeans():
        log.info(rep)


if __name__ == '__main__':
    sys.exit(main() or 0)
