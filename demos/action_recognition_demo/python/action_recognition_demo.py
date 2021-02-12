#!/usr/bin/env python3
"""
 Copyright (c) 2020 Intel Corporation

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
from argparse import ArgumentParser, SUPPRESS
from os import path

from openvino.inference_engine import IECore

from action_recognition_demo.models import IEModel, DummyDecoder
from action_recognition_demo.result_renderer import ResultRenderer
from action_recognition_demo.steps import run_pipeline

sys.path.append(path.join(path.dirname(path.dirname(path.dirname(path.abspath(__file__)))), 'common/python'))
import monitors
from images_capture import open_images_capture


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
                      help='Optional. Name of output to save.')
    args.add_argument('-limit', '--output_limit', required=False, default=1000, type=int,
                      help='Optional. Number of frames to store in output. '
                           'If 0 is set, all frames are stored.')
    args.add_argument('-mt', '--model_type', help='Required. Specify model type.',
                      type=str, required=True, choices=('single', 'composite'))
    args.add_argument('-m_en', '--m_encoder', help='Required. Path to encoder model.', required=True, type=str)
    decoder_args = args.add_mutually_exclusive_group()
    decoder_args.add_argument('-m_de', '--m_decoder',
                              help="Optional. Path to decoder model. If not specified, "
                                   "for composite models simple averaging of encoder's outputs over a time window is applied.",
                              default=None, type=str)
    decoder_args.add_argument('--seq', dest='decoder_seq_size',
                              help='Optional. Length of sequence that decoder takes as input.',
                              default=16, type=int)
    args.add_argument('-l', '--cpu_extension',
                      help='Optional. For CPU custom layers, if any. Absolute path to a shared library with the '
                           'kernels implementation.', type=str, default=None)
    args.add_argument('-d', '--device',
                      help='Optional. Specify a target device to infer on. CPU, GPU, FPGA, HDDL or MYRIAD is '
                           'acceptable. The demo will look for a suitable plugin for the device specified. '
                           'Default value is CPU.',
                      default='CPU', type=str)
    args.add_argument('-lb', '--labels', help='Optional. Path to file with label names.', type=str)
    args.add_argument('--no_show', action='store_true', help="Optional. Don't show output.")
    args.add_argument('-s', '--smooth', dest='label_smoothing', help='Optional. Number of frames used for output label smoothing.',
                      default=30, type=int)
    args.add_argument('-u', '--utilization-monitors', default='', type=str,
                      help='Optional. List of monitors to show initially.')

    return parser


def main():
    args = build_argparser().parse_args()

    if args.labels:
        with open(args.labels) as f:
            labels = [line.strip() for line in f]
    else:
        labels = None

    ie = IECore()

    if 'MYRIAD' in args.device:
        myriad_config = {'VPU_HW_STAGES_OPTIMIZATION': 'YES'}
        ie.set_config(myriad_config, 'MYRIAD')

    if args.cpu_extension and 'CPU' in args.device:
        ie.add_extension(args.cpu_extension, 'CPU')

    decoder_target_device = 'CPU'
    if args.device != 'CPU':
        encoder_target_device = args.device
    else:
        encoder_target_device = decoder_target_device

    if args.model_type == 'single':
        model_xml = args.m_encoder
        model_bin = args.m_encoder.replace('.xml', '.bin')
        model = IEModel(model_xml, model_bin, ie, encoder_target_device,
                        num_requests=(3 if args.device == 'MYRIAD' else 1))
        seq_size = model.input_size[2]
    elif args.model_type == 'composite': 
        encoder_xml = args.m_encoder
        encoder_bin = args.m_encoder.replace('.xml', '.bin')
        model = []
        model.append(IEModel(encoder_xml, encoder_bin, ie, encoder_target_device,
                        num_requests=(3 if args.device == 'MYRIAD' else 1)))

        if args.m_decoder is not None:
            decoder_xml = args.m_decoder
            decoder_bin = args.m_decoder.replace('.xml', '.bin')
            model.append(IEModel(decoder_xml, decoder_bin, ie, decoder_target_device, num_requests=2))
            seq_size = model[1].input_size[1]
        else:
            model.append(DummyDecoder(num_requests=2))
            seq_size = args.decoder_seq_size
    else:
        raise RuntimeError('No model type or invalid model type (-at) provided: {}'.format(args.architecture_type))

    presenter = monitors.Presenter(args.utilization_monitors, 70)
    result_presenter = ResultRenderer(no_show=args.no_show, model_type=args.model_type, presenter=presenter, output=args.output, limit=args.output_limit, labels=labels,
                                      label_smoothing_window=args.label_smoothing)
    cap = open_images_capture(args.input, args.loop)
    run_pipeline(cap, args.model_type, model, result_presenter.render_frame, seq_size=seq_size, fps=cap.fps())
    print(presenter.reportMeans())


if __name__ == '__main__':
    sys.exit(main() or 0)
