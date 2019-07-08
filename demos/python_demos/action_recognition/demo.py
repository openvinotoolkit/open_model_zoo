#!/usr/bin/env python

from __future__ import print_function

import sys
from argparse import ArgumentParser, SUPPRESS

from openvino.inference_engine import IEPlugin

from action_recognition_demo.models import IEModel
from action_recognition_demo.result_renderer import ResultRenderer
from action_recognition_demo.steps import run_pipeline


def video_demo(encoder, decoder, videos, fps=30, labels=None):
    """Continuously run demo on provided video list"""
    result_presenter = ResultRenderer(labels=labels)
    run_pipeline(videos, encoder, decoder, result_presenter.render_frame, fps=fps)


def create_ie_plugin(device='CPU', cpu_extension=None, plugin_dir=None):
    print("Initializing plugin for {} device...".format(device))
    plugin = IEPlugin(device, plugin_dirs=plugin_dir)

    if 'MYRIAD' in device:
        myriad_config = {"VPU_HW_STAGES_OPTIMIZATION": "YES"}
        plugin.set_config(myriad_config)

    if cpu_extension and 'CPU' in device:
        plugin.add_cpu_extension(cpu_extension)

    return plugin


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("--encoder", help="Required. Path to encoder model", required=True, type=str)
    args.add_argument("--decoder", help="Required. Path to decoder model", required=True, type=str)

    args.add_argument("-v", "--video",
                        help="Optional. Path to a video or file. 'cam' for capturing video", type=str)
    args.add_argument('-vl', '--video_list',
                        help="Optional. Path to a list with video files (text file, one video per "
                             "line)",
                        type=str)
    args.add_argument("-e", "--cpu_extension",
                        help="Optional. For CPU custom layers, if any. Absolute path to a shared library with the "
                             "kernels implementation.", type=str, default=None)
    args.add_argument("-pp", "--plugin_dir", help="Optional. Path to a plugin folder", type=str, default=None)
    args.add_argument("-d", "--device",
                        help="Optional. Specify a target device to infer on. CPU, GPU, FPGA, HDDL or MYRIAD is "
                             "acceptable. The demo will look for a suitable plugin for the device specified. "
                             "Default value is CPU",
                        default="CPU", type=str)
    args.add_argument("--fps", help="Optional. FPS for renderer", default=30, type=int)
    args.add_argument("-l", "--labels", help="Optional. Path to file with label names", type=str)

    return parser


def main():
    args = build_argparser().parse_args()

    if args.video_list:
        with open(args.video_list) as f:
            videos = [line.strip() for line in f.read().split('\n')]
    if args.video:
        videos = [args.video]

    if not args.video_list and not args.video:
        raise ValueError("Either --video or --video_list option is expected")

    if args.labels:
        with open(args.labels) as f:
            labels = [l.strip() for l in f.read().strip().split('\n')]
    else:
        labels = None

    decoder_plugin = create_ie_plugin('CPU', cpu_extension=args.cpu_extension, plugin_dir=args.plugin_dir)
    if args.device != 'CPU':
        encoder_plugin = create_ie_plugin(args.device, cpu_extension=args.cpu_extension, plugin_dir=args.plugin_dir)
    else:
        encoder_plugin = decoder_plugin

    encoder_xml = args.encoder
    encoder_bin = args.encoder.replace(".xml", ".bin")
    decoder_xml = args.decoder
    decoder_bin = args.decoder.replace(".xml", ".bin")

    encoder = IEModel(encoder_xml, encoder_bin, encoder_plugin, num_requests=(3 if args.device == 'MYRIAD' else 1))
    decoder = IEModel(decoder_xml, decoder_bin, decoder_plugin, num_requests=2)
    video_demo(encoder, decoder, videos, args.fps, labels)


if __name__ == '__main__':
    sys.exit(main() or 0)
