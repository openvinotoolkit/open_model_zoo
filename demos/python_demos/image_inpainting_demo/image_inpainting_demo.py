#!/usr/bin/env python3

"""
 Copyright (c) 2019 Intel Corporation
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
import cv2
import numpy as np
from argparse import ArgumentParser, SUPPRESS

from openvino.inference_engine import IECore

from inpainting import ImageInpainting

def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      required=True, type=str)
    args.add_argument("-i", "--input", type=str, default='', help="path to image.")
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is "
                           "acceptable. The demo will look for a suitable plugin for device specified. "
                           "Default value is CPU", default="CPU", type=str)
    args.add_argument("-p", "--parts", help="Optional. Number of parts to draw mask.", default=8, type=int)
    args.add_argument("-mbw", "--max_brush_width", help="Optional. Max width of brush to draw mask.", default=24, type=int)
    args.add_argument("-ml", "--max_length", help="Optional. Max strokes length to draw mask.", default=100, type=int)
    args.add_argument("-mv", "--max_vertex", help="Optional. Max number of vertex to draw mask.", default=20, type=int)
    args.add_argument("--no_show", help="Optional. Don't show output", action='store_true')

    return parser

def main():
    args = build_argparser().parse_args()

    ie = IECore()
    inpainting_processor = ImageInpainting(ie, args.model, args.parts,
                                           args.max_brush_width, args.max_length, args.max_vertex, args.device)

    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    masked_image, output_image = inpainting_processor.process(img)
    concat_imgs = np.hstack((masked_image, output_image))
    cv2.putText(concat_imgs, 'summary: {:.1f} FPS'.format(
            float(1 / inpainting_processor.infer_time)), (5, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 200))
    if not args.no_show:
        cv2.imshow('Image Inpainting Demo', concat_imgs)
        cv2.waitKey(0)


if __name__ == "__main__":
    sys.exit(main() or 0)
