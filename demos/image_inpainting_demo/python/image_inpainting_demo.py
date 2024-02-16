#!/usr/bin/env python3
"""
 Copyright (c) 2019-2024 Intel Corporation
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
from pathlib import Path
from time import perf_counter

import numpy as np
import cv2
from openvino import Core, get_version

from inpainting_gui import InpaintingGUI
from inpainting import ImageInpainting

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)


def build_arg_parser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help', default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m", "--model", help="Required. Path to an .xml file with a trained model.",
                      required=True, type=Path)
    args.add_argument("-i", "--input", required=True, help="Required. Path to image.")
    args.add_argument("-d", "--device",
                      help="Optional. Specify the target device to infer on; CPU or GPU is "
                           "acceptable. The demo will look for a suitable plugin for device specified. "
                           "Default value is CPU", default="CPU", type=str)
    args.add_argument("-p", "--parts", help="Optional. Number of parts to draw mask. Ignored in GUI mode",
                      default=8, type=int)
    args.add_argument("-mbw", "--max_brush_width", help="Optional. Max width of brush to draw mask."
                      " Ignored in GUI mode",
                      default=24, type=int)
    args.add_argument("-ml", "--max_length", help="Optional. Max strokes length to draw mask. Ignored in GUI mode",
                      default=100, type=int)
    args.add_argument("-mv", "--max_vertex", help="Optional. Max number of vertex to draw mask. Ignored in GUI mode",
                      default=20, type=int)
    args.add_argument("--no_show", help="Optional. Don't show output. Cannot be used in GUI mode", action='store_true')
    args.add_argument("-o", "--output", help="Optional. Save output to the file with provided filename."
                      " Ignored in GUI mode", default="", type=str)
    args.add_argument("-ac", "--auto_mask_color", help="Optional. Use automatic (non-interactive) mode with color mask."
                      "Provide color to be treated as mask (3 RGB components in range of 0...255). "
                      "Cannot be used together with -ar.",
                      metavar='C', default=None, type=int, nargs=3)
    args.add_argument("-ar", "--auto_mask_random",
                      help="Optional. Use automatic (non-interactive) mode with random mask for inpainting"
                      " (with parameters set by -p, -mbw, -mk and -mv). Cannot be used together with -ac.",
                      action='store_true')

    return parser


def create_random_mask(parts, max_vertex, max_length, max_brush_width, h, w, max_angle=360):
    mask = np.zeros((h, w, 1), dtype=np.float32)
    for _ in range(parts):
        num_strokes = np.random.randint(max_vertex)
        start_y = np.random.randint(h)
        start_x = np.random.randint(w)
        for i in range(num_strokes):
            angle = np.random.random() * np.deg2rad(max_angle)
            if i % 2 == 0:
                angle = 2 * np.pi - angle
            length = np.random.randint(max_length + 1)
            brush_width = np.random.randint(10, max_brush_width + 1) // 2 * 2
            next_y = start_y + length * np.cos(angle)
            next_x = start_x + length * np.sin(angle)

            next_y = np.clip(next_y, 0, h - 1).astype(np.int32)
            next_x = np.clip(next_x, 0, w - 1).astype(np.int32)
            cv2.line(mask, (start_y, start_x), (next_y, next_x), 1, brush_width)
            cv2.circle(mask, (start_y, start_x), brush_width // 2, 1)

            start_y, start_x = next_y, next_x
    return mask


def inpaint_auto(img, inpainting_processor, args):
    start_time = perf_counter()
    #--- Generating mask
    if args.auto_mask_random:
        mask = create_random_mask(args.parts, args.max_vertex, args.max_length, args.max_brush_width,
                                inpainting_processor.input_height, inpainting_processor.input_width)
    else:
        # argument comes in RGB mode, but we will use BGR notation below
        top = np.full(img.shape, args.auto_mask_color[::-1], np.uint8)
        mask = cv2.inRange(img, top, top)
        mask = cv2.resize(mask, (inpainting_processor.input_width, inpainting_processor.input_height))
        _, mask = cv2.threshold(mask, 1, 1, cv2.THRESH_BINARY)
        mask = np.expand_dims(mask, 2)

    #--- Resizing image and removing masked areas from it
    img = cv2.resize(img, (inpainting_processor.input_width, inpainting_processor.input_height))
    masked_image = (img * (1 - mask) + 255 * mask).astype(np.uint8)

    #--- Inpaint and show results
    output_image = inpainting_processor.process(masked_image, mask)
    concat_imgs = np.hstack((masked_image, output_image))
    total_latency = (perf_counter() - start_time) * 1e3
    cv2.putText(concat_imgs, 'original', (5, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 100))
    cv2.putText(concat_imgs, 'result', (concat_imgs.shape[1] - 5 - cv2.getTextSize('result', cv2.FONT_HERSHEY_COMPLEX, 0.5, 1)[0][0], 15),
                cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 100))
    cv2.putText(concat_imgs, 'Latency: {:.1f} ms'.format(total_latency), (5, 35), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 200))
    log.info("Metrics report:")
    log.info("\tLatency: {:.1f} ms".format(total_latency))
    return concat_imgs, output_image

def main():
    args = build_arg_parser().parse_args()

    # Loading source image
    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        log.error("Cannot load image " + args.input)
        return -1

    if args.auto_mask_color and args.auto_mask_random:
        log.error("-ar and -ac options cannot be used together")
        return -1

    log.info('OpenVINO Runtime')
    log.info('\tbuild: {}'.format(get_version()))
    core = Core()

    log.info('Reading model {}'.format(args.model))
    inpainting_processor = ImageInpainting(core, args.model, args.device)
    log.info('The model {} is loaded to {}'.format(args.model, args.device))

    if args.auto_mask_color or args.auto_mask_random:
        # Command-line inpaining for just one image
        concat_image, result = inpaint_auto(img, inpainting_processor, args)
        if args.output != "":
            cv2.imwrite(args.output, result)
        if not args.no_show:
            cv2.imshow('Image Inpainting Demo', concat_image)
            cv2.waitKey(0)
    else:
        # Inpainting with GUI
        if args.no_show:
            log.error("--no_show argument cannot be used in GUI mode")
            return -1
        InpaintingGUI(img, inpainting_processor).run()
    return 0

if __name__ == "__main__":
    sys.exit(main() or 0)
