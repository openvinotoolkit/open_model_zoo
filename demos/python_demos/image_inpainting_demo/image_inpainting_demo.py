#!/usr/bin/env python3
"""
 Copyright (c) 2019-2020 Intel Corporation
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

from argparse import ArgumentParser, SUPPRESS

import numpy as np
import cv2
from openvino.inference_engine import IECore

from inpainting_gui import InpaintingGUI
from inpainting import ImageInpainting


def build_arg_parser():
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
    args.add_argument("-r", "--rnd",
                      help="Optional. Use random mask for inpainting (with parameters set by -p, -mbw, -mk and -mv)."
                      " Ignored in GUI mode",
                      action='store_true')
    args.add_argument("-p", "--parts", help="Optional. Number of parts to draw mask. Ignored in GUI mode",
                      default=8, type=int)
    args.add_argument("-mbw", "--max_brush_width", help="Optional. Max width of brush to draw mask."
                      " Ignored in GUI mode",
                      default=24, type=int)
    args.add_argument("-ml", "--max_length", help="Optional. Max strokes length to draw mask. Ignored in GUI mode",
                      default=100, type=int)
    args.add_argument("-mv", "--max_vertex", help="Optional. Max number of vertex to draw mask. Ignored in GUI mode",
                      default=20, type=int)
    args.add_argument("-mc", "--mask_color",
                      help="Optional. Color to be treated as mask (provide 3 RGB components in range of 0...255)."
                      " Default is 0 0 0. Ignored in GUI mode", default=[0, 0, 0], type=int, nargs=3)
    args.add_argument("--no_show", help="Optional. Don't show output. Cannot be used in GUI mode", action='store_true')
    args.add_argument("-o", "--output", help="Optional. Save output to the file with provided filename."
                      " Ignored in GUI mode", default="", type=str)
    args.add_argument("-a", "--auto", help="Optional. Use automatic (non-interactive) mode instead of GUI",
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

            next_y = np.clip(next_y, 0, h - 1).astype(np.int)
            next_x = np.clip(next_x, 0, w - 1).astype(np.int)
            cv2.line(mask, (start_y, start_x), (next_y, next_x), 1, brush_width)
            cv2.circle(mask, (start_y, start_x), brush_width // 2, 1)

            start_y, start_x = next_y, next_x
    return mask


def inpaint_random_holes(img, args):
    mask_color = args.mask_color[::-1] # argument comes in RGB mode, but we will use BGR notation below

    ie = IECore()

    inpainting_processor = ImageInpainting(ie, args.model, args.device)

    #--- Resize to model input and generate random mask
    img = cv2.resize(img, (inpainting_processor.input_width, inpainting_processor.input_height))

    if args.rnd:
        mask = create_random_mask(args.parts, args.max_vertex, args.max_length, args.max_brush_width,
                                inpainting_processor.input_height, inpainting_processor.input_width)
    else:
        top = np.full(img.shape, mask_color, np.uint8)
        mask = cv2.inRange(img, top, top) / 255
        mask = np.expand_dims(mask, 2)

    masked_image = (img * (1 - mask) + 255 * mask).astype(np.uint8)

    #--- Inpaint and show results
    output_image = inpainting_processor.process(masked_image, mask)
    concat_imgs = np.hstack((masked_image, output_image))
    cv2.putText(concat_imgs, 'summary: {:.1f} FPS'.format(float(1 / inpainting_processor.infer_time)), (5, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 200))
    return concat_imgs, output_image

def main():
    args = build_arg_parser().parse_args()

    # Loading source image
    img = cv2.imread(args.input, cv2.IMREAD_COLOR)
    if img is None:
        print("Cannot load image " + args.input)
        return -1

    if args.auto:
        # Command-line inpaining for just one image
        concat_image, result = inpaint_random_holes(img,args)
        if args.output != "":
            cv2.imwrite(args.output, result)
        if not args.no_show:
            cv2.imshow('Image Inpainting Demo', concat_image)
            cv2.waitKey(0)
    else:
        # Inpainting with GUI
        if args.no_show:
            print("Error: --no_show argument cannot be used in GUI mode")
            return -1
        InpaintingGUI(img, args.model, args.device).run()
    return 0

if __name__ == "__main__":
    exit(main())
