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

import logging as log
import os
import sys
from argparse import SUPPRESS, ArgumentParser

import cv2 as cv
from tqdm import tqdm
from utils import (COLOR_BLACK, COLOR_RED, COLOR_WHITE, DEFAULT_RESIZE_STEP,
                   DEFAULT_WIDTH, MAX_HEIGHT, MAX_WIDTH, MIN_HEIGHT, MIN_WIDTH,
                   PREPROCESSING, Model, calculate_probability,
                   create_renderer, prerocess_crop, strip_internal_spaces)

log.basicConfig(format='[ %(levelname)s ] %(message)s', level=log.DEBUG, stream=sys.stdout)


class InteractiveDemo:
    def __init__(self, input_model_shape, resolution):
        self.resolution = resolution
        self._tgt_shape = input_model_shape
        self.start_point, self.end_point = self._create_input_window()
        self._prev_rendered_formula = None
        self._prev_formula_img = None
        self._latex_h = 0
        self._renderer = create_renderer()

    def _create_input_window(self):
        aspect_ratio = self._tgt_shape[0] / self._tgt_shape[1]
        width = min(DEFAULT_WIDTH, self.resolution[0])
        height = int(width * aspect_ratio)
        start_point = (int(self.resolution[0] / 2 - width / 2), int(self.resolution[1] / 2 - height / 2))
        end_point = (int(self.resolution[0] / 2 + width / 2), int(self.resolution[1] / 2 + height / 2))
        return start_point, end_point

    def get_crop(self, frame):
        return frame[self.start_point[1]:self.end_point[1], self.start_point[0]:self.end_point[0], :]

    def _draw_rectangle(self, frame):
        return cv.rectangle(frame, self.start_point, self.end_point, color=COLOR_RED, thickness=2)

    def resize_window(self, action):
        height = self.end_point[1] - self.start_point[1]
        width = self.end_point[0] - self.start_point[0]
        aspect_ratio = height / width
        step = max(int(DEFAULT_RESIZE_STEP * aspect_ratio), 1)
        if action == 'increase':
            max_h, max_w = min(MAX_HEIGHT, self.resolution[1]), min(MAX_WIDTH, self.resolution[0])
            if height >= max_h or width >= max_w:
                return
            self.start_point = (self.start_point[0] - DEFAULT_RESIZE_STEP,
                                self.start_point[1] - step)
            self.end_point = (self.end_point[0] + DEFAULT_RESIZE_STEP,
                              self.end_point[1] + step)
        elif action == 'decrease':
            if height <= MIN_HEIGHT or width <= MIN_WIDTH:
                return
            self.start_point = (self.start_point[0] + DEFAULT_RESIZE_STEP,
                                self.start_point[1] + step)
            self.end_point = (self.end_point[0] - DEFAULT_RESIZE_STEP,
                              self.end_point[1] - step)
        else:
            raise ValueError("wrong action: {}".format(action))

    def _put_text(self, frame, text):
        if text == '':
            return frame
        text = strip_internal_spaces(text)
        (txt_w, self._latex_h), _ = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 1, 3)
        start_point = (self.start_point[0],
                       self.end_point[1] - self.start_point[1] + int(self._latex_h * 1.5))
        frame = cv.putText(frame, text, org=start_point, fontFace=cv.FONT_HERSHEY_SIMPLEX,
                           fontScale=0.7, color=COLOR_BLACK, thickness=3, lineType=cv.LINE_AA)
        frame = cv.putText(frame, text, org=start_point, fontFace=cv.FONT_HERSHEY_SIMPLEX,
                           fontScale=0.7, color=COLOR_WHITE, thickness=2, lineType=cv.LINE_AA)
        comment_coords = (0, self.end_point[1] - self.start_point[1] + int(self._latex_h * 1.5))
        frame = cv.putText(frame, "Predicted:", comment_coords,
                           fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=COLOR_WHITE, thickness=2, lineType=cv.LINE_AA)
        return frame

    def put_crop(self, frame, crop):
        height = self.end_point[1] - self.start_point[1]
        width = self.end_point[0] - self.start_point[0]
        crop = cv.resize(crop, (width, height))
        frame[0:height, self.start_point[0]:self.end_point[0], :] = crop
        comment_coords = (0, 20)
        frame = cv.putText(frame, "Model input:", comment_coords, fontFace=cv.FONT_HERSHEY_SIMPLEX,
                           fontScale=0.7, color=COLOR_WHITE, thickness=2, lineType=cv.LINE_AA)
        return frame

    def _put_formula_img(self, frame, formula):
        if self._renderer is None or formula == '':
            return frame
        formula_img = self._render_formula_async(formula)
        if formula_img is None:
            return frame
        y_start = self.end_point[1] - self.start_point[1] + self._latex_h * 2
        formula_img = self._resize_if_need(formula_img)
        frame[y_start:y_start + formula_img.shape[0],
              self.start_point[0]:self.start_point[0] + formula_img.shape[1],
              :] = formula_img
        comment_coords = (0, y_start + (formula_img.shape[0] + self._latex_h) // 2)
        frame = cv.putText(frame, "Rendered:", comment_coords,
                           fontFace=cv.FONT_HERSHEY_SIMPLEX, fontScale=0.7, color=COLOR_WHITE, thickness=2, lineType=cv.LINE_AA)
        return frame

    def _resize_if_need(self, formula_img):
        if (self.end_point[0] - self.start_point[0]) < formula_img.shape[1]:
            scale_factor = (self.end_point[0] - self.start_point[0]) / formula_img.shape[1]
            formula_img = cv.resize(formula_img, fx=scale_factor, fy=scale_factor, dsize=None)
        return formula_img

    def _render_formula_async(self, formula):
        if formula == self._prev_rendered_formula:
            return self._prev_formula_img
        result = self._renderer.thread_render(formula)
        if result is None:
            return None
        formula_img, res_formula = result
        if res_formula != formula:
            return None
        self._prev_rendered_formula = formula
        self._prev_formula_img = formula_img
        return formula_img

    def draw(self, frame, phrase):
        frame = self._put_text(frame, phrase)
        frame = self._put_formula_img(frame, phrase)
        frame = self._draw_rectangle(frame)
        return frame


def create_capture(input_source, demo_resolution):
    try:
        input_source = int(input_source)
    except ValueError:
        pass
    capture = cv.VideoCapture(input_source)
    capture.set(cv.CAP_PROP_BUFFERSIZE, 1)
    capture.set(cv.CAP_PROP_FRAME_WIDTH, demo_resolution[0])
    capture.set(cv.CAP_PROP_FRAME_HEIGHT, demo_resolution[1])
    return capture


def non_interactive_demo(model, args):
    renderer = create_renderer()
    show_window = not args.no_show
    for rec in tqdm(model.images_list):
        image = rec.img
        distribution, targets = model.infer_sync(image)
        prob = calculate_probability(distribution)
        log.info("Confidence score is {}".format(prob))
        if prob >= args.conf_thresh ** len(distribution):
            phrase = model.vocab.construct_phrase(targets)
            if args.output_file:
                with open(args.output_file, 'a') as output_file:
                    output_file.write(rec.img_name + '\t' + phrase + '\n')
            else:
                print("\n\tImage name: {}\n\tFormula: {}\n".format(rec.img_name, phrase))
                if renderer is not None:
                    rendered_formula, _ = renderer.render(phrase)
                    if rendered_formula is not None and show_window:
                        cv.imshow("Predicted formula", rendered_formula)
                        cv.waitKey(0)
        else:
            log.info("Confidence score is low. The formula was not recognized.")


def build_argparser():
    parser = ArgumentParser(add_help=False)
    args = parser.add_argument_group('Options')
    args.add_argument('-h', '--help', action='help',
                      default=SUPPRESS, help='Show this help message and exit.')
    args.add_argument("-m_encoder", help="Required. Path to an .xml file with a trained encoder part of the model",
                      required=True, type=str)
    args.add_argument("-m_decoder", help="Required. Path to an .xml file with a trained decoder part of the model",
                      required=True, type=str)
    args.add_argument("-i", "--input", help="Required. Path to a folder with images, path to an image files, integer "
                      "identifier of the camera or path to the video. See README.md for details.",
                      required=True, type=str)
    args.add_argument("-no_show", "--no_show", action='store_true',
                      help='Optional. Suppress pop-up window with rendered formula.')
    args.add_argument("-o", "--output_file",
                      help="Optional. Path to file where to store output. If not mentioned, result will be stored "
                      "in the console.",
                      type=str)
    args.add_argument("-v", "--vocab_path", help="Required. Path to vocab file to construct meaningful phrase",
                      type=str, required=True)
    args.add_argument("--max_formula_len",
                      help="Optional. Defines maximum length of the formula (number of tokens to decode)",
                      default="128", type=int)
    args.add_argument("-t", "--conf_thresh",
                      help="Optional. Probability threshold to treat model prediction as meaningful",
                      default=0.95, type=float)
    args.add_argument("-d", "--device",
                      help="Optional. Specify a device to infer on (the list of available devices is shown below). Use "
                           "'-d HETERO:<comma-separated_devices_list>' format to specify HETERO plugin. Use "
                           "'-d MULTI:<comma-separated_devices_list>' format to specify MULTI plugin. Default is CPU",
                      default="CPU", type=str)
    args.add_argument("--resolution", default=(1280, 720), type=int, nargs=2,
                      help='Optional. Resolution of the demo application window. Default: 1280 720')
    args.add_argument('--preprocessing_type', choices=PREPROCESSING.keys(),
                      help="Optional. Type of the preprocessing", default='crop')
    args.add_argument('--imgs_layer', help='Optional. Encoder input name for images. See README for details.',
                      default='imgs')
    args.add_argument('--row_enc_out_layer',
                      help='Optional. Encoder output key for row_enc_out. See README for details.',
                      default='row_enc_out')
    args.add_argument('--hidden_layer', help='Optional. Encoder output key for hidden. See README for details.',
                      default='hidden')
    args.add_argument('--context_layer', help='Optional. Encoder output key for context. See README for details.',
                      default='context')
    args.add_argument('--init_0_layer', help='Optional. Encoder output key for init_0. See README for details.',
                      default='init_0')
    args.add_argument('--dec_st_c_layer', help='Optional. Decoder input key for dec_st_c. See README for details.',
                      default='dec_st_c')
    args.add_argument('--dec_st_h_layer', help='Optional. Decoder input key for dec_st_h. See README for details.',
                      default='dec_st_h')
    args.add_argument('--dec_st_c_t_layer', help='Optional. Decoder output key for dec_st_c_t. See README for details.',
                      default='dec_st_c_t')
    args.add_argument('--dec_st_h_t_layer', help='Optional. Decoder output key for dec_st_h_t. See README for details.',
                      default='dec_st_h_t')
    args.add_argument('--output_layer', help='Optional. Decoder output key for output. See README for details.',
                      default='output')
    args.add_argument('--output_prev_layer',
                      help='Optional. Decoder input key for output_prev. See README for details.',
                      default='output_prev')
    args.add_argument('--logit_layer', help='Optional. Decoder output key for logit. See README for details.',
                      default='logit')
    args.add_argument('--tgt_layer', help='Optional. Decoder input key for tgt. See README for details.',
                      default='tgt')
    return parser


def main():
    args = build_argparser().parse_args()
    interactive_mode = not (os.path.isdir(args.input) or args.input.endswith('.png') or args.input.endswith('.jpg'))
    model = Model(args, interactive_mode)
    if not interactive_mode:
        non_interactive_demo(model, args)
        return

    _, _, height, width = model.encoder.input("imgs").shape
    prev_text = ''
    demo = InteractiveDemo((height, width), resolution=args.resolution)
    show_window = not args.no_show
    capture = create_capture(args.input, demo.resolution)
    if not capture.isOpened():
        log.error("Cannot open camera")
        return 1
    while True:
        ret, frame = capture.read()
        if not ret:
            break
        bin_crop = demo.get_crop(frame)
        model_input = prerocess_crop(bin_crop, (height, width), preprocess_type=args.preprocessing_type)
        frame = demo.put_crop(frame, model_input)
        model_res = model.infer_async(model_input)
        if not model_res:
            phrase = prev_text
        else:
            distribution, targets = model_res
            prob = calculate_probability(distribution)
            log.debug("Confidence score is {}".format(prob))
            if prob >= args.conf_thresh ** len(distribution):
                log.debug("Prediction updated")
                phrase = model.vocab.construct_phrase(targets)
            else:
                log.debug("Confidence score is low, prediction is not complete")
                phrase = ''
        frame = demo.draw(frame, phrase)
        prev_text = phrase
        if show_window:
            cv.imshow('Press q to quit.', frame)
            key = cv.waitKey(1) & 0xFF
            if key in (ord('Q'), ord('q'), ord('\x1b')):
                break
            elif key in (ord('o'), ord('O')):
                demo.resize_window("decrease")
            elif key in (ord('p'), ord('P')):
                demo.resize_window("increase")


if __name__ == '__main__':
    sys.exit(main() or 0)
