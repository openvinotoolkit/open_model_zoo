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

import copy
import json
import logging as log
import os
import re
import subprocess  # nosec B404  # disable import-subprocess check
import tempfile
from enum import Enum
from multiprocessing.pool import ThreadPool
from types import SimpleNamespace as namespace

import cv2 as cv
import numpy as np
import sympy
from openvino import Core, get_version
from tqdm import tqdm

START_TOKEN = 0
END_TOKEN = 2
DENSITY = 300
DEFAULT_WIDTH = 800
MIN_HEIGHT = 30
MAX_HEIGHT = 150
MAX_WIDTH = 1200
MIN_WIDTH = 260
COLOR_WHITE = (255, 255, 255)
COLOR_BLACK = (0, 0, 0)
COLOR_RED = (0, 0, 255)
# default value to resize input window's width in pixels
DEFAULT_RESIZE_STEP = 10


def strip_internal_spaces(text):
    """
    Removes spaces between digits, digit and dot,
    dot and digit; after opening brackets and parentheses
    and before closing ones; spaces around ^ symbol.
    """
    text = text.replace("{ ", "{")
    text = text.replace(" }", "}")
    text = text.replace("( ", "(")
    text = text.replace(" )", ")")
    text = text.replace(" ^ ", "^")
    return re.sub(r'(?<=[\d.]) (?=[\d.])', '', text)


def crop(img, target_shape):
    target_height, target_width = target_shape
    img_h, img_w = img.shape[0:2]
    new_w = min(target_width, img_w)
    new_h = min(target_height, img_h)
    return img[:new_h, :new_w, :]


def resize(img, target_shape):
    target_height, target_width = target_shape
    img_h, img_w = img.shape[0:2]
    scale = min(target_height / img_h, target_width / img_w)
    return cv.resize(img, None, fx=scale, fy=scale)


PREPROCESSING = {
    'crop': crop,
    'resize': resize
}


def create_renderer():
    """
    Checks if pdflatex is installed and rendering
    of latex formula could be performed
    """
    command = subprocess.run(["pdflatex", "--version"], stdout=subprocess.DEVNULL,
                             stderr=subprocess.DEVNULL, check=False, shell=True) # shell=True is used to process case with no pdflatex installed on Windows
    if command.returncode != 0:
        renderer = None
        log.warning("pdflatex not installed, please, install it to use rendering")
    else:
        renderer = Renderer()
    return renderer


def preprocess_image(preprocess, image_raw, tgt_shape):
    """
    Crop or resize with constant aspect ratio
    and bottom right pad resulting image
    """
    target_height, target_width = tgt_shape
    image_raw = preprocess(image_raw, tgt_shape)
    img_h, img_w = image_raw.shape[0:2]
    image_raw = cv.copyMakeBorder(image_raw, 0, target_height - img_h,
                                  0, target_width - img_w, cv.BORDER_CONSTANT,
                                  None, COLOR_WHITE)
    return image_raw


def prerocess_crop(crop, tgt_shape, preprocess_type='crop'):
    """
    Binarize image and call preprocess_image function
    """
    crop = cv.cvtColor(crop, cv.COLOR_BGR2GRAY)
    crop = cv.cvtColor(crop, cv.COLOR_GRAY2BGR)
    _, bin_crop = cv.threshold(crop, 120, 255, type=cv.THRESH_BINARY)
    return preprocess_image(PREPROCESSING[preprocess_type], bin_crop, tgt_shape)


def read_net(model_path, core, model_type):
    log.info('Reading {} model {}'.format(model_type, model_path))
    return core.read_model(model_path)


def change_layout(model_input):
    """
    Change layout of the image from [H, W, C] to [N, C, H, W]
    where N is equal to one (batch dimension)
    """
    model_input = model_input.transpose((2, 0, 1))
    model_input = np.expand_dims(model_input, axis=0)
    return model_input


def calculate_probability(distribution):
    return np.prod(np.amax(distribution, axis=1))


class Model:
    class Status(Enum):
        READY = 0
        ENCODER_INFER = 1
        DECODER_INFER = 2

    def __init__(self, args, interactive_mode):
        self.args = args
        log.info('OpenVINO Runtime')
        log.info('\tbuild: {}'.format(get_version()))
        self.core = Core()
        self.encoder = read_net(self.args.m_encoder, self.core, 'Formula Recognition Encoder')
        self.decoder = read_net(self.args.m_decoder, self.core, 'Formula Recognition Decoder')
        self.compiled_encoder = self.core.compile_model(self.encoder, device_name=self.args.device)
        log.info('The Formula Recognition Encoder model {} is loaded to {}'.format(args.m_encoder, args.device))
        self.compiled_decoder = self.core.compile_model(self.decoder, device_name=self.args.device)
        log.info('The Formula Recognition Decoder model {} is loaded to {}'.format(args.m_decoder, args.device))
        self.images_list = []
        self.vocab = Vocab(self.args.vocab_path)
        self.model_status = Model.Status.READY
        self.is_async = interactive_mode
        self.infer_request_encoder = self.compiled_encoder.create_infer_request()
        self.infer_request_decoder = self.compiled_decoder.create_infer_request()
        self.num_infers_decoder = 0
        self.check_model_dimensions()
        if not interactive_mode:
            self.preprocess_inputs()

    def preprocess_inputs(self):
        _, _, height, width = self.encoder.input("imgs").shape
        target_shape = (height, width)
        if os.path.isdir(self.args.input):
            inputs = sorted(os.path.join(self.args.input, inp)
                            for inp in os.listdir(self.args.input))
        else:
            inputs = [self.args.input]
        for filenm in tqdm(inputs):
            image_raw = cv.imread(filenm)
            assert image_raw is not None, "Error reading image {}".format(filenm)
            image = preprocess_image(
                PREPROCESSING[self.args.preprocessing_type], image_raw, target_shape)
            record = namespace(img_name=filenm, img=image)
            self.images_list.append(record)

    def check_model_dimensions(self):
        batch_dim, channels, height, width = self.encoder.input("imgs").shape
        assert batch_dim == 1, "Demo only works with batch size 1."
        assert channels in (1, 3), "Input image is not 1 or 3 channeled image."

    def _async_infer_encoder(self, image):
        self.infer_request_encoder.start_async(inputs={self.args.imgs_layer: image})

    def _async_infer_decoder(self, row_enc_out, dec_st_c, dec_st_h, output, tgt):
        self.num_infers_decoder += 1
        self.infer_request_decoder.start_async(inputs={self.args.row_enc_out_layer: row_enc_out,
                                                       self.args.dec_st_c_layer: dec_st_c,
                                                       self.args.dec_st_h_layer: dec_st_h,
                                                       self.args.output_prev_layer: output,
                                                       self.args.tgt_layer: tgt
                                                       }
                                               )

    def infer_async(self, model_input):
        model_input = change_layout(model_input)
        assert self.is_async
        if self.model_status == Model.Status.READY:
            self._start_encoder(model_input)
            return None

        if self.model_status == Model.Status.ENCODER_INFER:
            infer_status_encoder = self.infer_request_encoder.wait_for(0)
            if infer_status_encoder:
                self._start_decoder()
            return None

        return self._process_decoding_results()

    def infer_sync(self, model_input):
        assert not self.is_async
        model_input = change_layout(model_input)
        self._start_encoder(model_input)
        self.infer_request_encoder.wait()
        self._start_decoder()
        res = None
        while res is None:
            res = self._process_decoding_results()
        return res

    def _process_decoding_results(self):
        timeout = 0 if self.is_async else -1
        infer_status_decoder = self.infer_request_decoder.wait_for(timeout)
        if not infer_status_decoder and self.is_async:
            return None
        self._unpack_dec_results()

        if self.tgt[0][0][0] == END_TOKEN or self.num_infers_decoder >= self.args.max_formula_len:
            self.num_infers_decoder = 0
            self.logits = np.array(self.logits)
            logits = self.logits.squeeze(axis=1)
            targets = np.argmax(logits, axis=1)
            self.model_status = Model.Status.READY
            return logits, targets
        self._async_infer_decoder(self.row_enc_out,
                                  self.dec_states_c,
                                  self.dec_states_h,
                                  self.output,
                                  self.tgt.squeeze(axis=0)
                                  )

        return None

    def _start_encoder(self, model_input):
        self._async_infer_encoder(model_input)
        self.model_status = Model.Status.ENCODER_INFER

    def _start_decoder(self):
        self._unpack_enc_results()
        self._async_infer_decoder(self.row_enc_out, self.dec_states_c, self.dec_states_h, self.output, self.tgt)
        self.model_status = Model.Status.DECODER_INFER

    def _unpack_dec_results(self):
        self.dec_states_h = self.infer_request_decoder.get_tensor(self.args.dec_st_h_t_layer).data[:]
        self.dec_states_c = self.infer_request_decoder.get_tensor(self.args.dec_st_c_t_layer).data[:]
        self.output = self.infer_request_decoder.get_tensor(self.args.output_layer).data[:]
        logit = self.infer_request_decoder.get_tensor(self.args.logit_layer).data[:]
        self.logits.append(copy.deepcopy(logit))
        self.tgt = np.array([[np.argmax(logit, axis=1)]])

    def _unpack_enc_results(self):
        self.row_enc_out = self.infer_request_encoder.get_tensor(self.args.row_enc_out_layer).data[:]
        self.dec_states_h = self.infer_request_encoder.get_tensor(self.args.hidden_layer).data[:]
        self.dec_states_c = self.infer_request_encoder.get_tensor(self.args.context_layer).data[:]
        self.output = self.infer_request_encoder.get_tensor(self.args.init_0_layer).data[:]
        self.tgt = np.array([[START_TOKEN]])
        self.logits = []


class Renderer:
    class Status(Enum):
        READY = 0
        RENDERING = 1

    def __init__(self):
        self.cur_formula = None
        self.res_img = None
        self._state = Renderer.Status.READY
        self._worker = ThreadPool(processes=1)
        self._async_result = None

    def render(self, formula):
        """
        Synchronous method. Returns rendered image and text of the formula,
        corresponding to the rendered image when rendering is done.
        """
        if self.cur_formula is None:
            self.cur_formula = formula
        elif self.cur_formula == formula:
            return self.res_img, self.cur_formula
        self.cur_formula = formula
        self.res_img = None

        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as output_file:
            output_file_name = output_file.name

        try:
            sympy.preview('$${}$$'.format(formula), viewer='file',
                          filename=output_file_name, euler=False, dvioptions=['-D', '{}'.format(DENSITY)])
            self.res_img = cv.imread(output_file_name)
        finally:
            os.unlink(output_file_name)

        return self.res_img, self.cur_formula

    def thread_render(self, formula):
        """
        Provides asynchronous interface to the rendering process.
        In contrast with the self.render, this one returns image and formula only when rendering is complete.
        If rendering is incomplete at the moment of method call, method returns None
        """
        if self._state == Renderer.Status.READY:
            self._async_result = self._worker.apply_async(self.render, args=(formula,))
            self._state = Renderer.Status.RENDERING
            return None
        if self._state == Renderer.Status.RENDERING:
            if self._async_result.ready() and self._async_result.successful():
                self._state = Renderer.Status.READY
                return self.res_img, self.cur_formula
            elif self._async_result.ready() and not self._async_result.successful():
                self._state = Renderer.Status.READY
            return None


class Vocab:
    """Vocabulary class which helps to get
    human readable formula from sequence of integer tokens
    """

    def __init__(self, vocab_path):
        assert vocab_path.endswith(".json"), "Wrong extension of the vocab file"
        with open(vocab_path, "r") as f:
            vocab_dict = json.load(f)
            vocab_dict['id2sign'] = {int(k): v for k, v in vocab_dict['id2sign'].items()}

        self.id2sign = vocab_dict["id2sign"]

    def construct_phrase(self, indices):
        """Function to get latex formula from sequence of tokens

        Args:
            indices (list): sequence of int

        Returns:
            str: decoded formula
        """
        phrase_converted = []
        for token in indices:
            if token == END_TOKEN:
                break
            phrase_converted.append(
                self.id2sign.get(token, "?"))
        return " ".join(phrase_converted)
