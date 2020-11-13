import json
import pickle as pkl
import subprocess
import tempfile
from enum import Enum
from multiprocessing.pool import ThreadPool

import cv2 as cv
import sympy

START_TOKEN = 0
END_TOKEN = 2
DENSITY = 300
DEFAULT_RESOLUTION = (1280, 720)
DEFAULT_WIDTH = 800
MIN_HEIGHT = 30
MAX_HEIGHT = 150
MAX_WIDTH = 1200
MIN_WIDTH = 260
# default value to resize input window's width in pixels
DEFAULT_RESIZE_STEP = 10


def create_renderer():
    command = subprocess.run("pdflatex --version", stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, check=False, shell=True)
    if command.returncode != 0:
        renderer = None
        log.warning("pdflatex not installed, please, install it to use rendering")
    else:
        renderer = Renderer()
    return renderer


class Color(Enum):
    red = (0, 0, 255)
    green = (0, 255, 0)
    blue = (255, 0, 0)
    white = (255, 255, 255)
    black = (0, 0, 0)


class ModelStatus(Enum):
    ready = 0
    encoder_infer = 1
    decoder_infer = 2


class Renderer:
    class Status(Enum):
        ready = 0
        rendering = 1

    def __init__(self):
        with tempfile.NamedTemporaryFile() as temp_file:
            temp_file_name = temp_file.name
        self.output_file = f'{temp_file_name}.png'
        self.cur_formula = None
        self.res_img = None
        self._state = Renderer.Status.ready
        self._worker = ThreadPool(processes=1)
        self._async_result = None

    def render(self, formula):
        if self.cur_formula is None:
            self.cur_formula = formula
        elif self.cur_formula == formula:
            return self.output_file
        self.cur_formula = formula
        try:
            sympy.preview(f'$${formula}$$', viewer='file',
                      filename=self.output_file, euler=False, dvioptions=['-D', f'{DENSITY}'])
            self.res_img = cv.imread(self.output_file)
        except Exception:
            self.res_img = None
        return self.res_img, self.cur_formula

    def thread_render(self, formula):
        if self._state == Renderer.Status.ready:
            self._async_result = self._worker.apply_async(self.render, args=(formula,))
            self._state = Renderer.Status.rendering
            return None
        if self._state == Renderer.Status.rendering:
            if self._async_result.ready() and self._async_result.successful():
                self._state = Renderer.Status.ready
                return self.res_img, self.cur_formula
            elif self._async_result.ready() and not self._async_result.successful():
                self._state = Renderer.Status.ready
            return None


class VideoCapture:
    def __init__(self, input_model_shape, resolution=DEFAULT_RESOLUTION, device_id=0):
        self.capture = cv.VideoCapture(device_id)
        self.resolution = resolution
        self.capture.set(cv.CAP_PROP_BUFFERSIZE, 1)
        self.capture.set(3, resolution[0])
        self.capture.set(4, resolution[1])
        self.tgt_shape = input_model_shape
        self.start_point, self.end_point = self._create_input_window()
        self.prev_rendered_formula = None
        self.prev_formula_img = None
        self.renderer = create_renderer()

    def __call__(self):
        ret, frame = self.capture.read()
        return frame

    def _create_input_window(self):
        aspect_ratio = self.tgt_shape[0] / self.tgt_shape[1]
        default_width = DEFAULT_WIDTH
        height = int(default_width * aspect_ratio)
        start_point = (int(self.resolution[0] / 2 - default_width / 2), int(self.resolution[1] / 2 - height / 2))
        end_point = (int(self.resolution[0] / 2 + default_width / 2), int(self.resolution[1] / 2 + height / 2))
        return start_point, end_point

    def get_crop(self, frame):
        crop = frame[self.start_point[1]:self.end_point[1], self.start_point[0]:self.end_point[0], :]
        return crop

    def draw_rectangle(self, frame):
        frame = cv.rectangle(frame, self.start_point, self.end_point, color=(0, 0, 255), thickness=2)
        return frame

    def resize_window(self, action):
        height = self.end_point[1] - self.start_point[1]
        width = self.end_point[0] - self.start_point[0]
        aspect_ratio = height / width
        if action == 'increase':
            if height >= MAX_HEIGHT or width >= MAX_WIDTH:
                return
            self.start_point = (self.start_point[0]-DEFAULT_RESIZE_STEP,
                                self.start_point[1] - int(DEFAULT_RESIZE_STEP * aspect_ratio))
            self.end_point = (self.end_point[0]+DEFAULT_RESIZE_STEP,
                              self.end_point[1] + int(DEFAULT_RESIZE_STEP * aspect_ratio))
        elif action == 'decrease':
            if height <= MIN_HEIGHT or width <= MIN_WIDTH:
                return
            self.start_point = (self.start_point[0]+DEFAULT_RESIZE_STEP,
                                self.start_point[1] + int(DEFAULT_RESIZE_STEP * aspect_ratio))
            self.end_point = (self.end_point[0]-DEFAULT_RESIZE_STEP,
                              self.end_point[1] - int(DEFAULT_RESIZE_STEP * aspect_ratio))
        else:
            raise ValueError(f"wrong action: {action}")

    def put_text(self, frame, text):
        if text == '':
            return frame
        (txt_w, txt_h), baseLine = cv.getTextSize(text, cv.FONT_HERSHEY_SIMPLEX, 1, 3)

        start_point = (int(self.resolution[0] / 2 - txt_w / 2), self.start_point[1] - txt_h)
        frame = cv.putText(frame, rf'{text}', org=start_point, fontFace=cv.FONT_HERSHEY_SIMPLEX,
                           fontScale=1, color=(255, 255, 255), thickness=3, lineType=cv.LINE_AA)
        frame = cv.putText(frame, rf'{text}', org=start_point, fontFace=cv.FONT_HERSHEY_SIMPLEX,
                           fontScale=1, color=(0, 0, 0), thickness=1, lineType=cv.LINE_AA)
        return frame

    def put_crop(self, frame, crop):
        height = self.end_point[1] - self.start_point[1]
        width = self.end_point[0] - self.start_point[0]
        crop = cv.resize(crop, (width, height))
        frame[0:height, self.start_point[0]:self.end_point[0], :] = crop
        return frame

    def put_formula(self, frame, formula):
        if self.renderer is None or formula == '':
            return frame
        if formula != self.prev_rendered_formula:
            result = self.renderer.thread_render(formula)
            if result is None:
                return frame
            formula_img, res_formula = result
            if res_formula == formula:
                self.prev_rendered_formula = formula
                self.prev_formula_img = formula_img
        else:
            formula_img = self.prev_formula_img
        height = self.end_point[1] - self.start_point[1]
        frame[height:height + formula_img.shape[0],
              self.start_point[0]:self.start_point[0] + formula_img.shape[1],
              :] = formula_img
        return frame

    def release(self):
        self.capture.release()


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
