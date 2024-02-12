"""
Copyright (c) 2018-2024 Intel Corporation

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

from copy import deepcopy
from collections import namedtuple
import warnings
import cv2
try:
    from cv2.gapi.mx.pp import Resize, Crop, Padding
    _preprocessing_available = True
except ImportError:
    Resize, Crop, Padding = None, None, None
    _preprocessing_available = False
from ...utils import get_size_from_config, string_to_tuple


def gapi_preprocess_available():
    return _preprocessing_available


PreprocessingOp = namedtuple('PreprocessingOp', ['name', 'value'])


class GAPIPreprocessor:

    def __init__(self, config):
        self.SUPPORTED_PREPROCESSING_OPS = {
            'resize': self.get_resize_op,
            'crop': self.get_crop_op,
            'padding': self.get_padding_op
        }
        if _preprocessing_available:
            self.RESIZE_INTERPOLATIONS = {
                'LINEAR': cv2.gapi.mx.pp.Resize_Interpolation_BILINEAR,
                'BILINEAR': cv2.gapi.mx.pp.Resize_Interpolation_BILINEAR,
                'BICUBIC': cv2.gapi.mx.pp.Resize_Interpolation_BICUBIC,
            }
            self.RESIZE_ASPECT_RATIO = {
                'greater': cv2.gapi.mx.pp.Resize_AspectRatioScale_GREATER,
                'fit_to_window': cv2.gapi.mx.pp.Resize_AspectRatioScale_FIT_TO_WINDOW,
                'height': cv2.gapi.mx.pp.Resize_AspectRatioScale_HEIGHT,
                'width': cv2.gapi.mx.pp.Resize_AspectRatioScale_WIDTH,
            }
            self.PADDING_TYPE = {
                'center': cv2.gapi.mx.pp.Padding_PadType_CENTER,
                'right_bottom': cv2.gapi.mx.pp.Padding_PadType_RIGHT_BOTTOM,
                'left_top': cv2.gapi.mx.pp.Padding_PadType_LEFT_TOP,
            }
        else:
            self.RESIZE_ASPECT_RATIO = {}
            self.RESIZE_INTERPOLATIONS = {}
            self.PADDING_TYPE = {}

        self.config = config or []
        self.configure()

    def configure(self):
        steps = []
        step_names = set()
        keep_preprocessing_config = deepcopy(self.config)
        for preprocessor in reversed(self.config):
            if preprocessor['type'] not in self.SUPPORTED_PREPROCESSING_OPS:
                break
            ie_ops = self.get_op(preprocessor)
            if not ie_ops:
                break
            if ie_ops.name in step_names:
                break
            step_names.add(ie_ops.name)
            steps.append(ie_ops)
            keep_preprocessing_config = keep_preprocessing_config[:-1]
        self.steps = steps
        self.keep_preprocessing_info = keep_preprocessing_config
        if not steps:
            warnings.warn('no preprocessing steps for transition to PreProcessInfo')

    def get_op(self, preprocessing_config):
        preprocessing_getter = self.SUPPORTED_PREPROCESSING_OPS[preprocessing_config['type']]
        return preprocessing_getter(preprocessing_config)

    def get_resize_op(self, config):
        aspect_ratio_cfg = config.get('aspect_ratio_scale')
        if aspect_ratio_cfg is not None and aspect_ratio_cfg not in self.RESIZE_ASPECT_RATIO:
            return None
        aspect_ratio = self.RESIZE_ASPECT_RATIO.get(aspect_ratio_cfg, cv2.gapi.mx.pp.Resize_AspectRatioScale_UNKNOWN)
        cfg_interpolation = config.get('interpolation', 'BILINEAR').upper()
        if cfg_interpolation not in self.RESIZE_INTERPOLATIONS:
            return None
        interpolation = self.RESIZE_INTERPOLATIONS[cfg_interpolation]
        height, width = get_size_from_config(config)
        return PreprocessingOp(
            'resize_algorithm', Resize((width, height), interpolation=interpolation, aspect_ratio_scale=aspect_ratio)
        )

    def has_resize(self):
        preprocessor_names = [step.name for step in self.steps]
        return 'resize_algorithm' in preprocessor_names

    @staticmethod
    def get_crop_op(config):
        if 'max_square' in config:
            return PreprocessingOp('crop', Crop(cv2.gapi.mx.pp.MaxSquare()))
        if 'central_fraction' in config:
            ratio = float(config['central_fraction'])
            return PreprocessingOp('crop', Crop(cv2.gapi.mx.pp.CentralFraction(ratio)))
        height, width = get_size_from_config(config)
        return PreprocessingOp('crop', Crop((width, height)))

    def get_padding_op(self, config):
        if 'numpy_pad_mode' in config and config['numpy_pad_mode'] != 'constant':
            return None
        height, width = get_size_from_config(config, allow_none=True)
        # gapi dos not support fully strided padding right now
        if height is None or width is None:
            return None
        pad_type = config.get('pad_type', 'center')
        if pad_type not in self.PADDING_TYPE:
            return None
        pad_t = self.PADDING_TYPE[pad_type]
        pad_val = config.get('pad_value', [0, 0, 0])
        if isinstance(pad_val, str):
            pad_val = string_to_tuple(pad_val, int)
        if isinstance(pad_val, int):
            pad_val = [pad_val] * 3
        if isinstance(pad_val, tuple):
            pad_val = list(pad_val)
        return PreprocessingOp('padding', Padding((width, height), pad_val, pad_type=pad_t))
