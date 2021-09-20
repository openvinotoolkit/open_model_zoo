"""
Copyright (c) 2018-2021 Intel Corporation

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
try:
    from openvino.inference_engine import ResizeAlgorithm, PreProcessInfo, ColorFormat, MeanVariant
except ImportError:
    ResizeAlgorithm, PreProcessInfo, ColorFormat, MeanVariant = None, None, None, None
from ..utils import get_or_parse_value
from ..config import ConfigError


def ie_preprocess_available():
    return PreProcessInfo is not None


PreprocessingOp = namedtuple('PreprocessingOp', ['name', 'value'])


class IEPreprocessor:
    PRECOMPUTED_MEANS = {
        'imagenet': (104.00698793, 116.66876762, 122.67891434),
        'cifar10': (125.307, 122.961, 113.8575),
    }

    PRECOMPUTED_STDS = {
        'imagenet': (104.00698793, 116.66876762, 122.67891434),
        'cifar10': (125.307, 122.961, 113.8575),
    }

    def __init__(self, config):
        self.SUPPORTED_PREPROCESSING_OPS = {
            'resize': self.get_resize_op,
            'auto_resize': self.get_resize_op,
            'bgr_to_rgb': self.get_color_format_op,
            'rgb_to_bgr': self.get_color_format_op,
            'nv12_to_bgr': self.get_color_format_op,
            'nv12_to_rgb': self.get_color_format_op,
            'normalization': self.get_normalization_op
        }
        self.mean_values = None
        self.std_values = None
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

    @property
    def preprocess_info(self):
        preprocess_info = PreProcessInfo()
        for (name, value) in self.steps:
            setattr(preprocess_info, name, value)
        return preprocess_info

    @staticmethod
    def get_resize_op(config):
        supported_interpolations = {
            'LINEAR': ResizeAlgorithm.RESIZE_BILINEAR,
            'BILINEAR': ResizeAlgorithm.RESIZE_BILINEAR,
            'AREA': ResizeAlgorithm.RESIZE_AREA
        }
        if 'aspect_ratio' in config:
            return None
        interpolation = config.get('interpolation', 'BILINEAR').upper()
        if interpolation not in supported_interpolations:
            return None
        return PreprocessingOp('resize_algorithm', supported_interpolations[interpolation])

    @staticmethod
    def get_color_format_op(config):
        source_color = config['type'].split('_')[0].upper()
        try:
            ie_color_format = ColorFormat[source_color]
        except KeyError:
            return None
        return PreprocessingOp('color_format', ie_color_format)

    def get_normalization_op(self, config):
        self.mean_values = get_or_parse_value(config.get('mean'), self.PRECOMPUTED_MEANS)
        self.std_values = get_or_parse_value(config.get('std'), self.PRECOMPUTED_STDS)
        return PreprocessingOp('mean_variant', MeanVariant.MEAN_VALUE)

    def set_normalization(self, num_channels, preprocess_info):
        preprocess_info.init(num_channels)
        if self.mean_values is None:
            self.mean_values = [0] * num_channels
        if self.std_values is None:
            self.std_values = [1] * num_channels
        if isinstance(self.mean_values, tuple) and len(self.mean_values) == 1:
            self.mean_values = list(self.mean_values) * num_channels
        if isinstance(self.std_values, tuple) and len(self.std_values) == 1:
            self.std_values = list(self.std_values) * num_channels
        if len(self.mean_values) != num_channels or len(self.std_values) != num_channels:
            raise ConfigError('mean and std should be specified for all channels')
        for channel in range(num_channels):
            preprocess_info[channel].mean_value = self.mean_values[channel]
            preprocess_info[channel].std_scale = self.std_values[channel]

    def has_resize(self):
        preprocessor_names = [step.name for step in self.steps]
        return 'resize_algorithm' in preprocessor_names

    def has_normalization(self):
        preprocessor_names = [step.name for step in self.steps]
        return 'mean_variant' in preprocessor_names
