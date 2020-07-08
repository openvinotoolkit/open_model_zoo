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

import cv2
import numpy as np
from ..config import NumberField, BoolField

try:
    import tensorflow as tf
except ImportError as import_error:
    tf = None

from .preprocessor import Preprocessor


class BgrToRgb(Preprocessor):
    __provider__ = 'bgr_to_rgb'

    def process(self, image, annotation_meta=None):
        def process_data(data):
            return cv2.cvtColor(data, cv2.COLOR_BGR2RGB)
        image.data = process_data(image.data) if not isinstance(image.data, list) else [
            process_data(fragment) for fragment in image.data
        ]
        return image


class BgrToGray(Preprocessor):
    __provider__ = 'bgr_to_gray'

    def process(self, image, annotation_meta=None):
        image.data = np.expand_dims(cv2.cvtColor(image.data, cv2.COLOR_BGR2GRAY).astype(np.float32), -1)
        return image


class RgbToBgr(Preprocessor):
    __provider__ = 'rgb_to_bgr'

    def process(self, image, annotation_meta=None):
        def process_data(data):
            return cv2.cvtColor(data, cv2.COLOR_RGB2BGR)
        image.data = process_data(image.data) if not isinstance(image.data, list) else [
            process_data(fragment) for fragment in image.data
        ]
        return image


class RgbToGray(Preprocessor):
    __provider__ = 'rgb_to_gray'

    def process(self, image, annotation_meta=None):
        image.data = np.expand_dims(cv2.cvtColor(image.data, cv2.COLOR_RGB2GRAY).astype(np.float32), -1)
        return image


class TfConvertImageDType(Preprocessor):
    __provider__ = 'tf_convert_image_dtype'

    def __init__(self, config, name):
        super().__init__(config, name)
        if tf is None:
            raise ImportError('*tf_convert_image_dtype* operation requires TensorFlow. Please install it before usage')
        tf.enable_eager_execution()
        self.converter = tf.image.convert_image_dtype
        self.dtype = tf.float32

    def process(self, image, annotation_meta=None):
        converted_data = self.converter(image.data, dtype=self.dtype)
        image.data = converted_data.numpy()

        return image


class SelectInputChannel(Preprocessor):
    __provider__ = 'select_channel'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters['channel'] = NumberField(value_type=int, min_value=0)
        return parameters

    def configure(self):
        self.channel = self.get_value_from_config('channel')

    def process(self, image, annotation_meta=None):
        def process_data(data):
            return data[:, :, self.channel, np.newaxis]

        if isinstance(image.data, list):
            image.data = [process_data(item) for item in image.data]
        else:
            image.data = process_data(image.data)

        return image


class BGR2YUVConverter(Preprocessor):
    __provider__ = 'bgr_to_yuv'
    color = cv2.COLOR_BGR2YUV

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'split_channels': BoolField(
                optional=True, default=False, description='Allow treat channels as independent input'
            )
        })
        return parameters

    def configure(self):
        self.split_channels = self.get_value_from_config('split_channels')

    def process(self, image, annotation_meta=None):
        data = image.data
        yuvdata = cv2.cvtColor(data, self.color)
        if self.split_channels:
            y = yuvdata[:, :, 0]
            u = yuvdata[:, :, 1]
            v = yuvdata[:, :, 2]
            identifier = image.data
            new_identifier = ['{}_y'.format(identifier), '{}_u'.format(identifier), '{}_v'.format(identifier)]
            yuvdata = [np.expand_dims(y, -1), np.expand_dims(u, -1), np.expand_dims(v, -1)]
            image.identifier = new_identifier
        image.data = yuvdata

        return image


class RGB2YUVConverter(BGR2YUVConverter):
    __provider__ = 'rgb_to_yuv'
    color = cv2.COLOR_RGB2YUV
