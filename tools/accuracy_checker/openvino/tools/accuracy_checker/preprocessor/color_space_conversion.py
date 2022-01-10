"""
Copyright (c) 2018-2022 Intel Corporation

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

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            "cast_to_float": BoolField(
                default=True,
                description="Parameter specifies if the result image should be casted to np.float32"
            )
        })
        return parameters

    def configure(self):
        self.cast_to_float = self.get_value_from_config('cast_to_float')

    def process(self, image, annotation_meta=None):
        def process_data(data):
            gray_image = np.expand_dims(cv2.cvtColor(data, cv2.COLOR_BGR2GRAY), -1)
            if self.cast_to_float:
                gray_image = gray_image.astype(np.float32)
            return gray_image

        image.data = process_data(image.data) if not isinstance(image.data, list) else [
            process_data(fragment) for fragment in image.data
        ]
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

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            "cast_to_float": BoolField(
                default=True,
                description="Parameter specifies if the result image should be casted to np.float32"
            )
        })
        return parameters

    def configure(self):
        self.cast_to_float = self.get_value_from_config('cast_to_float')

    def process(self, image, annotation_meta=None):
        def process_data(data):
            gray_image = np.expand_dims(cv2.cvtColor(data, cv2.COLOR_RGB2GRAY), -1)
            if self.cast_to_float:
                gray_image = gray_image.astype(np.float32)
            return gray_image

        image.data = process_data(image.data) if not isinstance(image.data, list) else [
            process_data(fragment) for fragment in image.data
        ]
        return image


class BGRToLAB(Preprocessor):
    __provider__ = 'bgr_to_lab'

    def process(self, image, annotation_meta=None):
        def process_data(data):
            return cv2.cvtColor(data.astype(np.float32) / 255, cv2.COLOR_BGR2LAB)

        image.data = process_data(image.data) if not isinstance(image.data, list) else [
            process_data(fragment) for fragment in image.data
        ]
        return image


class RGBToLAB(Preprocessor):
    __provider__ = 'bgr_to_lab'

    def process(self, image, annotation_meta=None):
        def process_data(data):
            return cv2.cvtColor(data.astype(np.float32) / 255, cv2.COLOR_RGB2LAB)

        image.data = process_data(image.data) if not isinstance(image.data, list) else [
            process_data(fragment) for fragment in image.data
        ]
        return image

class TfConvertImageDType(Preprocessor):
    __provider__ = 'tf_convert_image_dtype'

    def __init__(self, config, name):
        super().__init__(config, name)
        try:
            import tensorflow as tf # pylint: disable=C0415
        except ImportError as import_error:
            raise ImportError(
                '*tf_convert_image_dtype* operation requires TensorFlow. '
                'Please install it before usage. {}'.format(import_error.msg)
            ) from import_error
        if tf.__version__ < '2.0.0':
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
            ),
            'shrink_uv': BoolField(
                optional=True, default=False, description='Allow shrink uv-channels after split'
            )
        })
        return parameters

    def configure(self):
        self.split_channels = self.get_value_from_config('split_channels')
        self.shrink_uv = self.get_value_from_config('shrink_uv')
        if self.shrink_uv and not self.split_channels:
            self.split_channels = True

    def process(self, image, annotation_meta=None):
        data = image.data
        yuvdata = cv2.cvtColor(data, self.color)
        if self.split_channels:
            y = yuvdata[:, :, 0]
            u = yuvdata[:, :, 1]
            v = yuvdata[:, :, 2]
            identifier = image.identifier
            new_identifier = ['{}_y'.format(identifier), '{}_u'.format(identifier), '{}_v'.format(identifier)]
            if self.shrink_uv:
                u = cv2.resize(u, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
                v = cv2.resize(v, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            yuvdata = [np.expand_dims(y, -1), np.expand_dims(u, -1), np.expand_dims(v, -1)]
            image.identifier = new_identifier
        image.data = yuvdata

        return image


class RGB2YUVConverter(BGR2YUVConverter):
    __provider__ = 'rgb_to_yuv'
    color = cv2.COLOR_RGB2YUV


class BGRtoNV12Converter(Preprocessor):
    __provider__ = 'bgr_to_nv12'

    def process(self, image, annotation_meta=None):
        data = image.data
        height, width, _ = data.shape
        y, u, v = cv2.cvtColor(data, cv2.COLOR_BGR2YUV)

        shrunk_u = cv2.resize(u, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        shrunk_v = cv2.resize(v, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

        uv = np.zeros((height // 2, width))

        uv[:, 0::2] = shrunk_u
        uv[:, 1::2] = shrunk_v

        nv12 = np.vstack((y, uv))

        nv12 = np.floor(nv12 + 0.5).astype(np.uint8)
        image.data = nv12

        return image


class RGBtoNV12Converter(Preprocessor):
    __provider__ = 'rgb_to_nv12'

    def process(self, image, annotation_meta=None):
        data = image.data
        height, width, _ = data.shape
        y, u, v = cv2.cvtColor(data, cv2.COLOR_RGB2YUV)

        shrunk_u = cv2.resize(u, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
        shrunk_v = cv2.resize(v, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)

        uv = np.zeros((height // 2, width))

        uv[:, 0::2] = shrunk_u
        uv[:, 1::2] = shrunk_v

        nv12 = np.vstack((y, uv))

        nv12 = np.floor(nv12 + 0.5).astype(np.uint8)
        image.data = nv12

        return image


class NV12toBGRConverter(Preprocessor):
    __provider__ = 'nv12_to_bgr'

    def process(self, image, annotation_meta=None):
        image.data = cv2.cvtColor(image.data, cv2.COLOR_YUV2BGR_NV12)
        return image


class NV12toRGBConverter(Preprocessor):
    __provider__ = 'nv12_to_rgb'

    def process(self, image, annotation_meta=None):
        image.data = cv2.cvtColor(image.data, cv2.COLOR_YUV2RGB_NV12)
        return image


class BGR2YCrCbConverter(Preprocessor):
    __provider__ = 'bgr_to_ycrcb'
    color = cv2.COLOR_BGR2YCrCb

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
        ycrcbdata = cv2.cvtColor(data, self.color)
        if self.split_channels:
            y = ycrcbdata[:, :, 0]
            cr = ycrcbdata[:, :, 1]
            cb = ycrcbdata[:, :, 2]
            identifier = image.identifier
            new_identifier = ['{}_y'.format(identifier), '{}_cr'.format(identifier), '{}_cb'.format(identifier)]
            ycrcbdata = [np.expand_dims(y, -1), np.expand_dims(cr, -1), np.expand_dims(cb, -1)]
            image.identifier = new_identifier
        image.data = ycrcbdata

        return image


class RGB2YCrCbConverter(BGR2YCrCbConverter):
    __provider__ = 'rgb_to_ycrcb'
    color = cv2.COLOR_RGB2YCrCb
