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


class TfConvertImageDType(Preprocessor):
    __provider__ = 'tf_convert_image_dtype'

    def __init__(self, config, name, input_shapes=None):
        super().__init__(config, name, input_shapes)
        try:
            import tensorflow as tf
        except ImportError as import_error:
            raise ImportError(
                'tf_convert_image_dtype disabled.Please, install Tensorflow before using. \n{}'.format(import_error.msg)
            )
        tf.enable_eager_execution()
        self.converter = tf.image.convert_image_dtype
        self.dtype = tf.float32

    def process(self, image, annotation_meta=None):
        converted_data = self.converter(image.data, dtype=self.dtype)
        image.data = converted_data.numpy()

        return image
