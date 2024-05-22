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

import cv2
import numpy as np
from PIL import Image

from ..config import StringField, ConfigError, BoolField
from .data_reader import BaseReader
from ..utils import get_path, UnsupportedPackage

try:
    import skimage.io as sk
except ImportError as import_error:
    sk = UnsupportedPackage('skimage.io', import_error.msg)

try:
    import rawpy
except ImportError as import_error:
    rawpy = UnsupportedPackage('rawpy', import_error.msg)


OPENCV_IMREAD_FLAGS = {
    'color': cv2.IMREAD_COLOR,
    'gray': cv2.IMREAD_GRAYSCALE,
    'unchanged': cv2.IMREAD_UNCHANGED
}


class OpenCVImageReader(BaseReader):
    __provider__ = 'opencv_imread'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'reading_flag': StringField(optional=True, choices=OPENCV_IMREAD_FLAGS, default='color',
                                        description='Flag which specifies the way image should be read.')
        })
        return parameters

    def configure(self):
        super().configure()
        self.flag = OPENCV_IMREAD_FLAGS[self.get_value_from_config('reading_flag')]

    def read(self, data_id):
        data_path = self.data_source / data_id if self.data_source else data_id
        return cv2.imread(str(get_path(data_path)), self.flag)


class PillowImageReader(BaseReader):
    __provider__ = 'pillow_imread'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'to_rgb': BoolField(optional=True, default=True, description='convert image to RGB format')
        })
        return parameters

    def configure(self):
        super().configure()
        self.convert_to_rgb = self.get_value_from_config('to_rgb')

    def read(self, data_id):
        data_path = get_path(self.data_source / data_id) if self.data_source is not None else data_id
        with open(str(data_path), 'rb') as f:
            img = Image.open(f)

            return np.array(img.convert('RGB') if self.convert_to_rgb else img)


class ScipyImageReader(BaseReader):
    __provider__ = 'scipy_imread'

    def read(self, data_id):
        # reimplementation scipy.misc.imread
        data_path = self.data_source / data_id if self.data_source is not None else data_id
        image = Image.open(str(get_path(data_path)))
        if image.mode == 'P':
            image = image.convert('RGBA') if 'transparency' in image.info else image.convert('RGB')

        return np.array(image)


class OpenCVFrameReader(BaseReader):
    __provider__ = 'opencv_capture'

    def __init__(self, data_source, config=None, **kwargs):
        super().__init__(data_source, config, **kwargs)
        self.current = -1

    def read(self, data_id):
        if data_id < 0:
            raise IndexError('frame with {} index can not be grabbed, non-negative index is expected')
        if data_id < self.current:
            self.videocap.set(cv2.CAP_PROP_POS_FRAMES, data_id)
            self.current = data_id - 1

        return self._read_sequence(data_id)

    def _read_sequence(self, data_id):
        frame = None
        while self.current != data_id:
            success, frame = self.videocap.read()
            self.current += 1
            if not success:
                raise EOFError('frame with {} index does not exist in {}'.format(self.current, self.data_source))

        return frame

    def configure(self):
        if not self.data_source:
            raise ConfigError('data_source parameter is required to create "{}" '
                              'data reader and read data'.format(self.__provider__))
        self.data_source = get_path(self.data_source)
        self.videocap = cv2.VideoCapture(str(self.data_source))
        self.multi_infer = self.get_value_from_config('multi_infer')
        self.data_layout = self.get_value_from_config('data_layout')

    def reset(self):
        self.current = -1
        self.videocap.set(cv2.CAP_PROP_POS_FRAMES, 0)


class TensorflowImageReader(BaseReader):
    __provider__ = 'tf_imread'

    def __init__(self, data_source, config=None, **kwargs):
        super().__init__(data_source, config, **kwargs)
        try:
            import tensorflow as tf  # pylint: disable=C0415
        except ImportError as import_err:
            raise ImportError(
                'tf backend for image reading requires TensorFlow. '
                'Please install it before usage. {}'.format(import_err.msg)
            ) from import_err
        if tf.__version__ < '2.0.0':
            tf.enable_eager_execution()

        def read_func(path):
            img_raw = tf.read_file(str(path)) if tf.__version__ < '2.0.0' else tf.io.read_file(str(path))
            img_tensor = tf.image.decode_image(img_raw, channels=3)
            return img_tensor.numpy()

        self.read_realisation = read_func

    def read(self, data_id):
        data_path = self.data_source / data_id if self.data_source is not None else data_id
        return self.read_realisation(data_path)


class SkimageReader(BaseReader):
    __provider__ = 'skimage_imread'

    def __init__(self, data_source, config=None, **kwargs):
        super().__init__(data_source, config, **kwargs)
        if isinstance(sk, UnsupportedPackage):
            sk.raise_error(self.__provider__)

    def read(self, data_id):
        data_path = self.data_source / data_id if self.data_source is not None else data_id
        return sk.imread(str(data_path))


class RawpyReader(BaseReader):
    __provider__ = 'rawpy'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'postprocess': BoolField(optional=True, default=True)
        })
        return params

    def configure(self):
        if isinstance(rawpy, UnsupportedPackage):
            rawpy.raise_error(self.__provider__)
        self.postprocess = self.get_value_from_config('postprocess')
        super().configure()

    def read(self, data_id):
        data_path = self.data_source / data_id if self.data_source is not None else data_id
        raw = rawpy.imread(str(data_path))
        if not self.postprocess:
            return raw.raw_image_visible.astype(np.float32)
        postprocessed = raw.postprocess(use_camera_wb=True, half_size=False, no_auto_bright=True, output_bps=16)
        return np.float32(postprocessed / 65535.0)
