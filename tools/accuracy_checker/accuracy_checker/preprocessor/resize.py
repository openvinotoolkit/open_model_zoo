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

import inspect

import cv2
import numpy as np
from PIL import Image

from ..config import BoolField, ConfigError, NumberField, StringField
from ..dependency import ClassProvider
from ..logging import warning
from ..preprocessor import Preprocessor, GeometricOperationMetadata
from ..utils import contains_all, get_size_from_config, get_parameter_value_from_config, UnsupportedPackage


def scale_width(dst_width, dst_height, image_width, image_height,):
    return int(dst_width * image_width / image_height), dst_height


def scale_height(dst_width, dst_height, image_width, image_height):
    return dst_width, int(dst_height * image_height / image_width)


def scale_greater(dst_width, dst_height, image_width, image_height):
    if image_height > image_width:
        return scale_height(dst_width, dst_height, image_width, image_height)
    return scale_width(dst_width, dst_height, image_width, image_height)


def scale_fit_to_window(dst_width, dst_height, image_width, image_height):
    im_scale = min(dst_height / image_height, dst_width / image_width)
    return int(im_scale * image_width), int(im_scale * image_height)


def frcnn_keep_aspect_ratio(dst_width, dst_height, image_width, image_height):
    min_size = min(dst_width, dst_height)
    max_size = max(dst_width, dst_height)
    w1, h1 = scale_greater(min_size, min_size, image_width, image_height)
    if max(w1, h1) <= max_size:
        return w1, h1
    return scale_fit_to_window(max_size, max_size, image_width, image_height)


def ctpn_keep_aspect_ratio(dst_width, dst_height, image_width, image_height):
    scale = min(dst_height, dst_width)
    max_scale = max(dst_height, dst_width)
    im_min_size = min(image_width, image_height)
    im_max_size = max(image_width, image_height)
    im_scale = float(scale) / float(im_min_size)
    if np.round(im_scale * im_max_size) > max_scale:
        im_scale = float(max_scale) / float(im_max_size)
    new_h = np.round(image_height * im_scale)
    new_w = np.round(image_width * im_scale)
    return int(new_w), int(new_h)


def east_keep_aspect_ratio(dst_width, dst_height, image_width, image_height):
    resize_w = image_width
    resize_h = image_height
    max_side_len = max(dst_width, dst_height)
    min_side_len = min(dst_width, dst_height)

    # limit the max side
    if max(resize_h, resize_w) > max_side_len:
        ratio = float(max_side_len) / resize_h if resize_h > resize_w else float(max_side_len) / resize_w
    else:
        ratio = 1.
    resize_h = int(resize_h * ratio)
    resize_w = int(resize_w * ratio)

    resize_h = resize_h if resize_h % min_side_len == 0 else (resize_h // min_side_len - 1) * min_side_len
    resize_w = resize_w if resize_w % min_side_len == 0 else (resize_w // min_side_len - 1) * min_side_len
    resize_h = max(32, resize_h)
    resize_w = max(32, resize_w)

    return resize_w, resize_h


def ppocr_aspect_ratio(dst_width, dst_height, image_width, image_height):
    ratio = image_width / image_height
    tmp_w = int(32 * ratio)
    if np.ceil(dst_height * ratio) > dst_width:
        resize_w = tmp_w
    else:
        resize_w = int(np.ceil(dst_height * ratio))
    return resize_w, dst_height


class ScaleFactor:
    def __init__(self, config, parameters):
        self.scale = get_parameter_value_from_config(config, parameters, 'scale')

    def __call__(self, dst_width, dst_height, image_width, image_height):
        resize_w = int(image_width * self.scale)
        resize_h = int(image_height * self.scale)
        return resize_w, resize_h


def min_ratio(dst_width, dst_height, image_width, image_height):
    ratio = min(float(image_height) / float(dst_height), float(image_width) / float(dst_width))
    return int(image_width / ratio), int(image_height / ratio)


def mask_rcnn_benchmark_ratio(dst_width, dst_height, image_width, image_height):
    min_dst_size = min(dst_width, dst_height)
    ratio = min_dst_size / min(image_width, image_height)
    return int(image_width * ratio), int(image_height * ratio)



ASPECT_RATIO_SCALE = {
    'width': scale_width,
    'height': scale_height,
    'greater': scale_greater,
    'fit_to_window': scale_fit_to_window,
    'frcnn_keep_aspect_ratio': frcnn_keep_aspect_ratio,
    'ctpn_keep_aspect_ratio': ctpn_keep_aspect_ratio,
    'east_keep_aspect_ratio': east_keep_aspect_ratio,
    'min_ratio': min_ratio,
    'mask_rcnn_benchmark_aspect_ratio': mask_rcnn_benchmark_ratio,
    'scale_factor': ScaleFactor,
    'ppcrnn_ratio': ppocr_aspect_ratio
}


class _Resizer(ClassProvider):
    __provider_type__ = 'resizer'

    default_interpolation = None

    def __init__(self, interpolation=None):
        if not interpolation:
            interpolation = self.default_interpolation
        if interpolation.upper() not in self.supported_interpolations():
            raise ConfigError('{} not found for {}'.format(self.supported_interpolations(), self.__provider__))
        self.interpolation = self.supported_interpolations().get(interpolation.upper(), self.default_interpolation)

    def resize(self, data, new_height, new_width):
        raise NotImplementedError

    def __call__(self, data, new_height, new_width):
        return self.resize(data, new_height, new_width)

    @classmethod
    def all_provided_interpolations(cls):
        interpolations = set()
        for _, provider_class in cls.providers.items():
            try:
                interpolations.update(provider_class.supported_interpolations())
            except ImportError:
                continue
        return interpolations

    @classmethod
    def supported_interpolations(cls):
        return {}


class _OpenCVResizer(_Resizer):
    __provider__ = 'opencv'

    _supported_interpolations = {
        'NEAREST': cv2.INTER_NEAREST,
        'LINEAR': cv2.INTER_LINEAR,
        'BILINEAR': cv2.INTER_LINEAR,
        'CUBIC': cv2.INTER_CUBIC,
        'BICUBIC': cv2.INTER_CUBIC,
        'AREA': cv2.INTER_AREA,
        'MAX': cv2.INTER_MAX,
        'BITS': cv2.INTER_BITS,
        'BITS2': cv2.INTER_BITS2,
        'LANCZOS4': cv2.INTER_LANCZOS4,
    }
    default_interpolation = 'LINEAR'

    def resize(self, data, new_height, new_width):
        return cv2.resize(data, (new_width, new_height), interpolation=self.interpolation).astype(np.float32)

    @classmethod
    def supported_interpolations(cls):
        return cls._supported_interpolations


class _PillowResizer(_Resizer):
    __provider__ = 'pillow'
    default_interpolation = 'BILINEAR'

    def __init__(self, interpolation):
        self._supported_interpolations = {
            'NEAREST': Image.NEAREST,
            'NONE': Image.NONE,
            'BILINEAR': Image.BILINEAR,
            'LINEAR': Image.LINEAR,
            'BICUBIC': Image.BICUBIC,
            'CUBIC': Image.CUBIC,
            'ANTIALIAS': Image.ANTIALIAS,
        }
        try:
            optional_interpolations = {
                'BOX': Image.BOX,
                'LANCZOS': Image.LANCZOS,
                'HAMMING': Image.HAMMING,
            }
            self._supported_interpolations.update(optional_interpolations)
        except AttributeError:
            pass
        super().__init__(interpolation)

    def resize(self, data, new_height, new_width):
        data = Image.fromarray(data)
        data = data.resize((new_width, new_height), self.interpolation)
        data = np.array(data)

        return data

    @classmethod
    def supported_interpolations(cls):
        if Image is None:
            return {}
        intrp = {
            'NEAREST': Image.NEAREST,
            'NONE': Image.NONE,
            'BILINEAR': Image.BILINEAR,
            'LINEAR': Image.LINEAR,
            'BICUBIC': Image.BICUBIC,
            'CUBIC': Image.CUBIC,
            'ANTIALIAS': Image.ANTIALIAS
        }
        try:
            optional_interpolations = {
                'BOX': Image.BOX,
                'LANCZOS': Image.LANCZOS,
                'HAMMING': Image.HAMMING,
            }
            intrp.update(optional_interpolations)
        except AttributeError:
            pass
        return intrp


class _TFResizer(_Resizer):
    __provider__ = 'tf'
    _supported_interpolations = {}

    def __init__(self, interpolation):
        try:
            import tensorflow as tf # pylint: disable=C0415
        except ImportError as import_error:
            UnsupportedPackage("tf", import_error.msg).raise_error(self.__provider__)
        if tf.__version__ < '2.0.0':
            tf.enable_eager_execution()
            self._resize = tf.image.resize_images
        else:
            self._resize = tf.image.resize
        self._supported_interpolations = {
            'BILINEAR': tf.image.ResizeMethod.BILINEAR,
            'AREA': tf.image.ResizeMethod.AREA,
            'BICUBIC': tf.image.ResizeMethod.BICUBIC,
        }
        self.default_interpolation = 'BILINEAR'
        super().__init__(interpolation)

    def resize(self, data, new_height, new_width):
        resized_data = self._resize(data, [new_height, new_width], method=self.interpolation)
        return resized_data.numpy()

    @classmethod
    def supported_interpolations(cls):
        return {}


def create_resizer(config):
    resize_realization = config.get('resize_realization')
    interpolation = config.get('interpolation')

    def provided_both_resizer(additional_flag):
        return contains_all(config, ['resize_realization', additional_flag])

    def select_resizer_by_flags(use_pil, use_tf):
        if use_pil and use_tf:
            raise ConfigError('Pillow and TensorFlow flags both provided. Please select only one resize method.')
        if use_pil:
            return 'pillow'
        if use_tf:
            return 'tf'
        return 'opencv'

    if resize_realization:
        if provided_both_resizer('use_pillow') or provided_both_resizer('use_tensorflow'):
            warning(
                'resize_realization and flag: {} both provided. resize_realization: {} will be used.'.format(
                    'use_pillow' if 'use_pillow' in config else 'use_tensorflow', config['resize_realization']
                )
            )
    else:
        use_pillow, use_tesorfow = config.get('use_pillow', False), config.get('use_tensorflow', False)
        resize_realization = select_resizer_by_flags(use_pillow, use_tesorfow)

    return _Resizer.provide(resize_realization, interpolation)


class Resize(Preprocessor):
    __provider__ = 'resize'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'size': NumberField(
                value_type=int, optional=True, min_value=1, description="Destination sizes for both dimensions."
            ),
            'dst_width': NumberField(
                value_type=int, optional=True, min_value=1, description="Destination width for image resizing."
            ),
            'dst_height': NumberField(
                value_type=int, optional=True, min_value=1, description="Destination height for image resizing."
            ),
            'scale': NumberField(
                value_type=float, optional=True, min_value=0, description="Rescale factor for x & y axes."
            ),
            'aspect_ratio_scale': StringField(
                choices=ASPECT_RATIO_SCALE, optional=True,
                description="Allows save image aspect ratio using one of these ways: "
                            "{}".format(', '.join(ASPECT_RATIO_SCALE))
            ),
            'interpolation': StringField(
                choices=_Resizer.all_provided_interpolations(), optional=True, default='LINEAR',
                description="Specifies method that will be used."
            ),
            'use_pillow': BoolField(
                optional=True, default=False,
                description="Parameter specifies usage of Pillow library for resizing."
            ),
            'use_tensorflow': BoolField(
                optional=True,
                description="Specifies usage of TensorFlow Image for resizing. Requires TensorFlow installation."
            ),

            'resize_realization': StringField(
                optional=True, choices=_Resizer.providers,
                description="Parameter specifies functionality of which library will be used for resize: "
                            "{}".format(', '.join(_Resizer.providers))
            ),
            'factor': NumberField(
                optional=True, min_value=1,
                description='destination size must be divisible by a given number without remainder, '
                            'when aspect ratio resize used', value_type=int
            )
        })

        return parameters

    def configure(self):
        allow_none = self.get_value_from_config('aspect_ratio_scale') == 'scale_factor'
        self.dst_height, self.dst_width = get_size_from_config(self.config, allow_none=allow_none)
        self.resizer = create_resizer(self.config)
        self.scaling_func = ASPECT_RATIO_SCALE.get(self.get_value_from_config('aspect_ratio_scale'))
        self.factor = self.get_value_from_config('factor')
        if inspect.isclass(self.scaling_func):
            self.scaling_func = self.scaling_func(self.config, self.parameters())

    def process(self, image, annotation_meta=None):
        data = image.data
        new_height, new_width = self.dst_height, self.dst_width

        is_simple_case = not isinstance(data, list) # otherwise -- pyramid, tiling, etc

        def process_data(data, new_height, new_width, scale_func, resize_func):
            dst_width, dst_height = new_width, new_height
            image_h, image_w = data.shape[:2]
            if scale_func:
                dst_width, dst_height = scale_func(new_width, new_height, image_w, image_h)
                if self.factor:
                    dst_width -= (dst_width - 1) % self.factor
                    dst_height -= (dst_height - 1) % self.factor
                if new_height is None:
                    new_height = dst_height
                if new_width is None:
                    new_width = dst_width

            resize_meta = {}
            resize_meta['preferable_width'] = max(dst_width, new_width)
            resize_meta['preferable_height'] = max(dst_height, new_height)
            resize_meta['image_info'] = [dst_height, dst_width, 1]
            resize_meta['scale_x'] = float(dst_width) / image_w
            resize_meta['scale_y'] = float(dst_height) / image_h
            resize_meta['original_width'] = image_w
            resize_meta['original_height'] = image_h

            if is_simple_case:
                # support GeometricOperationMetadata array for simple case only -- without tiling, pyramids, etc
                image.metadata.setdefault(
                    'geometric_operations', []).append(GeometricOperationMetadata('resize', resize_meta))

            image.metadata.update(resize_meta)

            data = resize_func(data, dst_height, dst_width)
            if len(data.shape) == 2:
                data = np.expand_dims(data, axis=-1)

            return data

        image.data = (
            process_data(data, new_height, new_width, self.scaling_func, self.resizer)
            if is_simple_case else [
                process_data(data_fragment, new_height, new_width, self.scaling_func, self.resizer)
                for data_fragment in data
            ]
        )

        return image


class AutoResize(Preprocessor):
    __provider__ = 'auto_resize'

    def __init__(self, config, name=None):
        super().__init__(config, name)
        self.dst_height = None
        self.dst_width = None

    def set_input_shape(self, input_shape):
        def is_image_input(shape):
            return len(shape) == 4 and shape[1] in [1, 3, 4]
        if input_shape is None:
            raise ConfigError('resize to input size impossible')
        image_inputs = [value for value in input_shape.values() if is_image_input(value)]
        if not image_inputs:
            raise ConfigError('image input is not detected')
        if len(image_inputs) == 1:
            self.dst_height, self.dst_width = image_inputs[0][2:]
        else:
            self.dst_height = [im_input[2] for im_input in image_inputs]
            self.dst_width = [im_input[3] for im_input in image_inputs]

    def process(self, image, annotation_meta=None):
        is_simple_case = not isinstance(image.data, list)  # otherwise -- pyramid, tiling, etc
        if self.dst_height is None or self.dst_width is None:
            self.set_input_shape(self.input_shapes)

        def process_data(data, idx=0):
            image_h, image_w = data.shape[:2]
            if isinstance(self.dst_height, list):
                dst_height, dst_width = self.dst_height[idx], self.dst_width[idx]
            else:
                dst_height, dst_width = self.dst_height, self.dst_width

            data = cv2.resize(data, (dst_width, dst_height)).astype(np.float32)
            if len(data.shape) == 2:
                data = np.expand_dims(data, axis=-1)

            if is_simple_case:
                # support GeometricOperationMetadata array for simple case only -- without tiling, pyramids, etc
                resize_meta = {
                    'preferable_width': dst_width,
                    'preferable_height': dst_height,
                    'image_info': [dst_height, dst_width, 1],
                    'scale_x': float(dst_width) / image_w,
                    'scale_y': float(dst_height) / image_h,
                    'original_width': image_w,
                    'original_height': image_h
                }
                image.metadata.setdefault('geometric_operations', []).append(
                    GeometricOperationMetadata('resize', resize_meta)
                )
                image.metadata.update(resize_meta)

            return data

        data = image.data
        image.data = (
            process_data(data) if is_simple_case else [
                process_data(data_fragment, idx) for idx, data_fragment in enumerate(data)
            ]
        )

        return image
