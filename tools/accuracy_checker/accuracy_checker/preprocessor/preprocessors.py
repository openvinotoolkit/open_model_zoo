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

import math
from collections import namedtuple
import cv2
import numpy as np
from PIL import Image

from ..config import BoolField, ConfigValidator, NumberField, StringField, ConfigError, BaseField
from ..dependency import ClassProvider
from ..utils import get_size_from_config, get_or_parse_value, string_to_tuple, get_size_3d_from_config, contains_all
from ..utils import get_parameter_value_from_config
from ..logging import warning

# The field .type should be string, the field .parameters should be dict
GeometricOperationMetadata = namedtuple('GeometricOperationMetadata',
                                        ['type', 'parameters'])


class Preprocessor(ClassProvider):
    __provider_type__ = 'preprocessor'

    def __init__(self, config, name=None, input_shapes=None):
        self.config = config
        self.name = name
        self.input_shapes = input_shapes

        self.validate_config()
        self.configure()

    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)

    def get_value_from_config(self, key):
        return get_parameter_value_from_config(self.config, self.parameters(), key)

    @classmethod
    def parameters(cls):
        return {
            'type': StringField(
                default=cls.__provider__ if hasattr(cls, '__provider__') else None, description="Preprocessor type."
            )
        }

    def process(self, image, annotation_meta=None):
        raise NotImplementedError

    def configure(self):
        pass

    def validate_config(self):
        ConfigValidator(self.name,
                        on_extra_argument=ConfigValidator.ERROR_ON_EXTRA_ARGUMENT,
                        fields=self.parameters()).validate(self.config)


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


ASPECT_RATIO_SCALE = {
    'width': scale_width,
    'height': scale_height,
    'greater': scale_greater,
    'fit_to_window': scale_fit_to_window,
    'frcnn_keep_aspect_ratio': frcnn_keep_aspect_ratio
}


class _Resizer(ClassProvider):
    __provider_type__ = 'resizer'

    supported_interpolations = {}
    default_interpolation = None

    def __init__(self, interpolation=None):
        if not interpolation:
            interpolation = self.default_interpolation
        if interpolation.upper() not in self.supported_interpolations:
            raise ConfigError('{} not found for {}'.format(self.supported_interpolations, self.__provider__))
        self.interpolation = self.supported_interpolations.get(interpolation.upper(), self.default_interpolation)

    def resize(self, data, new_height, new_width):
        raise NotImplementedError

    def __call__(self, data, new_height, new_width):
        return self.resize(data, new_height, new_width)

    @classmethod
    def all_provided_interpolations(cls):
        interpolations = set()
        for _, provider_class in cls.providers.items():
            try:
                interpolations.update(provider_class.supported_interpolations)
            except ImportError:
                continue
        return interpolations


class _OpenCVResizer(_Resizer):
    __provider__ = 'opencv'

    supported_interpolations = {
        'NEAREST': cv2.INTER_NEAREST,
        'LINEAR': cv2.INTER_LINEAR,
        'CUBIC': cv2.INTER_CUBIC,
        'AREA': cv2.INTER_AREA,
        'MAX': cv2.INTER_MAX,
        'BITS': cv2.INTER_BITS,
        'BITS2': cv2.INTER_BITS2,
        'LANCZOS4': cv2.INTER_LANCZOS4,
    }
    default_interpolation = 'LINEAR'

    def resize(self, data, new_height, new_width):
        return cv2.resize(data, (new_width, new_height), interpolation=self.interpolation).astype(np.float32)


class _PillowResizer(_Resizer):
    __provider__ = 'pillow'

    supported_interpolations = {
        'NEAREST': Image.NEAREST,
        'NONE': Image.NONE,
        'BOX': Image.BOX,
        'BILINEAR': Image.BILINEAR,
        'LINEAR': Image.LINEAR,
        'HAMMING': Image.HAMMING,
        'BICUBIC': Image.BICUBIC,
        'CUBIC': Image.CUBIC,
        'LANCZOS': Image.LANCZOS,
        'ANTIALIAS': Image.ANTIALIAS
    }
    default_interpolation = 'BILINEAR'

    def resize(self, data, new_height, new_width):
        data = Image.fromarray(data)
        data = data.resize((new_width, new_height), self.interpolation)
        data = np.array(data)

        return data


class _TFResizer(_Resizer):
    __provider__ = 'tf'

    def __init__(self, interpolation):
        try:
            import tensorflow as tf
        except ImportError as import_error:
            raise ImportError(
                'tf resize disabled. Please, install Tensorflow before using. \n{}'.format(import_error.msg)
            )
        tf.enable_eager_execution()
        self.supported_interpolations = {
            'BILINEAR': tf.image.ResizeMethod.BILINEAR,
            'AREA': tf.image.ResizeMethod.AREA,
            'BICUBIC': tf.image.ResizeMethod.BICUBIC,
        }
        self.default_interpolation = 'BILINEAR'
        self._resize = tf.image.resize_images

        super().__init__(interpolation)

    def resize(self, data, new_height, new_width):
        resized_data = self._resize(data, [new_height, new_width], method=self.interpolation)
        return resized_data.numpy()


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
            )
        })

        return parameters

    def configure(self):
        self.dst_height, self.dst_width = get_size_from_config(self.config)
        self.resizer = create_resizer(self.config)
        self.scaling_func = ASPECT_RATIO_SCALE.get(self.get_value_from_config('aspect_ratio_scale'))

    def process(self, image, annotation_meta=None):
        data = image.data
        new_height, new_width = self.dst_height, self.dst_width

        is_simple_case = not isinstance(data, list) # otherwise -- pyramid, tiling, etc

        def process_data(data, new_height, new_width, scale_func, resize_func):
            dst_width, dst_height = new_width, new_height
            image_h, image_w = data.shape[:2]
            if scale_func:
                dst_width, dst_height = scale_func(new_width, new_height, image_w, image_h)

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
                image.metadata.setdefault('geometric_operations', []).append(GeometricOperationMetadata('resize',
                                                                                                        resize_meta))

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

    def configure(self):
        if self.input_shapes is None or len(self.input_shapes) != 1:
            raise ConfigError('resize to input size possible, only for one input layer case')
        input_shape = next(iter(self.input_shapes.values()))
        self.dst_height, self.dst_width = input_shape[2:]

    def process(self, image, annotation_meta=None):
        is_simple_case = not isinstance(image.data, list) # otherwise -- pyramid, tiling, etc

        def process_data(data):
            data = cv2.resize(data, (self.dst_width, self.dst_height)).astype(np.float32)
            if len(data.shape) == 2:
                data = np.expand_dims(data, axis=-1)

            if is_simple_case:
                # support GeometricOperationMetadata array for simple case only -- without tiling, pyramids, etc
                image.metadata.setdefault('geometric_operations', []).append(GeometricOperationMetadata('auto_resize',
                                                                                                        {}))

            return data

        data = image.data
        image.data = (
            process_data(data) if is_simple_case else [
                process_data(data_fragment)for data_fragment in data
            ]
        )

        return image

class Normalize(Preprocessor):
    __provider__ = 'normalization'

    PRECOMPUTED_MEANS = {
        'imagenet': (104.00698793, 116.66876762, 122.67891434),
        'cifar10': (125.307, 122.961, 113.8575),
    }

    PRECOMPUTED_STDS = {
        'imagenet': (104.00698793, 116.66876762, 122.67891434),
        'cifar10': (125.307, 122.961, 113.8575),
    }

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'mean': BaseField(
                optional=True,
                description="Values which will be subtracted from image channels. You can specify one "
                            "value for all channels or list of comma separated channel-wise values."
            ),
            'std': BaseField(
                optional=True,
                description="Specifies values, on which pixels will be divided. You can specify one value for all "
                            "channels or list of comma separated channel-wise values."
            )
        })
        return parameters

    def configure(self):
        self.mean = get_or_parse_value(self.config.get('mean'), Normalize.PRECOMPUTED_MEANS)
        self.std = get_or_parse_value(self.config.get('std'), Normalize.PRECOMPUTED_STDS)
        if not self.mean and not self.std:
            raise ConfigError('mean or std value should be provided')

        if self.std and 0 in self.std:
            raise ConfigError('std value should not contain 0')

        if self.mean and not (len(self.mean) == 3 or len(self.mean) == 1):
            raise ConfigError('mean should be one value or comma-separated list channel-wise values')

        if self.std and not (len(self.std) == 3 or len(self.std) == 1):
            raise ConfigError('std should be one value or comma-separated list channel-wise values')

    def process(self, image, annotation_meta=None):
        def process_data(data, mean, std):
            if self.mean:
                data = data - mean
            if self.std:
                data = data / std

            return data

        image.data = process_data(image.data, self.mean, self.std) if not isinstance(image.data, list) else [
            process_data(data_fragment, self.mean, self.std) for data_fragment in image.data
        ]

        return image


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

FLIP_MODES = {
    'horizontal': 0,
    'vertical': 1
}


class Flip(Preprocessor):
    __provider__ = 'flip'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'mode' : StringField(choices=FLIP_MODES.keys(), default='horizontal',
                                 description="Specifies the axis for flipping (vertical or horizontal).")
        })
        return parameters

    def configure(self):
        mode = self.get_value_from_config('mode')
        if isinstance(mode, str):
            self.mode = FLIP_MODES[mode]

    def process(self, image, annotation_meta=None):
        image.data = cv2.flip(image.data, self.mode)
        image.metadata.setdefault(
            'geometric_operations', []).append(GeometricOperationMetadata('flip', {'mode': self.mode}))
        return image


class Crop(Preprocessor):
    __provider__ = 'crop'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'size': NumberField(
                value_type=int, optional=True, min_value=1,
                description="Destination size for cropping both dimensions."
            ),
            'dst_width': NumberField(
                value_type=int, optional=True, min_value=1,
                description="Destination width for image cropping respectively."
            ),
            'dst_height': NumberField(
                value_type=int, optional=True, min_value=1,
                description="Destination height for image cropping respectively."
            ),
            'use_pillow': BoolField(
                optional=True, default=False, description="Parameter specifies usage of Pillow library for cropping."
            ),
            'central_fraction' : NumberField(
                value_type=float, min_value=0, max_value=1, optional=True, description="Central Fraction."
            )
        })

        return parameters

    def configure(self):
        self.use_pillow = self.get_value_from_config('use_pillow')
        self.dst_height, self.dst_width = get_size_from_config(self.config, allow_none=True)
        self.central_fraction = self.get_value_from_config('central_fraction')
        if self.dst_height is None and self.dst_width is None and self.central_fraction is None:
            raise ConfigError('sizes for crop or central_fraction should be provided')
        if self.dst_height and self.dst_width and self.central_fraction:
            raise ConfigError('both sizes and central fraction provided  for cropping')

        if not self.central_fraction:
            if self.dst_height is None or self.dst_width is None:
                raise ConfigError('one from crop dimentions is not provided')

    def process(self, image, annotation_meta=None):
        is_simple_case = not isinstance(image.data, list) # otherwise -- pyramid, tiling, etc
        data = image.data

        def process_data(data, dst_height, dst_width, central_fraction, use_pillow):
            height, width = data.shape[:2]
            if not central_fraction:
                new_height = dst_height
                new_width = dst_width
            else:
                new_height = int(height * central_fraction)
                new_width = int(width * central_fraction)

            if use_pillow:
                i = int(round((height - new_height) / 2.))
                j = int(round((width - new_width) / 2.))
                cropped_data = Image.fromarray(data).crop((j, i, j + new_width, i + new_height))
                return np.array(cropped_data)

            if width < new_width or height < new_height:
                resized = np.array([width, height])
                if resized[0] < new_width:
                    resized = resized * new_width / resized[0]
                if resized[1] < new_height:
                    resized = resized * new_height / resized[1]

                data = cv2.resize(data, tuple(np.ceil(resized).astype(int)))

            height, width = data.shape[:2]
            start_height = (height - new_height) // 2
            start_width = (width - new_width) // 2

            if is_simple_case:
                # support GeometricOperationMetadata array for simple case only -- without tiling, pyramids, etc
                image.metadata.setdefault('geometric_operations', []).append(GeometricOperationMetadata('crop', {}))

            return data[start_height:start_height + new_height, start_width:start_width + new_width]

        image.data = process_data(
            data, self.dst_height, self.dst_width, self.central_fraction, self.use_pillow
        ) if not isinstance(data, list) else [
            process_data(
                fragment, self.dst_height, self.dst_width, self.central_fraction, self.use_pillow
            ) for fragment in image.data
        ]

        return image


class CropRect(Preprocessor):
    __provider__ = 'crop_rect'

    def process(self, image, annotation_meta=None):
        rect = annotation_meta.get('rect')
        if not rect:
            return image

        rows, cols = image.data.shape[:2]
        rect_x_min, rect_y_min, rect_x_max, rect_y_max = rect
        start_width, start_height = max(0, rect_x_min), max(0, rect_y_min)

        width = min(start_width + (rect_x_max - rect_x_min), cols)
        height = min(start_height + (rect_y_max - rect_y_min), rows)

        image.data = image.data[start_height:height, start_width:width]
        image.metadata.setdefault('geometric_operations', []).append(GeometricOperationMetadata('crop_rect', {}))
        return image

class ExtendAroundRect(Preprocessor):
    __provider__ = 'extend_around_rect'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'augmentation_param' : NumberField(
                value_type=float, optional=True, default=0, description="Scale factor for augmentation."
            )
        })
        return parameters

    def configure(self):
        self.augmentation_param = self.get_value_from_config('augmentation_param')

    def process(self, image, annotation_meta=None):
        rect = annotation_meta.get('rect')
        rows, cols = image.data.shape[:2]

        rect_x_left, rect_y_top, rect_x_right, rect_y_bottom = rect or (0, 0, cols, rows)
        rect_x_left = max(0, rect_x_left)
        rect_y_top = max(0, rect_y_top)
        rect_x_right = min(rect_x_right, cols)
        rect_y_bottom = min(rect_y_bottom, rows)

        rect_w = rect_x_right - rect_x_left
        rect_h = rect_y_bottom - rect_y_top

        width_extent = (rect_x_right - rect_x_left + 1) * self.augmentation_param
        height_extent = (rect_y_bottom - rect_y_top + 1) * self.augmentation_param
        rect_x_left = rect_x_left - width_extent
        border_left = abs(min(0, rect_x_left))
        rect_x_left = int(max(0, rect_x_left))

        rect_y_top = rect_y_top - height_extent
        border_top = abs(min(0, rect_y_top))
        rect_y_top = int(max(0, rect_y_top))

        rect_y_bottom += border_top
        rect_y_bottom = int(rect_y_bottom + height_extent + 0.5)
        border_bottom = abs(max(0, rect_y_bottom - rows))

        rect_x_right += border_left
        rect_x_right = int(rect_x_right + width_extent + 0.5)
        border_right = abs(max(0, rect_x_right - cols))

        image.data = cv2.copyMakeBorder(
            image.data, int(border_top), int(border_bottom), int(border_left), int(border_right), cv2.BORDER_REPLICATE
        )

        rect = (
            int(rect_x_left), int(rect_y_top),
            int(rect_x_left) + int(rect_w + width_extent * 2), int(rect_y_top) + int(rect_h + height_extent * 2)
        )
        annotation_meta['rect'] = rect

        image.metadata.setdefault('geometric_operations', []).append(
            GeometricOperationMetadata('extend_around_rect', {})
        )

        return image

class PointAligner(Preprocessor):
    __provider__ = 'point_alignment'

    ref_landmarks = np.array([
        30.2946 / 96, 51.6963 / 112,
        65.5318 / 96, 51.5014 / 112,
        48.0252 / 96, 71.7366 / 112,
        33.5493 / 96, 92.3655 / 112,
        62.7299 / 96, 92.2041 / 112
    ], dtype=np.float64).reshape(5, 2)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'draw_points': BoolField(optional=True, default=False, description="Allows visualize points."),
            'normalize': BoolField(
                optional=True, default=True, description="Allows to use normalization for keypoints."),
            'size': NumberField(
                value_type=int, optional=True, min_value=1,
                description="Destination size for keypoints resizing for both destination dimentions."
            ),
            'dst_width': NumberField(
                value_type=int, optional=True, min_value=1, description="Destination width for keypoints resizing."
            ),
            'dst_height': NumberField(
                value_type=int, optional=True, min_value=1, description="Destination height for keypoints resizing."
            ),
        })

        return parameters

    def configure(self):
        self.draw_points = self.get_value_from_config('draw_points')
        self.normalize = self.get_value_from_config('normalize')
        self.dst_height, self.dst_width = get_size_from_config(self.config)

    def process(self, image, annotation_meta=None):
        keypoints = annotation_meta.get('keypoints')
        image.data = self.align(image.data, keypoints)
        image.metadata.setdefault('geometric_operations', []).append(GeometricOperationMetadata('point_alignment', {}))
        return image

    def align(self, img, points):
        if not points:
            return img

        points_number = len(points) // 2
        points = np.array(points).reshape(points_number, 2)

        inp_shape = [1., 1.]
        if self.normalize:
            inp_shape = img.shape

        keypoints = points.copy().astype(np.float64)
        keypoints[:, 0] *= (float(self.dst_width) / inp_shape[1])
        keypoints[:, 1] *= (float(self.dst_height) / inp_shape[0])

        keypoints_ref = np.zeros((points_number, 2), dtype=np.float64)
        keypoints_ref[:, 0] = self.ref_landmarks[:, 0] * self.dst_width
        keypoints_ref[:, 1] = self.ref_landmarks[:, 1] * self.dst_height

        transformation_matrix = self.transformation_from_points(np.array(keypoints_ref), np.array(keypoints))
        img = cv2.resize(img, (self.dst_width, self.dst_height))
        if self.draw_points:
            for point in keypoints:
                cv2.circle(img, (int(point[0]), int(point[1])), 5, (255, 0, 0), -1)

        return cv2.warpAffine(img, transformation_matrix, (self.dst_width, self.dst_height), flags=cv2.WARP_INVERSE_MAP)

    @staticmethod
    def transformation_from_points(points1, points2):
        points1 = np.matrix(points1.astype(np.float64))
        points2 = np.matrix(points2.astype(np.float64))

        c1 = np.mean(points1, axis=0)
        c2 = np.mean(points2, axis=0)
        points1 -= c1
        points2 -= c2
        s1 = np.std(points1)
        s2 = np.std(points2)
        points1 /= np.maximum(s1, np.finfo(np.float64).eps)
        points2 /= np.maximum(s1, np.finfo(np.float64).eps)
        points_std_ratio = s2 / np.maximum(s1, np.finfo(np.float64).eps)

        u, _, vt = np.linalg.svd(points1.T * points2)
        r = (u * vt).T

        return np.hstack((points_std_ratio * r, c2.T - points_std_ratio * r * c1.T))


def center_padding(dst_width, dst_height, width, height):
    pad = [int(math.floor((dst_height - height) / 2.0)), int(math.floor((dst_width - width) / 2.0))]
    pad.extend([dst_height - height - pad[0], dst_width - width - pad[1]])

    return pad


def right_bottom_padding(dst_width, dst_height, width, height):
    return [0, 0, dst_height - height, dst_width - width]


def left_top_padding(dst_width, dst_height, width, height):
    return [dst_height - height, dst_width - width, 0, 0]


padding_func = {
    'center': center_padding,
    'left_top': left_top_padding,
    'right_bottom': right_bottom_padding
}


class Padding(Preprocessor):
    __provider__ = 'padding'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'stride': NumberField(
                value_type=int, min_value=1, optional=True, default=1, description="Stride for padding."
            ),
            'pad_value': StringField(
                optional=True, default='0,0,0', description="Value for filling space around original image."
            ),
            'size': NumberField(
                value_type=int, optional=True, min_value=1,
                description="Destination size for padded image for both dimensions."),
            'dst_width': NumberField(
                value_type=int, optional=True, min_value=1, description="Destination width for padded image."
            ),
            'dst_height': NumberField(
                value_type=int, optional=True, min_value=1, description="Destination height for padded image."
            ),
            'pad_type': StringField(
                choices=padding_func.keys(), optional=True, default='center',
                description="Padding space location. Supported: {}".format(', '.join(padding_func))
            ),
            'use_numpy': BoolField(
                optional=True, default=False, description="Allow to use numpy for padding instead default OpenCV."
            )
        })

        return parameters

    def configure(self):
        self.stride = self.get_value_from_config('stride')
        pad_val = self.get_value_from_config('pad_value')
        if isinstance(pad_val, int):
            self.pad_value = (pad_val, pad_val, pad_val)
        if isinstance(pad_val, str):
            self.pad_value = string_to_tuple(pad_val, int)
        self.dst_height, self.dst_width = get_size_from_config(self.config, allow_none=True)
        self.pad_func = padding_func[self.get_value_from_config('pad_type')]
        self.use_numpy = self.get_value_from_config('use_numpy')

    def process(self, image, annotation_meta=None):
        height, width, _ = image.data.shape
        pref_height = self.dst_height or image.metadata.get('preferable_height', height)
        pref_width = self.dst_width or image.metadata.get('preferable_width', width)
        height = min(height, pref_height)
        pref_height = math.ceil(pref_height / float(self.stride)) * self.stride
        pref_width = max(pref_width, width)
        pref_width = math.ceil(pref_width / float(self.stride)) * self.stride
        pad = self.pad_func(pref_width, pref_height, width, height)
        image.metadata['padding'] = pad
        padding_realization_func = self._opencv_padding if not self.use_numpy else self._numpy_padding
        image.data = padding_realization_func(image.data, pad)

        image.metadata.setdefault('geometric_operations', []).append(
            GeometricOperationMetadata('padding',
                                       {
                                           'pad': pad,
                                           'dst_width': self.dst_width,
                                           'dst_height': self.dst_height,
                                           'pref_width': pref_width,
                                           'pref_height': pref_height,
                                           'width': width,
                                           'height': height
                                       }))

        return image

    def _opencv_padding(self, image, pad):
        return cv2.copyMakeBorder(
            image, pad[0], pad[2], pad[1], pad[3], cv2.BORDER_CONSTANT, value=self.pad_value
        )

    def _numpy_padding(self, image, pad):
        pad_values = (
            (self.pad_value[0], self.pad_value[0]),
            (self.pad_value[1], self.pad_value[1]),
            (self.pad_value[2], self.pad_value[2])
        )
        return np.pad(
            image, ((pad[0], pad[2]), (pad[1], pad[3]), (0, 0)),
            mode='constant', constant_values=pad_values
        )

class Tiling(Preprocessor):
    __provider__ = 'tiling'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'margin': NumberField(value_type=int, min_value=1, description="Margin for tiled fragment of image."),
            'size': NumberField(
                value_type=int, optional=True, min_value=1,
                description="Destination size of tiled fragment for both dimentions."
            ),
            'dst_width'  : NumberField(
                value_type=int, optional=True, min_value=1, description="Destination width of tiled fragment."
            ),
            'dst_height' : NumberField(
                value_type=int, optional=True, min_value=1, description="Destination height of tiled fragment."
            ),
        })
        return parameters

    def configure(self):
        self.dst_height, self.dst_width = get_size_from_config(self.config)
        self.margin = self.get_value_from_config('margin')

    def process(self, image, annotation_meta=None):
        data = image.data
        image_size = data.shape
        output_height = self.dst_height - 2 * self.margin
        output_width = self.dst_width - 2 * self.margin
        data = cv2.copyMakeBorder(data, *np.full(4, self.margin), cv2.BORDER_REFLECT_101)
        num_tiles_h = image_size[0] // output_height + (1 if image_size[0] % output_height else 0)
        num_tiles_w = image_size[1] // output_width + (1 if image_size[1] % output_width else 0)
        tiled_data = []
        for height in range(num_tiles_h):
            for width in range(num_tiles_w):
                offset = [output_height * height, output_width * width]
                tile = data[offset[0]:offset[0] + self.dst_height, offset[1]:offset[1] + self.dst_width, :]
                margin = [0, self.dst_height - tile.shape[0], 0, self.dst_width - tile.shape[1]]
                tile = cv2.copyMakeBorder(tile, *margin, cv2.BORDER_REFLECT_101)
                tiled_data.append(tile)
        image.data = tiled_data
        image.metadata['tiles_shape'] = (num_tiles_h, num_tiles_w)
        image.metadata['multi_infer'] = True

        image.metadata.setdefault('geometric_operations', []).append(
            GeometricOperationMetadata('tiling', {'tiles_shape': (num_tiles_h, num_tiles_w)}))

        return image

class Crop3D(Preprocessor):
    __provider__ = 'crop3d'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'size': NumberField(
                value_type=int, optional=True, min_value=1,
                description="Destination size for 3d crop for all dimentions."
            ),
            'dst_width': NumberField(
                value_type=int, optional=True, min_value=1, description="Destination width for 3d crop."
            ),
            'dst_height': NumberField(
                value_type=int, optional=True, min_value=1, description="Destination height for 3d crop."
            ),
            'dst_volume': NumberField(
                value_type=int, optional=True, min_value=1, description="Destination volume for 3d crop."
            )
        })

        return parameters

    def configure(self):
        self.dst_height, self.dst_width, self.dst_volume = get_size_3d_from_config(self.config)

    def process(self, image, annotation_meta=None):
        image.data = self.crop_center(image.data, self.dst_height, self.dst_width, self.dst_volume)
        image.metadata.setdefault('geometric_operations', []).append(GeometricOperationMetadata('crop3d', {}))
        return image

    @staticmethod
    def crop_center(img, cropx, cropy, cropz):

        z, y, x, _ = img.shape

        # Make sure starting index is >= 0
        startx = max(x // 2 - (cropx // 2), 0)
        starty = max(y // 2 - (cropy // 2), 0)
        startz = max(z // 2 - (cropz // 2), 0)

        # Make sure ending index is <= size
        endx = min(startx + cropx, x)
        endy = min(starty + cropy, y)
        endz = min(startz + cropz, z)

        return img[startz:endz, starty:endy, startx:endx, :]


class Normalize3d(Preprocessor):
    __provider__ = "normalize3d"

    def process(self, image, annotation_meta=None):
        data = self.normalize_img(image.data)
        image_list = []
        for img in data:
            image_list.append(img)
        image.data = image_list
        image.metadata['multi_infer'] = True

        return image

    @staticmethod
    def normalize_img(img):
        for channel in range(img.shape[3]):
            channel_val = img[:, :, :, channel] - np.mean(img[:, :, :, channel])
            channel_val /= np.std(img[:, :, :, channel])
            img[:, :, :, channel] = channel_val

        return img


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
