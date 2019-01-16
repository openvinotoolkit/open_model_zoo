"""
Copyright (c) 2018 Intel Corporation

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

from ..config import BaseField, BoolField, ConfigValidator, NumberField, StringField
from ..dependency import ClassProvider
from ..utils import get_size_from_config, get_or_parse_value


class BasePreprocessorConfig(ConfigValidator):
    type = StringField()


class Preprocessor(ClassProvider):
    __provider_type__ = 'preprocessor'

    def __init__(self, config, name=None):
        self.config = config
        self.name = name

        self.validate_config()
        self.configure()

    def __call__(self, *args, **kwargs):
        return self.process(*args, **kwargs)

    def process(self, image, annotation_meta=None):
        raise NotImplementedError

    def configure(self):
        pass

    def validate_config(self):
        BasePreprocessorConfig(self.name,
                               on_extra_argument=BasePreprocessorConfig.ERROR_ON_EXTRA_ARGUMENT).validate(self.config)


def scale_width(dst_width, dst_height, image_width, image_height,):
    return int(dst_width * image_width / image_height), dst_height


def scale_height(dst_width, dst_height, image_width, image_height):
    return dst_width, int(dst_height * image_height / image_width)


def scale_greater(dst_width, dst_height, image_width, image_height):
    if image_height > image_width:
        return scale_height(dst_width, dst_height, image_width, image_height)
    return scale_width(dst_width, dst_height, image_width, image_height)

class Resize(Preprocessor):
    __provider__ = 'resize'

    PIL_INTERPOLATION = {
        'NEAREST': Image.NEAREST,
        'NONE': Image.NONE,
        'BOX': Image.BOX,
        'BILINEAR': Image.BILINEAR,
        'LINEAR': Image.LINEAR,
        'HAMMING': Image.HAMMING,
        'BICUBIC': Image.BICUBIC,
        'CUBIC': Image.CUBIC,
        'LANCZOS': Image.LANCZOS,
        'ANTIALIAS': Image.ANTIALIAS,
    }

    OPENCV_INTERPOLATION = {
        'NEAREST': cv2.INTER_NEAREST,
        'LINEAR': cv2.INTER_LINEAR,
        'CUBIC': cv2.INTER_CUBIC,
        'AREA': cv2.INTER_AREA,
        'MAX': cv2.INTER_MAX,
        'BITS': cv2.INTER_BITS,
        'BITS2': cv2.INTER_BITS2,
        'LANCZOS4': cv2.INTER_LANCZOS4,
    }

    ASPECT_RATIO_SCALE = {
        'width': scale_width,
        'height': scale_height,
        'greater': scale_greater,
    }

    def validate_config(self):
        class _ConfigValidator(BasePreprocessorConfig):
            size = NumberField(floats=False, optional=True)
            dst_width = NumberField(floats=False, optional=True)
            dst_height = NumberField(floats=False, optional=True)
            aspect_ratio_scale = StringField(choices=set(Resize.ASPECT_RATIO_SCALE), optional=True)
            interpolation = StringField(choices=set(Resize.PIL_INTERPOLATION) | set(Resize.OPENCV_INTERPOLATION),
                                        optional=True)
            use_pil = BoolField(optional=True)

        _ConfigValidator(self.name, on_extra_argument=_ConfigValidator.ERROR_ON_EXTRA_ARGUMENT).validate(self.config)

    def configure(self):
        self.dst_height, self.dst_width = get_size_from_config(self.config)
        self.use_pil = self.config.get('use_pil', False)

        interpolation = self.config.get('interpolation', 'LINEAR')

        self.scaling_func = Resize.ASPECT_RATIO_SCALE.get(self.config.get('aspect_ratio_scale'))

        if self.use_pil and interpolation.upper() not in Resize.PIL_INTERPOLATION:
            raise ValueError("Incorrect interpolation option: {} for resize preprocessing".format(interpolation))
        if not self.use_pil and interpolation.upper() not in Resize.OPENCV_INTERPOLATION:
            raise ValueError("Incorrect interpolation option: {} for resize preprocessing".format(interpolation))

        if self.use_pil:
            self.interpolation = Resize.PIL_INTERPOLATION[interpolation]
        else:
            self.interpolation = Resize.OPENCV_INTERPOLATION[interpolation]

    def process(self, image, annotation_meta=None):
        image_ = image.data
        new_height, new_width = self.dst_height, self.dst_width
        if self.scaling_func is not None:
            image_h, image_w = image_.shape[:2]
            new_width, new_height = self.scaling_func(self.dst_width, self.dst_height, image_w, image_h)

        if self.use_pil:
            image_ = Image.fromarray(image_)
            image_ = image_.resize((new_width, new_height), self.interpolation)
            image.data = np.array(image_)
            return image
        image.data = cv2.resize(image_, (new_width, new_height), interpolation=self.interpolation).astype(np.float32)
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

    def validate_config(self):
        class _ConfigValidator(BasePreprocessorConfig):
            mean = BaseField(optional=True)
            std = BaseField(optional=True)

        _ConfigValidator(self.name, on_extra_argument=_ConfigValidator.ERROR_ON_EXTRA_ARGUMENT).validate(self.config)

    def configure(self):
        self.mean = get_or_parse_value(self.config.get('mean'), Normalize.PRECOMPUTED_MEANS)
        self.std = get_or_parse_value(self.config.get('std'), Normalize.PRECOMPUTED_STDS)

    def process(self, image, annotation_meta=None):
        if self.mean is not None:
            image.data = image.data - self.mean
        if self.std is not None:
            image.data = image.data / self.std
        return image


class BgrToRgb(Preprocessor):
    __provider__ = 'bgr_to_rgb'

    def process(self, image, annotation_meta=None):
        image.data = cv2.cvtColor(image.data, cv2.COLOR_BGR2RGB)
        return image


class Flip(Preprocessor):
    __provider__ = 'flip'

    FLIP_MODES = {
        'horizontal': 0,
        'vertical': 1
    }

    def validate_config(self):
        class _ConfigValidator(BasePreprocessorConfig):
            mode = StringField(choices=Flip.FLIP_MODES.keys())

        _ConfigValidator(self.name, on_extra_argument=_ConfigValidator.ERROR_ON_EXTRA_ARGUMENT).validate(self.config)

    def configure(self):
        mode = self.config.get('mode', 'horizontal')
        if isinstance(mode, str):
            self.mode = Flip.FLIP_MODES[mode]

    def process(self, image, annotation_meta=None):
        image.data = cv2.flip(image.data, self.mode)
        return image


class Crop(Preprocessor):
    __provider__ = 'crop'

    def validate_config(self):
        class _ConfigValidator(BasePreprocessorConfig):
            size = NumberField(floats=False, optional=True)
            dst_width = NumberField(floats=False, optional=True)
            dst_height = NumberField(floats=False, optional=True)

        _ConfigValidator(self.name, on_extra_argument=_ConfigValidator.ERROR_ON_EXTRA_ARGUMENT).validate(self.config)

    def configure(self):
        self.dst_height, self.dst_width = get_size_from_config(self.config)

    def process(self, image, annotation_meta=None):
        image_ = image.data
        height, width, _ = image_.shape
        if width < self.dst_width or height < self.dst_height:
            resized = np.array([width, height])
            if resized[0] < self.dst_width:
                resized = resized * self.dst_width / resized[0]
            if resized[1] < self.dst_height:
                resized = resized * self.dst_height / resized[1]

            image_ = cv2.resize(image_, tuple(np.ceil(resized).astype(int)))

        height, width, _ = image_.shape
        start_height = (height - self.dst_height) // 2
        start_width = (width - self.dst_width) // 2

        image.data = image_[start_height:start_height + self.dst_height, start_width:start_width + self.dst_width]
        return image


class CropRect(Preprocessor):
    __provider__ = 'crop_rect'

    def process(self, image, annotation_meta=None):
        rect = annotation_meta.get('rect')
        rows, cols = image.data.shape[:2]
        if rect is None:
            return image
        rect_x_min, rect_y_min, rect_x_max, rect_y_max = rect
        start_width = max(0, rect_x_min)
        start_height = max(0, rect_y_min)
        width = min(start_width + (rect_x_max - rect_x_min), cols)
        height = min(start_height + (rect_y_max - rect_y_min), rows)
        image.data = image.data[start_height:height, start_width:width]
        return image


class ExtendAroundRect(Preprocessor):
    __provider__ = 'extend_around_rect'

    def validate_config(self):
        class _ConfigValidator(BasePreprocessorConfig):
            augmentation_param = NumberField(floats=True, optional=True)

        _ConfigValidator(self.name, on_extra_argument=_ConfigValidator.ERROR_ON_EXTRA_ARGUMENT).validate(self.config)

    def configure(self):
        self.augmentation_param = self.config.get('augmentation_param', 0)

    def process(self, image, annotation_meta=None):
        rect = annotation_meta.get('rect')
        rows, cols = image.data.shape[:2]

        rect_x_left, rect_y_top, rect_x_right, rect_y_bottom = rect
        if None in rect:
            rect_x_left = 0
            rect_y_top = 0
            rect_x_right = cols
            rect_y_bottom = rows
        else:
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

        image.data = cv2.copyMakeBorder(image.data, int(border_top), int(border_bottom),
                                        int(border_left), int(border_right), cv2.BORDER_REPLICATE)

        rect = (int(rect_x_left), int(rect_y_top), int(rect_x_left) + int(rect_w + width_extent * 2),
                int(rect_y_top) + int(rect_h + height_extent * 2))
        annotation_meta['rect'] = rect
        return image


class PointAligner(Preprocessor):
    __provider__ = 'point_aligment'
    ref_landmarks = np.array([30.2946 / 96, 51.6963 / 112,
                              65.5318 / 96, 51.5014 / 112,
                              48.0252 / 96, 71.7366 / 112,
                              33.5493 / 96, 92.3655 / 112,
                              62.7299 / 96, 92.2041 / 112],
                             dtype=np.float64).reshape(5, 2)

    def validate_config(self):
        class _ConfigValidator(BasePreprocessorConfig):
            draw_points = BoolField(optional=True)
            normalize = BoolField(optional=True)
            size = NumberField(floats=False, optional=True)
            dst_width = NumberField(floats=False, optional=True)
            dst_height = NumberField(floats=False, optional=True)

        _ConfigValidator(self.name, on_extra_argument=_ConfigValidator.ERROR_ON_EXTRA_ARGUMENT).validate(self.config)

    def configure(self):
        self.draw_points = self.config.get('draw_points', False)
        self.normalize = self.config.get('normalize', True)
        self.dst_height, self.dst_width = get_size_from_config(self.config)

    def process(self, image, annotation_meta=None):
        keypoints = annotation_meta.get('keypoints')
        image.data = self.align(image.data, keypoints)
        return image

    def align(self, img, points):
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
        points1 /= s1
        points2 /= s2

        U, _, Vt = np.linalg.svd(points1.T * points2)
        R = (U * Vt).T

        return np.hstack(((s2 / s1) * R, c2.T - (s2 / s1) * R * c1.T))
