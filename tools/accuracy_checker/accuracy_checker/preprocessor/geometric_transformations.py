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

from ..config import ConfigError, NumberField, StringField, BoolField
from ..preprocessor import Preprocessor
from ..utils import get_size_from_config, string_to_tuple, get_size_3d_from_config
from ..logging import warning

try:
    from PIL import Image
except ImportError:
    Image = None

# The field .type should be string, the field .parameters should be dict
GeometricOperationMetadata = namedtuple('GeometricOperationMetadata', ['type', 'parameters'])

FLIP_MODES = {'horizontal': 0, 'vertical': 1}


class Flip(Preprocessor):
    __provider__ = 'flip'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'mode': StringField(
                choices=FLIP_MODES.keys(), default='horizontal',
                description="Specifies the axis for flipping (vertical or horizontal)."
            )
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
        if self.use_pillow and Image is None:
            raise ValueError(
                'Crop operation with pillow backend, requires Pillow. Please install it or select default backend'
            )
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

        image.data = self.process_data(
            data, self.dst_height, self.dst_width, self.central_fraction,
            self.use_pillow, is_simple_case, image.metadata
        ) if not isinstance(data, list) else [
            self.process_data(
                fragment, self.dst_height, self.dst_width, self.central_fraction,
                self.use_pillow, is_simple_case, image.metadata
            ) for fragment in image.data
        ]

        return image

    @staticmethod
    def process_data(data, dst_height, dst_width, central_fraction, use_pillow, is_simple_case, metadata):
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
            metadata.setdefault('geometric_operations', []).append(GeometricOperationMetadata('crop', {}))

        return data[start_height:start_height + new_height, start_width:start_width + new_width]


class CropRect(Preprocessor):
    __provider__ = 'crop_rect'

    def process(self, image, annotation_meta=None):
        if not annotation_meta:
            warning('operation *crop_rect* required annotation metadata')
            return image
        rect = annotation_meta.get('rect')
        if not rect:
            warning(
                'operation *crop_rect* rect key in annotation meta, please use annotation converter '
                'which allows such transformation'
            )
            return image

        rows, cols = image.data.shape[:2]
        rect_x_min, rect_y_min, rect_x_max, rect_y_max = rect
        start_width, start_height = max(0, rect_x_min), max(0, rect_y_min)

        width = min(start_width + (rect_x_max - rect_x_min), cols)
        height = min(start_height + (rect_y_max - rect_y_min), rows)

        image.data = image.data[int(start_height):int(height), int(start_width):int(width)]
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
        if not annotation_meta:
            warning('operation *extend_around_rect* required annotation metadata')
            return image
        rect = annotation_meta.get('rect')
        if not rect:
            warning(
                'operation *extend_around_rect* require rect key in annotation meta, please use annotation converter '
                'which allows such transformation'
            )
            return image
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
        if not annotation_meta:
            warning('operation *point_alignment* required annotation metadata')
            return image
        keypoints = annotation_meta.get('keypoints')
        if not keypoints:
            warning(
                'operation *point_alignment* require keypoints key in annotation meta, please use annotation converter '
                'which allows such transformation'
            )
            return image
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
        points1 = points1.astype(np.float64)
        points2 = points2.astype(np.float64)

        c1 = np.mean(points1, axis=0, keepdims=True)
        c2 = np.mean(points2, axis=0, keepdims=True)
        points1 -= c1
        points2 -= c2
        s1 = np.std(points1)
        s2 = np.std(points2)
        points1 /= np.maximum(s1, np.finfo(np.float64).eps)
        points2 /= np.maximum(s1, np.finfo(np.float64).eps)
        points_std_ratio = s2 / np.maximum(s1, np.finfo(np.float64).eps)

        u, _, vt = np.linalg.svd(points1.T @ points2)
        r = (u @ vt).T

        return np.hstack((points_std_ratio * r, c2.T - points_std_ratio * r @ c1.T))


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
            ),
            'numpy_pad_mode': StringField(
                optional=True, default='constant',
                choices=['constant', 'edge', 'maximum', 'minimum', 'mean', 'median', 'wrap'],
                description="If use_numpy is True, Numpy padding mode,including constant, edge, mean, etc."
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
        self.numpy_pad_mode = self.get_value_from_config('numpy_pad_mode')

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
        if self.numpy_pad_mode != 'constant':
            return np.pad(
                image, ((pad[0], pad[2]), (pad[1], pad[3]), (0, 0)),
                mode=self.numpy_pad_mode
            )
        return np.pad(
            image, ((pad[0], pad[2]), (pad[1], pad[3]), (0, 0)),
            mode=self.numpy_pad_mode, constant_values=pad_values
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


class TransformedCropWithAutoScale(Preprocessor):
    __provider__ = 'transformed_crop_with_auto_scale'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'size': NumberField(
                value_type=int, optional=True, min_value=1,
                description="Destination sizes for both dimensions of heatmaps output."
            ),
            'dst_width': NumberField(
                value_type=int, optional=True, min_value=1, description="Width of heatmaps output."
            ),
            'dst_height': NumberField(
                value_type=int, optional=True, min_value=1, description="Height of heatmaps output."
            ),
            'stride': NumberField(
                value_type=int, optional=False,
                description="Stride for network. It is input size of heatmaps / output size of heatmaps."
            )
        })

        return parameters

    def configure(self):
        self.input_height, self.input_width = get_size_from_config(self.config)
        self.stride = self.get_value_from_config('stride')

    def process(self, image, annotation_meta=None):
        data = image.data
        center, scale = self.get_center_scale(annotation_meta['rects'][0], data.shape[1], data.shape[0])
        trans = self.get_transformation_matrix(center, scale, [self.input_width, self.input_height])
        rev_trans = self.get_transformation_matrix(center, scale, [self.input_width // self.stride,
                                                                   self.input_height // self.stride], key=1)
        data = cv2.warpAffine(data, trans, (self.input_width, self.input_height), flags=cv2.INTER_LINEAR)
        image.data = data
        image.metadata.setdefault('rev_trans', rev_trans)
        return image

    @staticmethod
    def get_center_scale(bbox, image_w, image_h):
        aspect_ratio = 0.75
        bbox[0] = np.max((0, bbox[0]))
        bbox[1] = np.max((0, bbox[1]))
        x2 = np.min((image_w - 1, bbox[0] + np.max((0, bbox[2] - 1))))
        y2 = np.min((image_h - 1, bbox[1] + np.max((0, bbox[3] - 1))))
        if x2 >= bbox[0] and y2 >= bbox[1]:
            bbox = [bbox[0], bbox[1], x2 - bbox[0], y2 - bbox[1]]
        cx_bbox = bbox[0] + bbox[2] * 0.5
        cy_bbox = bbox[1] + bbox[3] * 0.5
        center = np.array([np.float32(cx_bbox), np.float32(cy_bbox)])
        if bbox[2] > aspect_ratio * bbox[3]:
            bbox[3] = bbox[2] * 1.0 / aspect_ratio
        elif bbox[2] < aspect_ratio * bbox[3]:
            bbox[2] = bbox[3] * aspect_ratio

        scale = np.array([bbox[2] / 200., bbox[3] / 200.], np.float32) * 1.25

        return center, scale

    @staticmethod
    def get_transformation_matrix(center, scale, output_size, key=0):
        w, _ = scale * 200
        shift_y = [0, -w * 0.5]
        shift_x = [-w * 0.5, 0]
        points = np.array([center, center + shift_x, center + shift_y], dtype=np.float32)
        transformed_points = np.array([
            [output_size[0] * 0.5, output_size[1] * 0.5],
            [0, output_size[1] * 0.5],
            [output_size[0] * 0.5, output_size[1] * 0.5 - output_size[0] * 0.5]], dtype=np.float32)
        if key == 0:
            trans = cv2.getAffineTransform(np.float32(points), np.float32(transformed_points))
        else:
            trans = cv2.getAffineTransform(np.float32(transformed_points), np.float32(points))
        return trans


class ImagePyramid(Preprocessor):
    __provider__ = 'pyramid'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update(
            {
                'min_size': NumberField(value_type=int, min_value=1, description='min side size for pyramid layer'),
                'factor': NumberField(value_type=float, description='scale factor for pyramid layers')
            }
        )

        return parameters

    def configure(self):
        self.min_size = self.get_value_from_config('min_size')
        self.factor = self.get_value_from_config('factor')

    def process(self, image, annotation_meta=None):
        data = image.data.astype(float)
        height, width, _ = data.shape
        min_layer = min(height, width)
        m = 12.0 / self.min_size
        min_layer = min_layer * m
        scales = []
        factor_count = 0
        while min_layer >= 12:
            scales.append(m * pow(self.factor, factor_count))
            min_layer *= self.factor
            factor_count += 1
        scaled_data = []
        for scale in scales:
            hs = int(np.ceil(height * scale))
            ws = int(np.ceil(width * scale))
            scaled_data.append(cv2.resize(data, (ws, hs)))

        image.data = scaled_data
        image.metadata.update({'multi_infer': True, 'scales': scales})

        return image
