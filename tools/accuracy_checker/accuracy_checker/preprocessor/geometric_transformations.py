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

import math
from collections import namedtuple

import cv2
import numpy as np

from ..config import ConfigError, NumberField, StringField, BoolField, ListField
from ..preprocessor import Preprocessor
from ..utils import get_size_from_config, string_to_tuple, UnsupportedPackage
from ..logging import warning

try:
    from skimage.transform import estimate_transform, warp
except ImportError as import_error:
    estimate_transform = UnsupportedPackage("skimage.transform", import_error.msg)
    warp = UnsupportedPackage("skimage.transform", import_error.msg)

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
            ),
            'merge_with_original': BoolField(
                optional=True, description='allow joint flipped image to original', default=False
            )
        })
        return parameters

    def configure(self):
        mode = self.get_value_from_config('mode')
        if isinstance(mode, str):
            self.mode = FLIP_MODES[mode]
        self.merge = self.get_value_from_config('merge_with_original')

    def process(self, image, annotation_meta=None):
        flipped_data = cv2.flip(image.data, self.mode)
        if self.merge:
            image.data = [image.data, flipped_data]
            image.metadata['multi_infer'] = True
        else:
            image.data = flipped_data

        image.metadata.setdefault(
            'geometric_operations', []).append(GeometricOperationMetadata('flip', {'mode': self.mode}))
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
                description="Destination size for keypoints resizing for both destination dimensions."
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


def center_padding(dst_width, dst_height, width, height, left_top_extend=False):
    delta = [int(math.floor((dst_height - height) / 2.0)), int(math.floor((dst_width - width) / 2.0))]
    ost = [(dst_height - height) % 2, (dst_width - width) % 2]
    if left_top_extend:
        pad = [delta[0] + ost[0], delta[1] + ost[1]]
    else:
        pad = delta
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
            ),
            'enable_resize': BoolField(
                optional=True, default=False, description='allow resize images if source image large then padding size'
            ),
            'left_top_extend': BoolField(
                optional=True, default=False,
                description='allow to use left-top extend instead of right-bottom for center padding'
            )
        })

        return parameters

    def configure(self):
        self.stride = self.get_value_from_config('stride')
        pad_val = self.get_value_from_config('pad_value')
        if isinstance(pad_val, int):
            self.pad_value = (pad_val, pad_val, pad_val)
        if isinstance(pad_val, str):
            self.pad_value = string_to_tuple(pad_val, float)
        self.dst_height, self.dst_width = get_size_from_config(self.config, allow_none=True)
        self.pad_type = self.get_value_from_config('pad_type')
        self.pad_func = padding_func[self.pad_type]
        self.use_numpy = self.get_value_from_config('use_numpy')
        self.numpy_pad_mode = self.get_value_from_config('numpy_pad_mode')
        self.enable_resize = self.get_value_from_config('enable_resize')
        self.left_top_extend = self.get_value_from_config('left_top_extend')

    def process(self, image, annotation_meta=None):
        height, width, _ = image.data.shape
        pref_height = self.dst_height or image.metadata.get('preferable_height', height)
        pref_width = self.dst_width or image.metadata.get('preferable_width', width)
        height = min(height, pref_height)
        width_pref_init = pref_width
        pref_height = math.ceil(pref_height / float(self.stride)) * self.stride
        pref_width = max(pref_width, width)
        pref_width = math.ceil(pref_width / float(self.stride)) * self.stride
        if self.pad_type == 'center':
            pad = self.pad_func(pref_width, pref_height, width, height, self.left_top_extend)
        else:
            pad = self.pad_func(pref_width, pref_height, width, height)
        image.metadata['padding'] = pad
        padding_realization_func = self._opencv_padding if not self.use_numpy else self._numpy_padding
        image.data = padding_realization_func(image.data, pad)
        meta = {
            'pad': pad,
            'dst_width': self.dst_width,
            'dst_height': self.dst_height,
            'pref_width': pref_width,
            'pref_height': pref_height,
            'width': width,
            'height': height,
            'resized': False
        }
        if self.enable_resize and image.data.shape[:2] != (pref_height, width_pref_init):
            image.data = cv2.resize(image.data, (width_pref_init, pref_height))
            meta['resized'] = True
            meta['pref_width'] = width_pref_init

        image.metadata.setdefault('geometric_operations', []).append(
            GeometricOperationMetadata('padding', meta))

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
                description="Destination size of tiled fragment for both dimensions."
            ),
            'dst_width': NumberField(
                value_type=int, optional=True, min_value=1, description="Destination width of tiled fragment."
            ),
            'dst_height': NumberField(
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


class FaceDetectionImagePyramid(Preprocessor):
    __provider__ = 'face_detection_image_pyramid'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update(
            {
                'min_face_ratio': NumberField(
                    value_type=float, default=0.05, min_value=0.01, max_value=1,
                    description='Minimum face ratio to image size'
                ),
                'resize_scale': NumberField(
                    value_type=int, default=2, min_value=1,
                    description='Scale factor for pyramid layers'
                )
            }
        )
        return parameters

    def configure(self):
        self.min_face_ratio = self.get_value_from_config('min_face_ratio')
        self.resize_scale = self.get_value_from_config('resize_scale')
        self.min_supported_face_size = 24
        self.stage1_window_size = [12, 192]

    def perform_scaling(self, initial_width, initial_height, img_width, img_height):
        width = initial_width
        height = initial_height

        image_pyramid = []
        scales = []
        pyramid_scale = 1

        shorter = min(img_height, img_width)
        min_face_size = max(int(shorter * self.min_face_ratio), self.min_supported_face_size)

        while width >= self.stage1_window_size[0] and height >= self.stage1_window_size[0]:
            min_detectable_size = int(img_width / width + 0.5) * self.stage1_window_size[0]
            if min_detectable_size >= min_face_size:
                if min_detectable_size > self.min_supported_face_size:
                    pyramid_scale /= 2
                    width = int(initial_width / pyramid_scale + 0.5)
                    height = int(initial_height / pyramid_scale + 0.5)

                image_pyramid.append((int(width), int(height)))
                scales.append(img_width / int(width))

                max_detectable_size = int(img_width / width + 0.5) * self.stage1_window_size[1]
                if max_detectable_size < shorter:
                    while max_detectable_size > min_detectable_size:
                        pyramid_scale *= self.resize_scale
                        width = int(initial_height / pyramid_scale + 0.5)
                        height = int(initial_height / pyramid_scale + 0.5)
                        min_detectable_size = int(img_width / width + 0.5) * self.stage1_window_size[0]
                        min_detectable_size *= 2
                break

            pyramid_scale *= self.resize_scale
            width = int(initial_width / pyramid_scale + 0.5)
            height = int(initial_height / pyramid_scale + 0.5)

        return image_pyramid, scales, pyramid_scale

    def process(self, image, annotation_meta=None):
        img_height, img_width, _ = image.data.shape
        initial_width = img_width * self.stage1_window_size[0] / self.min_supported_face_size
        initial_height = img_height * self.stage1_window_size[0] / self.min_supported_face_size
        image_pyramid, scales, pyramid_scale = self.perform_scaling(
            initial_width,
            initial_height,
            img_width, img_height
        )

        if len(image_pyramid) == 0:
            pyramid_scale /= self.resize_scale
            width = int(initial_width / pyramid_scale + 0.5)
            height = int(initial_height / pyramid_scale + 0.5)
            image_pyramid.append((width, height))
            scales.append(img_width / width)

        scaled_data = []
        data = image.data

        # perform resizing
        for dimension in image_pyramid:
            w, h = dimension
            scaled_data.append(cv2.resize(data, (w, h)))

        image.data = scaled_data
        image.metadata.update({'multi_infer': True, 'scales': scales})
        return image


class WarpAffine(Preprocessor):
    __provider__ = 'warp_affine'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'src_landmarks': ListField(
                description='Source landmark points',
                value_type=ListField(value_type=float)
            ),
            'dst_landmarks': ListField(
                description='Destination landmark points',
                value_type=ListField(value_type=float)
            ),
            'dst_height': NumberField(
                description='Destination height size',
                value_type=int,
                optional=False
            ),
            'dst_width': NumberField(
                description='Destination width size',
                value_type=int,
                optional=False
            )
        })
        return parameters

    def configure(self):
        self.src_landmarks = self.get_value_from_config('src_landmarks')
        self.dst_landmarks = self.get_value_from_config('dst_landmarks')
        self.dst_height = self.get_value_from_config('dst_height')
        self.dst_width = self.get_value_from_config('dst_width')

        self.validate(self.src_landmarks, self.dst_landmarks)

    def validate(self, point1, point2):
        if len(self.src_landmarks) != len(self.dst_landmarks):
            raise ConfigError('To align points, number of src landmarks and dst landmarks must match')
        if len(self.src_landmarks) <= 0:
            raise ConfigError('One or more landmark points are required')
        if not all(len(c) == 2 for c in self.src_landmarks) or not all(len(c) == 2 for c in self.dst_landmarks):
            raise ConfigError('Coordinate values must be a list of size 2')

    def process(self, image, annotation_meta=None):
        is_simple_case = not isinstance(image.data, list)

        def process_data(data):
            src = np.array(self.src_landmarks, dtype=np.float32)
            dst = np.array(self.dst_landmarks, dtype=np.float32)
            M = cv2.estimateAffinePartial2D(src, dst, method=cv2.LMEDS)[0]
            data = cv2.warpAffine(data, M, (self.dst_width, self.dst_height), borderValue=0.0).copy()
            return data

        if is_simple_case:
            image.data = process_data(image.data)
            return image

        image.data = [process_data(images) for images in image.data]
        return image

class SimilarityTransfom(Preprocessor):
    __provider__ = 'similarity_transform_box'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'box_scale': NumberField(value_type=float, min_value=0, description='Scale factor for box', default=1.),
            'size': NumberField(
                value_type=int, optional=True, min_value=1, description="Destination sizes for both dimensions."
            ),
            'dst_width': NumberField(
                value_type=int, optional=True, min_value=1, description="Destination width for image resizing."
            ),
            'dst_height': NumberField(
                value_type=int, optional=True, min_value=1, description="Destination height for image resizing."
            )
        })
        return params

    def configure(self):
        if isinstance(estimate_transform, UnsupportedPackage):
            estimate_transform.raise_error(self.__provider__)
        self.box_scale = self.get_value_from_config('box_scale')
        self.dst_height, self.dst_width = get_size_from_config(self.config)

    def process(self, image, annotation_meta=None):
        left, top, right, bottom = annotation_meta.get('rect', [0, 0, image.data.shape[0], image.data.shape[1]])
        old_size = (right - left + bottom - top) / 2
        center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        size = int(old_size * self.box_scale)
        src_pts = np.array([[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
                            [center[0] + size / 2, center[1] - size / 2]])
        dst_pts = np.array([[0, 0], [0, self.dst_height - 1], [self.dst_width - 1, 0]])
        tform = estimate_transform('similarity', src_pts, dst_pts)
        image.data = warp(image.data / 255, tform.inverse, output_shape=(self.dst_width, self.dst_height))
        image.data *= 255

        image.metadata['transform_matrix'] = tform.params
        image.metadata['roi_box'] = [left, top, right, bottom]

        return image

    @staticmethod
    def estimate_transform(src, dst):
        num = src.shape[0]
        dim = src.shape[1]

        src_mean = src.mean(axis=0)
        dst_mean = dst.mean(axis=0)

        src_demean = src - src_mean
        dst_demean = dst - dst_mean
        A = dst_demean.T @ src_demean / num

        d = np.ones((dim,), dtype=np.double)
        if np.linalg.det(A) < 0:
            d[dim - 1] = -1

        T = np.eye(dim + 1, dtype=np.double)

        U, S, V = np.linalg.svd(A)

        rank = np.linalg.matrix_rank(A)
        if rank == 0:
            return np.nan * T
        if rank == dim - 1:
            if np.linalg.det(U) * np.linalg.det(V) > 0:
                T[:dim, :dim] = U @ V
            else:
                s = d[dim - 1]
                d[dim - 1] = -1
                T[:dim, :dim] = U @ np.diag(d) @ V
                d[dim - 1] = s
        else:
            T[:dim, :dim] = U @ np.diag(d) @ V

        scale = 1.0 / src_demean.var(axis=0).sum() * (S @ d)

        T[:dim, dim] = dst_mean - scale * (T[:dim, :dim] @ src_mean.T)
        T[:dim, :dim] *= scale

        return T
