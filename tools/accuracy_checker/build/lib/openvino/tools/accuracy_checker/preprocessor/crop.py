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
from PIL import Image

from ..config import NumberField, BoolField, StringField, ConfigError
from ..logging import warning
from .preprocessor import Preprocessor
from .geometric_transformations import GeometricOperationMetadata
from ..utils import get_size_from_config, get_size_3d_from_config, is_image, finalize_image_shape


class CornerCrop(Preprocessor):
    __provider__ = 'corner_crop'
    shape_modificator = True
    _dynamic_shape = False

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'dst_width': NumberField(
                value_type=int, optional=True, min_value=1,
                description="Destination width for image cropping respectively."
            ),
            'dst_height': NumberField(
                value_type=int, optional=True, min_value=1,
                description="Destination height for image cropping respectively."
            ),
            'corner_type': StringField(
                optional=True, choices=['top_left', 'top_right', 'bottom_left', 'bottom_right'],
                default='top_left', description="Destination height for image cropping respectively."
            ),
        })

        return parameters

    def configure(self):
        self.corner_type = self.get_value_from_config('corner_type')
        self.dst_height, self.dst_width = get_size_from_config(self.config, allow_none=True)

    def process(self, image, annotation_meta=None):
        if isinstance(image.data, list):
            image.data = [
                self.process_data(
                    fragment, self.dst_height, self.dst_width, self.corner_type)
                for fragment in image.data
            ]
        else:
            image.data = self.process_data(
                image.data, self.dst_height, self.dst_width, self.corner_type)

        return image

    @property
    def dynamic_result_shape(self):
        return self._dynamic_shape

    @staticmethod
    def process_data(data, dst_height, dst_width, corner_type):
        height, width = data.shape[:2]
        new_height, new_width, start_height, start_width = CornerCrop.get_roi(
            height, width, dst_height, dst_width, corner_type)

        return data[start_height:start_height + new_height, start_width:start_width + new_width]

    @staticmethod
    def get_roi(height, width, dst_height, dst_width, corner_type):
        if corner_type == 'top_left':
            new_height = min(height, dst_height)
            start_height = 0
            new_width = min(width, dst_width)
            start_width = 0
        elif corner_type == 'top_right':
            new_height = min(height, dst_height)
            start_height = 0
            if width > dst_width:
                start_width = width - dst_width
                new_width = width
            else:
                start_width = 0
                new_width = width
        elif corner_type == 'bottom_left':
            if height > dst_height:
                start_height = height - dst_height
                new_height = height
            else:
                start_height = 0
                new_height = height
            new_width = min(width, dst_width)
            start_width = 0
        elif corner_type == 'bottom_right':
            if height > dst_height:
                start_height = height - dst_height
                new_height = height
            else:
                start_height = 0
                new_height = height
            if width > dst_width:
                start_width = width - dst_width
                new_width = width
            else:
                start_width = 0
                new_width = width
        return new_height, new_width, start_height, start_width

    def calculate_out_single_shape(self, data_shape):
        height, width, _, _ = self.get_roi(
            data_shape[0], data_shape[1], self.dst_height, self.dst_width, self.corner_type
        )
        return finalize_image_shape(height, width, data_shape)

    def calculate_out_shape(self, data_shape):
        return [self.calculate_out_single_shape(ds) if is_image(ds) else ds for ds in data_shape]


class Crop(Preprocessor):
    __provider__ = 'crop'
    shape_modificator = True

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
            'central_fraction': NumberField(
                value_type=float, min_value=0, max_value=1, optional=True, description="Central Fraction."
            ),
            'max_square': BoolField(
                optional=True, default=False, description='crop center area by shortest side'
            )
        })

        return parameters

    def configure(self):
        self.use_pillow = self.get_value_from_config('use_pillow')
        self.dst_height, self.dst_width = get_size_from_config(self.config, allow_none=True)
        self.central_fraction = self.get_value_from_config('central_fraction')
        self.max_square = self.get_value_from_config('max_square')
        if self.dst_height is None and self.dst_width is None and self.central_fraction is None and not self.max_square:
            raise ConfigError('sizes for crop or central_fraction or max_square should be provided')
        if self.dst_height and self.dst_width and self.central_fraction:
            raise ConfigError('both sizes and central fraction provided for cropping')
        if self.dst_height and self.dst_width and self.max_square:
            raise ConfigError('both sizes and max_square provided for cropping')
        if self.central_fraction and self.max_square:
            raise ConfigError('both central fraction and nax_square provided for cropping')

        if not self.central_fraction and not self.max_square:
            if self.dst_height is None or self.dst_width is None:
                raise ConfigError('one from crop dimensions is not provided')

    def process(self, image, annotation_meta=None):
        is_simple_case = not isinstance(image.data, list)  # otherwise -- pyramid, tiling, etc
        data = image.data

        image.data = self.process_data(
            data, self.dst_height, self.dst_width, self.central_fraction, self.max_square,
            self.use_pillow, is_simple_case, image.metadata
        ) if not isinstance(data, list) else [
            self.process_data(
                fragment, self.dst_height, self.dst_width, self.central_fraction, self.max_square,
                self.use_pillow, is_simple_case, image.metadata
            ) for fragment in image.data
        ]

        return image

    @staticmethod
    def process_data(data, dst_height, dst_width, central_fraction, max_square, use_pillow, is_simple_case, metadata):
        height, width = data.shape[:2]
        if not central_fraction and not max_square:
            new_height = dst_height
            new_width = dst_width
        elif max_square:
            new_height = min(height, width)
            new_width = new_height
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

    @property
    def dynamic_result_shape(self):
        if self.max_square or self.central_fraction:
            return True
        return False

    def calculate_out_single_shape(self, data_shape):
        height, width = data_shape[:2]
        if self.dynamic_result_shape and (height == -1 or width == -1):
            return data_shape
        if self.max_square:
            side = min(height, width)
            return finalize_image_shape(side, side, data_shape)
        if self.central_fraction:
            new_height = int(height * self.central_fraction)
            new_width = int(width * self.central_fraction)
            return finalize_image_shape(new_height, new_width, data_shape)

        return finalize_image_shape(self.dst_height, self.dst_width, data_shape)

    def calculate_out_shape(self, data_shape):
        return [self.calculate_out_single_shape(ds) if is_image(ds) else ds for ds in data_shape]


class CropRect(Preprocessor):
    __provider__ = 'crop_rect'
    shape_modificator = True
    dynamic_result_shapes = True

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

    @staticmethod
    def calculate_out_single_shape(data_shape):
        return finalize_image_shape(-1, -1, data_shape)

    def calculate_out_shape(self, data_shape):
        return [self.calculate_out_single_shape(ds) if is_image(ds) else ds for ds in data_shape]


class ExtendAroundRect(Preprocessor):
    __provider__ = 'extend_around_rect'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'augmentation_param': NumberField(
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


class Crop3D(Preprocessor):
    __provider__ = 'crop3d'
    shape_modificator = True
    _dynamic_shapes = False

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'size': NumberField(
                value_type=int, optional=True, min_value=1,
                description="Destination size for 3d crop for all dimensions."
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

    @property
    def dynamic_result_shape(self):
        return self._dynamic_shapes


class TransformedCropWithAutoScale(Preprocessor):
    __provider__ = 'transformed_crop_with_auto_scale'
    shape_modificator = True
    _dynamic_shapes = False

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
        self.dst_height, self.dst_width = get_size_from_config(self.config)
        self.stride = self.get_value_from_config('stride')

    def process(self, image, annotation_meta=None):
        data = image.data
        center, scale = self.get_center_scale(annotation_meta['rects'][0], data.shape[1], data.shape[0])
        trans = self.get_transformation_matrix(center, scale, [self.dst_width, self.dst_height])
        rev_trans = self.get_transformation_matrix(center, scale, [self.dst_width // self.stride,
                                                                   self.dst_height // self.stride], key=1)
        data = cv2.warpAffine(data, trans, (self.dst_width, self.dst_height), flags=cv2.INTER_LINEAR)
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

    @property
    def dynamic_result_shape(self):
        return self._dynamic_shapes

    def calculate_out_single_shape(self, data_shape):
        return finalize_image_shape(self.dst_height, self.dst_width, data_shape)

    def calculate_out_shape(self, data_shape):
        return [self.calculate_out_single_shape(ds) if is_image(ds) else ds for ds in data_shape]


class CandidateCrop(Preprocessor):
    __provider__ = 'candidate_crop'
    shape_modificator = True

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'scale_width': NumberField(
                value_type=float, min_value=0, default=1, optional=True,
                description='Value to scale width relative to the original candidate width'
            ),
            'scale_height': NumberField(
                value_type=float, min_value=0, default=1, optional=True,
                description='Value to scale height relative to the original candidate height'
            )
        })
        return parameters

    def configure(self):
        self.scale_width = self.get_value_from_config('scale_width')
        self.scale_height = self.get_value_from_config('scale_height')

    def process(self, image, annotation_meta=None):
        candidates = annotation_meta['candidate_info']
        patches = []
        data = image.data
        img_height, img_width, _ = data.shape
        for i in range(candidates.x_mins.size):
            x_min = int(round(candidates.x_mins[i]))
            y_min = int(round(candidates.y_mins[i]))
            width = int(round(candidates.x_maxs[i] - candidates.x_mins[i]))
            height = int(round(candidates.y_maxs[i] - candidates.y_mins[i]))
            x_min -= int(round(width * (self.scale_width - 1) / 2))
            y_min -= int(round(height * (self.scale_height - 1) / 2))
            width = int(round(width * self.scale_width))
            height = int(round(height * self.scale_height))
            bbox = [x_min, y_min, x_min + width, y_min + height]
            ext_bbox = [bbox[0], bbox[1], bbox[2], bbox[3]]
            bbox[0] = max(0, bbox[0])
            bbox[1] = max(0, bbox[1])
            bbox[2] = min(img_width, bbox[2])
            bbox[3] = min(img_height, bbox[3])
            crop_image = data[bbox[1]:bbox[3], bbox[0]:bbox[2]]
            # padding for turncated region
            if bbox[0] == 0 or bbox[1] == 0 or bbox[2] == img_width or bbox[3] == img_height:
                crop_image = cv2.copyMakeBorder(
                    crop_image,
                    bbox[1] - ext_bbox[1],
                    ext_bbox[3] - bbox[3],
                    bbox[0] - ext_bbox[0],
                    ext_bbox[2] - bbox[2],
                    cv2.BORDER_CONSTANT
                )
            patches.append(crop_image)

        if candidates.x_mins.size == 0:
            patches.append(data)

        image.data = patches
        image.metadata.update({
            'multi_infer': True,
            'candidates': candidates
        })
        return image


class CropOrPad(Preprocessor):
    __provider__ = 'crop_or_pad'
    shape_modificator = True
    _dynamic_shapes = False

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'size': NumberField(
                value_type=int, optional=True, min_value=1,
                description="Destination sizes for both dimensions of heatmaps output."
            ),
            'dst_width': NumberField(
                value_type=int, optional=True, min_value=1, description="Width of heatmaps output."
            ),
            'dst_height': NumberField(
                value_type=int, optional=True, min_value=1, description="Height of heatmaps output."
            )
        })
        return params

    def configure(self):
        self.dst_height, self.dst_width = get_size_from_config(self.config)

    def process(self, image, annotation_meta=None):
        height, width = image.data.shape[:2]
        width_diff = self.dst_width - width
        offset_crop_width = max(-width_diff // 2, 0)
        offset_pad_width = max(width_diff // 2, 0)

        height_diff = self.dst_height - height
        offset_crop_height = max(-height_diff // 2, 0)
        offset_pad_height = max(height_diff // 2, 0)
        cropped, meta = self.crop_to_bounding_box(
            image.data, offset_crop_height, offset_crop_width, min(self.dst_height, height), min(self.dst_width, width))
        resized, pad_meta = self.pad_to_bounding_box(
            cropped, offset_pad_height, offset_pad_width, self.dst_height, self.dst_width
        )
        meta.update(pad_meta)
        image.data = resized
        image.metadata.setdefault('geometric_operations', []).append(GeometricOperationMetadata('crop_or_pad', meta))
        return image

    @staticmethod
    def crop_to_bounding_box(data, start_h, start_w, end_h, end_w):
        return data[int(start_h):int(end_h), int(start_w):int(end_w)], {}

    @staticmethod
    def pad_to_bounding_box(data, offset_h, offset_w, dst_h, dst_w):
        height, width = data.shape[:2]
        after_padding_width = dst_w - offset_w - width
        after_padding_height = dst_h - offset_h - height
        meta = {
            'pad': [offset_h, offset_w, after_padding_height, after_padding_width],
            'dst_height': dst_h,
            'dst_width': dst_w,
            'height': height,
            'width': width
        }
        return cv2.copyMakeBorder(
            data, offset_h, after_padding_height, offset_w, after_padding_width, cv2.BORDER_CONSTANT, value=0
        ), meta

    @property
    def dynamic_result_shape(self):
        return self._dynamic_shapes

    def calculate_out_single_shape(self, data_shape):
        return finalize_image_shape(self.dst_height, self.dst_width, data_shape)

    def calculate_out_shape(self, data_shape):
        return [self.calculate_out_single_shape(ds) if is_image(ds) else ds for ds in data_shape]


class CropWithPadSize(Preprocessor):
    __provider__ = 'crop_image_with_padding'
    shape_modificator = True
    _dynamic_shapes = False

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'size': NumberField(value_type=int, min_value=1),
            'crop_padding': NumberField(value_type=int, min_value=1)
        })
        return params

    def configure(self):
        self.size = self.get_value_from_config('size')
        self.crop_padding = self.get_value_from_config('crop_padding')

    def process(self, image, annotation_meta=None):
        image_height, image_width = image.data.shape[:2]
        padded_center_crop_size = int((self.size / (self.size + self.crop_padding)) * min(image_height, image_width))
        offset_height = ((image_height - padded_center_crop_size) + 1) // 2
        offset_width = ((image_width - padded_center_crop_size) + 1) // 2
        cropped_data, _ = CropOrPad.crop_to_bounding_box(
            image.data, offset_height, offset_width,
            offset_height + padded_center_crop_size, offset_width + padded_center_crop_size
        )
        image.data = cv2.resize(cropped_data, (self.size, self.size))
        return image

    @property
    def dynamic_result_shape(self):
        return self._dynamic_shapes

    def calculate_out_single_shape(self, data_shape):
        return finalize_image_shape(self.size, self.size, data_shape)

    def calculate_out_shape(self, data_shape):
        return [self.calculate_out_single_shape(ds) if is_image(ds) else ds for ds in data_shape]


class ObjectCropWithScale(Preprocessor):
    __provider__ = 'object_crop_with_scale'
    shape_modificator = True
    _dynamic_shape = False

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
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
            )
        })
        return params

    def configure(self):
        self.dst_height, self.dst_width = get_size_from_config(self.config)

    def process(self, image, annotation_meta=None):
        meta = annotation_meta or {}
        center = meta.get('center', [-1, 0])
        scale = meta.get('scale', 1)
        if center[0] != -1:
            center[1] = center[1] + 15 * scale
            scale = scale * 1.25

        image.data = self.crop(image.data, np.array(center), scale)
        image.metadata['scale'] = scale
        image.metadata['center'] = center

        return image

    def crop(self, img, center, scale):
        # Preprocessing for efficient cropping
        height, width = img.shape[:2]
        sf = scale * 200.0 / self.dst_width
        if sf >= 2:
            new_size = int(np.math.floor(max(height, width) / sf))
            new_height = int(np.math.floor(height / sf))
            new_width = int(np.math.floor(width / sf))
            if new_size < 2:
                return (
                    np.zeros((self.dst_width, self.dst_height, img.shape[2]), dtype=np.float32) if len(img.shape) > 2
                    else np.zeros((self.dst_width, self.dst_height), dtype=np.float32)
                )
            img = cv2.resize(img, dsize=(new_width, new_height), interpolation=cv2.INTER_LINEAR)
            center = center * 1.0 / sf
            scale = scale / sf

        # Upper left point
        ul = np.array(self.transform([0, 0], center, scale, [self.dst_width, self.dst_height], invert=1))
        # Bottom right point
        br = np.array(
            self.transform(
                [self.dst_width, self.dst_height], center, scale, [self.dst_width, self.dst_height], invert=1
            ))

        new_shape = [br[1] - ul[1], br[0] - ul[0]]

        if len(img.shape) > 2:
            new_shape += [img.shape[2]]
        new_img = np.zeros(new_shape, dtype=np.float32)
        new_x = [max(0, -ul[0]), min(br[0], img.shape[1]) - ul[0]]
        new_y = [max(0, -ul[1]), min(br[1], img.shape[0]) - ul[1]]
        old_x = [max(0, ul[0]), min(img.shape[1], br[0])]
        old_y = [max(0, ul[1]), min(img.shape[0], br[1])]

        if new_x[1] < new_x[0]:
            tmp_new, tmp_old = new_x[1], old_x[0]
            new_x[1], old_x[0] = new_x[0], old_x[1]
            new_x[0], old_x[1] = tmp_new, tmp_old
        new_img[new_y[0]:new_y[1], new_x[0]:new_x[1]] = img[old_y[0]:old_y[1], old_x[0]:old_x[1]]
        new_img = cv2.resize(new_img, dsize=(self.dst_width, self.dst_height), interpolation=cv2.INTER_LINEAR)
        return new_img

    @staticmethod
    def get_transform(center, scale, res):
        height = 200 * scale
        transformation = np.zeros((3, 3))
        transformation[0, 0], transformation[1, 1] = float(res[1]) / height, float(res[0]) / height
        transformation[0, 2] = res[1] * (-float(center[0]) / height + .5)
        transformation[1, 2] = res[0] * (-float(center[1]) / height + .5)
        transformation[2, 2] = 1
        return transformation

    @staticmethod
    def transform(pt, center, scale, res, invert=0):
        # Transform pixel location to different reference
        transform_matrix = ObjectCropWithScale.get_transform(center, scale, res)
        if invert:
            transform_matrix = np.linalg.inv(transform_matrix)
        new_pt = np.array([pt[0] - 1, pt[1] - 1, 1.]).T
        new_pt = np.dot(transform_matrix, new_pt)
        return new_pt[:2].astype(int) + 1

    @property
    def dynamic_result_shape(self):
        return self._dynamic_shape

    def calculate_out_single_shape(self, data_shape):
        return finalize_image_shape(self.dst_height, self.dst_width, data_shape)

    def calculate_out_shape(self, data_shape):
        return [self.calculate_out_single_shape(ds) if is_image(ds) else ds for ds in data_shape]
