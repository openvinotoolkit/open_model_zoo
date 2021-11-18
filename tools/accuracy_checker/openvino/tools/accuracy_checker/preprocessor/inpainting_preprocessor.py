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

import cv2
import numpy as np

from .preprocessor import Preprocessor
from ..config import NumberField, BoolField, PathField, StringField, ConfigError
from ..utils import get_size_from_config
from ..data_readers import BaseReader


class FreeFormMask(Preprocessor):
    __provider__ = 'free_form_mask'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'parts': NumberField(optional=True, default=8, description="Number of parts to draw mask.", value_type=int),
            'max_brush_width': NumberField(
                optional=True, default=24, description="Maximum brush width to draw mask.", value_type=int
            ),
            'max_length': NumberField(
                optional=True, default=100, description="Maximum line length to draw mask.", value_type=int
            ),
            "max_vertex": NumberField(
                optional=True, default=20, description="Maximum number vertex to draw mask.", value_type=int
            ),
            'inverse_mask': BoolField(optional=True, default=False, description="Inverse mask"),
        })
        return parameters

    def configure(self):
        self.parts = self.get_value_from_config('parts')
        self.max_brush_width = self.get_value_from_config('max_brush_width')
        self.max_length = self.get_value_from_config('max_length')
        self.max_vertex = self.get_value_from_config('max_vertex')
        self.inverse_mask = self.get_value_from_config('inverse_mask')

    @staticmethod
    def _free_form_mask(mask, max_vertex, max_length, max_brush_width, h, w, max_angle=360):
        num_strokes = np.random.randint(max_vertex)
        start_y = np.random.randint(h)
        start_x = np.random.randint(w)
        for i in range(num_strokes):
            angle = np.random.random() * np.deg2rad(max_angle)
            if i % 2 == 0:
                angle = 2 * np.pi - angle
            length = np.random.randint(max_length + 1)
            brush_width = np.random.randint(10, max_brush_width + 1) // 2 * 2
            next_y = start_y + length * np.cos(angle)
            next_x = start_x + length * np.sin(angle)

            next_y = np.clip(next_y, 0, h - 1).astype(np.int)
            next_x = np.clip(next_x, 0, w - 1).astype(np.int)
            cv2.line(mask, (start_y, start_x), (next_y, next_x), 1, brush_width)
            cv2.circle(mask, (start_y, start_x), brush_width // 2, 1)

            start_y, start_x = next_y, next_x
        return mask

    def process(self, image, annotation_meta=None):
        if len(image.data) == 2:
            return preprocess_input_mask(image, self.inverse_mask)
        img = image.data[0]
        img_height, img_width = img.shape[:2]
        mask = np.zeros((img_height, img_width, 1), dtype=np.float32)

        for _ in range(self.parts):
            mask = self._free_form_mask(mask, self.max_vertex, self.max_length, self.max_brush_width,
                                        img_height, img_width)

        img = img * (1 - mask) + 255 * mask
        if self.inverse_mask:
            mask = 1 - mask
        image.data = [img, mask]
        identifier = image.identifier[0]
        image.identifier = ['{}_image'.format(identifier), '{}_mask'.format(identifier)]
        return image


class RectMask(Preprocessor):
    __provider__ = "rect_mask"

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'dst_height': NumberField(
                optional=True, default=128, description="Height of mask", value_type=int
            ),
            'dst_width': NumberField(
                optional=True, default=128, description="Width of mask", value_type=int
            ),
            'size': NumberField(
                optional=True, default=128,
                description="Size of mask, used if both dimensions are equal", value_type=int
            ),
            'inverse_mask': BoolField(optional=True, default=False, description="Inverse mask")

        })
        return parameters

    def configure(self):
        self.mask_height, self.mask_width = get_size_from_config(self.config, allow_none=True)
        if self.mask_height is None:
            self.mask_height = 128
        if self.mask_width is None:
            self.mask_width = 128
        self.inverse_mask = self.get_value_from_config('inverse_mask')

    def process(self, image, annotation_meta=None):
        if len(image.data) == 2:
            return preprocess_input_mask(image, self.inverse_mask)

        img = image.data[0]
        img_height, img_width = img.shape[:2]
        mp0 = max(0, (img_height - self.mask_height)//2)
        mp1 = max(0, (img_width - self.mask_width)//2)

        mask = np.zeros((img_height, img_width)).astype(np.float32)
        mask[mp0:mp0 + self.mask_height, mp1:mp1 + self.mask_width] = 1
        mask = np.expand_dims(mask, axis=2)

        img = img * (1 - mask) + 255 * mask
        if self.inverse_mask:
            mask = 1 - mask
        image.data = [img, mask]
        identifier = image.identifier[0]
        image.identifier = ['{}_image'.format(identifier), '{}_mask'.format(identifier)]
        return image


class CustomMask(Preprocessor):
    __provider__ = 'custom_mask'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'mask_dir': PathField(is_directory=True, optional=False, description="Path to mask dataset directory"),
            'inverse_mask': BoolField(optional=True, default=False, description="Inverse mask"),
            'mask_loader': StringField(optional=True, default='numpy_reader', description="Mask loader")
        })
        return parameters

    def configure(self):
        self.mask_dir = self.get_value_from_config('mask_dir')
        self.inverse_mask = self.get_value_from_config('inverse_mask')
        self.mask_loader = self.get_value_from_config('mask_loader')

    def process(self, image, annotation_meta=None):
        if len(image.data) == 2:
            return preprocess_input_mask(image, self.inverse_mask)

        if annotation_meta.get('mask') is None:
            raise ConfigError('Path to mask dataset is not set during annotation conversion.'
                              'Please specify masks_dir parameter')

        loader = BaseReader.provide(self.mask_loader, self.mask_dir)
        mask = loader.read(annotation_meta['mask']['mask_name'])
        mask = np.minimum(mask, 1.0)

        img = image.data[0]
        img_height, img_width = img.shape[:2]
        mask = cv2.resize(mask, (img_width, img_height))

        if self.inverse_mask:
            mask = 1 - mask
        if len(mask.shape) == 2:
            mask = np.expand_dims(mask, axis=2)
        if mask.shape[2] != 1:
            mask = mask[:, :, 0]
            mask = np.expand_dims(mask, axis=2)

        img = img * (1 - mask) + 255 * mask
        image.data = [img, mask]
        identifier = image.identifier[0]
        image.identifier = ['{}_image'.format(identifier), '{}_mask'.format(identifier)]
        return image


def preprocess_input_mask(data, inverse=False):
    img, mask = data.data[0], data.data[1]
    if len(mask.shape) == 3 and mask.shape[-1] != 1:
        mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY)
    mask = mask.astype(float) / 255
    mask[mask >= 0.5] = 1
    mask[mask < 0.5] = 0
    if len(mask.shape) == 2:
        mask = np.expand_dims(mask, -1)
    img = img * (1 - mask) + 255 * mask

    data.data[0] = img
    data.data[1] = mask if not inverse else 1 - mask

    return data
