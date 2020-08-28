"""
Copyright (c) 2018-2020 Intel Corporation

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

import numpy as np

from ..config import ConfigError, BaseField, NumberField, ListField, StringField
from ..preprocessor import Preprocessor
from ..utils import get_or_parse_value

try:
    from scipy.ndimage import interpolation
except ImportError:
    interpolation = None


class Resize3D(Preprocessor):
    __provider__ = 'resize3d'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'size': BaseField(optional=True, description='Specifies resize'),
        })
        return parameters

    def configure(self):
        if interpolation is None:
            raise ValueError('resize3d require scipy, please install it before usage.')

        self.shape = self._check_size(
            get_or_parse_value(self.config.get('size'), default=(128, 128, 128), casting_type=int))

    def process(self, image, annotation_meta=None):
        data = np.asarray(image.data)
        shape = self.shape if len(data.shape) == 3 else (data.shape[0],) + self.shape
        if len(data.shape) != len(shape):
            raise RuntimeError('Shape of original data and resize shape are mismatched for {} preprocessor '
                               '(data shape - {}, resize shape - {})'.format(self.__provider__, data.shape, shape))

        factor = [float(o) / i for i, o in zip(data.shape, shape)]
        image.data = interpolation.zoom(data, zoom=factor, order=1)

        return image

    def _check_size(self, size):
        if len(size) != 3:
            raise ConfigError("Incorrect size dimension for {} - must be 3, but {} found"
                              .format(self.__provider__, len(size)))
        if not all(np.array(size) > 0):
            raise ConfigError("Size must be positive value for {}, but {} found".format(self.__provider__, size))
        return size


class CropBraTS(Preprocessor):
    __provider__ = 'crop_brats'

    def process(self, image, annotation_meta=None):
        def bbox3(img):
            # Finds indexes non-zero voxels across axis 0, 1 and 2 correspondenly
            nonzero_across_axis_0 = np.any(img, axis=(1, 2)).nonzero()
            nonzero_across_axis_1 = np.any(img, axis=(0, 2)).nonzero()
            nonzero_across_axis_2 = np.any(img, axis=(0, 1)).nonzero()

            nonzero_across_axis_0 = nonzero_across_axis_0[0]
            nonzero_across_axis_1 = nonzero_across_axis_1[0]
            nonzero_across_axis_2 = nonzero_across_axis_2[0]

            # If any axis contains only zero voxels than image is blank
            bbox = np.array([[-1, -1, -1], [0, 0, 0]])
            if nonzero_across_axis_0.size == 0:
                return bbox

            bbox[:, 0] = nonzero_across_axis_0[[0, -1]]
            bbox[:, 1] = nonzero_across_axis_1[[0, -1]]
            bbox[:, 2] = nonzero_across_axis_2[[0, -1]]

            return bbox

        bboxes = np.zeros((image.data.shape[0],) + (2, 3))
        for i in range(image.data.shape[0]):
            bboxes[i] = bbox3(image.data[i, :, :, :])

        bbox_min = np.min(bboxes[:, 0, :], axis=0).ravel().astype(int)
        bbox_max = np.max(bboxes[:, 1, :], axis=0).ravel().astype(int)
        bbox = np.zeros((2, 3), dtype=int)
        bbox[0] = bbox_min
        bbox[1] = bbox_max

        image.data = image.data[:, bbox[0, 0]:bbox[1, 0], bbox[0, 1]:bbox[1, 1], bbox[0, 2]:bbox[1, 2]]

        image.metadata['box'] = bbox

        return image


class NormalizeBrats(Preprocessor):
    __provider__ = "normalize_brats"

    _MASK_OPTIONS = {
        'none': 0,
        'nullify': 1,
        'ignore': 2,
        'all': 3,
    }

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'masked': StringField(optional=True, choices=NormalizeBrats._MASK_OPTIONS.keys(),
                                  default=False,
                                  description='Does not apply normalization to zero values. '
                                              'Applicable for brain tumor segmentation models'),
            'cutoff': NumberField(optional=True, default=0, min_value=0,
                                  description='Species range of values - [-cutoff, cutoff]'),
            'shift_value': NumberField(optional=True, default=0, description='Specifies shift value'),
            'normalize_value': NumberField(optional=True, default=1, description='Specifies normalize value')
        })

        return parameters

    def configure(self):
        self.masked = NormalizeBrats._MASK_OPTIONS[self.get_value_from_config('masked')]
        self.cutoff = self.get_value_from_config('cutoff')
        self.shift_value = self.get_value_from_config('shift_value')
        self.normalize_value = self.get_value_from_config('normalize_value')

    def process(self, image, annotation_meta=None):
        image.data = self.normalize_img(image.data)
        return image

    def normalize_img(self, image):
        for channel in range(image.shape[0]):
            img = image[channel, :, :, :].copy()
            if self.masked in (2, 3):
                mask = img > 0
                image_masked = np.ma.masked_array(img, ~mask)
                mean, std = np.mean(image_masked), np.std(image_masked)
            else:
                mean, std = np.mean(img), np.std(img)

            img -= mean
            img /= std

            if self.cutoff > 0:
                img = np.clip(img, -self.cutoff, self.cutoff) # pylint: disable=E1130
            img += self.shift_value
            img /= self.normalize_value
            if self.masked in (1, 3):
                mask = image[channel, :, :, :] > 0
                img[~mask] = 0
            image[channel, :, :, :] = img

        return image


class SwapModalitiesBrats(Preprocessor):
    __provider__ = 'swap_modalities'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'modality_order': ListField(
                value_type=NumberField(value_type=int, min_value=0, max_value=3),
                validate_values=True,
                description="Specifies order of modality according to model input"
            )
        })

        return parameters

    def configure(self):
        self.modal_order = self.get_value_from_config('modality_order')
        if len(self.modal_order) != 4:
            raise ConfigError('{} supports only 4 modality, but found {}'
                              .format(self.__provider__, len(self.modal_order)))
        if len(self.modal_order) != len(set(self.modal_order)):
            raise ConfigError('Incorrect modality index found in {} for {}. Indexes must be unique'
                              .format(self.modal_order, self.__provider__))

    def process(self, image, annotation_meta=None):
        image.data = self.swap_modalities(image.data)
        return image

    def swap_modalities(self, image):
        image = image[self.modal_order, :, :, :]
        return image
