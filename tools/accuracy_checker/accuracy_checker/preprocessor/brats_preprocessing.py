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

import numpy as np
from scipy.ndimage import interpolation

from ..config import ConfigError, BaseField, NumberField, ListField, BoolField
from ..preprocessor import Preprocessor
from ..utils import get_or_parse_value


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
        self.shape = get_or_parse_value(self.config.get('size'), (128, 128, 128), casting_type=int)
        self.shape = (4,) + self.shape

    def process(self, image, annotation_meta=None):
        data = np.asarray(image.data)
        assert (len(data.shape) == len(self.shape)), \
            'Image shape - {}, resize shape - {}'.format(data.shape, self.shape)

        factor = [float(o) / i for i, o in zip(data.shape, self.shape)]
        image.data = interpolation.zoom(data, zoom=factor, order=1)

        return image


class CropBraTS(Preprocessor):
    __provider__ = 'crop_brats'

    def process(self, image, annotation_meta=None):
        def bbox3(img):
            rows = np.where(np.any(np.any(img, axis=1), axis=1))
            cols = np.where(np.any(np.any(img, axis=0), axis=1))
            slices = np.where(np.any(np.any(img, axis=0), axis=0))

            if rows[0].shape[0] > 0:
                rmin, rmax = rows[0][[0, -1]]
                cmin, cmax = cols[0][[0, -1]]
                smin, smax = slices[0][[0, -1]]

                return np.array([[rmin, cmin, smin], [rmax, cmax, smax]])
            return np.array([[-1, -1, -1], [0, 0, 0]])

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

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'masked': BoolField(optional=True, default=False,
                                description='Does not apply normalization to zero values. '
                                            'Applicable for brain tumor segmentation models'),
            'cutoff': NumberField(optional=True,
                                  description='Species range of values - [-cutoff, cutoff]'),
            'shift_value': NumberField(optional=True, default=0, description='Specifies shift value'),
            'normalize_value': NumberField(optional=True, default=1, description='Specifies normalize value')
        })

        return parameters

    def configure(self):
        self.masked = self.config.get('masked')
        self.cutoff = self.config.get('cutoff', None)
        self.shift_value = self.config.get('shift_value')
        self.normalize_value = self.config.get('normalize_value')

    def process(self, image, annotation_meta=None):
        image.data = self.normalize_img(image.data)
        return image

    def normalize_img(self, image):
        image_copy = image.copy()
        for channel in range(image.shape[0]):
            img = image[channel, :, :, :].copy()
            if self.masked:
                mask = img > 0
                image_masked = np.ma.masked_array(img, ~mask)
                mean, std = np.mean(image_masked), np.std(image_masked)
            else:
                mean, std = np.mean(img), np.std(img)

            img -= mean
            img /= std

            if self.cutoff:
                img = np.clip(img, -self.cutoff, self.cutoff)
            img += self.shift_value
            img /= self.normalize_value
            if self.masked:
                img[~mask] = 0
            image[channel, :, :, :] = img

        return image


class SwapModalitiesBrats(Preprocessor):
    __provider__ = 'swap_modalities'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'modality_order': ListField(optional=True, default=None,
                                        description="Specifies order of modality according to model input")
        })

        return parameters

    def configure(self):
        self.modal_order = self.get_value_from_config('modality_order')
        if len(self.modal_order) != 4:
            raise ConfigError('{} supports only 4 modality, but found {}'
                              .format(self.__provider__, len(self.modal_order)))
        if max(self.modal_order) != 3 or min(self.modal_order) != 0:
            raise ConfigError('Incorrect modality index found in {} for {}'
                              .format(self.modal_order, self.__provider__))
        if len(self.modal_order) != len(set(self.modal_order)):
            raise ConfigError('Incorrect modality index found in {} for {}. Indexes must be unique'
                              .format(self.modal_order, self.__provider__))

    def process(self, image, annotation_meta=None):
        if self.modal_order is not None:
            image.data = self.swap_modalities(image.data)
        return image

    def swap_modalities(self, image):
        order = self.modal_order if image.shape[0] == 4 else [0]
        image = image[order, :, :, :]
        return image
