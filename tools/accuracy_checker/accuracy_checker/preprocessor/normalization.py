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

import numpy as np

from ..config import NormalizationArgsField, ConfigError, BoolField
from ..preprocessor import Preprocessor


def is_image_input(data):
    if np.isscalar(data):
        return False
    if len(np.shape(data)) not in [2, 3]:
        return False
    if len(np.shape(data)) == 3 and np.shape(data)[-1] not in [1, 3, 4]:
        return False
    return True


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
            'mean': NormalizationArgsField(
                optional=True,
                description="Values which will be subtracted from image channels. You can specify one "
                            "value for all channels or list of comma separated channel-wise values.",
                precomputed_args=Normalize.PRECOMPUTED_MEANS
            ),
            'std': NormalizationArgsField(
                optional=True,
                description="Specifies values, on which pixels will be divided. You can specify one value for all "
                            "channels or list of comma separated channel-wise values.",
                precomputed_args=Normalize.PRECOMPUTED_STDS,
                allow_zeros=False
            ),
            'images_only': BoolField(
                optional=True, default=False, description='in multi input mode, process only images.'
            )
        })
        return parameters

    def configure(self):
        self.mean = self.get_value_from_config('mean')
        self.std = self.get_value_from_config('std')
        self.images_only = self.get_value_from_config('images_only')
        if not self.mean and not self.std:
            raise ConfigError('mean or std value should be provided')

    def process(self, image, annotation_meta=None):
        def process_data(data, mean, std):
            if (
                    self.images_only and not is_image_input(data)
            ):
                return data
            if self.mean:
                data = data - mean
            if self.std:
                data = data / std

            return data

        image.data = process_data(image.data, self.mean, self.std) if not isinstance(image.data, list) else [
            process_data(data_fragment, self.mean, self.std) for data_fragment in image.data
        ]

        return image


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
