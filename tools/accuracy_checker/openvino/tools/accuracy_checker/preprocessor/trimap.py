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
from .preprocessor import Preprocessor
from ..config import NumberField


class AlphaChannel(Preprocessor):
    __provider__ = 'alpha'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'channel': NumberField(optional=True, default=3, description='Alpha-channel number'),
        })
        return params

    def configure(self):
        self.channel = int(self.get_value_from_config('channel'))

    def process(self, image, annotation_meta=None):
        image.metadata.update({'alpha': image.data[:, :, self.channel]})
        image.data = image.data[:, :, 0:self.channel]

        return image


class TrimapPreprocessor(Preprocessor):
    __provider__ = 'trimap'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'cut_treshold': NumberField(optional=True, default=0.1, description='Alpha-channel cut treshold'),
            'keep_treshold': NumberField(optional=True, default=0.9, description='Alpha-channel keep treshold'),
        })
        return params

    def configure(self):
        self.cut_treshold = self.get_value_from_config('cut_treshold')
        self.keep_treshold = self.get_value_from_config('keep_treshold')

    def process(self, image, annotation_meta=None):
        alpha = image.metadata['alpha']

        cut = alpha < self.cut_treshold * 255
        keep = alpha > self.keep_treshold * 255
        cal = (alpha >= self.cut_treshold * 255) * (alpha <= self.keep_treshold * 255)

        trimap = np.stack([cut, cal, keep], 2)
        image.data = np.concatenate([image.data, trimap], 2)

        trimap = np.argmax(trimap, 2)
        image.metadata.update({'tmap': trimap})

        return image
