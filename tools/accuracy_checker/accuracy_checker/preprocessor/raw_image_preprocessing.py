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
from .preprocessor import Preprocessor
from ..config import NumberField


class PackBayerImage(Preprocessor):
    __provider__ = 'pack_raw_image'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'black_level': NumberField(min_value=1, description='black level for removal'),
            'ratio': NumberField(optional=True, default=1, description='exposure scale ratio')
        })
        return params

    def configure(self):
        self.black_level = self.get_value_from_config('black_level')
        self.ratio = self.get_value_from_config('ratio')

    def process(self, image, annotation_meta=None):
        ratio = (annotation_meta or {}).get('ratio', self.ratio)
        im = image.data
        im = np.maximum(im - self.black_level, 0) / (16383 - self.black_level)
        im = np.expand_dims(im, axis=2)
        height, width = im.shape[:2]
        out = np.concatenate((
            im[0:height:2, 0:width:2, :],
            im[0:height:2, 1:width:2, :],
            im[1:height:2, 1:width:2, :],
            im[1:height:2, 0:width:2, :]), axis=2)
        image.data = out * ratio
        return image
