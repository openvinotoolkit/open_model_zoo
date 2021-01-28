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
from ..config import NumberField, BoolField


class PackBayerImage(Preprocessor):
    __provider__ = 'pack_raw_image'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'black_level': NumberField(min_value=1, description='black level for removal'),
            'ratio': NumberField(optional=True, default=1, description='exposure scale ratio'),
            '9-channels': BoolField(optional=True, default=False, description='pack 9-channles inage')
        })
        return params

    def configure(self):
        self.black_level = self.get_value_from_config('black_level')
        self.ratio = self.get_value_from_config('ratio')
        self.nine_channels = self.get_value_from_config('9-channels')

    def process(self, image, annotation_meta=None):
        ratio = (annotation_meta or {}).get('ratio', self.ratio)
        im = image.data
        im = np.maximum(im - self.black_level, 0) / (16383 - self.black_level)
        im = np.expand_dims(im, axis=2)

        height, width = im.shape[:2]
        if not self.nine_channels:
            out = np.concatenate((
                im[0:height:2, 0:width:2, :],
                im[0:height:2, 1:width:2, :],
                im[1:height:2, 1:width:2, :],
                im[1:height:2, 0:width:2, :]), axis=2)
        else:
            H = (height // 6) * 6
            W = (width // 6) * 6

            out = np.zeros((H // 3, W // 3, 9))

            # 0 R
            out[0::2, 0::2, 0] = np.squeeze(im[0:H:6, 0:W:6])
            out[0::2, 1::2, 0] = np.squeeze(im[0:H:6, 4:W:6])
            out[1::2, 0::2, 0] = np.squeeze(im[3:H:6, 1:W:6])
            out[1::2, 1::2, 0] = np.squeeze(im[3:H:6, 3:W:6])

            # 1 G
            out[0::2, 0::2, 1] = np.squeeze(im[0:H:6, 2:W:6])
            out[0::2, 1::2, 1] = np.squeeze(im[0:H:6, 5:W:6])
            out[1::2, 0::2, 1] = np.squeeze(im[3:H:6, 2:W:6])
            out[1::2, 1::2, 1] = np.squeeze(im[3:H:6, 5:W:6])

            # 1 B
            out[0::2, 0::2, 2] = np.squeeze(im[0:H:6, 1:W:6])
            out[0::2, 1::2, 2] = np.squeeze(im[0:H:6, 3:W:6])
            out[1::2, 0::2, 2] = np.squeeze(im[3:H:6, 0:W:6])
            out[1::2, 1::2, 2] = np.squeeze(im[3:H:6, 4:W:6])

            # 4 R
            out[0::2, 0::2, 3] = np.squeeze(im[1:H:6, 2:W:6])
            out[0::2, 1::2, 3] = np.squeeze(im[2:H:6, 5:W:6])
            out[1::2, 0::2, 3] = np.squeeze(im[5:H:6, 2:W:6])
            out[1::2, 1::2, 3] = np.squeeze(im[4:H:6, 5:W:6])

            # 5 B
            out[0::2, 0::2, 4] = np.squeeze(im[2:H:6, 2:W:6])
            out[0::2, 1::2, 4] = np.squeeze(im[1:H:6, 5:W:6])
            out[1::2, 0::2, 4] = np.squeeze(im[4:H:6, 2:W:6])
            out[1::2, 1::2, 4] = np.squeeze(im[5:H:6, 5:W:6])

            out[:, :, 5] = np.squeeze(im[1:H:3, 0:W:3])
            out[:, :, 6] = np.squeeze(im[1:H:3, 1:W:3])
            out[:, :, 7] = np.squeeze(im[2:H:3, 0:W:3])
            out[:, :, 8] = np.squeeze(im[2:H:3, 1:W:3])
        image.data = out * ratio
        return image
