"""
Copyright (C) 2023 KNS Group LLC (YADRO)

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

from ..config import NumberField
from .preprocessor import Preprocessor


class AutoExposure(Preprocessor):
    __provider__ = 'autoexposure'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'key': NumberField(value_type=float, optional=True, description="Destination width"),
            'k': NumberField(value_type=int, optional=True, min_value=1, description="Downsampling amount"),
        })
        return parameters

    def configure(self):
        self.params = {'key': self.get_value_from_config('key'),
                       'k': self.get_value_from_config('k')}

    def process(self, image, annotation_meta=None):
        exposure = self.autoexposure(image.data[0], self.params)
        image.data[0] = image.data[0] * exposure
        annotation_meta['exposure'] = exposure
        return image

    @staticmethod
    def autoexposure(image, params):
        def luminance(r, g, b):
            return 0.212671 * r + 0.715160 * g + 0.072169 * b

        eps = 1e-8
        key, k = params['key'], params['k']

        # Compute the luminance of each pixel
        r = image[..., 0]
        g = image[..., 1]
        b = image[..., 2]
        lum = luminance(r, g, b)

        # Downsample the image to minimize sensitivity to noise
        h = lum.shape[0]  # original height
        w = lum.shape[1]  # original width
        hk = (h + k // 2) // k  # down sampled height
        wk = (w + k // 2) // k  # down sampled width

        lk = np.zeros((hk, wk), dtype=lum.dtype)
        for i in range(hk):
            for j in range(wk):
                begin_h = i * h // hk
                begin_w = j * w // wk
                end_h = (i + 1) * h // hk
                end_w = (j + 1) * w // wk

                lk[i, j] = lum[begin_h:end_h, begin_w:end_w].mean()

        lum = lk

        # Keep only values greater than epsilon
        lum = lum[lum > eps]
        if lum.size == 0:
            return 1.

        # Compute the exposure value
        return float(key / np.exp2(np.log2(lum).mean()))
