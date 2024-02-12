"""
Copyright (c) 2018-2024 Intel Corporation

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

from ..config import NumberField
from .preprocessor import Preprocessor


class JPEGCompression(Preprocessor):
    __provider__ = 'jpeg_compression'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'quality_factor': NumberField(
                value_type=int, optional=True, min_value=0, max_value=100,
                description="Quality of compression, from 0 to 100 (the higher is the better)."
            )
        })

        return parameters

    def configure(self):
        self.quality_factor = self.get_value_from_config('quality_factor')

    def process(self, image, annotation_meta=None):
        if isinstance(image.data, list):
            image.data = [
                self.process_data(fragment, self.quality_factor)
                for fragment in image.data
            ]
        else:
            image.data = self.process_data(
                image.data, self.quality_factor)

        return image

    @staticmethod
    def process_data(data, quality_factor):
        channels = data.shape[2]
        _, encimg = cv2.imencode('.jpg', data, [int(cv2.IMWRITE_JPEG_QUALITY), quality_factor])
        img_lq = cv2.imdecode(encimg, 0 if channels == 1 else 3)
        if channels == 1:
            img_lq = np.expand_dims(img_lq, axis=2)

        return img_lq
