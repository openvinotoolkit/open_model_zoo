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

import cv2
from PIL import Image
import numpy as np

from ..adapters import Adapter
from ..representation import SuperResolutionPrediction
from ..config import ConfigValidator, BoolField


class SuperResolutionAdapter(Adapter):
    __provider__ = 'super_resolution'
    prediction_types = (SuperResolutionPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'reverse_channels': BoolField(
                optional=True, default=False, description="Allow switching output image channels e.g. RGB to BGR"
            )
        })
        return parameters

    def validate_config(self):
        super().validate_config(on_extra_argument=ConfigValidator.IGNORE_ON_EXTRA_ARGUMENT)

    def configure(self):
        self.reverse_channels = self.get_value_from_config('reverse_channels')

    def process(self, raw, identifiers=None, frame_meta=None):
        result = []
        raw_outputs = self._extract_predictions(raw, frame_meta)
        for identifier, img_sr in zip(identifiers, raw_outputs[self.output_blob]):
            img_sr *= 255
            img_sr = np.clip(img_sr, 0., 255.)
            img_sr = img_sr.transpose((1, 2, 0)).astype(np.uint8)
            if self.reverse_channels:
                img_sr = cv2.cvtColor(img_sr, cv2.COLOR_BGR2RGB)
                img_sr = Image.fromarray(img_sr, 'RGB')
                img_sr = np.array(img_sr).astype(np.uint8)
            result.append(SuperResolutionPrediction(identifier, img_sr))

        return result
