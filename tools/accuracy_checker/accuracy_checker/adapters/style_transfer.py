"""
Copyright (c) 2020 Intel Corporation

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
from ..adapters import Adapter
from ..representation import StyleTransferPrediction


class StyleTransferAdapter(Adapter):
    __provider__ = 'style_transfer'
    prediction_types = (StyleTransferPrediction, )

    def process(self, raw, identifiers=None, frame_meta=None):
        result = []
        raw_outputs = self._extract_predictions(raw, frame_meta)[self.output_blob]
        for identifier, img in zip(identifiers, raw_outputs):
            img = self._basic_postprocess(img)
            result.append(StyleTransferPrediction(identifier, img))
        return result

    @classmethod
    def _basic_postprocess(cls, img):
        if img.shape[0] == 3:
            img = np.transpose(img, (1, 2, 0))
        return img
