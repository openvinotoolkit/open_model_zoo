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
from .adapter import Adapter
from ..representation import SalientRegionPrediction


class SalientObjectDetection(Adapter):
    __provider__ = 'salient_objects_detection'

    def process(self, raw, identifiers, frame_meta):
        raw_output = self._extract_predictions(raw, frame_meta)
        result = []
        for identifier, mask in zip(identifiers, raw_output[self.output_blob]):
            mask = 1/(1 + np.exp(-mask))
            result.append(SalientRegionPrediction(identifier, np.round(np.squeeze(mask)).astype(np.uint8)))

        return result
