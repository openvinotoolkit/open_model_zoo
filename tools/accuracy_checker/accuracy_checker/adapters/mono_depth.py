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

from .adapter import Adapter
from ..representation import DepthEstimationPrediction


class MonoDepthAdapter(Adapter):
    __provider__ = 'mono_depth'

    def process(self, raw, identifiers, frame_meta):
        raw_prediction = self._extract_predictions(raw, frame_meta)
        self.select_output_blob(raw_prediction)
        batch_prediction = raw_prediction[self.output_blob]
        result = []
        for identifier, prediction in zip(identifiers, batch_prediction):
            result.append(DepthEstimationPrediction(identifier, prediction))

        return result
