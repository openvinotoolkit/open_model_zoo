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

from  .adapter import Adapter
from ..config import StringField
from ..representation import  NoiseSuppressionPrediction


class NoiseSuppressionAdapter(Adapter):
    __provider__ = 'noise_suppression'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'output_blob': StringField(optional=True)
        })
        return params

    def configure(self):
        self._output_blob = self.get_value_from_config('output_blob')

    def process(self, raw, identifiers, frame_meta):
        if self._output_blob is None:
            self._output_blob = self.output_blob
        raw_prediction = self._extract_predictions(raw, frame_meta)
        result = []
        for identifier, signal in zip(identifiers, raw_prediction[self._output_blob]):
            result.append(NoiseSuppressionPrediction(identifier, signal))
        return result

    def _extract_predictions(self, outputs_list, meta):
        if not isinstance(outputs_list, list):
            return outputs_list
        if not meta[0].get('multi_infer', False):
            return outputs_list[0]
        out_signal = np.expand_dims(np.concatenate([np.squeeze(out[self._output_blob]) for out in outputs_list], 0), 0)
        return {self._output_blob: out_signal}
