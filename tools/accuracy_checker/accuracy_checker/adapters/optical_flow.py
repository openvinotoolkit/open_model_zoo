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
from .adapter import Adapter
from ..config import StringField
from ..representation import OpticalFlowPrediction


class PWCNetAdapter(Adapter):
    __provider__ = 'pwcnet'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'flow_out': StringField(optional=True, description='target output layer')
        })
        return params

    def configure(self):
        self.flow_out = self.get_value_from_config('flow_out')

    def process(self, raw, identifiers, frame_meta):
        raw_outputs = self._extract_predictions(raw, frame_meta)
        if self.flow_out is None:
            self.select_output_blob(raw_outputs)
            self.flow_out = self.output_blob
        result = []
        for identifier, flow in zip(identifiers, raw_outputs[self.flow_out]):
            if flow.shape[0] == 2:
                flow = np.transpose(flow, (1, 2, 0))
            result.append(OpticalFlowPrediction(identifier, flow))

        return result
