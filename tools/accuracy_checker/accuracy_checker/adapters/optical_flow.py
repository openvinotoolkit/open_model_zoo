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
from ..representation import OpticalFlowPrediction


class PWCNetAdapter(Adapter):
    __provider__ = 'pwcnet'

    def process(self, raw, identifiers, frame_meta):
        raw_outputs = self._extract_predictions(raw, frame_meta)
        result = []
        for identifier, flow in zip(identifiers, raw_outputs['pwcnet/mul_6']):
            if flow.shape[0] == 2:
                flow = np.transpose(flow, (1, 2, 0))
            result.append(OpticalFlowPrediction(identifier, flow))

        return result
