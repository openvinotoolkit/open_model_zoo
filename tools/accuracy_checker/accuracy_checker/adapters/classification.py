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

import numpy as np

from ..topology_types import ImageClassification
from ..adapters import Adapter
from ..config import BoolField
from ..representation import ClassificationPrediction, ArgMaxClassificationPrediction


class ClassificationAdapter(Adapter):
    """
    Class for converting output of classification model to ClassificationPrediction representation
    """
    __provider__ = 'classification'
    topology_types = (ImageClassification, )
    prediction_types = (ClassificationPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'argmax_output': BoolField(
                optional=True, default=False, description="identifier that model output is ArgMax layer"
            ),
        })

        return parameters

    def configure(self):
        self.argmax_output = self.get_value_from_config('argmax_output')

    def process(self, raw, identifiers=None, frame_meta=None):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model
            frame_meta: list of meta information about each frame
        Returns:
            list of ClassificationPrediction objects
        """
        prediction = self._extract_predictions(raw, frame_meta)[self.output_blob]
        prediction = np.reshape(prediction, (prediction.shape[0], -1))

        result = []
        for identifier, output in zip(identifiers, prediction):
            if self.argmax_output:
                single_prediction = ArgMaxClassificationPrediction(identifier, output[0])
            else:
                single_prediction = ClassificationPrediction(identifier, output)
            result.append(single_prediction)

        return result
