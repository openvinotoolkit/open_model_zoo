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

from ..adapters import Adapter
from ..config import BoolField, StringField
from ..representation import ClassificationPrediction, ArgMaxClassificationPrediction


class ClassificationAdapter(Adapter):
    """
    Class for converting output of classification model to ClassificationPrediction representation
    """
    __provider__ = 'classification'
    prediction_types = (ClassificationPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'argmax_output': BoolField(
                optional=True, default=False, description="identifier that model output is ArgMax layer"
            ),
            'block': BoolField(
                optional=True, default=False, description="process whole batch as a single data block"
            ),
            'classification_output': StringField(optional=True, description='target output layer name')
        })

        return parameters

    def configure(self):
        self.argmax_output = self.get_value_from_config('argmax_output')
        self.block = self.get_value_from_config('block')
        self.classification_out = self.get_value_from_config('classification_output')

    def process(self, raw, identifiers, frame_meta):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model
            frame_meta: list of meta information about each frame
        Returns:
            list of ClassificationPrediction objects
        """
        if self.classification_out is not None:
            self.output_blob = self.classification_out
        multi_infer = frame_meta[-1].get('multi_infer', False) if frame_meta else False
        raw_prediction = self._extract_predictions(raw, frame_meta)
        self.select_output_blob(raw_prediction)
        prediction = raw_prediction[self.output_blob]
        if multi_infer:
            prediction = np.mean(prediction, axis=0)
        if len(np.shape(prediction)) == 1:
            prediction = np.expand_dims(prediction, axis=0)
        prediction = np.reshape(prediction, (prediction.shape[0], -1))

        result = []
        if self.block:
            if self.argmax_output:
                single_prediction = ArgMaxClassificationPrediction(identifiers[0], prediction)
            else:
                single_prediction = ClassificationPrediction(identifiers[0], prediction)

            result.append(single_prediction)

        else:
            for identifier, output in zip(identifiers, prediction):
                if self.argmax_output:
                    single_prediction = ArgMaxClassificationPrediction(identifier, output[0])
                else:
                    single_prediction = ClassificationPrediction(identifier, output)
                result.append(single_prediction)

        return result

    @staticmethod
    def _extract_predictions(outputs_list, meta):
        is_multi_infer = meta[-1].get('multi_infer', False) if meta else False
        if not is_multi_infer:
            return outputs_list[0] if not isinstance(outputs_list, dict) else outputs_list

        output_map = {}
        for output_key in outputs_list[0].keys():
            output_data = np.asarray([output[output_key] for output in outputs_list])
            output_map[output_key] = output_data

        return output_map
