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

import numpy as np

from ..adapters import Adapter
from ..config import BoolField, StringField, NumberField
from ..representation import ClassificationPrediction, ArgMaxClassificationPrediction
from ..utils import softmax


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
            'fixed_output': BoolField(
                optional=True, default=False, description="special mode to gather predictions from specified index"
            ),
            'fixed_output_index': NumberField(
                optional=True, default=0, description="Output index in fixed_output mode"
            ),
            'block': BoolField(
                optional=True, default=False, description="process whole batch as a single data block"
            ),
            'label_as_array': BoolField(
                optional=True, default=False, description="produce ClassificationPrediction's label as array"
            ),
            'classification_output': StringField(optional=True, description='target output layer name'),
            'multi_label_threshold': NumberField(
                optional=True, value_type=float,
                description='threshold for treating classification as multi label problem'),
            'do_softmax': BoolField(
                optional=True, description='apply softmax on probabilities in logits format', default=False)
        })

        return parameters

    def configure(self):
        self.argmax_output = self.get_value_from_config('argmax_output')
        self.block = self.get_value_from_config('block')
        self.classification_out = self.get_value_from_config('classification_output')
        self.fixed_output = self.get_value_from_config('fixed_output')
        self.fixed_output_index = int(self.get_value_from_config('fixed_output_index'))
        self.label_as_array = self.get_value_from_config('label_as_array')
        self.do_softmax = self.get_value_from_config('do_softmax')
        self.multilabel_thresh = self.get_value_from_config('multi_label_threshold')
        self.output_verified = False

    def select_output_blob(self, outputs):
        self.output_verified = True
        if self.classification_out:
            self.classification_out = self.check_output_name(self.classification_out, outputs)
            return
        super().select_output_blob(outputs)
        self.classification_out = self.output_blob
        return

    def process(self, raw, identifiers, frame_meta):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model
            frame_meta: list of meta information about each frame
        Returns:
            list of ClassificationPrediction objects
        """
        if not self.output_verified:
            self.select_output_blob(raw)
        multi_infer = frame_meta[-1].get('multi_infer', False) if frame_meta else False
        raw_prediction = self._extract_predictions(raw, frame_meta)
        prediction = raw_prediction[self.output_blob]
        if multi_infer:
            prediction = np.mean(prediction, axis=0)
        if len(np.shape(prediction)) == 1:
            prediction = np.expand_dims(prediction, axis=0)
        prediction = np.reshape(prediction, (prediction.shape[0], -1))

        result = []
        if self.block:
            result.append(self.prepare_representation(identifiers[0], prediction))
        else:
            for identifier, output in zip(identifiers, prediction):
                result.append(self.prepare_representation(identifier, output))

        return result

    def prepare_representation(self, identifier, prediction):
        if self.argmax_output:
            single_prediction = ArgMaxClassificationPrediction(identifier, prediction)
        elif self.fixed_output:
            single_prediction = ArgMaxClassificationPrediction(identifier,
                                                               prediction[:, self.fixed_output_index])
        else:
            if self.do_softmax:
                prediction = softmax(prediction)
            single_prediction = ClassificationPrediction(
                identifier, prediction, self.label_as_array,
                multilabel_threshold=self.multilabel_thresh)
        return single_prediction

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


class MaskToBinaryClassification(Adapter):
    __provider__ = 'mask_to_binary_classification'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'threshold': NumberField(optional=True, default=0.5, min_value=0, max_value=1)
        })
        return params

    def configure(self):
        self.threshold = self.get_value_from_config('threshold')

    def process(self, raw, identifiers, frame_meta):
        raw_outputs = self._extract_predictions(raw, frame_meta)
        results = []
        for identifier, mask in zip(identifiers, raw_outputs[self.output_blob]):
            prob = np.max(mask)
            results.append(ArgMaxClassificationPrediction(identifier, int(prob >= self.threshold)))

        return results
