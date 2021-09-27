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
from ..config import BoolField, ListField, StringField
from ..representation import RegressionPrediction


class RegressionAdapter(Adapter):
    """
    Class for converting output of regression model to RegressionPrediction representation
    """

    __provider__ = 'regression'
    prediction_types = (RegressionPrediction, )

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({'keep_shape': BoolField(optional=True, default=False)})
        return params

    def configure(self):
        self.keep_shape = self.get_value_from_config('keep_shape')

    def process(self, raw, identifiers, frame_meta):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model
            frame_meta: list of meta information about each frame
        Returns:
            list of RegressionPrediction objects
        """
        predictions = self._extract_predictions(raw, frame_meta)
        self.select_output_blob(predictions)
        predictions = predictions[self.output_blob]
        if len(np.shape(predictions)) == 1 or (self.keep_shape and np.shape(predictions)[0] != len(identifiers)):
            predictions = np.expand_dims(predictions, axis=0)
        if not self.keep_shape:
            predictions = np.reshape(predictions, (predictions.shape[0], -1))

        result = []
        for identifier, output in zip(identifiers, predictions):
            prediction = RegressionPrediction(identifier, output)
            result.append(prediction)

        return result


class MultiOutputRegression(Adapter):
    __provider__ = 'multi_output_regression'
    prediction_types = (RegressionPrediction,)

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'outputs': ListField(value_type=str, allow_empty=False, description='list of target output names')
        })
        return params

    def configure(self):
        self.output_list = self.get_value_from_config('outputs')

    def process(self, raw, identifiers, frame_meta):
        raw_outputs = self._extract_predictions(raw, frame_meta)
        result = []
        for batch_id, identfier in enumerate(identifiers):
            res_dict = {}
            for output_name in self.output_list:
                res_dict.update({output_name: raw_outputs[output_name][batch_id]})
            result.append(RegressionPrediction(identfier, res_dict))
        return result


class KaldiFeatsRegression(Adapter):
    __provider__ = 'kaldi_feat_regression'

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'target_out': StringField(optional=True, description='target output name'),
            'flattenize': BoolField(optional=True, description='make output flatten')
        })
        return params

    def configure(self):
        self.target_out = self.get_value_from_config('target_out')
        self.flattenize = self.get_value_from_config('flattenize')

    def process(self, raw, identifiers, frame_meta):
        """
        Args:
            identifiers: list of input data identifiers
            raw: output of model
            frame_meta: list of meta information about each frame
        Returns:
            list of RegressionPrediction objects
        """
        self.select_output_blob(raw if isinstance(raw, dict) else raw[0])
        predictions = self._extract_predictions(raw, frame_meta)
        predictions = predictions[self.output_blob]

        result = []
        for identifier, output in zip(identifiers, predictions):
            if self.flattenize:
                output = output.flatten()
            prediction = RegressionPrediction(identifier, output)
            result.append(prediction)

        return result

    def _extract_predictions(self, outputs_list, meta):
        is_multi_infer = meta[-1].get('multi_infer', False) if meta else False
        context_shift_left = meta[-1].get('context_left', 0)
        context_shift_right = meta[-1].get('context_right', 0)
        context_shift = context_shift_right + context_shift_left
        if not is_multi_infer:
            output_map = outputs_list[0] if not isinstance(outputs_list, dict) else outputs_list
        else:
            output_map = {
                self.output_blob: np.expand_dims(
                    np.concatenate([out[self.output_blob] for out in outputs_list], axis=0), 0)
            }
        if context_shift:
            out = output_map[self.output_blob]
            out = out[context_shift_left+context_shift:, ...] if np.ndim(out) == 2 else out[:, context_shift:, ...]
            output_map[self.output_blob] = out

        return output_map

    def select_output_blob(self, outputs):
        if self.target_out:
            self.output_blob = self.target_out
        if self.output_blob is None:
            self.output_blob = next(iter(outputs))
