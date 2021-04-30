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
from .base_representation import BaseRepresentation
from ..data_readers import BaseReader


class NoiseSuppressionRepresentation(BaseRepresentation):
    pass


class NoiseSuppressionAnnotation(NoiseSuppressionRepresentation):
    def __init__(self, identifier, gt_path):
        super().__init__(identifier)
        self._gt_path = gt_path
        self._value = None

    @property
    def value(self):
        if self._value is not None:
            return self._value
        data_source = self.metadata.get('additional_data_source')
        if not data_source:
            data_source = self.metadata['data_source']
        loader = BaseReader.provide('wav_reader', data_source, {'type': 'wav_reader', 'mono': True, 'to_float': True})
        data, _ = loader.read(self._gt_path)
        return np.squeeze(data)

    @value.setter
    def value(self, value):
        self._value = value


class NoiseSuppressionPrediction(NoiseSuppressionRepresentation):
    def __init__(self, identifier, value):
        super().__init__(identifier)
        self.value = value
