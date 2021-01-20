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

from .postprocessor import PostprocessorWithSpecificTargets
from ..config import NumberField
from ..representation import FeaturesRegressionAnnotation, RegressionAnnotation, RegressionPrediction


class MinMaxRegressionNormalization(PostprocessorWithSpecificTargets):
    __provider__ = 'min_max_normalization'
    annotation_types = (FeaturesRegressionAnnotation, RegressionAnnotation)
    prediction_types = (RegressionPrediction, )

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'min': NumberField(optional=True, default=0, description='minimal value in range'),
            'max': NumberField(description='maximum value in range')
        })
        return params

    def configure(self):
        super().configure()
        self.min = self.get_value_from_config('min')
        self.max = self.get_value_from_config('max')

    def process_image(self, annotation, prediction):
        for ann in annotation:
            ann.value = (ann.value - self.min) / (self.max - self.min)

        for pred in prediction:
            pred.value = (pred.value - self.min) / (self.max - self.min)

        return annotation, prediction
