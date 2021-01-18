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

from functools import singledispatch
import numpy as np
from ..config import StringField
from ..representation import (
    DetectionAnnotation,
    DetectionPrediction,
    TextDetectionPrediction,
    TextDetectionAnnotation,
    ClassificationAnnotation,
    ClassificationPrediction
)
from .postprocessor import Postprocessor

round_policies_func = {
    'nearest': np.rint,
    'nearest_to_zero': np.trunc,
    'lower': np.floor,
    'greater': np.ceil
}


class CastToInt(Postprocessor):
    __provider__ = 'cast_to_int'
    annotation_types = (DetectionAnnotation, TextDetectionAnnotation, ClassificationAnnotation)
    prediction_types = (DetectionPrediction, TextDetectionPrediction, ClassificationPrediction)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'round_policy': StringField(
                optional=True, choices=round_policies_func.keys(), default='nearest',
                description="Method for rounding: {}".format(', '.join(round_policies_func))
            )
        })

        return parameters

    def configure(self):
        self.round_func = round_policies_func[self.get_value_from_config('round_policy')]

    def process_image(self, annotation, prediction):
        @singledispatch
        def cast_func(entry):
            pass

        @cast_func.register(DetectionAnnotation)
        @cast_func.register(DetectionPrediction)
        def _(entry):
            entry.x_mins = self.round_func(entry.x_mins)
            entry.x_maxs = self.round_func(entry.x_maxs)
            entry.y_mins = self.round_func(entry.y_mins)
            entry.y_maxs = self.round_func(entry.y_maxs)

        @cast_func.register(TextDetectionAnnotation)
        @cast_func.register(TextDetectionPrediction)
        def _(entry):
            entry.points = self.round_func(entry.points)

        @cast_func.register(ClassificationAnnotation)
        @cast_func.register(ClassificationPrediction)
        def _(entry):
            entry.label = self.round_func(entry.label)

        for annotation_ in annotation:
            cast_func(annotation_)

        for prediction_ in prediction:
            cast_func(prediction_)

        return annotation, prediction
