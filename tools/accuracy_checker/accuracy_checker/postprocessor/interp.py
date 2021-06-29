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
from ..config import StringField, NumberField, BoolField
from ..representation import ImageProcessingPrediction, ImageProcessingAnnotation
from .postprocessor import Postprocessor

interp_modes_func = {
    'linear': np.interp,
}


class Interpolation(Postprocessor):
    __provider__ = 'interpolation'
    annotation_types = (ImageProcessingAnnotation)
    prediction_types = (ImageProcessingPrediction)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'mode': StringField(
                optional=True, choices=interp_modes_func.keys(), default='linear',
                description="Interpolation mode: {}".format(', '.join(interp_modes_func))
            ),
            'target_min': NumberField(optional=True, default=0.0, description="Target range minimum"),
            'target_max': NumberField(optional=True, default=255.0, description="Target range maximum"),
            'as_log': BoolField(optional=True, default=False, description="log values before interpolation"),
        })

        return parameters

    def configure(self):
        self.interp_func = interp_modes_func[self.get_value_from_config('mode')]
        self.target_min = self.get_value_from_config('target_min')
        self.target_max = self.get_value_from_config('target_max')
        self.as_log = self.get_value_from_config('as_log')

    def process_image(self, annotation, prediction):
        def cast_func(entry):
            pass

        @cast_func.register(ImageProcessingPrediction)
        @cast_func.register(ImageProcessingAnnotation)
        def _(entry):
            val = entry.value
            if self.as_log:
                val_min = np.min(val)
                val_range = np.max(val) - np.min(val)
                val = np.log(val - val_min + val_range / 255.0)
            entry.value = self.interp_func(val, (np.min(val), np.max(val)), (self.target_min, self.target_max))

        for annotation_ in annotation:
            cast_func(annotation_)

        for prediction_ in prediction:
            cast_func(prediction_)

        return annotation, prediction
