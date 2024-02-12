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

import cv2
from .postprocessor import PostprocessorWithSpecificTargets
from ..config import NumberField
from ..representation import BackgroundMattingAnnotation, BackgroundMattingPrediction


class RescaleMask(PostprocessorWithSpecificTargets):
    __provider__ = 'rescale_mask'
    annotation_types = (BackgroundMattingAnnotation, )
    prediction_types = (BackgroundMattingPrediction, )

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'min': NumberField(optional=True, default=0, description='minimal value in range'),
            'max': NumberField(description='maximum value in range')
        })
        return params

    def configure(self):
        self.min = self.get_value_from_config('min')
        self.max = self.get_value_from_config('max')

    def process_image(self, annotation, prediction):
        for ann in annotation:
            ann.mask = cv2.normalize(ann.mask, None, self.min, self.max, cv2.NORM_MINMAX)

        for pred in prediction:
            pred.mask = cv2.normalize(pred.mask, None, self.min, self.max, cv2.NORM_MINMAX)

        return annotation, prediction
