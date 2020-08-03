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

from ..config import BoolField
from ..postprocessor.postprocessor import Postprocessor
from ..representation import (
    DetectionPrediction, DetectionAnnotation, ActionDetectionPrediction, ActionDetectionAnnotation
)


class NormalizeBoxes(Postprocessor):
    __provider__ = 'normalize_boxes'

    annotation_types = (DetectionAnnotation, ActionDetectionAnnotation)
    prediction_types = (DetectionPrediction, ActionDetectionPrediction)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'use_annotation_rect': BoolField(
                optional=True, default=False,
                description="Allows to use size of rectangle saved in annotation metadata "
                            "for point scaling instead source image size."
            )
        })

        return parameters

    def configure(self):
        self.use_annotation_rect = self.config.get('use_annotation_rect', False)

    def process_image(self, annotation, prediction):
        height, width, _ = self.image_size
        for target in annotation:
            target.x_mins /= np.maximum(float(width), np.finfo(np.float64).eps)
            target.x_maxs /= np.maximum(float(width), np.finfo(np.float64).eps)
            target.y_mins /= np.maximum(float(height), np.finfo(np.float64).eps)
            target.y_maxs /= np.maximum(float(height), np.finfo(np.float64).eps)

        return annotation, prediction
