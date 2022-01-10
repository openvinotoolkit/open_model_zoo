"""
Copyright (c) 2018-2022 Intel Corporation

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
from ..representation import FacialLandmarksAnnotation, FacialLandmarksPrediction


class NormalizeLandmarksPoints(Postprocessor):
    __provider__ = 'normalize_landmarks_points'

    annotation_types = (FacialLandmarksAnnotation, )
    prediction_types = (FacialLandmarksPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'use_annotation_rect': BoolField(
                optional=True, default=False,
                description="Allows to use size of rectangle saved in annotation metadata for point scaling"
                            " instead source image size."
            )
        })

        return parameters

    def configure(self):
        self.use_annotation_rect = self.get_value_from_config('use_annotation_rect')

    def process_image(self, annotation, prediction):
        for target in annotation:
            height, width, _ = self.image_size
            x_start, y_start = 0, 0
            if self.use_annotation_rect:
                resized_box = annotation[0].metadata.get('rect')
                x_start, y_start, x_max, y_max = resized_box
                width = x_max - x_start
                height = y_max - y_start

            target.x_values = (
                (np.array(target.x_values, dtype=float) - x_start) / np.maximum(width, np.finfo(np.float64).eps)
            )
            target.y_values = (
                (np.array(target.y_values, dtype=float) - y_start) / np.maximum(height, np.finfo(np.float64).eps)
            )

        return annotation, prediction
