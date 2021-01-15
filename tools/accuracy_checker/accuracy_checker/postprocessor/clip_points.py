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
from ..config import BoolField, NumberField
from ..representation import TextDetectionAnnotation, TextDetectionPrediction
from ..utils import get_size_from_config
from .postprocessor import PostprocessorWithSpecificTargets


class ClipPoints(PostprocessorWithSpecificTargets):
    __provider__ = 'clip_points'

    annotation_types = (TextDetectionAnnotation, )
    prediction_types = (TextDetectionPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'dst_width': NumberField(
                value_type=int, optional=True, min_value=1, description="Destination width for points clipping."
            ),
            'dst_height': NumberField(
                value_type=int, optional=True, min_value=1, description="Destination height for points clipping."
            ),
            'size': NumberField(
                value_type=int, optional=True, min_value=1,
                description="Destination size for points clipping for both dimensions."
            ),
            'points_normalized': BoolField(
                optional=True, default=False, description="Flags shows points are normalized or not."
            )
        })

        return parameters

    def configure(self):
        self.dst_height, self.dst_width = get_size_from_config(self.config, allow_none=True)
        self.points_normalized = self.get_value_from_config('points_normalized')

    def process_image(self, annotation, prediction):
        target_width = self.dst_width or self.image_size[1] - 1
        target_height = self.dst_height or self.image_size[0] - 1

        max_width = target_width if not self.points_normalized else 1
        max_height = target_height if not self.points_normalized else 1
        for target in annotation:
            points = []
            for polygon in target.points:
                polygon[:, 0] = np.clip(polygon[:, 0], 0, max_width)
                polygon[:, 1] = np.clip(polygon[:, 1], 0, max_height)
                points.append(polygon)
            target.points = points
        for target in prediction:
            points = []
            for polygon in target.points:
                polygon[:, 0] = np.clip(polygon[:, 0], 0, max_width)
                polygon[:, 1] = np.clip(polygon[:, 1], 0, max_height)
                points.append(polygon)
            target.points = points

        return annotation, prediction
