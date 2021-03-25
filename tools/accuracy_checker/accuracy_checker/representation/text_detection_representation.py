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
from ..utils import remove_difficult
from .base_representation import BaseRepresentation


class TextDetectionRepresentation(BaseRepresentation):
    def __init__(self, identifier='', points=None, description=''):
        super().__init__(identifier)
        self.points = points if points is not None else []
        if isinstance(points, list):
            self.points = np.array(points)
        self.description = description

    def remove(self, indexes):
        self.points = np.delete(self.points, indexes, axis=0)
        difficult = self.metadata.get('difficult_boxes')
        if not difficult:
            return
        self.metadata['difficult_boxes'] = remove_difficult(difficult, indexes)
        self.description = np.delete(self.description, indexes)

    @property
    def boxes(self):
        if np.size(self.points) == 0:
            return []

        x_coords = np.reshape(self.points[:, :, 0], (-1, 4))
        y_coords = np.reshape(self.points[:, :, 1], (-1, 4))
        x_mins = np.min(x_coords, axis=1)
        x_maxs = np.max(x_coords, axis=1)
        y_mins = np.min(y_coords, axis=1)
        y_maxs = np.max(y_coords, axis=1)

        return [[x_min, y_min, x_max, y_max] for x_min, y_min, x_max, y_max in zip(x_mins, y_mins, x_maxs, y_maxs)]


class TextDetectionAnnotation(TextDetectionRepresentation):
    pass


class TextDetectionPrediction(TextDetectionRepresentation):
    def to_annotation(self, **kwargs):
        return TextDetectionAnnotation(self.identifier, self.points, self.description)
