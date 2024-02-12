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

import numpy as np
from .base_representation import BaseRepresentation


class FacialLandmarksHeatMapRepresentation(BaseRepresentation):
    def __init__(self, identifier='', x_values=None, y_values=None, heatmap=None):
        super().__init__(identifier)
        self.x_values = x_values if np.size(x_values) > 0 else []
        self.y_values = y_values if np.size(y_values) > 0 else []
        self.heatmap = heatmap

    @property
    def size(self):
        return len(self.x_values)

class FacialLandmarksHeatMapAnnotation(FacialLandmarksHeatMapRepresentation):
    def normalization_coef(self, is_2d=False):
        min_x, max_x = np.min(self.x_values), np.max(self.x_values)
        min_y, max_y = np.min(self.y_values), np.max(self.y_values)
        return np.sqrt((max_x - min_x) ** 2 + (max_y - min_y) ** 2)

class FacialLandmarksHeatMapPrediction(FacialLandmarksHeatMapRepresentation):

    def to_annotation(self, **kwargs):
        return FacialLandmarksHeatMapPrediction(self.identifier, self.x_values, self.y_values)
