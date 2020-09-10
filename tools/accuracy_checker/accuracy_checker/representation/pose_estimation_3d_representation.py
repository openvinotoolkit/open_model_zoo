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
from .base_representation import BaseRepresentation
from .pose_estimation_representation import PoseEstimationRepresentation


class PoseEstimation3dRepresentation(BaseRepresentation):
    def __init__(self, identifier='', x_values=None, y_values=None, visibility=None, labels=None,
                 x_3d_values=None, y_3d_values=None, z_3d_values=None, fx=None):
        super().__init__(identifier)
        self.pose_2d = PoseEstimationRepresentation(identifier, x_values, y_values, visibility, labels)
        self.x_3d_values = x_3d_values if np.size(x_3d_values) > 0 else np.array([])
        self.y_3d_values = y_3d_values if np.size(y_3d_values) > 0 else np.array([])
        self.z_3d_values = z_3d_values if np.size(z_3d_values) > 0 else np.array([])
        self.fx = fx

    @property
    def bboxes(self):
        if self.size == 0:
            return []
        x_mins, y_mins, x_maxs, y_maxs = [], [], [], []
        for box_id in range(self.pose_2d.x_values.shape[0]):
            x_mins.append(np.min(self.pose_2d.x_values[box_id][self.pose_2d.visibility[box_id] > 0]))
            x_maxs.append(np.max(self.pose_2d.x_values[box_id][self.pose_2d.visibility[box_id] > 0]))
            y_mins.append(np.min(self.pose_2d.y_values[box_id][self.pose_2d.visibility[box_id] > 0]))
            y_maxs.append(np.max(self.pose_2d.y_values[box_id][self.pose_2d.visibility[box_id] > 0]))
        return [[x_min, y_min, x_max, y_max] for x_min, y_min, x_max, y_max in zip(x_mins, y_mins, x_maxs, y_maxs)]

    @property
    def size(self):
        return len(self.pose_2d.x_values)


class PoseEstimation3dAnnotation(PoseEstimation3dRepresentation):
    pass


class PoseEstimation3dPrediction(PoseEstimation3dRepresentation):
    def __init__(self, identifier='', x_values=None, y_values=None, visibility=None, scores=None,
                 x_3d_values=None, y_3d_values=None, z_3d_values=None, labels=None, translations=None):
        super().__init__(identifier, x_values, y_values, visibility, labels, x_3d_values, y_3d_values, z_3d_values)
        self.scores = scores if scores is not None and np.size(scores) else np.array([])
        self.translations = translations if translations is not None and np.size(translations) else np.array([])
