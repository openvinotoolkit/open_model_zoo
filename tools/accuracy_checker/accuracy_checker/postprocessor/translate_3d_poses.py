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

from ..representation import PoseEstimation3dAnnotation, PoseEstimation3dPrediction
from .postprocessor import Postprocessor


class Translate3dPoses(Postprocessor):
    __provider__ = 'translate_3d_poses'
    annotation_types = (PoseEstimation3dAnnotation,)
    prediction_types = (PoseEstimation3dPrediction,)

    def process_image(self, annotations, predictions):
        for batch_id, prediction in enumerate(predictions):
            for pose_id in range(prediction.size):
                translation = prediction.translations[pose_id]
                translation[2] *= annotations[batch_id].fx if annotations[batch_id] is not None else 1
                prediction.x_3d_values[pose_id] += translation[0]
                prediction.y_3d_values[pose_id] += translation[1]
                prediction.z_3d_values[pose_id] += translation[2]

        return annotations, predictions
