"""
Copyright (c) 2019 Intel Corporation
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

from .postprocessor import Postprocessor
from ..representation import DepthEstimationAnnotation, DepthEstimationPrediction


class AlignDepth(Postprocessor):
    __provider__ = 'align_prediction_depth_map'

    annotation_types = (DepthEstimationAnnotation, )
    prediction_types = (DepthEstimationPrediction, )

    def process_image(self, annotation, prediction):
        for annotation_, prediction_ in zip(annotation, prediction):
            prediction_.depth_map = self.scale_depth(annotation_.depth_map, annotation_.mask, prediction_.depth_map)

        return annotation, prediction

    @staticmethod
    def scale_depth(gt_depth_map, gt_mask, prediction_depth_map):
        # implement scaling
        return prediction_depth_map