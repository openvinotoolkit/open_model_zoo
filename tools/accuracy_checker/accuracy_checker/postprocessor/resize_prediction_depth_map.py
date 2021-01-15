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
import cv2
from .postprocessor import Postprocessor
from ..representation import DepthEstimationAnnotation, DepthEstimationPrediction


class ResizeDepthMap(Postprocessor):
    __provider__ = 'resize_prediction_depth_map'

    annotation_types = (DepthEstimationAnnotation, )
    prediction_types = (DepthEstimationPrediction, )

    def process_image(self, annotation, prediction):
        h, w, _ = self.image_size

        for target_prediction in prediction:
            depth_map = target_prediction.depth_map
            if len(depth_map.shape) == 3 and 1 in depth_map.shape:
                depth_map = np.squeeze(depth_map)
            target_prediction.depth_map = cv2.resize(depth_map, (w, h))

        return annotation, prediction
