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

        def compute_scale_and_shift(target, mask, prediction):
            # system matrix: A = [[a_00, a_01], [a_10, a_11]]
            a_00 = np.sum(mask * prediction * prediction)
            a_01 = np.sum(mask * prediction)
            a_11 = np.sum(mask)

            # right hand side: b = [b_0, b_1]
            b_0 = np.sum(mask * prediction * target)
            b_1 = np.sum(mask * target)

            # solution: x = A^-1 . b = [[a_11, -a_01], [-a_10, a_00]] / (a_00 * a_11 - a_01 * a_10) . b
            x_0 = np.zeros_like(b_0)
            x_1 = np.zeros_like(b_1)

            det = a_00 * a_11 - a_01 * a_01
            # A needs to be a positive definite matrix.
            valid = det > 0

            x_0[valid] = (a_11[valid] * b_0[valid] - a_01[valid] * b_1[valid]) / det[valid]
            x_1[valid] = (-a_01[valid] * b_0[valid] + a_00[valid] * b_1[valid]) / det[valid]

            return x_0, x_1

        scale, shift = compute_scale_and_shift(gt_depth_map, gt_mask, prediction_depth_map)
        prediction_depth_map = scale * prediction_depth_map + shift

        return prediction_depth_map
