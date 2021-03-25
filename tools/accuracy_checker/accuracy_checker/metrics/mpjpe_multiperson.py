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

import warnings
import numpy as np
from ..representation import PoseEstimation3dPrediction, PoseEstimation3dAnnotation
from .metric import PerImageEvaluationMetric


class MpjpeMultiperson(PerImageEvaluationMetric):
    __provider__ = 'mpjpe_multiperson'
    annotation_types = (PoseEstimation3dAnnotation,)
    prediction_types = (PoseEstimation3dPrediction,)

    def __init__(self, config, dataset, name=None, state=None):
        super().__init__(config, dataset, name, state)
        self.per_image_mpjpe = []
        self.meta.update({
            'scale': 10,
            'postfix': 'mm',
            'target': 'higher-worse',
        })

    def update(self, annotation, prediction):
        # since pelvis does not detected, remove it before evaluation
        kpt_mask = np.ones_like(annotation.x_3d_values[0], dtype=bool)
        kpt_mask[2] = False
        annotation.x_3d_values = annotation.x_3d_values[:, kpt_mask]
        annotation.y_3d_values = annotation.y_3d_values[:, kpt_mask]
        annotation.z_3d_values = annotation.z_3d_values[:, kpt_mask]
        if prediction.size:
            prediction.x_3d_values = prediction.x_3d_values[:, kpt_mask]
            prediction.y_3d_values = prediction.y_3d_values[:, kpt_mask]
            prediction.z_3d_values = prediction.z_3d_values[:, kpt_mask]

        matching_results = []
        sorted_prediction_ids = np.argsort(-prediction.scores, kind='stable')
        mask = np.ones(annotation.size, dtype=bool)
        for prediction_id in range(prediction.size):
            max_iou = 0
            matched_id = -1
            bbox = prediction.bboxes[sorted_prediction_ids[prediction_id]]
            for annotation_id in range(annotation.size):
                if not mask[annotation_id]:
                    continue
                iou = _get_iou(bbox, annotation.bboxes[annotation_id])
                if iou > max_iou:
                    max_iou = iou
                    matched_id = annotation_id
            if matched_id >= 0:
                mask[matched_id] = 0
                gt_coordinates = np.transpose(np.array(
                    [annotation.x_3d_values[matched_id], annotation.y_3d_values[matched_id],
                     annotation.z_3d_values[matched_id]]), (1, 0))
                predicted_coordinates = np.transpose(np.array(
                    [prediction.x_3d_values[sorted_prediction_ids[prediction_id]],
                     prediction.y_3d_values[sorted_prediction_ids[prediction_id]],
                     prediction.z_3d_values[sorted_prediction_ids[prediction_id]]]), (1, 0))
                matching_results.append((gt_coordinates, predicted_coordinates))

        image_mpjpe = 0
        for gt, target in matching_results:
            image_mpjpe += _mpjpe(gt, target)
        if matching_results:
            image_mpjpe /= len(matching_results)
        self.per_image_mpjpe.append(image_mpjpe)
        return image_mpjpe

    def evaluate(self, annotations, predictions):
        total_mpjpe = 0
        for image_mpjpe in self.per_image_mpjpe:
            total_mpjpe += image_mpjpe
        if self.per_image_mpjpe:
            total_mpjpe /= len(self.per_image_mpjpe)
        else:
            warnings.warn('No predicted results to compute MPJPE')
        return total_mpjpe

    def reset(self):
        self.per_image_mpjpe = []


def _get_iou(box_a, box_b):
    tl_x = max(box_a[0], box_b[0])
    tl_y = max(box_a[1], box_b[1])
    br_x = min(box_a[2], box_b[2])
    br_y = min(box_a[3], box_b[3])
    intersection_area = max(0, br_x - tl_x) * max(0, br_y - tl_y)

    area_a = (box_a[2] - box_a[0]) * (box_a[3] - box_a[1])
    area_b = (box_b[2] - box_b[0]) * (box_b[3] - box_b[1])
    iou = intersection_area / np.float32(area_a + area_b - intersection_area)

    return iou


def _mpjpe(gt, target):
    jpe = np.linalg.norm(gt - target)
    return jpe / target.shape[0]
