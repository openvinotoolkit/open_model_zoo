"""
 Copyright (c) 2020-2024 Intel Corporation

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
import itertools
import math
import numpy as np

from .types import NumericalValue
from .detection_model import DetectionModel
from .utils import Detection, nms


class FaceBoxes(DetectionModel):
    __model__ = 'FaceBoxes'

    def __init__(self, model_adapter, configuration=None, preload=False):
        super().__init__(model_adapter, configuration, preload)
        self.bboxes_blob_name, self.scores_blob_name = self._get_outputs()
        self.min_sizes = [[32, 64, 128], [256], [512]]
        self.steps = [32, 64, 128]
        self.variance = [0.1, 0.2]
        self.keep_top_k = 750

    def _get_outputs(self):
        (bboxes_blob_name, bboxes_layer), (scores_blob_name, scores_layer) = self.outputs.items()

        if bboxes_layer.shape[1] != scores_layer.shape[1]:
            self.raise_error("Expected the same second dimension for boxes and scores, but got {} and {}".format(
                bboxes_layer.shape, scores_layer.shape))

        if bboxes_layer.shape[2] == 4:
            return bboxes_blob_name, scores_blob_name

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'iou_threshold': NumericalValue(default_value=0.3, description="Threshold for NMS filtering")
        })
        parameters['labels'].update_default_value(['Face'])
        return parameters

    def postprocess(self, outputs, meta):
        detections = self._parse_outputs(outputs, meta)
        detections = self._resize_detections(detections, meta)
        return detections

    def _parse_outputs(self, outputs, meta):
        boxes = outputs[self.bboxes_blob_name][0]
        scores = outputs[self.scores_blob_name][0]

        detections = []

        feature_maps = [[math.ceil(self.h / step), math.ceil(self.w / step)] for step in
                        self.steps]
        prior_data = self.prior_boxes(feature_maps, [self.h, self.w])

        boxes[:, :2] = self.variance[0] * boxes[:, :2]
        boxes[:, 2:] = self.variance[1] * boxes[:, 2:]
        boxes[:, :2] = boxes[:, :2] * prior_data[:, 2:] + prior_data[:, :2]
        boxes[:, 2:] = np.exp(boxes[:, 2:]) * prior_data[:, 2:]

        score = np.transpose(scores)[1]

        mask = score > self.confidence_threshold
        filtered_boxes, filtered_score = boxes[mask, :], score[mask]
        if filtered_score.size != 0:
            x_mins = (filtered_boxes[:, 0] - 0.5 * filtered_boxes[:, 2])
            y_mins = (filtered_boxes[:, 1] - 0.5 * filtered_boxes[:, 3])
            x_maxs = (filtered_boxes[:, 0] + 0.5 * filtered_boxes[:, 2])
            y_maxs = (filtered_boxes[:, 1] + 0.5 * filtered_boxes[:, 3])

            keep = nms(x_mins, y_mins, x_maxs, y_maxs, filtered_score, self.iou_threshold,
                       keep_top_k=self.keep_top_k)

            filtered_score = filtered_score[keep]
            x_mins = x_mins[keep]
            y_mins = y_mins[keep]
            x_maxs = x_maxs[keep]
            y_maxs = y_maxs[keep]

            if filtered_score.size > self.keep_top_k:
                filtered_score = filtered_score[:self.keep_top_k]
                x_mins = x_mins[:self.keep_top_k]
                y_mins = y_mins[:self.keep_top_k]
                x_maxs = x_maxs[:self.keep_top_k]
                y_maxs = y_maxs[:self.keep_top_k]

            detections = [Detection(*det, 0) for det in zip(x_mins, y_mins, x_maxs, y_maxs, filtered_score)]
        return detections

    @staticmethod
    def calculate_anchors(list_x, list_y, min_size, image_size, step):
        anchors = []
        s_kx = min_size / image_size[1]
        s_ky = min_size / image_size[0]
        dense_cx = [x * step / image_size[1] for x in list_x]
        dense_cy = [y * step / image_size[0] for y in list_y]
        for cy, cx in itertools.product(dense_cy, dense_cx):
            anchors.append([cx, cy, s_kx, s_ky])
        return anchors

    def calculate_anchors_zero_level(self, f_x, f_y, min_sizes, image_size, step):
        anchors = []
        for min_size in min_sizes:
            if min_size == 32:
                list_x = [f_x + 0, f_x + 0.25, f_x + 0.5, f_x + 0.75]
                list_y = [f_y + 0, f_y + 0.25, f_y + 0.5, f_y + 0.75]
            elif min_size == 64:
                list_x = [f_x + 0, f_x + 0.5]
                list_y = [f_y + 0, f_y + 0.5]
            else:
                list_x = [f_x + 0.5]
                list_y = [f_y + 0.5]
            anchors.extend(self.calculate_anchors(list_x, list_y, min_size, image_size, step))
        return anchors

    def prior_boxes(self, feature_maps, image_size):
        anchors = []
        for k, f in enumerate(feature_maps):
            for i, j in itertools.product(range(f[0]), range(f[1])):
                if k == 0:
                    anchors.extend(self.calculate_anchors_zero_level(j, i, self.min_sizes[k],
                                                                     image_size, self.steps[k]))
                else:
                    anchors.extend(self.calculate_anchors([j + 0.5], [i + 0.5], self.min_sizes[k][0],
                                                          image_size, self.steps[k]))
        anchors = np.clip(anchors, 0, 1)

        return anchors
