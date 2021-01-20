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

from functools import singledispatch

import numpy as np

from ..config import BoolField, NumberField
from .postprocessor import Postprocessor
from ..representation import (
    DetectionPrediction, DetectionAnnotation, ActionDetectionPrediction, ActionDetectionAnnotation
)


@singledispatch
def get_scores(prediction):
    return prediction.scores


@get_scores.register(ActionDetectionAnnotation)
@get_scores.register(ActionDetectionPrediction)
def get_box_scores(prediction):
    return prediction.bbox_scores


@singledispatch
def set_scores(prediction, scores):
    prediction.scores = scores


@set_scores.register(ActionDetectionAnnotation)
@set_scores.register(ActionDetectionPrediction)
def set_box_scores(prediction, scores):
    prediction.bbox_scores = scores

class NMS(Postprocessor):
    __provider__ = 'nms'

    prediction_types = (DetectionPrediction, ActionDetectionPrediction)
    annotation_types = (DetectionAnnotation, ActionDetectionPrediction)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'overlap': NumberField(
                min_value=0, max_value=1, optional=True, default=0.5,
                description="Overlap threshold for merging detections."
            ),
            'include_boundaries': BoolField(
                optional=True, default=True, description="Shows if boundaries are included."
            ),
            'keep_top_k': NumberField(min_value=0, optional=True, description="Keep top K."),
            'use_min_area': BoolField(
                optional=True, default=False,
                description="Use minimum area of two bounding boxes as base area to calculate overlap"
            )
        })
        return parameters

    def configure(self):
        self.overlap = self.get_value_from_config('overlap')
        self.include_boundaries = self.get_value_from_config('include_boundaries')
        self.keep_top_k = self.get_value_from_config('keep_top_k')
        self.use_min_area = self.get_value_from_config('use_min_area')

    def process_image(self, annotations, predictions):
        for prediction in predictions:
            scores = get_scores(prediction)
            keep = self.nms(
                prediction.x_mins, prediction.y_mins, prediction.x_maxs, prediction.y_maxs, scores,
                self.overlap, self.include_boundaries, self.keep_top_k, self.use_min_area
            )
            prediction.remove([box for box in range(len(prediction.x_mins)) if box not in keep])

        return annotations, predictions

    @staticmethod
    def nms(x1, y1, x2, y2, scores, thresh, include_boundaries=True, keep_top_k=None, use_min_area=False):
        """
        Pure Python NMS baseline.
        """
        b = 1 if include_boundaries else 0

        areas = (x2 - x1 + b) * (y2 - y1 + b)
        order = scores.argsort()[::-1]

        if keep_top_k:
            order = order[:keep_top_k]

        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + b)
            h = np.maximum(0.0, yy2 - yy1 + b)
            intersection = w * h

            if use_min_area:
                base_area = np.minimum(areas[i], areas[order[1:]])
            else:
                base_area = (areas[i] + areas[order[1:]] - intersection)

            overlap = np.divide(
                intersection,
                base_area,
                out=np.zeros_like(intersection, dtype=float),
                where=base_area != 0
            )
            order = order[np.where(overlap <= thresh)[0] + 1] # pylint: disable=W0143

        return keep

class SoftNMS(Postprocessor):
    __provider__ = 'soft_nms'

    prediction_types = (DetectionPrediction, ActionDetectionPrediction)
    annotation_types = (DetectionAnnotation, ActionDetectionAnnotation)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'keep_top_k': NumberField(
                min_value=0, optional=True, default=200,
                description="The maximal number of detections which should be kept.",
                value_type=int
            ),
            'sigma': NumberField(
                value_type=float, optional=True, default=0.5,
                description="Sigma-value for updated detection score calculation."
            ),
            'min_score': NumberField(
                min_value=0, max_value=1, value_type=float, optional=True, default=0, description="Break point."
            )
        })
        return parameters

    def configure(self):
        self.keep_top_k = self.get_value_from_config('keep_top_k')
        self.sigma = self.get_value_from_config('sigma')
        self.min_score = self.get_value_from_config('min_score')

    def process_image(self, annotations, predictions):
        for prediction in predictions:
            if not prediction.size:
                continue

            scores = get_scores(prediction)
            keep, new_scores = self._nms(
                np.c_[prediction.x_mins, prediction.y_mins, prediction.x_maxs, prediction.y_maxs], scores,
            )
            prediction.remove([box for box in range(len(prediction.x_mins)) if box not in keep])
            set_scores(prediction, new_scores)

        return annotations, predictions

    @staticmethod
    def _matrix_iou(set_a, set_b):
        intersect_xmin = np.maximum(set_a[:, 0].reshape([-1, 1]), set_b[:, 0].reshape([1, -1]))
        intersect_ymin = np.maximum(set_a[:, 1].reshape([-1, 1]), set_b[:, 1].reshape([1, -1]))
        intersect_xmax = np.minimum(set_a[:, 2].reshape([-1, 1]), set_b[:, 2].reshape([1, -1]))
        intersect_ymax = np.minimum(set_a[:, 3].reshape([-1, 1]), set_b[:, 3].reshape([1, -1]))

        intersect_widths = np.maximum(0.0, intersect_xmax - intersect_xmin)
        intersect_heights = np.maximum(0.0, intersect_ymax - intersect_ymin)

        intersect_areas = intersect_widths * intersect_heights
        areas_set_a = ((set_a[:, 2] - set_a[:, 0]) * (set_a[:, 3] - set_a[:, 1])).reshape([-1, 1])
        areas_set_b = ((set_b[:, 2] - set_b[:, 0]) * (set_b[:, 3] - set_b[:, 1])).reshape([1, -1])

        areas_set_a[np.less(areas_set_a, 0.0)] = 0.0
        areas_set_b[np.less(areas_set_b, 0.0)] = 0.0

        union_areas = areas_set_a + areas_set_b - intersect_areas

        overlaps = intersect_areas / union_areas
        overlaps[np.less_equal(union_areas, 0.0)] = 0.0

        return overlaps

    def _nms(self, input_bboxes, input_scores):
        if len(input_bboxes) == 0:  # pylint: disable=len-as-condition
            return [], []

        if len(input_bboxes) > self.keep_top_k:
            indices = np.argsort(-input_scores)[:self.keep_top_k]
            scores = input_scores[indices]
            bboxes = input_bboxes[indices]
        else:
            scores = np.copy(input_scores)
            indices = np.arange(len(scores))
            bboxes = input_bboxes

        similarity_matrix = self._matrix_iou(bboxes, bboxes)

        out_ids = []
        out_scores = []
        for _ in range(self.keep_top_k):
            bbox_id = np.argmax(scores)
            bbox_score = scores[bbox_id]
            if bbox_score < self.min_score:
                break

            out_ids.append(indices[bbox_id])
            out_scores.append(bbox_score)
            scores[bbox_id] = 0.0

            iou_values = similarity_matrix[bbox_id]
            scores *= np.exp(np.negative(np.square(iou_values) / self.sigma))

        return np.array(out_ids, dtype=np.int32), np.array(out_scores, dtype=np.float32)

class DIoUNMS(Postprocessor):
    __provider__ = 'diou_nms'

    prediction_types = (DetectionPrediction, ActionDetectionPrediction)
    annotation_types = (DetectionAnnotation, ActionDetectionPrediction)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'overlap': NumberField(
                min_value=0, max_value=1, optional=True, default=0.5,
                description="Overlap threshold for merging detections."
            ),
            'include_boundaries': BoolField(
                optional=True, default=True, description="Shows if boundaries are included."
            ),
            'keep_top_k': NumberField(min_value=0, optional=True, description="Keep top K.")
        })
        return parameters

    def configure(self):
        self.overlap = self.get_value_from_config('overlap')
        self.include_boundaries = self.get_value_from_config('include_boundaries')
        self.keep_top_k = self.get_value_from_config('keep_top_k')

    def process_image(self, annotations, predictions):
        for prediction in predictions:
            scores = get_scores(prediction)
            keep = self.diou_nms(
                prediction.x_mins, prediction.y_mins, prediction.x_maxs, prediction.y_maxs, scores,
                self.overlap, self.include_boundaries, self.keep_top_k
            )
            prediction.remove([box for box in range(len(prediction.x_mins)) if box not in keep])

        return annotations, predictions

    @staticmethod
    def diou_nms(x1, y1, x2, y2, scores, thresh, include_boundaries=True, keep_top_k=None, use_min_area=False):

        b = 1 if include_boundaries else 0

        areas = (x2 - x1 + b) * (y2 - y1 + b)
        order = scores.argsort()[::-1]

        if keep_top_k:
            order = order[:keep_top_k]

        keep = []

        while order.size > 0:
            i = order[0]
            keep.append(i)

            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1 + b)
            h = np.maximum(0.0, yy2 - yy1 + b)
            intersection = w * h

            cw = np.maximum(x2[i], x2[order[1:]]) - np.minimum(x1[i], x1[order[1:]])
            ch = np.maximum(y2[i], y2[order[1:]]) - np.minimum(y1[i], y1[order[1:]])
            c_area = cw**2 + ch**2 + 1e-16
            d_1 = ((x2[order[1:]] + x1[order[1:]]) - (x2[i] + x1[i]))**2 / 4
            d_2 = ((y2[order[1:]] + y1[order[1:]]) - (y2[i] + y1[i]))**2 / 4
            d_area = d_1 + d_2

            base_area = (areas[i] + areas[order[1:]] - intersection)

            overlap = np.divide(
                intersection,
                base_area,
                out=np.zeros_like(intersection, dtype=float),
                where=base_area != 0
            ) - pow(d_area / c_area, 0.6)
            order = order[np.where(overlap <= thresh)[0] + 1] # pylint: disable=W0143

        return keep
