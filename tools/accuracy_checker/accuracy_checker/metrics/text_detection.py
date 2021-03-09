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

import math
from collections import namedtuple
import numpy as np
from .metric import PerImageEvaluationMetric
from ..config import BoolField, NumberField
from ..representation import TextDetectionPrediction, TextDetectionAnnotation, DetectionPrediction
from ..utils import polygon_from_points


def calculte_recall_precision_matrix(gt_rects, prediction_rects):
    num_gt = len(gt_rects)
    num_det = len(prediction_rects)
    output_shape = [num_gt, num_det]
    recall_mat = np.empty(output_shape)
    precision_mat = np.empty(output_shape)

    for gt_id, gt_rect in enumerate(gt_rects):
        for pred_id, pred_rect in enumerate(prediction_rects):
            intersected_area = rect_area(gt_rect, pred_rect)
            rg_dimensions = (gt_rect.xmax - gt_rect.xmin + 1) * (gt_rect.ymax - gt_rect.ymin + 1)
            rd_dimensions = (pred_rect.xmax - pred_rect.xmin + 1) * (pred_rect.ymax - pred_rect.ymin + 1)
            recall_mat[gt_id, pred_id] = 0 if rg_dimensions == 0 else intersected_area / rg_dimensions
            precision_mat[gt_id, pred_id] = 0 if rd_dimensions == 0 else intersected_area / rd_dimensions

    return recall_mat, precision_mat


def get_union(detection_polygon, annotation_polygon):
    area_prediction = detection_polygon.area
    area_annotation = annotation_polygon.area
    return area_prediction + area_annotation - get_intersection_area(detection_polygon, annotation_polygon)


def get_intersection_over_union(detection_polygon, annotation_polygon):
    union = get_union(detection_polygon, annotation_polygon)
    intersection = get_intersection_area(detection_polygon, annotation_polygon)
    return intersection / union if union != 0 else 0.0


def get_intersection_area(detection_polygon, annotation_polygon):
    return detection_polygon.intersection(annotation_polygon).area


Rectangle = namedtuple('Rectangle', 'xmin ymin xmax ymax')
Point = namedtuple('Point', 'x y')


def rect_center(r):
    x = float(r.xmin) + float(r.xmax - r.xmin + 1) / 2.
    y = float(r.ymin) + float(r.ymax - r.ymin + 1) / 2.
    return Point(x, y)


def rect_point_distance(r1, r2):
    distx = math.fabs(r1.x - r2.x)
    disty = math.fabs(r1.y - r2.y)
    return math.sqrt(distx * distx + disty * disty)


def rect_center_distance(r1, r2):
    return rect_point_distance(rect_center(r1), rect_center(r2))


def rect_diag(r):
    w = (r.xmax - r.xmin + 1)
    h = (r.ymax - r.ymin + 1)
    return math.sqrt(h * h + w * w)


def rect_area(a, b):
    dx = min(a.xmax, b.xmax) - max(a.xmin, b.xmin) + 1
    dy = min(a.ymax, b.ymax) - max(a.ymin, b.ymin) + 1
    if (dx >= 0) and (dy >= 0):
        return dx*dy
    return 0.


def rect_from_points(points):
    return Rectangle(*points)


class FocusedTextLocalizationMetric(PerImageEvaluationMetric):
    annotation_types = (TextDetectionAnnotation, )
    prediction_types = (TextDetectionPrediction, DetectionPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'area_recall_constrain':  NumberField(
                min_value=0, max_value=1, optional=True, default=0.5,
                description="Minimal value for recall that allows to make decision "
                            "that prediction polygon matched with annotation."
            ),
            'ignore_difficult':  BoolField(
                optional=True, default=True,
                description="Allows to ignore difficult ground truth text polygons in metric calculation."
            ),
            'area_precision_constrain':  NumberField(
                min_value=0, max_value=1, optional=True, default=0.5,
                description="Minimal value for precision that allows to make decision "
                            "that prediction polygon matched with annotation."
            ),
            'center_diff_threshold': NumberField(min_value=0, optional=True, default=1),
            'one_to_one_match_score': NumberField(
                min_value=0, optional=True, max_value=1, default=1,
                description='weight for one to one matching results',
            ),
            'one_to_many_match_score': NumberField(
                min_value=0, optional=True, max_value=1, default=0.8,
                description='weight for one to many matching results',
            ),
            'many_to_one_match_score': NumberField(
                min_value=0, optional=True, max_value=1, default=1,
                description='weight for many to one matching results',
            )
        })

        return parameters

    def configure(self):
        self.area_recall_constrain = self.get_value_from_config('area_recall_constrain')
        self.area_precision_constrain = self.get_value_from_config('area_precision_constrain')
        self.ignore_difficult = self.get_value_from_config('ignore_difficult')
        self.center_diff_threshold = self.get_value_from_config('center_diff_threshold')
        self.one_to_one_match_score = self.get_value_from_config('one_to_one_match_score')
        self.one_to_many_match_score = self.get_value_from_config('one_to_many_match_score')
        self.many_to_one_match_score = self.get_value_from_config('many_to_one_match_score')
        self.word_spotting = self.get_value_from_config('word_spotting')
        self.num_valid_gt = 0
        self.num_valid_detections = 0
        self.precision_sum = 0
        self.recall_sum = 0

    def update(self, annotation, prediction):
        gt_rects = list(map(rect_from_points, annotation.boxes))
        prediction_rects = list(map(rect_from_points, prediction.boxes))
        num_gt = len(gt_rects)
        num_det = len(prediction_rects)
        gt_difficult_mask = np.full(num_gt, False)
        prediction_difficult_mask = np.full(num_det, False)
        if self.ignore_difficult:
            gt_difficult_inds = annotation.metadata.get('difficult_boxes', [])
            prediction_difficult_inds = prediction.metadata.get('difficult_boxes', [])
            gt_difficult_mask[gt_difficult_inds] = True
            prediction_difficult_mask[prediction_difficult_inds] = True
            prediction_difficult_mask = self._update_difficult_prediction_mask(
                gt_difficult_inds, prediction_difficult_mask, gt_rects, prediction_rects
            )

        num_ignored_gt = np.sum(gt_difficult_mask)
        num_ignored_pred = np.sum(prediction_difficult_mask)
        num_valid_gt = num_gt - num_ignored_gt
        num_valid_pred = num_det - num_ignored_pred

        self.num_valid_detections += num_valid_pred
        self.num_valid_gt += num_valid_gt

        if num_gt == 0:
            recall = 1
            precision = 0 if num_det > 0 else 1
            self.precision_sum += precision
            self.recall_sum += recall
            return precision, recall, num_valid_gt, num_valid_pred

        recall_accum = 0
        precision_accum = 0

        if num_det > 0:
            gt_rect_mat = np.zeros(num_gt, np.int8)
            det_rect_mat = np.zeros(num_det, np.int8)
            recall_mat, precision_mat = calculte_recall_precision_matrix(gt_rects, prediction_rects)
            one_to_one_recall, one_to_ona_precision, det_rect_mat, gt_rect_mat = self._one_to_one_match(
                gt_rects, prediction_rects,
                gt_difficult_mask, prediction_difficult_mask,
                gt_rect_mat, det_rect_mat,
                recall_mat, precision_mat
            )
            recall_accum += one_to_one_recall
            precision_accum += one_to_ona_precision

            one_to_many_recall, one_to_many_precision, det_rect_mat, gt_rect_mat = self._one_to_many_match(
                gt_rects, gt_difficult_mask, prediction_difficult_mask, gt_rect_mat, det_rect_mat,
                recall_mat, precision_mat
            )
            recall_accum += one_to_many_recall
            precision_accum += one_to_many_precision

            many_to_one_recall, many_to_one_precision, det_rect_mat, gt_rect_mat = self._many_to_one_match(
                prediction_rects, prediction_difficult_mask, gt_difficult_mask, gt_rect_mat, det_rect_mat,
                recall_mat, precision_mat,
            )
            recall_accum += many_to_one_recall
            precision_accum += many_to_one_precision

        if num_valid_gt == 0:
            recall = float(1)
            precision = float(0) if num_valid_pred > 0 else float(1)
        else:
            recall = float(recall_accum)
            precision = float(0) if num_valid_pred == 0 else float(precision_accum)

        self.recall_sum += recall
        self.precision_sum += precision

        return precision, recall, num_valid_gt, num_valid_pred

    def evaluate(self, annotations, predictions):
        raise NotImplementedError()

    def _update_difficult_prediction_mask(self, gt_difficult_inds, dt_difficult_mask, gt_rects, dt_rects):
        for det_id, detection_rect in enumerate(dt_rects):
            for gt_difficult_id in gt_difficult_inds:
                gt_difficult_rect = gt_rects[gt_difficult_id]
                intersected_area = rect_area(gt_difficult_rect, detection_rect)
                width = detection_rect.xmax - detection_rect.xmin + 1
                height = detection_rect.ymax - detection_rect.ymin + 1
                rd_dimensions = width * height
                if rd_dimensions == 0:
                    precision = 0
                else:
                    precision = intersected_area / rd_dimensions
                if precision > self.area_precision_constrain:
                    dt_difficult_mask[det_id] = True

        return dt_difficult_mask

    def _one_to_one_match(
            self, gt_rects, prediction_rects, gt_difficult_mask, prediction_difficult_mask, gt_rect_mat, det_rect_mat,
            recall_mat, precision_mat
    ):
        def match_rects(row, col, recall_mat, precision_mat):
            cont = 0
            for j in range(len(recall_mat[0])):
                recall_constrain_pass = recall_mat[row, j] >= self.area_recall_constrain
                precision_constrain_pass = precision_mat[row, j] >= self.area_precision_constrain
                if recall_constrain_pass and precision_constrain_pass:
                    cont += 1
            if cont != 1:
                return False
            cont = 0
            for i in range(len(recall_mat)):
                recall_constrain_pass = recall_mat[i, col] >= self.area_recall_constrain
                precision_constrain_pass = precision_mat[i, col] >= self.area_precision_constrain
                if recall_constrain_pass and precision_constrain_pass:
                    cont += 1
            if cont != 1:
                return False

            recall_constrain_pass = recall_mat[row, col] >= self.area_recall_constrain
            precision_constrain_pass = precision_mat[row, col] >= self.area_precision_constrain

            if recall_constrain_pass and precision_constrain_pass:
                return True
            return False

        recall_accum = 0
        precision_accum = 0
        for gt_id, gt_rect in enumerate(gt_rects):
            for pred_id, pred_rect in enumerate(prediction_rects):
                both_not_matched = not gt_rect_mat[gt_id] and not det_rect_mat[pred_id]
                difficult = gt_difficult_mask[gt_id] and prediction_difficult_mask[pred_id]
                if both_not_matched and not difficult:
                    match = match_rects(gt_id, pred_id, recall_mat, precision_mat)
                    if match:
                        norm_distance = rect_center_distance(gt_rect, pred_rect)
                        norm_distance /= rect_diag(gt_rect) + rect_diag(pred_rect)
                        norm_distance *= 2.0
                        if norm_distance < self.center_diff_threshold:
                            gt_rect_mat[gt_id] = self.one_to_one_match_score
                            det_rect_mat[pred_id] = 1
                            recall_accum += 1
                            precision_accum += 1

        return recall_accum, precision_accum, det_rect_mat, gt_rect_mat

    def _one_to_many_match(
            self, gt_rects, gt_difficult_mask, pred_difficult_mask, gt_rect_mat, det_rect_mat, recall_mat, precision_mat
    ):
        def match_rects(gt_id, recall_mat, precision_mat, gt_rect_mat, det_rect_mat, pred_difficult_mask):
            many_sum = 0
            det_rects = []
            for det_num in range(len(recall_mat[0])):
                if gt_rect_mat[gt_id] == 0 and det_rect_mat[det_num] == 0 and pred_difficult_mask[det_num]:
                    if precision_mat[gt_id, det_num] >= self.area_precision_constrain:
                        many_sum += recall_mat[gt_id, det_num]
                        det_rects.append(det_num)
            if many_sum >= self.area_recall_constrain:
                return True, det_rects
            return False, []

        recall_accum = 0
        precision_accum = 0

        for gt_id, _ in enumerate(gt_rects):
            if not gt_difficult_mask[gt_id]:
                match, matches_det = match_rects(
                    gt_id, recall_mat, precision_mat, gt_rect_mat, det_rect_mat, pred_difficult_mask
                )
                if match:
                    gt_rect_mat[gt_id] = 1
                    recall_accum += self.one_to_many_match_score
                    precision_accum += self.one_to_many_match_score * len(matches_det)
                    for det_id in matches_det:
                        det_rect_mat[det_id] = 1

        return recall_accum, precision_accum, det_rect_mat, gt_rect_mat

    def _many_to_one_match(
            self, prediction_rects, prediction_difficult_mask, gt_difficult_mask, gt_rect_mat, det_rect_mat,
            recall_mat, precision_mat
    ):
        def match_rects(det_id, recall_mat, precision_mat, gt_rect_mat, det_rect_mat, gt_difficult_mask):
            many_sum = 0
            gt_rects = []
            for gt_id in range(len(recall_mat)):
                if gt_rect_mat[gt_id] == 0 and det_rect_mat[det_id] == 0 and not gt_difficult_mask[gt_id]:
                    if recall_mat[gt_id, det_id] >= self.area_recall_constrain:
                        many_sum += precision_mat[gt_id, det_id]
                        gt_rects.append(gt_id)
            if many_sum >= self.area_precision_constrain:
                return True, gt_rects
            return False, []

        recall_accum = 0
        precision_accum = 0

        for pred_id, _ in enumerate(prediction_rects):
            if not prediction_difficult_mask[pred_id]:
                match, matches_gt = match_rects(
                    pred_id, recall_mat, precision_mat, gt_rect_mat, det_rect_mat, gt_difficult_mask
                )
                if match:
                    det_rect_mat[pred_id] = 1
                    recall_accum += self.many_to_one_match_score * len(matches_gt)
                    precision_accum += self.many_to_one_match_score
                    for gt_id in matches_gt:
                        gt_rect_mat[gt_id] = 1

        return recall_accum, precision_accum, det_rect_mat, gt_rect_mat

    def reset(self):
        self.num_valid_gt = 0
        self.num_valid_detections = 0
        self.precision_sum = 0
        self.recall_sum = 0


class FocusedTextLocalizationPrecision(FocusedTextLocalizationMetric):
    __provider__ = 'focused_text_precision'

    def update(self, annotation, prediction):
        precision, _, _, num_valid_dt = super().update(annotation, prediction)
        return precision / num_valid_dt if num_valid_dt != 0 else 0

    def evaluate(self, annotations, predictions):
        return self.precision_sum / self.num_valid_detections if self.num_valid_detections != 0 else 0


class FocusedTextLocalizationRecall(FocusedTextLocalizationMetric):
    __provider__ = 'focused_text_recall'

    def update(self, annotation, prediction):
        precision, _, num_valid_gt, _ = super().update(annotation, prediction)
        return precision / num_valid_gt if num_valid_gt != 0 else 0

    def evaluate(self, annotations, predictions):
        return self.recall_sum / self.num_valid_gt if self.num_valid_gt != 0 else 0


class FocusedTextLocalizationHMean(FocusedTextLocalizationMetric):
    __provider__ = 'focused_text_hmean'

    def update(self, annotation, prediction):
        precision, recall, num_valid_gt, num_valid_dt = super().update(annotation, prediction)
        overall_p = precision / num_valid_dt if num_valid_dt != 0 else 0
        overall_r = recall / num_valid_gt if num_valid_gt != 0 else 0

        return 2 * overall_r * overall_p / (overall_r + overall_p) if overall_r + overall_p != 0 else 0

    def evaluate(self, annotations, predictions):
        recall = self.recall_sum / self.num_valid_gt if self.num_valid_gt != 0 else 0
        precision = self.precision_sum / self.num_valid_detections if self.num_valid_detections != 0 else 0
        return 2 * recall * precision / (recall + precision) if recall + precision != 0 else 0


class IncidentalSceneTextLocalizationMetric(PerImageEvaluationMetric):
    annotation_types = (TextDetectionAnnotation, )
    prediction_types = (TextDetectionPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'iou_constrain':  NumberField(
                min_value=0, max_value=1, optional=True, default=0.5,
                description="Minimal value for intersection over union that allows to make decision "
                            "that prediction polygon is true positive."
            ),
            'ignore_difficult':  BoolField(
                optional=True, default=True,
                description="Allows to ignore difficult ground truth text polygons in metric calculation."
            ),
            'area_precision_constrain':  NumberField(
                min_value=0, max_value=1, optional=True, default=0.5,
                description="Minimal value for intersection over union that allows to make decision "
                            "that prediction polygon matched with ignored annotation."
            ),
            'word_spotting': BoolField(
                optional=True, default=False,
                description="Allows to use transcriptions in order to compute word spotting metrics"
            )
        })

        return parameters

    def configure(self):
        self.iou_constrain = self.get_value_from_config('iou_constrain')
        self.area_precision_constrain = self.get_value_from_config('area_precision_constrain')
        self.ignore_difficult = self.get_value_from_config('ignore_difficult')
        self.word_spotting = self.get_value_from_config('word_spotting')
        self.number_matched_detections = 0
        self.number_valid_annotations = 0
        self.number_valid_detections = 0

    def update(self, annotation, prediction):
        gt_polygons = list(map(polygon_from_points, annotation.points))
        gt_texts = list(annotation.description)

        prediction_polygons = list(map(polygon_from_points, prediction.points))
        prediction_texts = list(prediction.description)

        num_gt = len(gt_polygons)
        num_det = len(prediction_polygons)
        gt_difficult_mask = np.full(num_gt, False)
        prediction_difficult_mask = np.full(num_det, False)
        num_det_matched = 0
        if self.ignore_difficult:
            gt_difficult_inds = annotation.metadata.get('difficult_boxes', [])
            prediction_difficult_inds = prediction.metadata.get('difficult_boxes', [])
            gt_difficult_mask[gt_difficult_inds] = True
            prediction_difficult_mask[prediction_difficult_inds] = True
            for det_id, detection_polygon in enumerate(prediction_polygons):
                for gt_difficult_id in gt_difficult_inds:
                    gt_difficult_polygon = gt_polygons[gt_difficult_id]
                    intersected_area = get_intersection_area(gt_difficult_polygon,
                                                             detection_polygon)
                    pd_dimensions = detection_polygon.area
                    precision = 0 if pd_dimensions == 0 else intersected_area / pd_dimensions

                    if precision >= self.area_precision_constrain:
                        prediction_difficult_mask[det_id] = True
                        break

        if num_gt > 0 and num_det > 0:
            iou_matrix = np.empty((num_gt, num_det))
            gt_matched = np.zeros(num_gt, np.int8)
            det_matched = np.zeros(num_det, np.int8)

            for gt_id, gt_polygon in enumerate(gt_polygons):
                for pred_id, pred_polygon in enumerate(prediction_polygons):
                    iou_matrix[gt_id, pred_id] = get_intersection_over_union(pred_polygon, gt_polygon)
                    not_matched_before = gt_matched[gt_id] == 0 and det_matched[pred_id] == 0
                    not_difficult = not gt_difficult_mask[gt_id] and not prediction_difficult_mask[pred_id]
                    if not_matched_before and not_difficult:
                        iou_big_enough = iou_matrix[gt_id, pred_id] >= self.iou_constrain
                        if not self.word_spotting:
                            transcriptions_equal = True
                        else:
                            transcriptions_equal = gt_texts[gt_id].lower() == prediction_texts[pred_id].lower()
                        if iou_big_enough and transcriptions_equal:
                            gt_matched[gt_id] = 1
                            det_matched[pred_id] = 1
                            num_det_matched += 1

        num_ignored_gt = np.sum(gt_difficult_mask)
        num_ignored_pred = np.sum(prediction_difficult_mask)
        num_valid_gt = num_gt - num_ignored_gt
        num_valid_pred = num_det - num_ignored_pred

        self.number_matched_detections += num_det_matched
        self.number_valid_annotations += num_valid_gt
        self.number_valid_detections += num_valid_pred

        return num_det_matched, num_valid_gt, num_valid_pred

    def evaluate(self, annotations, predictions):
        raise NotImplementedError()

    def reset(self):
        self.number_matched_detections = 0
        self.number_valid_annotations = 0
        self.number_valid_detections = 0


class IncidentalSceneTextLocalizationPrecision(IncidentalSceneTextLocalizationMetric):
    __provider__ = 'incidental_text_precision'

    def update(self, annotation, prediction):
        num_det_matched, _, num_valid_dt = super().update(annotation, prediction)
        return 0 if num_valid_dt == 0 else float(num_det_matched) / num_valid_dt

    def evaluate(self, annotations, predictions):
        precision = (
            0 if self.number_valid_detections == 0
            else float(self.number_matched_detections) / self.number_valid_detections
        )

        return precision


class IncidentalSceneTextLocalizationRecall(IncidentalSceneTextLocalizationMetric):
    __provider__ = 'incidental_text_recall'

    def update(self, annotation, prediction):
        num_det_matched, num_valid_gt, _ = super().update(annotation, prediction)
        return 0 if num_valid_gt == 0 else float(num_det_matched) / num_valid_gt

    def evaluate(self, annotations, predictions):
        recall = (
            0 if self.number_valid_annotations == 0
            else float(self.number_matched_detections) / self.number_valid_annotations
        )

        return recall


class IncidentalSceneTextLocalizationHMean(IncidentalSceneTextLocalizationMetric):
    __provider__ = 'incidental_text_hmean'

    def update(self, annotation, prediction):
        num_det_matched, num_valid_gt, num_valid_pred = super().update(annotation, prediction)
        precision = 0 if num_valid_pred == 0 else num_det_matched / num_valid_pred
        recall = 0 if num_valid_gt == 0 else num_det_matched / num_valid_gt

        return 0 if precision + recall == 0 else 2 * recall * precision / (recall + precision)

    def evaluate(self, annotations, predictions):
        recall = (
            0 if self.number_valid_annotations == 0
            else float(self.number_matched_detections) / self.number_valid_annotations
        )
        precision = (
            0 if self.number_valid_detections == 0
            else float(self.number_matched_detections) / self.number_valid_detections
        )

        return 0 if recall + precision == 0 else 2 * recall * precision / (recall + precision)
