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

from functools import singledispatch
from collections import defaultdict
import numpy as np
from ..config import NumberField, BaseField, ConfigError
from ..representation import (
    DetectionPrediction,
    DetectionAnnotation,
    PoseEstimationPrediction,
    PoseEstimationAnnotation,
    CoCoInstanceSegmentationPrediction,
    CoCoInstanceSegmentationAnnotation
)
from ..utils import get_or_parse_value, finalize_metric_result, UnsupportedPackage
from .overlap import Overlap
from .metric import PerImageEvaluationMetric
from .detection import average_precision, APIntegralType, _prepare_prediction_boxes

try:
    import pycocotools.mask as maskUtils
except ImportError as import_error:
    maskUtils = UnsupportedPackage("pycocotools", import_error.msg)

COCO_THRESHOLDS = {
    '0.5': [0.5],
    '0.75': [0.75],
    '0.5:0.05:0.95': np.linspace(.5, 0.95, np.round((0.95 - .5) / .05).astype(int) + 1, endpoint=True)
}


class MSCOCOBaseMetric(PerImageEvaluationMetric):
    annotation_types = (PoseEstimationAnnotation, DetectionAnnotation)
    prediction_types = (PoseEstimationPrediction, DetectionPrediction)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'max_detections': NumberField(
                value_type=int, optional=True, default=20,
                description="Max number of predicted results per image. If you have more predictions, "
                            "the results with minimal confidence will be ignored."
            ),
            'threshold': BaseField(
                optional=True, default='.50:.05:.95',
                description="Intersection over union threshold. "
                            "You can specify one value or comma separated range of values. "
                            "This parameter supports precomputed values for "
                            "standard COCO thresholds: {}".format(', '.join(COCO_THRESHOLDS)))
        })

        return parameters

    def configure(self):
        self.max_detections = self.get_value_from_config('max_detections')
        threshold = process_threshold(self.get_value_from_config('threshold'))
        self.config.pop('threshold', None)
        self.thresholds = get_or_parse_value(threshold, COCO_THRESHOLDS)
        if not self.dataset:
            raise ConfigError('coco metrics require dataset metadata providing in dataset_meta'
                              'Please provide dataset meta file or regenerate annotation')
        if not self.dataset.metadata:
            raise ConfigError('coco metrics require dataset metadata providing in dataset_meta'
                              'Please provide dataset meta file or regenerate annotation')
        label_map = self.dataset.metadata.get('label_map', {})
        self.labels = [
            label for label in label_map
            if label != self.dataset.metadata.get('background_label')
        ]
        if not self.labels:
            raise ConfigError('coco metrics require label_map providing in dataset_meta'
                              'Please provide dataset meta file or regenerate annotation')
        self.meta['names'] = [label_map[label] for label in self.labels]
        self.meta['orig_label_names'] = [label_map[label] for label in self.labels]
        self.matching_results = [[] for _ in self.labels]

    def update(self, annotation, prediction):
        compute_iou, create_boxes = select_specific_parameters(annotation)
        per_class_results = []
        profile_boxes = self.profiler is not None

        for label_id, label in enumerate(self.labels):
            detections, scores, dt_difficult = prepare_predictions(prediction, label, self.max_detections)
            ground_truth, gt_difficult, iscrowd, boxes, areas = prepare_annotations(annotation, label, create_boxes)
            iou = compute_iou(ground_truth, detections, annotation_boxes=boxes, annotation_areas=areas, iscrowd=iscrowd)
            eval_result = evaluate_image(
                ground_truth, gt_difficult, iscrowd, detections, dt_difficult, scores, iou, self.thresholds,
                profile_boxes
            )
            self.matching_results[label_id].append(eval_result)
            per_class_results.append(eval_result)

        return per_class_results

    def evaluate(self, annotations, predictions):
        pass

    def reset(self):
        self.matching_results = [[] for _ in self.labels]
        label_map = self.dataset.metadata.get('label_map', {})
        self.meta['names'] = [label_map[label] for label in self.labels]
        if self.profiler:
            self.profiler.reset()

    def _update_label_stat_for_non_matched_classes(self, labels_stat, predictions):
        matched_classes = set(labels_stat)
        background = self.dataset.metadata.get('background_label')
        prediction_classes = np.unique([pred.labels for pred in predictions])
        get_polygon = hasattr(predictions[0], 'to_polygon')
        for pc in prediction_classes:
            if pc == background or pc in matched_classes:
                continue
            if not get_polygon:
                prediction_boxes, _, _ = _prepare_prediction_boxes(
                    pc, predictions, True
                )
                conf = prediction_boxes[:, 0] if not get_polygon else []
                prediction_boxes = prediction_boxes[:, 1:]
            else:
                prediction_boxes = [p.to_polygon().get(pc, []) for p in predictions][0]
                conf = [p.scores[p.labels == pc] for p in predictions][0]
            label_report = {
                'precision': [],
                'recall': [],
                'result': [],
                'ap': -1,
                'scores': conf,
                'matched': defaultdict(list),
                'gt': [],
                'dt': prediction_boxes,
                'prediction_matches': 0,
                'annotation_matches': 0,
                'iou': []
            }
            labels_stat[int(pc)] = label_report
        return labels_stat


class MSCOCOAveragePrecision(MSCOCOBaseMetric):
    __provider__ = 'coco_precision'

    def update(self, annotation, prediction):
        per_class_matching = super().update(annotation, prediction)
        per_class_result = [
            compute_precision_recall(self.thresholds, [per_class_matching[i]]) for i, _ in enumerate(self.labels)
        ]
        precisions = [result[0] for result in per_class_result]
        if self.profiler:
            per_class_matching_dict = {}
            for class_match, label, (precision, recall, all_precisions, all_recalls, _, _, _) in zip(
                    per_class_matching, self.labels, per_class_result
            ):
                class_match['result'] = precision
                class_match['precision'] = precision
                class_match['recall'] = recall
                max_precision = all_precisions[0] if np.size(all_precisions) else np.array([])
                max_recall = all_recalls[0] if np.size(all_recalls) else np.array([])
                ap = average_precision(max_precision, max_recall, APIntegralType.voc_max)
                class_match['ap'] = ap
                per_class_matching_dict[label] = class_match

            per_class_matching_dict = self._update_label_stat_for_non_matched_classes(
                per_class_matching_dict, [prediction]
            )
            self.profiler.update(annotation.identifier, per_class_matching_dict, self.name, np.nanmean(precisions))
        return precisions

    def evaluate(self, annotations, predictions):
        per_class_result = [
            compute_precision_recall(self.thresholds, self.matching_results[i]) for i, _ in enumerate(self.labels)
        ]
        precision = [result[0] for result in per_class_result]
        if self.profiler:
            self.profiler.finish()
            summary = self.prepare_summary(per_class_result)
            self.profiler.write_summary(summary)
        precision, self.meta['names'] = finalize_metric_result(precision, self.meta['orig_label_names'])

        return precision

    @staticmethod
    def prepare_summary(label_stats):
        ap, recall_v, recall, precision_v, precision, lamrs = [], [], [], [], [], []
        per_class_summary = {}
        tp_fp_rate = []
        total_objects_cnt = 0
        approx_pr_recall, approx_fppi_miss_rate = [], []
        for label_idx, stat in enumerate(label_stats):
            precision_v_, recall_v_, precisions, recalls, fps, tps, num_img = stat
            if num_img == 0:
                lamrs.append(0)
                tp_fp_rate.append([0, 0])
                ap.append(np.NAN)
                continue
            max_precision = precisions[0] if np.size(precisions) else np.array([])
            max_recall = recalls[0] if np.size(recalls) else np.array([])
            ap_v = min(average_precision(max_precision, max_recall, APIntegralType.voc_max), 1)
            ap.append(ap_v)
            recall.append(recalls[0] if np.size(recalls) else [])
            recall_v.append(recall_v_)
            precision.append(precisions[0] if np.size(precisions) else [])
            precision_v.append(precision_v_)
            if not np.size(precisions):
                continue
            fppi = 1 - precisions[0]
            mr = 1 - recalls[0]
            pr = np.array([precisions[0], recalls[0]]).T
            fm = np.array([fppi, mr]).T
            if np.size(recalls[0]) and np.size(precisions[0]):
                approx_recall = np.linspace(0, 1, 100, endpoint=True)
                approx_precision = np.interp(approx_recall, recalls[0], precisions[0])
                approx_pr_recall.append([approx_precision, approx_recall])
                approx_fppi = np.linspace(0, 1, 100, endpoint=True)
                approx_mr = np.interp(approx_fppi, fppi, mr)
                approx_fppi_miss_rate.append([approx_fppi, approx_mr])
            fppi_tmp = np.insert(fppi, 0, -1.0)
            mr_tmp = np.insert(mr, 0, 1.0)
            ref = np.logspace(-2.0, 0.0, num=9)
            fp = int(np.sum(fps[0]))
            tp = int(np.sum(tps[0]))
            tp_fp_rate.append([tp, fp])
            for i, ref_i in enumerate(ref):
                j = np.where(fppi_tmp <= ref_i)[-1][-1]
                ref[i] = mr_tmp[j]

            lamr = np.exp(np.mean(np.log(np.maximum(1e-10, ref))))
            lamrs.append(lamr)
            total_objects_cnt += len(recall[-1])
            per_class_summary[label_idx] = {
                'precision': precision_v_ if not np.isnan(precision_v_) else -1,
                'result': precision_v_ if not np.isnan(precision_v_) else -1,
                'recall': recall_v_ if not np.isnan(recall_v_) else -1,
                'ap': ap_v if not np.isnan(ap_v) else -1,
                "scale": 100,
                "result_postfix": "%",
                'objects_count': len(recall[-1]),
                'charts': {
                    'precision_recall': pr.tolist(),
                    'fppi_miss_rate': fm.tolist()
                },
                'images_count': int(num_img)
            }
        ap_res = np.nanmean(ap)
        recall_res = np.nanmean(recall_v)
        precision_res = np.nanmean(precision_v)
        return {
            'summary_result': {
                'result': precision_res if not np.isnan(precision_res) else -1,
                'precision': precision_res if not np.isnan(precision_res) else -1,
                'recall': recall_res if not np.isnan(recall_res) else -1,
                'ap': ap_res if not np.isnan(ap_res) else -1,
                'result_scale': 100,
                'result_postfix': '%',
                'charts': {
                    'ap': [0 if np.isnan(ap_) else ap_ for ap_ in ap],
                    'log_miss_rate': lamrs,
                    'tp_fp_rate': tp_fp_rate,
                    'precision_recall': np.mean(approx_pr_recall, 0).T.tolist() if np.size(approx_pr_recall) else [],
                    'fppi_miss_rate': (
                        np.mean(approx_fppi_miss_rate, 0).T.tolist() if np.size(approx_fppi_miss_rate) else []
                    )
                },
                'objects_count': total_objects_cnt

            },
            'per_class_result': per_class_summary
        }


class MSCOCORecall(MSCOCOBaseMetric):
    __provider__ = 'coco_recall'

    def update(self, annotation, prediction):
        per_class_matching = super().update(annotation, prediction)
        per_class_result = [
            compute_precision_recall(self.thresholds, [per_class_matching[i]]) for i, _ in enumerate(self.labels)
        ]
        recalls = [metric[1] for metric in per_class_result]
        if self.profiler:
            per_class_matching_dict = {}
            for class_match, label, (precision, recall, all_precisions, all_recalls, _, _, _) in zip(
                    per_class_matching, self.labels, per_class_result
            ):
                class_match['result'] = recall
                class_match['precision'] = precision
                class_match['recall'] = recall
                ap = average_precision(all_precisions, all_recalls, APIntegralType.voc_max)
                class_match['ap'] = ap
                per_class_matching_dict[label] = class_match
            self.profiler.update(annotation.identifier, per_class_matching_dict, self.name, np.nanmean(recalls))
        return per_class_result

    def evaluate(self, annotations, predictions):
        if self.profiler:
            self.profiler.finish()
        recalls = [
            compute_precision_recall(self.thresholds, self.matching_results[i])[1]
            for i, _ in enumerate(self.labels)
        ]
        recalls, self.meta['names'] = finalize_metric_result(recalls, self.meta['orig_label_names'])

        return recalls


class MSCOCOKeypointsBaseMetric(MSCOCOBaseMetric):
    annotation_types = (PoseEstimationAnnotation,)
    prediction_types = (PoseEstimationPrediction,)

    def update(self, annotation, prediction):
        per_class_results = []

        def _prepare_predictions(prediction, label, max_detections):
            if prediction.size == 0:
                return [], [], []
            prediction_ids = prediction.labels == label
            scores = prediction.scores[prediction_ids]
            if np.size(scores) == 0:
                return [], [], []
            scores_ids = np.argsort(- scores, kind='mergesort')
            difficult_box_mask = np.full(prediction.size, False)
            difficult_box_mask[prediction.metadata.get('difficult_boxes', [])] = True
            difficult_for_label = difficult_box_mask[prediction_ids]
            if len(scores_ids) > max_detections:
                scores_ids = scores_ids[:max_detections]
            detections = prepare_keypoints(prediction, prediction_ids)
            detections = detections[scores_ids]

            return detections, scores[scores_ids], difficult_for_label[scores_ids]

        def _prepare_annotations(annotation, label):
            annotation_ids = annotation.labels == label
            if not np.size(annotation_ids):
                return [], [], [], [], []
            difficult_box_mask = np.full(annotation.size, False)
            difficult_box_indices = annotation.metadata.get("difficult_boxes", [])
            iscrowd = np.array(annotation.metadata.get('iscrowd', [0] * annotation.size))
            difficult_box_mask[difficult_box_indices] = True
            difficult_box_mask[iscrowd > 0] = True
            difficult_label = difficult_box_mask[annotation_ids]
            not_difficult_box_indices = np.argwhere(~difficult_label).reshape(-1)
            difficult_box_indices = np.argwhere(difficult_label).reshape(-1)
            iscrowd_label = iscrowd[annotation_ids]
            order = np.hstack((not_difficult_box_indices, difficult_box_indices)).astype(int)
            boxes = np.array(annotation.bboxes)
            boxes = boxes[annotation_ids]
            areas = np.array(annotation.areas)
            areas = areas[annotation_ids] if np.size(areas) > 0 else np.array([])
            boxes = boxes[order]
            areas = areas[order]

            return (
                prepare_keypoints(annotation, annotation_ids)[order],
                difficult_label[order],
                iscrowd_label[order], boxes, areas
            )

        for label_id, label in enumerate(self.labels):
            detections, scores, dt_difficult = _prepare_predictions(prediction, label, self.max_detections)
            ground_truth, gt_difficult, iscrowd, boxes, areas = _prepare_annotations(annotation, label)
            iou = compute_oks(ground_truth, detections, boxes, areas)
            eval_result = evaluate_image(
                ground_truth, gt_difficult, iscrowd, detections, dt_difficult, scores, iou, self.thresholds
            )
            self.matching_results[label_id].append(eval_result)
            per_class_results.append(eval_result)

        return per_class_results


class MSCOCOKeypointsPrecision(MSCOCOKeypointsBaseMetric):
    __provider__ = 'coco_keypoints_precision'

    def update(self, annotation, prediction):
        per_class_matching = super().update(annotation, prediction)
        return [
            compute_precision_recall(self.thresholds, [per_class_matching[i]])[0] for i, _ in enumerate(self.labels)
        ]

    def evaluate(self, annotations, predictions):
        precision = [
            compute_precision_recall(self.thresholds, self.matching_results[i])[0]
            for i, _ in enumerate(self.labels)
        ]
        precision, self.meta['names'] = finalize_metric_result(precision, self.meta['orig_label_names'])

        return precision


class MSCOCOKeypointsRecall(MSCOCOKeypointsBaseMetric):
    __provider__ = 'coco_keypoints_recall'

    def update(self, annotation, prediction):
        per_class_matching = super().update(annotation, prediction)
        return [
            compute_precision_recall(self.thresholds, [per_class_matching[i]])[1] for i, _ in enumerate(self.labels)
        ]

    def evaluate(self, annotations, predictions):
        recalls = [
            compute_precision_recall(self.thresholds, self.matching_results[i])[1]
            for i, _ in enumerate(self.labels)
        ]
        recalls, self.meta['names'] = finalize_metric_result(recalls, self.meta['orig_label_names'])

        return recalls


class MSCOCOSegmBase(MSCOCOBaseMetric):
    __provider__ = 'coco_segm'

    annotation_types = (CoCoInstanceSegmentationAnnotation,)
    prediction_types = (CoCoInstanceSegmentationPrediction,)

    def configure(self):
        super().configure()
        if isinstance(maskUtils, UnsupportedPackage):
            maskUtils.raise_error(self.__provider__)

    def update(self, annotation, prediction):
        compute_iou, create_boxes = select_specific_parameters(annotation)
        per_class_results = []
        profile_boxes = self.profiler is not None
        if profile_boxes:
            annotation_polygons = annotation.to_polygon()
            prediction_polygons = prediction.to_polygon()

        for label_id, label in enumerate(self.labels):

            detections, scores, dt_difficult = prepare_predictions(prediction, label, self.max_detections)
            ground_truth, gt_difficult, iscrowd, boxes, areas = prepare_annotations(annotation, label, create_boxes)
            iou = compute_iou(ground_truth, detections, annotation_boxes=boxes, annotation_areas=areas, iscrowd=iscrowd)
            eval_result = evaluate_image(
                ground_truth if not profile_boxes else annotation_polygons.get(label, []),
                gt_difficult, iscrowd, detections if not profile_boxes else prediction_polygons.get(label, []),
                dt_difficult, scores, iou, self.thresholds,
                profile_boxes
            )
            self.matching_results[label_id].append(eval_result)
            per_class_results.append(eval_result)

        return per_class_results


class MSCOCOSegmAveragePrecision(MSCOCOAveragePrecision, MSCOCOSegmBase):
    __provider__ = 'coco_segm_precision'


class MSCOCOSegmRecall(MSCOCORecall, MSCOCOSegmBase):
    __provider__ = 'coco_segm_recall'


@singledispatch
def select_specific_parameters(annotation):
    return compute_iou_boxes, False


@select_specific_parameters.register(PoseEstimationAnnotation)
def pose_estimation_params(annotation):
    return compute_oks, True


@select_specific_parameters.register(CoCoInstanceSegmentationAnnotation)
def instance_segmentation_params(annotation):
    return compute_iou_masks, False


@singledispatch
def prepare(entry, order):
    return np.c_[entry.x_mins[order], entry.y_mins[order], entry.x_maxs[order], entry.y_maxs[order]]


@prepare.register(PoseEstimationPrediction)
@prepare.register(PoseEstimationAnnotation)
def prepare_keypoints(entry, order):
    if entry.size == 0:
        return []

    if np.size(entry.x_values[order]) == 0:
        return []

    return np.concatenate((entry.x_values[order], entry.y_values[order], entry.visibility[order]), axis=-1)


@prepare.register(CoCoInstanceSegmentationPrediction)
@prepare.register(CoCoInstanceSegmentationAnnotation)
def prepare_masks(entry, order):
    return np.array([entry.mask[idx] for idx in order])


def prepare_predictions(prediction, label, max_detections):
    if prediction.size == 0:
        return [], [], []
    prediction_ids = np.argwhere(prediction.labels == label).reshape(-1)
    scores = prediction.scores[prediction_ids]
    if np.size(scores) == 0:
        return [], [], []
    scores_ids = np.argsort(- scores, kind='mergesort')
    difficult_box_mask = np.full(prediction.size, False)
    difficult_box_mask[prediction.metadata.get('difficult_boxes', [])] = True
    difficult_for_label = difficult_box_mask[prediction_ids]
    if len(scores_ids) > max_detections:
        scores_ids = scores_ids[:max_detections]
    detections = prepare(prediction, prediction_ids)
    detections = detections[scores_ids]

    return detections, scores[scores_ids], difficult_for_label[scores_ids]


def prepare_annotations(annotation, label, create_boxes=False):
    annotation_ids = np.argwhere(np.array(annotation.labels) == label).reshape(-1)
    if not np.size(annotation_ids):
        boxes = None if not create_boxes else np.array([])
        areas = None if not create_boxes else np.array([])
        return [], [], [], boxes, areas
    difficult_box_mask = np.full(annotation.size, False)
    difficult_box_indices = annotation.metadata.get("difficult_boxes", [])
    iscrowd = np.array(annotation.metadata.get('iscrowd', [0] * annotation.size))
    difficult_box_mask[difficult_box_indices] = True
    difficult_box_mask[iscrowd > 0] = True
    difficult_label = difficult_box_mask[annotation_ids]
    not_difficult_box_indices = np.argwhere(~difficult_label).reshape(-1)
    difficult_box_indices = np.argwhere(difficult_label).reshape(-1)
    iscrowd_label = iscrowd[annotation_ids]
    order = np.hstack((not_difficult_box_indices, difficult_box_indices)).astype(int)
    boxes = None
    areas = None
    if create_boxes:
        boxes = np.array(annotation.bboxes)
        boxes = boxes[annotation_ids]
        areas = np.array(annotation.areas)
        areas = areas[annotation_ids] if np.size(areas) > 0 else np.array([])
        boxes = boxes[order]
        areas = areas[order]

    return prepare(annotation, annotation_ids)[order], difficult_label[order], iscrowd_label[order], boxes, areas


def compute_precision_recall(thresholds, matching_results):
    num_thresholds = len(thresholds)
    rectangle_thresholds = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
    num_rec_thresholds = len(rectangle_thresholds)
    precision = -np.ones((num_thresholds, num_rec_thresholds))  # -1 for the precision of absent categories
    recall = -np.ones(num_thresholds)
    dt_scores = np.concatenate([e['scores'] for e in matching_results])
    inds = np.argsort(-1 * dt_scores, kind='mergesort')
    dtm = np.concatenate([e['dt_matches'] for e in matching_results], axis=1)[:, inds]
    dt_ignored = np.concatenate([e['dt_ignore'] for e in matching_results], axis=1)[:, inds]
    gt_ignored = np.concatenate([e['gt_ignore'] for e in matching_results])
    npig = np.count_nonzero(gt_ignored == 0)
    tps = np.logical_and(dtm, np.logical_not(dt_ignored))
    fps = np.logical_and(np.logical_not(dtm), np.logical_not(dt_ignored))
    tp_sum = np.cumsum(tps, axis=1).astype(dtype=float)
    fp_sum = np.cumsum(fps, axis=1).astype(dtype=float)
    num_images = np.sum([np.size(e['gt_ignore']) != 0 for e in matching_results])
    if npig == 0:
        return np.nan, np.nan, np.array([]), np.array([]), [False], [False], num_images
    precisions, recalls = [], []
    for t, (tp, fp) in enumerate(zip(tp_sum, fp_sum)):
        tp = np.array(tp)
        fp = np.array(fp)
        num_detections = len(tp)
        rc = tp / npig
        pr = tp / (fp + tp + np.spacing(1))
        precisions.append(pr)
        recalls.append(rc)
        q = np.zeros(num_rec_thresholds)
        if num_detections:
            recall[t] = rc[-1]
        else:
            recall[t] = 0

        # numpy is slow without cython optimization for accessing elements
        #  use python array gets significant speed improvement
        pr = pr.tolist()
        q = q.tolist()

        for i in range(num_detections - 1, 0, -1):
            if pr[i] > pr[i - 1]:
                pr[i - 1] = pr[i]

        inds = np.searchsorted(rc, rectangle_thresholds, side='left')
        try:
            for ri, pi in enumerate(inds):
                q[ri] = pr[pi]
        except IndexError:
            pass
        precision[t] = np.array(q)

    mean_precision = 0 if np.size(precision[precision > -1]) == 0 else np.mean(precision[precision > -1])
    mean_recall = 0 if np.size(recall[recall > -1]) == 0 else np.mean(recall[recall > -1])

    return (
        mean_precision, mean_recall,
        np.array(precisions),
        np.array(recalls),
        fps, tps, num_images
    )


def compute_iou_boxes(annotation, prediction, *args, **kwargs):
    if np.size(annotation) == 0 or np.size(prediction) == 0:
        return []
    overlap = Overlap.provide('iou')
    iou = np.zeros((prediction.size // 4, annotation.size // 4), dtype=np.float32)
    for i, box_a in enumerate(annotation):
        for j, box_b in enumerate(prediction):
            iou[j, i] = overlap(box_a, box_b)

    return iou


def compute_oks(annotation_points, prediction_points, annotation_boxes, annotation_areas, *args, **kwargs):
    if np.size(prediction_points) == 0 or np.size(annotation_points) == 0:
        return []
    oks = np.zeros((len(prediction_points), len(annotation_points)))
    sigmas = np.array([.26, .25, .25, .35, .35, .79, .79, .72, .72, .62, .62, 1.07, 1.07, .87, .87, .89, .89]) / 10.0
    variance = (sigmas * 2) ** 2
    # compute oks between each detection and ground truth object
    for gt_idx, gt_points in enumerate(annotation_points):
        # create bounds for ignore regions(double the gt bbox)
        xgt = gt_points[:17]
        ygt = gt_points[17:34]
        vgt = gt_points[34:]
        k1 = np.count_nonzero(vgt > 0)
        x0_bbox, y0_bbox, x1_bbox, y1_bbox = annotation_boxes[gt_idx]
        area_gt = annotation_areas[gt_idx]
        w_bbox = x1_bbox - x0_bbox
        h_bbox = y1_bbox - y0_bbox
        x0 = x0_bbox - w_bbox
        x1 = x0_bbox + w_bbox * 2
        y0 = y0_bbox - h_bbox
        y1 = y0_bbox + h_bbox * 2
        for dt_idx, dt_points in enumerate(prediction_points):
            xdt = dt_points[:17]
            ydt = dt_points[17:34]
            if k1 > 0:
                # measure the per-keypoint distance if keypoints visible
                x_diff = xdt - xgt
                y_diff = ydt - ygt
            else:
                # measure minimum distance to keypoints in (x0,y0) & (x1,y1)
                zeros = np.zeros(len(sigmas))
                x_diff = np.max((zeros, x0 - xdt), axis=0) + np.max((zeros, xdt - x1), axis=0)
                y_diff = np.max((zeros, y0 - ydt), axis=0) + np.max((zeros, ydt - y1), axis=0)
            evaluation = (x_diff ** 2 + y_diff ** 2) / variance / (area_gt + np.spacing(1)) / 2
            if k1 > 0:
                evaluation = evaluation[vgt > 0]
            oks[dt_idx, gt_idx] = np.sum(np.exp(- evaluation)) / evaluation.shape[0]

    return oks


def compute_iou_masks(annotation, prediction, iscrowd, *args, **kwargs):
    if np.size(annotation) == 0 or np.size(prediction) == 0:
        return []
    iou = maskUtils.iou(list(prediction), list(annotation), iscrowd)

    return iou


def evaluate_image(
        ground_truth, gt_difficult, iscrowd, detections, dt_difficult, scores, iou, thresholds, profile=False
):
    thresholds_num = len(thresholds)
    gt_num = len(ground_truth)
    dt_num = len(detections)
    gt_matched = np.zeros((thresholds_num, gt_num))
    dt_matched = np.zeros((thresholds_num, dt_num))
    gt_ignored = gt_difficult
    dt_ignored = np.zeros((thresholds_num, dt_num))
    if np.size(iou):
        for tind, t in enumerate(thresholds):
            for dtind, _ in enumerate(detections):
                # information about best match so far (matched_id = -1 -> unmatched)
                iou_current = min([t, 1 - 1e-10])
                matched_id = -1
                for gtind, _ in enumerate(ground_truth):
                    # if this gt already matched, and not a crowd, continue
                    if gt_matched[tind, gtind] > 0 and not iscrowd[gtind]:
                        continue
                    # if dt matched to reg gt, and on ignore gt, stop
                    if matched_id > -1 and not gt_ignored[matched_id] and gt_ignored[gtind]:
                        break
                    # continue to next gt unless better match made
                    if iou[dtind, gtind] < iou_current:
                        continue
                    # if match successful and best so far, store appropriately
                    iou_current = iou[dtind, gtind]
                    matched_id = gtind
                # if match made store id of match for both dt and gt
                if matched_id == -1:
                    continue
                dt_ignored[tind, dtind] = gt_ignored[matched_id]
                dt_matched[tind, dtind] = 1
                gt_matched[tind, matched_id] = dtind
    # store results for given image
    results = {
        'dt_matches': dt_matched,
        'gt_matches': gt_matched,
        'gt_ignore': gt_ignored,
        'dt_ignore': np.logical_or(dt_ignored, dt_difficult),
        'scores': scores
    }
    if profile:
        results.update({
            'dt': detections,
            'gt': ground_truth,
            'iou': iou
        })

    return results


def process_threshold(threshold):
    if isinstance(threshold, str):
        threshold_values = [str(float(value)) for value in threshold.split(":")]
        threshold = ":".join(threshold_values)
    return threshold
