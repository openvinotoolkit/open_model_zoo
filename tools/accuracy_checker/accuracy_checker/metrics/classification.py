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

import numpy as np

from ..representation import (
    ClassificationAnnotation,
    ClassificationPrediction,
    TextClassificationAnnotation,
    UrlClassificationAnnotation,
    ArgMaxClassificationPrediction,
    AnomalySegmentationAnnotation,
    AnomalySegmentationPrediction
)

from ..config import NumberField, StringField, ConfigError, BoolField
from .metric import Metric, PerImageEvaluationMetric
from .average_meter import AverageMeter
from ..utils import UnsupportedPackage

try:
    from sklearn.metrics import accuracy_score, confusion_matrix
except ImportError as import_error:
    accuracy_score = UnsupportedPackage("sklearn.metric.accuracy_score", import_error.msg)
    confusion_matrix = UnsupportedPackage("sklearn.metric.confusion_matrix", import_error.msg)

class ClassificationAccuracy(PerImageEvaluationMetric):
    """
    Class for evaluating accuracy metric of classification models.
    """

    __provider__ = 'accuracy'

    annotation_types = (ClassificationAnnotation, TextClassificationAnnotation)
    prediction_types = (ClassificationPrediction, ArgMaxClassificationPrediction)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'top_k': NumberField(
                value_type=int, min_value=1, optional=True, default=1,
                description="The number of classes with the highest probability, which will be used to decide "
                            "if prediction is correct."
            ),
            'match': BoolField(optional=True, default=False),
            'cast_to_int': BoolField(optional=True, default=False)
        })

        return parameters

    def configure(self):
        self.top_k = self.get_value_from_config('top_k')
        self.match = self.get_value_from_config('match')
        self.cast_to_int = self.get_value_from_config('cast_to_int')
        self.summary_helper = None

        def loss(annotation_label, prediction_top_k_labels):
            return int(annotation_label in prediction_top_k_labels)

        if not self.match:
            self.accuracy = AverageMeter(loss)
        else:
            if isinstance(accuracy_score, UnsupportedPackage):
                accuracy_score.raise_error(self.__provider__)
            self.accuracy = []
        if self.profiler:
            self.summary_helper = ClassificationProfilingSummaryHelper()

    def set_profiler(self, profiler):
        self.profiler = profiler
        self.summary_helper = ClassificationProfilingSummaryHelper()

    def update(self, annotation, prediction):
        if not self.match:
            accuracy = self.accuracy.update(annotation.label, prediction.top_k(self.top_k))
        else:
            label = prediction.label if not self.cast_to_int else np.round(prediction.label)

            accuracy = accuracy_score(annotation.label, label)
            self.accuracy.append(accuracy)
        if self.profiler:
            self.summary_helper.submit_data(annotation.label, prediction.top_k(self.top_k), prediction.scores)
            self.profiler.update(
                annotation.identifier, annotation.label, prediction.top_k(self.top_k), self.name, accuracy,
                prediction.scores
            )
        return accuracy

    def evaluate(self, annotations, predictions):
        if self.profiler:
            self.profiler.finish()
            summary = self.summary_helper.get_summary_report()
            self.profiler.write_summary(summary)
        if not self.match:
            accuracy = self.accuracy.evaluate()
        else:
            accuracy = np.mean(self.accuracy)
        return accuracy

    def reset(self):
        if not self.match:
            self.accuracy.reset()
        else:
            self.accuracy = []

        if self.profiler:
            self.profiler.reset()


class ClassificationProfilingSummaryHelper:
    def __init__(self):
        self.gt, self.pred, self.scores = [], [], []

    def reset(self):
        self.gt, self.pred, self.scores = [], [], []

    def submit_data(self, annotation_label, prediction_label, scores):
        self.gt.append(annotation_label)
        self.pred.append(prediction_label)
        self.scores.append(scores)

    def get_summary_report(self):
        if not self.gt:
            return {
                'summary_result': {
                    'precision': 0., 'recall': 0, 'f1_score': 0.,
                    'charts': {'roc': [], 'precision_recall': []}, 'num_objects': 0
                },
                'per_class_result': {}
            }
        y_true, y_score = self.binarize_labels()
        cm = self.cm()
        average_roc, avg_roc_auc, per_class_roc, per_class_auc = self.roc(y_true, y_score)
        average_pr_chart, per_class_pr_chart, average_pr_area, per_class_pr_area = self.pr(y_true, y_score)
        cm_diagonal = cm.diagonal()
        cm_horizontal_sum = cm.sum(axis=1)
        cm_vertical_sum = cm.sum(axis=0)
        precision = np.divide(
            cm_diagonal, cm_horizontal_sum, out=np.zeros_like(cm_diagonal, dtype=float), where=cm_horizontal_sum != 0
        )
        recall = np.divide(
            cm_diagonal, cm_vertical_sum, out=np.zeros_like(cm_diagonal, dtype=float), where=cm_vertical_sum != 0
        )
        sum_precision_recall = precision + recall
        f1_score = 2 * np.divide(
            precision * recall, sum_precision_recall, out=np.zeros_like(cm_diagonal, dtype=float),
            where=sum_precision_recall != 0
        )
        accuracy_per_class = precision
        summary = {
            'summary_result': {
                'precision': precision.mean(),
                'recall': recall.mean(),
                'f1_score': f1_score.mean(),
                'charts': {
                    'roc': average_roc.tolist(),
                    'precision_recall': average_pr_chart.tolist(),
                    'roc_auc': avg_roc_auc,
                    'precision_recall_area': average_pr_area,
                },
                'num_objects': np.sum(cm_horizontal_sum)
            },
            'per_class_result': {}
        }
        per_class_result = {}
        for i in range(cm.shape[0]):
            per_class_result[i] = {
                'result': accuracy_per_class[i],
                'precision': precision[i],
                'recall': recall[i],
                'f1_score': f1_score[i],
                'result_scale': 100,
                'result_postfix': '%',
                'num_objects': cm_horizontal_sum[i],
                'charts': {
                    'roc': per_class_roc[i].tolist(),
                    'precision_recall': per_class_pr_chart[i].tolist(),
                    'roc_auc': per_class_auc[i],
                    'precision_recall_area': per_class_pr_area[i]
                }
            }
        summary['per_class_result'].update(per_class_result)
        return summary

    def cm(self):
        num_labels = max(np.max(self.gt) + 1, np.max(self.pred) + 1)
        cm = np.zeros((num_labels, num_labels))
        for gt, pred in zip(self.gt, self.pred):
            cm[gt][pred] += 1
        return cm

    def binarize_labels(self):
        max_v = max(np.max(self.gt) + 1, np.max(self.pred) + 1)
        gt_bin = np.zeros((len(self.gt), max_v))
        pred_bin = np.zeros((len(self.pred), max_v))
        np.put_along_axis(gt_bin, np.expand_dims(np.array(self.gt).astype(int), 1), 1, axis=1)
        for top in np.transpose(self.pred, (1, 0)):
            np.put_along_axis(
                pred_bin, np.expand_dims(top.astype(int), 1), 1, axis=1)
        return gt_bin, pred_bin

    def roc(self, y_true, y_score):
        per_class_chart = {}
        per_class_area = {}
        for i in range(y_true.shape[-1]):
            per_class_chart[i], per_class_area[i] = self.roc_curve(y_true[:, i], y_score[:, i])
        average_chart, average_area = self.roc_curve(y_true.ravel(), y_score.ravel())
        return average_chart, average_area, per_class_chart, per_class_area

    def roc_curve(self, gt, pred):
        desc_score_indices = np.argsort(pred, kind="mergesort")[::-1]
        y_score = pred[desc_score_indices]
        y_true = gt[desc_score_indices]
        distinct_value_indices = np.where(np.diff(y_score))[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
        tps = np.cumsum(y_true)[threshold_idxs]
        fps = 1 + threshold_idxs - tps
        if max(fps) > 0:
            fps /= fps[-1]
        if max(tps) > 0:
            tps /= tps[-1]
        plot = np.array([fps, tps, y_score[threshold_idxs]])
        area = self.roc_auc_score(fps, tps)
        return plot.T, area

    def pr_curve(self, gt, pred):
        chart, area = self.roc_curve(gt, pred)
        fps, tps, _ = chart.T

        precision = tps / (tps + fps)
        precision[np.isnan(precision)] = 0
        recall = tps / tps[-1]
        recall[np.isnan(recall)] = 0
        last_ind = tps.searchsorted(tps[-1])
        sl = slice(last_ind, None, -1)
        area = self.precision_recall_auc(np.r_[precision[sl], 1], np.r_[recall[sl], 0])
        return np.array([np.r_[precision[sl], 1], np.r_[recall[sl], 0]]).T, area

    def pr(self, y_true, y_score):
        per_class_chart = {}
        per_class_area = {}
        for i in range(y_true.shape[-1]):
            per_class_chart[i], per_class_area[i] = self.pr_curve(y_true[:, i], y_score[:, i])
        average_chart, average_area = self.pr_curve(y_true.ravel(), y_score.ravel())
        return average_chart, per_class_chart, average_area, per_class_area

    @staticmethod
    def roc_auc_score(fpr, tpr):
        direction = 1
        dx = np.diff(fpr)
        if np.any(dx < 0):
            if np.all(dx <= 0):
                direction = -1
        return direction * np.trapz(tpr, fpr)

    @staticmethod
    def precision_recall_auc(precision, recall):
        return -1 * np.sum(np.diff(recall) * np.array(precision)[:-1])


class ClassificationAccuracyClasses(PerImageEvaluationMetric):
    """
    Class for evaluating accuracy for each class of classification models.
    """

    __provider__ = 'accuracy_per_class'

    annotation_types = (ClassificationAnnotation, TextClassificationAnnotation)
    prediction_types = (ClassificationPrediction, ArgMaxClassificationPrediction)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'top_k': NumberField(
                value_type=int, min_value=1, optional=True, default=1,
                description="The number of classes with the highest probability,"
                            " which will be used to decide if prediction is correct."
            ),
            'label_map': StringField(optional=True, default='label_map', description="Label map.")
        })
        return parameters

    def configure(self):
        self.top_k = self.get_value_from_config('top_k')
        label_map = self.get_value_from_config('label_map')
        if self.dataset.metadata:
            self.labels = self.dataset.metadata.get(label_map)
            if not self.labels:
                raise ConfigError('accuracy per class metric requires label_map providing in dataset_meta'
                                  'Please provide dataset meta file or regenerate annotation')
        else:
            raise ConfigError('accuracy per class metric requires dataset metadata'
                              'Please provide dataset meta file or regenerate annotation')
        self.meta['names'] = list(self.labels.values())

        def loss(annotation_label, prediction_top_k_labels):
            result = np.zeros_like(list(self.labels.keys()))
            if annotation_label in prediction_top_k_labels:
                result[annotation_label] = 1

            return result

        def counter(annotation_label):
            result = np.zeros_like(list(self.labels.keys()))
            result[annotation_label] = 1
            return result

        self.accuracy = AverageMeter(loss, counter)

    def update(self, annotation, prediction):
        result = self.accuracy.update(annotation.label, prediction.top_k(self.top_k))
        if self.profiler:
            self.profiler.update(
                annotation.identifier, annotation.label, prediction.top_k(self.top_k), self.name, result,
                prediction.scores
            )

        return result

    def evaluate(self, annotations, predictions):
        if self.profiler:
            self.profiler.finish()
        return self.accuracy.evaluate()

    def reset(self):
        if self.profiler:
            self.profiler.reset()
        self.accuracy.reset()

    def set_profiler(self, profiler):
        self.profiler = profiler
        self.summary_helper = ClassificationProfilingSummaryHelper()


class AverageProbMeter(AverageMeter):
    def __init__(self):
        def loss(annotation_label, prediction_scores):
            return prediction_scores
        super().__init__(loss=loss)


class ClipAccuracy(PerImageEvaluationMetric):
    __provider__ = 'clip_accuracy'

    annotation_types = (ClassificationAnnotation, )
    prediction_types = (ClassificationPrediction, )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.clip_accuracy = AverageMeter()
        self.video_accuracy = AverageMeter()
        self.video_avg_prob = AverageProbMeter()
        self.previous_video_id = None
        self.previous_video_label = None

    def update(self, annotation, prediction):
        if isinstance(annotation.identifier, list):
            video_id = annotation.identifier[0].video
        else:
            video_id = annotation.identifier.video

        if self.previous_video_id is not None and video_id != self.previous_video_id:
            video_top_label = np.argmax(self.video_avg_prob.evaluate())
            self.video_accuracy.update(video_top_label, self.previous_video_label)
            self.video_avg_prob = AverageProbMeter()

        self.video_avg_prob.update(annotation.label, prediction.scores)

        clip_accuracy = self.clip_accuracy.update(annotation.label, prediction.label)

        self.previous_video_id = video_id
        self.previous_video_label = annotation.label
        if self.profiler:
            self.profiler.update(annotation.identifier, prediction.label, self.name, clip_accuracy)

        return clip_accuracy

    def evaluate(self, annotations, predictions):
        if self.profiler:
            self.profiler.finish()
        return [self.clip_accuracy.evaluate(), self.video_accuracy.evaluate()]

    def reset(self):
        self.clip_accuracy.reset()
        self.video_accuracy.reset()
        self.video_avg_prob.reset()
        if self.profiler:
            self.profiler.reset()

    @classmethod
    def get_common_meta(cls):
        return {'target': 'higher-better', 'names': ['clip_accuracy', 'video_accuracy']}


class ClassificationF1Score(PerImageEvaluationMetric):
    __provider__ = 'classification_f1-score'

    annotation_types = (ClassificationAnnotation, TextClassificationAnnotation, AnomalySegmentationAnnotation)
    prediction_types = (ClassificationPrediction, ArgMaxClassificationPrediction, AnomalySegmentationPrediction)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'label_map': StringField(optional=True, default='label_map', description="Label map."),
            'pos_label': NumberField(
                optional=True, value_type=int, min_value=0,
                description="Return metric value for specified class during metric calculation."
            ),
            'pixel_level': BoolField(optional=True, default=False, description='calculate metric on pixel level')
        })
        return parameters

    def configure(self):
        label_map = self.get_value_from_config('label_map')
        self.pos_label = self.get_value_from_config('pos_label')
        self.pixel_level = self.get_value_from_config('pixel_level')
        if self.dataset.metadata:
            self.labels = self.dataset.metadata.get(label_map)
            if not self.labels:
                raise ConfigError('classification_f1-score metric requires label_map providing in dataset_meta'
                                  'Please provide dataset meta file or regenerate annotation')
        else:
            raise ConfigError('classification_f1-scores metric requires dataset metadata'
                              'Please provide dataset meta file or regenerate annotation')
        self.cm = np.zeros((len(self.labels), len(self.labels)))
        if self.pos_label is not None:
            self.meta['names'] = [self.labels[self.pos_label]]
        else:
            self.meta['names'] = list(self.labels.values())

    def update(self, annotation, prediction):
        if (
            self.pixel_level and isinstance(annotation, AnomalySegmentationAnnotation)
            and isinstance(prediction, AnomalySegmentationPrediction)
        ):
            return self.update_pixels(annotation, prediction)
        self.cm[annotation.label][prediction.label] += 1
        result = annotation.label == prediction.label
        if self.profiler:
            self.profiler.update(annotation.identifier, annotation.label, prediction.label, self.name, result)
        return result

    def update_pixels(self, annotation, prediction):
        label_true = annotation.mask.flatten()
        label_pred = prediction.mask.flatten()
        n_classes = len(self.labels)
        mask = (label_true >= 0) & (label_true < n_classes) & (label_pred < n_classes) & (label_pred >= 0)
        hist = np.bincount(n_classes * label_true[mask].astype(int) + label_pred[mask], minlength=n_classes ** 2)
        hist = hist.reshape(n_classes, n_classes)
        hist = hist.reshape(n_classes, n_classes)
        self.cm += hist
        return self.f1_score(hist)

    def evaluate(self, annotations, predictions):
        f1_score = self.f1_score(self.cm)
        if self.profiler:
            self.profiler.finish()
        if self.pos_label is not None:
            return f1_score[self.pos_label]
        return f1_score if len(f1_score) == 2 else f1_score[0]

    def reset(self):
        self.cm = np.zeros((len(self.labels), len(self.labels)))
        if self.profiler:
            self.profiler.reset()

    def set_profiler(self, profiler):
        self.profiler = profiler
        self.summary_helper = ClassificationProfilingSummaryHelper()

    @staticmethod
    def f1_score(cm):
        cm_diagonal = cm.diagonal()
        cm_horizontal_sum = cm.sum(axis=1)
        cm_vertical_sum = cm.sum(axis=0)
        precision = np.divide(
            cm_diagonal, cm_horizontal_sum, out=np.zeros_like(cm_diagonal, dtype=float), where=cm_horizontal_sum != 0
        )
        recall = np.divide(
            cm_diagonal, cm_vertical_sum, out=np.zeros_like(cm_diagonal, dtype=float), where=cm_vertical_sum != 0
        )
        sum_precision_recall = precision + recall
        f1_score = 2 * np.divide(
            precision * recall, sum_precision_recall, out=np.zeros_like(cm_diagonal, dtype=float),
            where=sum_precision_recall != 0
        )
        return f1_score


class MetthewsCorrelation(PerImageEvaluationMetric):
    __provider__ = 'metthews_correlation_coef'
    annotation_types = (ClassificationAnnotation, TextClassificationAnnotation)
    prediction_types = (ClassificationPrediction, )

    def configure(self):
        label_map = self.dataset.metadata.get('label_map', [])
        if label_map and len(label_map) != 2:
            raise ConfigError('metthews_correlation_coefficient applicable only for binary classification task')
        self.reset()

    def update(self, annotation, prediction):
        if annotation.label and prediction.label:
            self.tp += 1
            return 1
        if not annotation.label and not prediction.label:
            self.tn += 1
            return 1
        if not annotation.label and prediction.label:
            self.fp += 1
            return 0
        if annotation.label and not prediction.label:
            self.fn += 1
            return 0
        return -1

    def evaluate(self, annotations, predictions):
        delimiter_sum = (self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn)
        return ((self.tp * self.tn) - (self.fp * self.fn)) / np.sqrt(delimiter_sum) if delimiter_sum != 0 else -1

    def reset(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0


class RocAucScore(PerImageEvaluationMetric):
    __provider__ = 'roc_auc_score'
    annotation_types = (ClassificationAnnotation, TextClassificationAnnotation, AnomalySegmentationAnnotation)
    prediction_types = (ClassificationPrediction, ArgMaxClassificationPrediction, AnomalySegmentationPrediction)

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'pixel_level': BoolField(
                optional=True, default=False,
                description='calculate metric on pixel level, for anomaly segmentation only'),
            'calculate_hot_label': BoolField(
                optional=True, default=False,
                description='calculate one hot label for annotation and prediction before metric evaluation '
                            'for anomaly segmentation')
        })
        return params

    def configure(self):
        self.reset()
        self.pixel_level = self.get_value_from_config('pixel_level')
        self.calculate_hot_label = self.get_value_from_config('calculate_hot_label')

    def update(self, annotation, prediction):
        if (
            self.pixel_level and
            isinstance(prediction, AnomalySegmentationPrediction)
            and isinstance(annotation, AnomalySegmentationAnnotation)
        ):
            self.targets.append(annotation.mask.flatten())
            self.results.append(prediction.mask.flatten())
        else:
            self.targets.append(annotation.label)
            self.results.append(prediction.label)
        if np.isscalar(self.results[-1]) or np.ndim(self.results[-1]) == 0:
            res = annotation.label == prediction.label
            if isinstance(res, np.ndarray):
                res = res.item()
            return res
        return self.auc_score(self.targets[-1], self.results[-1])

    @staticmethod
    def one_hot_labels(targets, results):
        max_v = int(max(np.max(targets) + 1, np.max(results) + 1))
        gt_bin = np.zeros((len(targets), max_v))
        pred_bin = np.zeros((len(results), max_v))
        np.put_along_axis(gt_bin, np.expand_dims(np.array(targets).astype(int), 1), 1, axis=1)
        np.put_along_axis(pred_bin, np.expand_dims(np.array(results).astype(int), 1), 1, axis=1)

        return gt_bin, pred_bin

    def roc_curve_area(self, gt, pred):
        desc_score_indices = np.argsort(pred, kind="mergesort")[::-1]
        y_score = pred[desc_score_indices]
        y_true = gt[desc_score_indices]
        distinct_value_indices = np.where(y_score[1:] - y_score[:-1])[0]
        threshold_idxs = np.r_[distinct_value_indices, y_true.size - 1]
        tps = np.cumsum(y_true)[threshold_idxs]
        fps = 1 + threshold_idxs - tps

        tps = np.r_[0, tps]
        fps = np.r_[0, fps]

        if max(fps) > 0:
            fps = fps / fps[-1]
        if max(tps) > 0:
            tps = tps / tps[-1]
        area = self.roc_auc_score(fps, tps)
        return area

    @staticmethod
    def roc_auc_score(fpr, tpr):
        direction = 1
        dx = np.diff(fpr)
        if np.any(dx < 0):
            if np.all(dx <= 0):
                direction = -1
        return direction * np.trapz(tpr, fpr)

    def evaluate(self, annotations, predictions):
        all_results = self.results if np.isscalar(self.results[-1]) else np.concatenate(self.results)
        all_targets = self.targets if np.isscalar(self.targets[-1]) else np.concatenate(self.targets)
        roc_auc = self.auc_score(all_targets, all_results)
        return roc_auc

    def reset(self):
        self.targets = []
        self.results = []

    def auc_score(self, targets, results):
        (gt, dt) = self.one_hot_labels(targets, results) if self.calculate_hot_label else (targets, results)
        avg_area = self.roc_curve_area(np.array(gt).ravel(), np.array(dt).ravel())
        return avg_area


class AcerScore(PerImageEvaluationMetric):
    __provider__ = 'acer_score'
    annotation_types = (ClassificationAnnotation, TextClassificationAnnotation)
    prediction_types = (ClassificationPrediction, )

    def configure(self):
        if isinstance(confusion_matrix, UnsupportedPackage):
            confusion_matrix.raise_error(self.__provider__)
        self.reset()

    def update(self, annotation, prediction):
        self.targets.append(annotation.label)
        self.results.append(prediction.label)
        return prediction.label == annotation.label

    def evaluate(self, annotations, predictions):
        all_results = np.array(self.results)
        all_targets = np.array(self.targets)
        tn, fp, fn, tp = confusion_matrix(y_true=all_targets,
                                          y_pred=all_results,
                                          ).ravel()

        apcer = fp / (tn + fp) if (tn + fp) != 0 else 0
        bpcer = fn / (fn + tp) if (fn + tp) != 0 else 0
        acer = (apcer + bpcer) / 2

        return acer

    def reset(self):
        self.targets = []
        self.results = []

    @classmethod
    def get_common_meta(cls):
        return {'target': 'higher-worse'}


class UrlClassificationScore(Metric):
    __provider__ = 'url_classification_score'

    annotation_types = (UrlClassificationAnnotation, )
    prediction_types = (ClassificationPrediction, )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._annotations = []
        self._predictions = []

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'threshold': NumberField(
                value_type=float, min_value=0.01, max_value=0.99, optional=True, default=0.5,
                description="The threshold for ouput predictions when output is considered malcious or not."
            )
        })
        return parameters

    def configure(self):
        self.threshold = self.get_value_from_config('threshold')

    def update(self, annotation, prediction):
        _, accuracy = prediction.scores
        self._annotations.append(annotation.label)
        self._predictions.append(accuracy)
        return accuracy

    def evaluate(self, annotations, predictions):
        y_pred = [1 if x > self.threshold else 0 for x in self._predictions]
        y_true = self._annotations
        TN, FP, FN, TP = confusion_matrix(y_true, y_pred).ravel()
        # roc_auc = roc_auc_score(y_true, y_pred).tolist()
        accuracy = (TP+TN)/(TP+FP+FN+TN)
        return accuracy

    def reset(self):
        self._annotations = []
        self._predictions = []

    @classmethod
    def get_common_meta(cls):
        return {'target': 'higher-better', 'names': ['accuracy']}
