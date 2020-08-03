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

from ..config import BoolField, NumberField, ConfigError
from ..representation import (
    SegmentationAnnotation,
    SegmentationPrediction,
    BrainTumorSegmentationAnnotation,
    BrainTumorSegmentationPrediction,
    OAR3DTilingSegmentationAnnotation,
)
from .metric import PerImageEvaluationMetric
from ..utils import finalize_metric_result


class SegmentationMetric(PerImageEvaluationMetric):
    annotation_types = (SegmentationAnnotation, )
    prediction_types = (SegmentationPrediction, )

    CONFUSION_MATRIX_KEY = 'segmentation_confusion_matrix'

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'use_argmax': BoolField(
                optional=True, default=True, description="Allows to use argmax for prediction mask."
            ),
            'ignore_label': NumberField(
                optional=True, value_type=int, min_value=0,
                description='Ignore prediction and annotation of specified class during metric calculation'
            )
        })

        return parameters

    def evaluate(self, annotations, predictions):
        raise NotImplementedError

    def configure(self):
        self.use_argmax = self.get_value_from_config('use_argmax')
        if not self.dataset.labels:
            raise ConfigError('semantic segmentation metrics require label_map providing in dataset_meta'
                              'Please provide dataset meta file or regenerated annotation')
        self.ignore_label = self.get_value_from_config('ignore_label')

    def update(self, annotation, prediction):
        n_classes = len(self.dataset.labels)
        prediction_mask = np.argmax(prediction.mask, axis=0) if self.use_argmax else prediction.mask.astype('int64')

        def confusion_matrix():
            label_true = annotation.mask.flatten()
            label_pred = prediction_mask.flatten()
            mask = (label_true >= 0) & (label_true < n_classes) & (label_pred < n_classes) & (label_pred >= 0)
            hist = np.bincount(n_classes * label_true[mask].astype(int) + label_pred[mask], minlength=n_classes ** 2)
            hist = hist.reshape(n_classes, n_classes)
            if self.ignore_label is not None:
                hist[self.ignore_label, :] = 0
                hist[:, self.ignore_label] = 0

            return hist

        def accumulate(confusion_matrixs):
            return confusion_matrixs + cm

        cm = confusion_matrix()

        self._update_state(accumulate, self.CONFUSION_MATRIX_KEY, lambda: np.zeros((n_classes, n_classes)))
        return cm

    def reset(self):
        self.state = {}
        self._update_iter = 0


class SegmentationAccuracy(SegmentationMetric):
    __provider__ = 'segmentation_accuracy'

    def update(self, annotation, prediction):
        cm = super().update(annotation, prediction)
        return np.diag(cm).sum() / cm.sum()

    def evaluate(self, annotations, predictions):
        confusion_matrix = self.state[self.CONFUSION_MATRIX_KEY]
        return np.diag(confusion_matrix).sum() / confusion_matrix.sum()


class SegmentationIOU(SegmentationMetric):
    __provider__ = 'mean_iou'

    def update(self, annotation, prediction):
        cm = super().update(annotation, prediction)
        diagonal = np.diag(cm).astype(float)
        union = cm.sum(axis=1) + cm.sum(axis=0) - diagonal
        iou = np.divide(diagonal, union, out=np.full_like(diagonal, np.nan), where=union != 0)
        if self.ignore_label is not None:
            iou = np.delete(iou, self.ignore_label)

        return iou

    def evaluate(self, annotations, predictions):
        confusion_matrix = self.state[self.CONFUSION_MATRIX_KEY]
        diagonal = np.diag(confusion_matrix)
        union = confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) - diagonal
        iou = np.divide(diagonal, union, out=np.full_like(diagonal, np.nan), where=union != 0)
        cls_names = list(self.dataset.labels.values())
        if self.ignore_label is not None:
            iou = np.delete(iou, self.ignore_label)
            cls_names = [cls_name for cls_id, cls_name in self.dataset.labels.items() if cls_id != self.ignore_label]

        values, names = finalize_metric_result(iou, cls_names)
        self.meta['names'] = names

        return values


class SegmentationMeanAccuracy(SegmentationMetric):
    __provider__ = 'mean_accuracy'

    def update(self, annotation, prediction):
        cm = super().update(annotation, prediction)
        diagonal = np.diag(cm).astype(float)
        per_class_count = cm.sum(axis=1)
        acc_cls = np.divide(diagonal, per_class_count, out=np.full_like(diagonal, np.nan), where=per_class_count != 0)

        return acc_cls

    def evaluate(self, annotations, predictions):
        confusion_matrix = self.state[self.CONFUSION_MATRIX_KEY]
        diagonal = np.diag(confusion_matrix)
        per_class_count = confusion_matrix.sum(axis=1)
        acc_cls = np.divide(diagonal, per_class_count, out=np.full_like(diagonal, np.nan), where=per_class_count != 0)

        values, names = finalize_metric_result(acc_cls, list(self.dataset.labels.values()))
        self.meta['names'] = names

        return values


class SegmentationFWAcc(SegmentationMetric):
    __provider__ = 'frequency_weighted_accuracy'

    def update(self, annotation, prediction):
        cm = super().update(annotation, prediction)
        diagonal = np.diag(cm).astype(float)
        union = cm.sum(axis=1) + cm.sum(axis=0) - diagonal
        iou = np.divide(diagonal, union, out=np.zeros_like(diagonal), where=union != 0)
        freq = cm.sum(axis=1) / cm.sum()

        return (freq[freq > 0] * iou[freq > 0]).sum()

    def evaluate(self, annotations, predictions):
        confusion_matrix = self.state[self.CONFUSION_MATRIX_KEY]
        diagonal = np.diag(confusion_matrix)
        union = confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) - diagonal
        iou = np.divide(diagonal, union, out=np.zeros_like(diagonal), where=union != 0)
        freq = confusion_matrix.sum(axis=1) / confusion_matrix.sum()

        return (freq[freq > 0] * iou[freq > 0]).sum()


class SegmentationDSCAcc(PerImageEvaluationMetric):
    __provider__ = 'dice'
    annotation_types = (BrainTumorSegmentationAnnotation,)
    prediction_types = (BrainTumorSegmentationPrediction,)
    overall_metric = []

    def update(self, annotation, prediction):
        result = []
        for prediction_mask, annotation_mask in zip(prediction.mask, annotation.mask):
            annotation_mask = np.transpose(annotation_mask, (2, 0, 1))
            annotation_mask = np.expand_dims(annotation_mask, 0)
            numerator = np.sum(prediction_mask * annotation_mask) * 2.0 + 1.0
            denominator = np.sum(annotation_mask) + np.sum(prediction_mask) + 1.0
            result.append(numerator / denominator)
        self.overall_metric.extend(result)
        return np.mean(result)

    def evaluate(self, annotations, predictions):
        return sum(self.overall_metric) / len(self.overall_metric)

    def reset(self):
        self.overall_metric = []


class SegmentationDIAcc(PerImageEvaluationMetric):
    __provider__ = 'dice_index'
    annotation_types = (BrainTumorSegmentationAnnotation, SegmentationAnnotation, OAR3DTilingSegmentationAnnotation)
    prediction_types = (BrainTumorSegmentationPrediction, SegmentationPrediction, )

    overall_metric = []

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'mean': BoolField(optional=True, default=True, description='Allows calculation mean value.'),
            'median': BoolField(optional=True, default=False, description='Allows calculation median value.'),
        })

        return parameters

    def configure(self):
        self.mean = self.get_value_from_config('mean')
        self.median = self.get_value_from_config('median')
        self.output_order = self.get_value_from_config('output_order')

        labels = list(self.dataset.labels.values()) if self.dataset.metadata else ['overall']
        self.classes = len(labels)

        names_mean = ['mean@{}'.format(name) for name in labels] if self.mean else []
        names_median = ['median@{}'.format(name) for name in labels] if self.median else []
        self.meta['names'] = names_mean + names_median

        self.meta['calculate_mean'] = False

        self.overall_metric = []

    def update(self, annotation, prediction):
        result = np.zeros(shape=self.classes)

        annotation_data = annotation.mask
        prediction_data = prediction.mask

        if prediction_data.shape[0] != 1 and len(prediction_data.shape) != 3:
            raise RuntimeError("For '{}' metric prediction mask should has only 1 channel, but more found. "
                               "Specify 'make_argmax' option in adapter or postprocessor."
                               .format(self.__provider__))

        for c, p in enumerate(prediction.label_order, 1):
            annotation_data_ = (annotation_data == c)
            prediction_data_ = (prediction_data == p)

            intersection_count = np.logical_and(annotation_data_, prediction_data_).sum()
            union_count = annotation_data_.sum() + prediction_data_.sum()
            if union_count > 0:
                result[c] += 2.0*intersection_count / union_count

        annotation_data_ = (annotation_data > 0)
        prediction_data_ = (prediction_data > 0)

        intersection_count = np.logical_and(annotation_data_, prediction_data_).sum()
        union_count = annotation_data_.sum() + prediction_data_.sum()
        if union_count > 0:
            result[0] += 2.0 * intersection_count / union_count

        self.overall_metric.append(result)

        return result

    def evaluate(self, annotations, predictions):
        mean = np.mean(self.overall_metric, axis=0) if self.mean else []
        median = np.median(self.overall_metric, axis=0) if self.median else []
        result = np.concatenate((mean, median))
        return result

    def reset(self):
        labels = self.dataset.labels.values() if self.dataset.metadata else ['overall']
        self.classes = len(labels)
        names_mean = ['mean@{}'.format(name) for name in labels] if self.mean else []
        names_median = ['median@{}'.format(name) for name in labels] if self.median else []
        self.meta['names'] = names_mean + names_median
        self.meta['calculate_mean'] = False
        self.overall_metric = []

class SegmentationOAR3DTiling(PerImageEvaluationMetric):
    __provider__ = 'dice_oar3d'
    annotation_types = (OAR3DTilingSegmentationAnnotation,)
    prediction_types = (SegmentationPrediction,)

    overall_metric = []

    def configure(self):
        self.overall_metric = []

    def update(self, annotation, prediction):

        eps = 1e-6
        numerator = 2.0 * np.sum(annotation.mask * prediction.mask)
        denominator = np.sum(annotation.mask) + np.sum(prediction.mask)
        result = (numerator + eps) / (denominator + eps)

        self.overall_metric.append(result)

        return result

    def evaluate(self, annotations, predictions):
        result = np.mean(self.overall_metric, axis=0)
        return result

    def reset(self):
        self.overall_metric = []
