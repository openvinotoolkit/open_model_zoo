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

import numpy as np

from ..config import BoolField
from ..representation import (
    SegmentationAnnotation,
    SegmentationPrediction,
    BrainTumorSegmentationAnnotation,
    BrainTumorSegmentationPrediction
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
            )
        })

        return parameters

    def evaluate(self, annotations, predictions):
        raise NotImplementedError

    def configure(self):
        self.use_argmax = self.get_value_from_config('use_argmax')

    def update(self, annotation, prediction):
        n_classes = len(self.dataset.labels)
        prediction_mask = np.argmax(prediction.mask, axis=0) if self.use_argmax else prediction.mask.astype('int64')

        def update_confusion_matrix(confusion_matrix):
            label_true = annotation.mask.flatten()
            label_pred = prediction_mask.flatten()
            mask = (label_true >= 0) & (label_true < n_classes) & (label_pred < n_classes) & (label_pred >= 0)
            hist = np.bincount(n_classes * label_true[mask].astype(int) + label_pred[mask], minlength=n_classes ** 2)
            hist = hist.reshape(n_classes, n_classes)
            confusion_matrix += hist

            return confusion_matrix

        self._update_state(update_confusion_matrix, self.CONFUSION_MATRIX_KEY, lambda: np.zeros((n_classes, n_classes)))


class SegmentationAccuracy(SegmentationMetric):
    __provider__ = 'segmentation_accuracy'

    def evaluate(self, annotations, predictions):
        confusion_matrix = self.state[self.CONFUSION_MATRIX_KEY]
        return np.diag(confusion_matrix).sum() / confusion_matrix.sum()


class SegmentationIOU(SegmentationMetric):
    __provider__ = 'mean_iou'

    def evaluate(self, annotations, predictions):
        confusion_matrix = self.state[self.CONFUSION_MATRIX_KEY]
        union = confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
        diagonal = np.diag(confusion_matrix)
        iou = np.divide(diagonal, union, out=np.zeros_like(diagonal), where=union != 0)

        values, names = finalize_metric_result(iou, list(self.dataset.labels.values()))
        self.meta['names'] = names

        return values


class SegmentationMeanAccuracy(SegmentationMetric):
    __provider__ = 'mean_accuracy'

    def evaluate(self, annotations, predictions):
        confusion_matrix = self.state[self.CONFUSION_MATRIX_KEY]
        diagonal = np.diag(confusion_matrix)
        per_class_count = confusion_matrix.sum(axis=1)
        acc_cls = np.divide(diagonal, per_class_count, out=np.zeros_like(diagonal), where=per_class_count != 0)

        values, names = finalize_metric_result(acc_cls, list(self.dataset.labels.values()))
        self.meta['names'] = names

        return values


class SegmentationFWAcc(SegmentationMetric):
    __provider__ = 'frequency_weighted_accuracy'

    def evaluate(self, annotations, predictions):
        confusion_matrix = self.state[self.CONFUSION_MATRIX_KEY]

        union = (confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) - np.diag(confusion_matrix))
        diagonal = np.diag(confusion_matrix)
        iou = np.divide(diagonal, union, out=np.zeros_like(diagonal), where=union != 0)
        freq = confusion_matrix.sum(axis=1) / confusion_matrix.sum()

        return (freq[freq > 0] * iou[freq > 0]).sum()


class SegmentationDSCAcc(PerImageEvaluationMetric):
    __provider__ = 'dice'
    annotation_types = (BrainTumorSegmentationAnnotation,)
    prediction_types = (BrainTumorSegmentationPrediction,)
    overall_metric = []

    def update(self, annotation, prediction):
        cnt = 0
        for prediction_mask, annotation_mask in zip(prediction.mask, annotation.mask):
            annotation_mask = np.transpose(annotation_mask, (2, 0, 1))
            annotation_mask = np.expand_dims(annotation_mask, 0)
            numerator = np.sum(prediction_mask * annotation_mask) * 2.0 + 1.0
            denominator = np.sum(annotation_mask) + np.sum(prediction_mask) + 1.0
            self.overall_metric.append(numerator / denominator)
            cnt += 1

    def evaluate(self, annotations, predictions):
        return sum(self.overall_metric) / len(self.overall_metric)


class SegmentationDIAcc(PerImageEvaluationMetric):
    __provider__ = 'dice_index'
    annotation_types = (BrainTumorSegmentationAnnotation,)
    prediction_types = (BrainTumorSegmentationPrediction,)

    overall_metric = []

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'mean': BoolField(optional=True, default=True, description='Allows calculation mean value.'),
            'median': BoolField(optional=True, default=False, description='Allows calculation median value.')
        })

        return parameters

    def configure(self):
        self.mean = self.get_value_from_config('mean')
        self.median = self.get_value_from_config('median')

        labels = self.dataset.labels if self.dataset.metadata else ['overall']
        self.classes = len(labels)

        names_mean = ['mean@{}'.format(name) for name in labels] if self.mean else []
        names_median = ['median@{}'.format(name) for name in labels] if self.median else []
        self.meta['names'] = names_mean + names_median

        self.meta['calculate_mean'] = False

    def update(self, annotation, prediction):
        result = np.zeros(shape=self.classes)

        annotation_data = annotation.mask
        prediction_data = np.argmax(prediction.mask, axis=0)

        for c in range(1, self.classes):
            annotation_data_ = (annotation_data == c)
            prediction_data_ = (prediction_data == c)

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

    def evaluate(self, annotations, predictions):
        mean = np.mean(self.overall_metric, axis=0) if self.mean else []
        median = np.median(self.overall_metric, axis=0) if self.median else []
        result = np.concatenate((mean, median))
        return result
