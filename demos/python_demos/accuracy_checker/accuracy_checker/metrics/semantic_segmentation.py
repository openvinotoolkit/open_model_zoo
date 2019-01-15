"""
Copyright (c) 2018 Intel Corporation

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
from ..representation import SegmentationAnnotation, SegmentationPrediction
from .metric import PerImageEvaluationMetric, BaseMetricConfig
from ..utils import finalize_metric_result


class SegmentationMetricConfig(BaseMetricConfig):
    use_argmax = BoolField(optional=True)

class SegmentationMetric(PerImageEvaluationMetric):
    def evaluate(self, annotations, predictions):
        raise NotImplementedError

    annotation_types = (SegmentationAnnotation, )
    prediction_types = (SegmentationPrediction, )

    def validate_config(self):
        config_validator = SegmentationMetricConfig(
            'SemanticSegmentation_config', SegmentationMetricConfig.ERROR_ON_EXTRA_ARGUMENT
        )
        config_validator.validate(self.config)

    def configure(self):
        self.use_argmax = self.config.get('use_argmax', True)

    def update(self, annotation, prediction):
        n_classes = len(self.dataset.labels)
        prediction_mask = np.argmax(prediction.mask, axis=0) if self.use_argmax else prediction.mask.astype('int64')

        def update_confusion_matrix(confusion_matrix):
            label_true = annotation.mask.flatten()
            label_pred = prediction_mask.flatten()

            mask = (label_true >= 0) & (label_true < n_classes)
            hist = np.bincount(n_classes * label_true[mask].astype(int) +
                               label_pred[mask], minlength=n_classes ** 2).reshape(n_classes, n_classes)
            confusion_matrix += hist
            return confusion_matrix

        self._update_state(update_confusion_matrix, 'segm_confusion_matrix', lambda: np.zeros((n_classes, n_classes)))


class SegmentationAccuracy(SegmentationMetric):
    __provider__ = 'segmentation_accuracy'

    def evaluate(self, annotations, predictions):
        confusion_matrix = self.state['segm_confusion_matrix']
        return np.diag(confusion_matrix).sum() / confusion_matrix.sum()


class SegmentationIOU(SegmentationMetric):
    __provider__ = 'mean_iou'

    def evaluate(self, annotations, predictions):
        confusion_matrix = self.state['segm_confusion_matrix']
        union = confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) - np.diag(confusion_matrix)
        diagonal = np.diag(confusion_matrix)
        iou = np.divide(diagonal, union, out=np.zeros_like(diagonal), where=union != 0)
        values, names = finalize_metric_result(iou, list(self.dataset.labels.values()))
        self.meta['names'] = names
        return values


class SegmentationMeanAccuracy(SegmentationMetric):
    __provider__ = 'mean_accuracy'

    def evaluate(self, annotations, predictions):
        confusion_matrix = self.state['segm_confusion_matrix']
        diagonal = np.diag(confusion_matrix)
        per_class_count = confusion_matrix.sum(axis=1)
        acc_cls = np.divide(diagonal, per_class_count, out=np.zeros_like(diagonal), where=per_class_count != 0)
        values, names = finalize_metric_result(acc_cls, list(self.dataset.labels.values()))
        self.meta['names'] = names
        return values


class SegmentationFWAcc(SegmentationMetric):
    __provider__ = 'frequency_weighted_accuracy'

    def evaluate(self, annotations, predictions):
        confusion_matrix = self.state['segm_confusion_matrix']
        union = (confusion_matrix.sum(axis=1) + confusion_matrix.sum(axis=0) - np.diag(confusion_matrix))
        diagonal = np.diag(confusion_matrix)
        iou = np.divide(diagonal, union, out=np.zeros_like(diagonal), where=union != 0)
        freq = confusion_matrix.sum(axis=1) / confusion_matrix.sum()
        fwavacc = (freq[freq > 0] * iou[freq > 0]).sum()
        return fwavacc
