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

from ..representation import (
    ClassificationAnnotation,
    ClassificationPrediction,
    TextClassificationAnnotation,
    ArgMaxClassificationPrediction
)

from ..config import NumberField, StringField, ConfigError
from .metric import PerImageEvaluationMetric
from .average_meter import AverageMeter


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
            )
        })

        return parameters

    def configure(self):
        self.top_k = self.get_value_from_config('top_k')

        def loss(annotation_label, prediction_top_k_labels):
            return int(annotation_label in prediction_top_k_labels)

        self.accuracy = AverageMeter(loss)

    def update(self, annotation, prediction):
        return self.accuracy.update(annotation.label, prediction.top_k(self.top_k))

    def evaluate(self, annotations, predictions):
        return self.accuracy.evaluate()

    def reset(self):
        self.accuracy.reset()


class ClassificationAccuracyClasses(PerImageEvaluationMetric):
    """
    Class for evaluating accuracy for each class of classification models.
    """

    __provider__ = 'accuracy_per_class'

    annotation_types = (ClassificationAnnotation, TextClassificationAnnotation)
    prediction_types = (ClassificationPrediction, )

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
        return self.accuracy.update(annotation.label, prediction.top_k(self.top_k))

    def evaluate(self, annotations, predictions):
        self.meta['names'] = list(self.labels.values())
        return self.accuracy.evaluate()

    def reset(self):
        self.accuracy.reset()


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
        video_id = annotation.identifier.video

        if self.previous_video_id is not None and video_id != self.previous_video_id:
            video_top_label = np.argmax(self.video_avg_prob.evaluate())
            self.video_accuracy.update(video_top_label, self.previous_video_label)
            self.video_avg_prob = AverageProbMeter()

        self.video_avg_prob.update(annotation.label, prediction.scores)

        clip_accuracy = self.clip_accuracy.update(annotation.label, prediction.label)

        self.previous_video_id = video_id
        self.previous_video_label = annotation.label

        return clip_accuracy

    def evaluate(self, annotations, predictions):
        self.meta['names'] = ['clip_accuracy', 'video_accuracy']
        return [self.clip_accuracy.evaluate(), self.video_accuracy.evaluate()]

    def reset(self):
        self.clip_accuracy.reset()
        self.video_accuracy.reset()
        self.video_avg_prob.reset()


class ClassificationF1Score(PerImageEvaluationMetric):
    __provider__ = 'classification_f1-score'

    annotation_types = (ClassificationAnnotation, TextClassificationAnnotation)
    prediction_types = (ClassificationPrediction, )
    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'label_map': StringField(optional=True, default='label_map', description="Label map.")
        })
        return parameters

    def configure(self):
        label_map = self.get_value_from_config('label_map')
        if self.dataset.metadata:
            self.labels = self.dataset.metadata.get(label_map)
            if not self.labels:
                raise ConfigError('classification_f1-score metric requires label_map providing in dataset_meta'
                                  'Please provide dataset meta file or regenerate annotation')
        else:
            raise ConfigError('classification_f1-scores metric requires dataset metadata'
                              'Please provide dataset meta file or regenerate annotation')
        self.cm = np.zeros((len(self.labels), len(self.labels)))

    def update(self, annotation, prediction):
        self.cm[prediction.label] += 1
        return annotation.label == prediction.label

    def evaluate(self, annotations, predictions):
        cm_diagonal = self.cm.diagonal()
        cm_horizontal_sum = self.cm.sum(axis=1)
        cm_vertical_sum = self.cm.sum(axis=0)
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
        return f1_score if len(f1_score) == 2 else f1_score[0]

    def reset(self):
        self.cm = np.zeros((len(self.labels), len(self.labels)))


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
        delimeter_sum = (self.tp + self.fp) * (self.tp + self.fn) * (self.tn + self.fp) * (self.tn + self.fn)
        return ((self.tp * self.tn) - (self.fp * self.fn)) / np.sqrt(delimeter_sum) if delimeter_sum != 0 else -1

    def reset(self):
        self.tp = 0
        self.tn = 0
        self.fp = 0
        self.fn = 0
