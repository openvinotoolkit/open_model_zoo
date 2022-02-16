"""
Copyright (c) 2018-2022 Intel Corporation

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
from .metric import PerImageEvaluationMetric
from ..representation import MultiLabelRecognitionAnnotation, MultiLabelRecognitionPrediction
from ..config import StringField, BoolField, ConfigValidator, ConfigError


class MultiLabelMetric(PerImageEvaluationMetric):
    annotation_types = (MultiLabelRecognitionAnnotation, )
    prediction_types = (MultiLabelRecognitionPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'label_map': StringField(
                optional=True, default='label_map',
                description="The field in annotation metadata, which contains dataset label map."
            ),
            'calculate_average': BoolField(
                optional=True, default=True, description="Allows calculation of average accuracy."
            )
        })

        return parameters

    def configure(self):
        if not self.dataset.metadata:
            raise ConfigError('multi label metrics require  dataset_meta'
                              'Please provide dataset meta file or regenerate annotation')
        self.labels = self.dataset.metadata.get(self.get_value_from_config('label_map'))
        if not self.labels:
            raise ConfigError('multi label metrics require label_map providing in dataset_meta'
                              'Please provide dataset meta file or regenerate annotation')
        self.calculate_average = self.get_value_from_config('calculate_average')
        self.tp = np.zeros_like(list(self.labels.keys()), dtype=np.float)
        self.fp = np.zeros_like(list(self.labels.keys()), dtype=np.float)
        self.tn = np.zeros_like(list(self.labels.keys()), dtype=np.float)
        self.fn = np.zeros_like(list(self.labels.keys()), dtype=np.float)
        self.counter = np.zeros_like(list(self.labels.keys()), dtype=np.float)
        self._create_meta()

    def update(self, annotation, prediction):
        def loss(annotation_labels, prediction_labels):
            tp_result = np.zeros_like(list(self.labels.keys()), dtype=np.float)
            fp_results = np.zeros_like(list(self.labels.keys()), dtype=np.float)
            tn_results = np.zeros_like(list(self.labels.keys()), dtype=np.float)
            fn_results = np.zeros_like(list(self.labels.keys()), dtype=np.float)

            for index, label in enumerate(annotation_labels):
                if label == 1 and label == prediction_labels[index]:
                    tp_result[index] = 1.
                    continue

                if label == 1 and label != prediction_labels[index]:
                    fn_results[index] = 1.
                    continue

                if label == 0 and label == prediction_labels[index]:
                    tn_results[index] = 1.
                    continue

                if label == 0 and label != prediction_labels[index]:
                    fp_results[index] = 1.
                    continue

            return tp_result, fp_results, tn_results, fn_results

        def counter(annotation_label):
            count = np.zeros_like(annotation_label, dtype=float)
            cond = np.where(np.array(annotation_label) != -1)
            count[cond] = 1.
            return count

        tp_upd, fp_upd, tn_upd, fn_upd = loss(annotation.multi_label, prediction.multi_label)
        self.tp = np.add(self.tp, tp_upd)
        self.fp = np.add(self.fp, fp_upd)
        self.tn = np.add(self.tn, tn_upd)
        self.fn = np.add(self.fn, fn_upd)
        counter_upd = counter(annotation.multi_label)

        self.counter = np.add(self.counter, counter_upd)
        return tp_upd, fp_upd, tn_upd, fn_upd, counter_upd

    def evaluate(self, annotations, predictions):
        pass

    def _create_meta(self):
        self.meta['names'] = list(self.labels.values())
        if self.calculate_average:
            self.meta['names'].append('average')

    def reset(self):
        self.tp = np.zeros_like(list(self.labels.keys()), dtype=np.float)
        self.fp = np.zeros_like(list(self.labels.keys()), dtype=np.float)
        self.tn = np.zeros_like(list(self.labels.keys()), dtype=np.float)
        self.fn = np.zeros_like(list(self.labels.keys()), dtype=np.float)
        self.counter = np.zeros_like(list(self.labels.keys()), dtype=np.float)
        self._create_meta()

    @classmethod
    def get_common_meta(cls):
        meta = super().get_common_meta()
        meta['scale'] = 1
        meta['postfix'] = ''
        meta['calculate_mean'] = False
        return meta


class MultiLabelAccuracy(MultiLabelMetric):
    __provider__ = 'multi_accuracy'

    def update(self, annotation, prediction):
        tp_upd, _, tn_upd, _, counter_upd = super().update(annotation, prediction)
        tp_tn = np.add(tp_upd, tn_upd, dtype=float)
        average = np.sum(tp_tn) / np.sum(counter_upd)
        return average

    def evaluate(self, annotations, predictions):
        tp_tn = np.add(self.tp, self.tn, dtype=float)
        per_class = np.divide(tp_tn, self.counter, out=np.zeros_like(tp_tn, dtype=float), where=self.counter != 0)
        average = np.sum(tp_tn) / np.sum(self.counter)

        return [*per_class, average]


class MultiLabelPrecision(MultiLabelMetric):
    __provider__ = 'multi_precision'

    def update(self, annotation, prediction):
        tp_upd, fp_upd, _, _, _ = super().update(annotation, prediction)
        tp_fp = np.add(tp_upd, fp_upd, dtype=float)
        average = np.sum(tp_upd) / np.sum(tp_fp)
        return average

    def evaluate(self, annotations, predictions):
        tp_fp = np.add(self.tp, self.fp, dtype=float)
        per_class = np.divide(self.tp, tp_fp, out=np.zeros_like(self.tp, dtype=float), where=tp_fp != 0)
        if not self.calculate_average:
            return per_class
        average = np.sum(self.tp) / np.sum(tp_fp)

        return [*per_class, average]


class MultiLabelRecall(MultiLabelMetric):
    __provider__ = 'multi_recall'

    def update(self, annotation, prediction):
        tp_upd, _, _, fn_upd, _ = super().update(annotation, prediction)
        tp_fn = np.add(tp_upd, fn_upd, dtype=float)
        average = np.sum(tp_upd) / np.sum(tp_fn)
        return average

    def evaluate(self, annotations, predictions):
        tp_fn = np.add(self.tp, self.fn, dtype=float)
        per_class = np.divide(self.tp, tp_fn, out=np.zeros_like(self.tp, dtype=float), where=tp_fn != 0)
        if not self.calculate_average:
            return per_class
        average = np.sum(self.tp) / np.sum(tp_fn)

        return [*per_class, average]


class F1Score(PerImageEvaluationMetric):
    __provider__ = 'f1-score'
    annotation_types = (MultiLabelRecognitionAnnotation, )
    prediction_types = (MultiLabelRecognitionPrediction, )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.precision = MultiLabelPrecision(self.config, self.dataset)
        self.recall = MultiLabelRecall(self.config, self.dataset)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'label_map': StringField(
                optional=True, default='label_map',
                description="The field in annotation metadata, which contains dataset label map."
            ),
            'calculate_average': BoolField(
                optional=True, default=True, description="Allows calculation of average f-score"
            )
        })

        return parameters

    @classmethod
    def validate_config(cls, config):
        ConfigValidator(
            'f1_score',
            on_extra_argument=ConfigValidator.ERROR_ON_EXTRA_ARGUMENT,
            fields=cls.parameters()
        ).validate(config)

    def configure(self):
        if not self.dataset.metadata:
            raise ConfigError('f1-score metric requires dataset metadata providing'
                              'Please provide dataset meta file or regenerated annotation')
        label_map = self.get_value_from_config('label_map')
        self.labels = self.dataset.metadata.get(label_map)
        if not self.labels:
            raise ConfigError('f1-score metric requires label_map providing in dataset_meta'
                              'Please provide dataset meta file or regenerated annotation')
        self.calculate_average = self.get_value_from_config('calculate_average')
        self._create_meta()

    def update(self, annotation, prediction):
        avg_precision = self.precision.update(annotation, prediction)
        avg_recall = self.recall.update(annotation, prediction)
        return 2 * avg_precision * avg_recall / (avg_precision + avg_recall)

    def evaluate(self, annotations, predictions):
        precisions = self.precision.evaluate(annotations, predictions)
        recalls = self.recall.evaluate(annotations, predictions)

        precision_add = np.add(precisions[:-1], recalls[:-1], dtype=float)
        precision_multiply = np.multiply(precisions[:-1], recalls[:-1], dtype=float)

        per_class = 2 * np.divide(
            precision_multiply, precision_add, out=np.zeros_like(precision_multiply, dtype=float),
            where=precision_add != 0
        )
        if not self.calculate_average:
            return per_class

        average = 2 * (precisions[-1] * recalls[-1]) / (precisions[-1] + recalls[-1])

        return [*per_class, average]

    def reset(self):
        self.precision.reset()
        self.recall.reset()
        self._create_meta()

    def _create_meta(self):
        self.meta['names'] = list(self.labels.values())
        if self.calculate_average:
            self.meta['names'].append('average')

    @classmethod
    def get_common_meta(cls):
        meta = super().get_common_meta()
        meta['scale'] = 1
        meta['postfix'] = ''
        meta['calculate_mean'] = False
        return meta
