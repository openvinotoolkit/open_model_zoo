"""
Copyright (c) 2026 Intel Corporation

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


import pytest
from accuracy_checker.metrics.multilabel_recognition import F1Score, MultiLabelAccuracy
from accuracy_checker.representation import MultiLabelRecognitionAnnotation, MultiLabelRecognitionPrediction
from .common import DummyDataset


LABEL_MAP = {0: 'a', 1: 'b', 2: 'c'}


def evaluate(metric_cls, config):
    metric = metric_cls(config, DummyDataset(label_map=LABEL_MAP))
    annotations = [
        MultiLabelRecognitionAnnotation('img1', [1, 0, 1]),
        MultiLabelRecognitionAnnotation('img2', [0, 1, 1]),
    ]
    predictions = [
        MultiLabelRecognitionPrediction('img1', [1, 1, 1]),
        MultiLabelRecognitionPrediction('img2', [0, 1, 0]),
    ]
    for annotation, prediction in zip(annotations, predictions):
        metric.update(annotation, prediction)
    return list(metric.evaluate(annotations, predictions)), metric.meta['names']


class TestMultiLabelMetrics:
    def test_f1_score_with_average(self):
        result, names = evaluate(F1Score, {'type': 'f1-score'})
        assert names == ['a', 'b', 'c', 'average']
        assert result == pytest.approx([1.0, 2 / 3, 2 / 3, 0.75])

    def test_f1_score_without_average_keeps_all_classes(self):
        result, names = evaluate(F1Score, {'type': 'f1-score', 'calculate_average': False})
        assert names == ['a', 'b', 'c']
        assert result == pytest.approx([1.0, 2 / 3, 2 / 3])

    def test_multi_accuracy_with_average(self):
        result, names = evaluate(MultiLabelAccuracy, {'type': 'multi_accuracy'})
        assert names == ['a', 'b', 'c', 'average']
        assert result == pytest.approx([1.0, 0.5, 0.5, 2 / 3])

    def test_multi_accuracy_without_average(self):
        result, names = evaluate(MultiLabelAccuracy, {'type': 'multi_accuracy', 'calculate_average': False})
        assert names == ['a', 'b', 'c']
        assert result == pytest.approx([1.0, 0.5, 0.5])
