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

import pytest
import numpy as np
from accuracy_checker.metrics import MetricsExecutor
from accuracy_checker.representation import (
    RegressionPrediction, RegressionAnnotation, FacialLandmarksAnnotation, FacialLandmarksPrediction
)
from accuracy_checker.presenters import EvaluationResult


class TestRegressionMetric:
    def setup_method(self):
        self.module = 'accuracy_checker.metrics.metric_evaluator'

    def test_mae_with_zero_diff_between_annotation_and_prediction(self):
        annotations = [RegressionAnnotation('identifier', 3)]
        predictions = [RegressionPrediction('identifier', 3)]
        config = [{'type': 'mae'}]
        expected = EvaluationResult(
            pytest.approx([0.0, 0.0]),
            None,
            'mae',
            'mae',
            None,
            None,
            {'postfix': ' ', 'scale': 1, 'names': ['mean', 'std'], 'calculate_mean': False, 'target': 'higher-worse'}
        )
        dispatcher = MetricsExecutor(config, None)

        dispatcher.update_metrics_on_batch(range(len(annotations)), annotations, predictions)

        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result == expected

    def test_mae_with_negative_diff_between_annotation_and_prediction(self):
        annotations = [RegressionAnnotation('identifier', 3), RegressionAnnotation('identifier2', 1)]
        predictions = [RegressionPrediction('identifier', 5), RegressionPrediction('identifier2', 5)]
        config = [{'type': 'mae'}]
        expected = EvaluationResult(
            pytest.approx([3.0, 1.0]),
            None,
            'mae',
            'mae',
            None,
            None,
            {'postfix': ' ', 'scale': 1, 'names': ['mean', 'std'], 'calculate_mean': False, 'target': 'higher-worse'}
        )
        dispatcher = MetricsExecutor(config, None)

        dispatcher.update_metrics_on_batch(range(len(annotations)), annotations, predictions)

        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result == expected

    def test_mae_with_positive_diff_between_annotation_and_prediction(self):
        annotations = [RegressionAnnotation('identifier', 3), RegressionAnnotation('identifier2', 1)]
        predictions = [RegressionPrediction('identifier', 1), RegressionPrediction('identifier2', -3)]
        config = [{'type': 'mae'}]
        expected = EvaluationResult(
            pytest.approx([3.0, 1.0]),
            None,
            'mae',
            'mae',
            None,
            None,
            {'postfix': ' ', 'scale': 1, 'names': ['mean', 'std'], 'calculate_mean': False, 'target': 'higher-worse'}
        )
        dispatcher = MetricsExecutor(config, None)

        dispatcher.update_metrics_on_batch(range(len(annotations)), annotations, predictions)

        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result == expected

    def test_mse_with_zero_diff_between_annotation_and_prediction(self):
        annotations = [RegressionAnnotation('identifier', 3)]
        predictions = [RegressionPrediction('identifier', 3)]
        config = [{'type': 'mse'}]
        expected = EvaluationResult(
            pytest.approx([0.0, 0.0]),
            None,
            'mse',
            'mse',
            None,
            None,
            {'postfix': ' ', 'scale': 1, 'names': ['mean', 'std'], 'calculate_mean': False, 'target': 'higher-worse'}
        )
        dispatcher = MetricsExecutor(config, None)

        dispatcher.update_metrics_on_batch(range(len(annotations)), annotations, predictions)

        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result == expected

    def test_mse_with_negative_diff_between_annotation_and_prediction(self):
        annotations = [RegressionAnnotation('identifier', 3), RegressionAnnotation('identifier2', 1)]
        predictions = [RegressionPrediction('identifier', 5), RegressionPrediction('identifier2', 5)]
        config = [{'type': 'mse'}]
        expected = EvaluationResult(
            pytest.approx([10.0, 6.0]),
            None,
            'mse',
            'mse',
            None,
            None,
            {'postfix': ' ', 'scale': 1, 'names': ['mean', 'std'], 'calculate_mean': False, 'target': 'higher-worse'}
        )
        dispatcher = MetricsExecutor(config, None)

        dispatcher.update_metrics_on_batch(range(len(annotations)), annotations, predictions)

        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result == expected

    def test_mse_with_positive_diff_between_annotation_and_prediction(self):
        annotations = [RegressionAnnotation('identifier', 3), RegressionAnnotation('identifier2', 1)]
        predictions = [RegressionPrediction('identifier', 1), RegressionPrediction('identifier2', -3)]
        config = [{'type': 'mse'}]
        expected = EvaluationResult(
            pytest.approx([10.0, 6.0]),
            None,
            'mse',
            'mse',
            None,
            None,
            {'postfix': ' ', 'scale': 1, 'names': ['mean', 'std'], 'calculate_mean': False, 'target': 'higher-worse'}
        )
        dispatcher = MetricsExecutor(config, None)

        dispatcher.update_metrics_on_batch(range(len(annotations)), annotations, predictions)

        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result == expected

    def test_missed_interval(self):
        config = [{'type': 'mae_on_interval'}]
        with pytest.raises(ValueError):
            MetricsExecutor(config, None)

    def test_mae_on_interval_default_all_missed(self):
        annotations = [RegressionAnnotation('identifier', -2)]
        predictions = [RegressionPrediction('identifier', 1)]
        config = [{'type': 'mae_on_interval', 'end': 1}]
        expected = EvaluationResult(
            pytest.approx([0.0]),
            None,
            'mae_on_interval',
            'mae_on_interval',
            None,
            None,
            {'postfix': ' ', 'scale': 1, 'names': [], 'calculate_mean': False, 'target': 'higher-worse'}
        )
        dispatcher = MetricsExecutor(config, None)

        dispatcher.update_metrics_on_batch(range(len(annotations)), annotations, predictions)

        with pytest.warns(UserWarning) as warnings:
            for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
                assert len(warnings) == 1
                assert evaluation_result == expected

    def test_mae_on_interval_default_all_not_in_range_not_ignore_out_of_range(self):
        annotations = [RegressionAnnotation('identifier', -1), RegressionAnnotation('identifier', 2)]
        predictions = [RegressionPrediction('identifier', 1), RegressionPrediction('identifier', 2)]
        expected = EvaluationResult(
            pytest.approx([2.0, 0.0, 0.0, 0.0]),
            None,
            'mae_on_interval',
            'mae_on_interval',
            None,
            None,
            {
                'postfix': ' ',
                'scale': 1,
                'names': ['mean: < 0.0', 'std: < 0.0', 'mean: > 1.0', 'std: > 1.0'],
                'calculate_mean': False,
                'target': 'higher-worse'
            }
        )
        config = [{'type': 'mae_on_interval', 'end': 1, 'ignore_values_not_in_interval': False}]
        dispatcher = MetricsExecutor(config, None)

        dispatcher.update_metrics_on_batch(range(len(annotations)), annotations, predictions)

        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result == expected

    def test_mae_on_interval_values_in_range(self):
        annotations = [RegressionAnnotation('identifier', 0.5), RegressionAnnotation('identifier', 0.5)]
        predictions = [RegressionPrediction('identifier', 1), RegressionPrediction('identifier', 0.25)]
        config = [{'type': 'mae_on_interval', 'end': 1}]
        expected = EvaluationResult(
            pytest.approx([0.375, 0.125]),
            None,
            'mae_on_interval',
            'mae_on_interval',
            None,
            None,
            {'postfix': ' ', 'scale': 1, 'names': ['mean: <= 0.0 < 1.0', 'std: <= 0.0 < 1.0'], 'calculate_mean': False, 'target': 'higher-worse'}
        )
        dispatcher = MetricsExecutor(config, None)

        dispatcher.update_metrics_on_batch(range(len(annotations)), annotations, predictions)

        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result == expected

    def test_mae_on_interval_default_not_ignore_out_of_range(self):
        annotations = [
            RegressionAnnotation('identifier', -1),
            RegressionAnnotation('identifier',  2),
            RegressionAnnotation('identifier', 0.5)
        ]
        predictions = [
            RegressionPrediction('identifier', 1),
            RegressionPrediction('identifier', 2),
            RegressionPrediction('identifier', 1)
        ]
        config = [{'type': 'mae_on_interval', 'end': 1, 'ignore_values_not_in_interval': False}]
        expected = EvaluationResult(
            pytest.approx([2.0, 0.0, 0.5, 0.0, 0.0, 0.0]),
            None,
            'mae_on_interval',
            'mae_on_interval',
            None,
            None,
            {
                'postfix': ' ',
                'scale': 1,
                'names': [
                    'mean: < 0.0',
                    'std: < 0.0',
                    'mean: <= 0.0 < 1.0',
                    'std: <= 0.0 < 1.0',
                    'mean: > 1.0',
                    'std: > 1.0'
                ],
                'calculate_mean': False,
                'target': 'higher-worse'
            }
        )
        dispatcher = MetricsExecutor(config, None)

        dispatcher.update_metrics_on_batch(range(len(annotations)), annotations, predictions)

        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result == expected

    def test_mae_on_interval_with_given_interval(self):
        annotations = [
            RegressionAnnotation('identifier', -1),
            RegressionAnnotation('identifier',  2),
            RegressionAnnotation('identifier',  1)
        ]
        predictions = [
            RegressionPrediction('identifier', 1),
            RegressionPrediction('identifier', 3),
            RegressionPrediction('identifier', 1)
        ]
        config = [{'type': 'mae_on_interval', 'intervals': [0.0, 2.0, 4.0]}]
        expected = EvaluationResult(
            pytest.approx([0.0, 0.0, 1.0, 0.0]),
            None,
            'mae_on_interval',
            'mae_on_interval',
            None,
            None,
            {
                'postfix': ' ',
                'scale': 1,
                'names': ['mean: <= 0.0 < 2.0', 'std: <= 0.0 < 2.0', 'mean: <= 2.0 < 4.0', 'std: <= 2.0 < 4.0'],
                'calculate_mean': False,
                'target': 'higher-worse'
            }
        )
        dispatcher = MetricsExecutor(config, None)

        dispatcher.update_metrics_on_batch(range(len(annotations)), annotations, predictions)

        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result == expected

    def test_mae_on_interval_with_repeated_values(self):
        annotations = [
            RegressionAnnotation('identifier', -1),
            RegressionAnnotation('identifier',  2),
            RegressionAnnotation('identifier', 1)
        ]
        predictions = [
            RegressionPrediction('identifier', 1),
            RegressionPrediction('identifier', 3),
            RegressionPrediction('identifier', 1)
        ]
        config = [{'type': 'mae_on_interval', 'intervals': [0.0, 2.0, 2.0, 4.0]}]
        expected = EvaluationResult(
            pytest.approx([0.0, 0.0, 1.0, 0.0]),
            None,
            'mae_on_interval',
            'mae_on_interval',
            None,
            None,
            {
                'postfix': ' ',
                'scale': 1,
                'names': ['mean: <= 0.0 < 2.0', 'std: <= 0.0 < 2.0', 'mean: <= 2.0 < 4.0', 'std: <= 2.0 < 4.0'],
                'calculate_mean': False,
                'target': 'higher-worse'
            }
        )
        dispatcher = MetricsExecutor(config, None)

        dispatcher.update_metrics_on_batch(range(len(annotations)), annotations, predictions)

        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result == expected

    def test_mae_on_interval_with_unsorted_values(self):
        annotations = [
            RegressionAnnotation('identifier', -1),
            RegressionAnnotation('identifier',  2),
            RegressionAnnotation('identifier',  1)
        ]
        predictions = [
            RegressionPrediction('identifier', 1),
            RegressionPrediction('identifier', 3),
            RegressionPrediction('identifier', 1)
        ]
        config = [{'type': 'mae_on_interval', 'intervals': [2.0, 0.0, 4.0]}]
        expected = EvaluationResult(
            pytest.approx([0.0, 0.0, 1.0, 0.0]),
            None,
            'mae_on_interval',
            'mae_on_interval',
            None,
            None,
            {
                'postfix': ' ', 'scale': 1,
                'names': ['mean: <= 0.0 < 2.0', 'std: <= 0.0 < 2.0', 'mean: <= 2.0 < 4.0', 'std: <= 2.0 < 4.0'],
                'calculate_mean': False,
                'target': 'higher-worse'
            }
        )
        dispatcher = MetricsExecutor(config, None)

        dispatcher.update_metrics_on_batch(range(len(annotations)), annotations, predictions)

        for _, evaluation_result in dispatcher.iterate_metrics(annotations, predictions):
            assert evaluation_result == expected


class TestUpdateRegressionMetrics:
    def test_update_mae_metric_result(self):
        annotations = [RegressionAnnotation('identifier', 3), RegressionAnnotation('identifier2', 1)]
        predictions = [RegressionPrediction('identifier', 5), RegressionPrediction('identifier2', 5)]
        config = [{'type': 'mae'}]
        dispatcher = MetricsExecutor(config, None)

        metric_result, _ = dispatcher.update_metrics_on_batch(range(len(annotations)), annotations, predictions)
        assert metric_result[0][0].result == 2
        assert metric_result[1][0].result == 4

    def test_update_mse_metric_result(self):
        annotations = [RegressionAnnotation('identifier', 3), RegressionAnnotation('identifier2', 1)]
        predictions = [RegressionPrediction('identifier', 5), RegressionPrediction('identifier2', 5)]
        config = [{'type': 'mse'}]
        dispatcher = MetricsExecutor(config, None)

        metric_result, _ = dispatcher.update_metrics_on_batch(range(len(annotations)), annotations, predictions)
        assert metric_result[0][0].result == 4
        assert metric_result[1][0].result == 16

    def test_update_rmse_metric_result(self):
        annotations = [RegressionAnnotation('identifier', 3), RegressionAnnotation('identifier2', 1)]
        predictions = [RegressionPrediction('identifier', 5), RegressionPrediction('identifier2', 5)]
        config = [{'type': 'rmse'}]
        dispatcher = MetricsExecutor(config, None)

        metric_result, _ = dispatcher.update_metrics_on_batch(range(len(annotations)), annotations, predictions)
        assert metric_result[0][0].result == 2
        assert metric_result[1][0].result == 4

    def test_update_mae_on_interval_metric(self):
        config = [{'type': 'mae_on_interval', 'intervals': [0.0, 2.0, 4.0]}]
        annotations = [RegressionAnnotation('identifier', 3), RegressionAnnotation('identifier2', 1)]
        predictions = [RegressionPrediction('identifier', 5), RegressionPrediction('identifier2', 5)]
        dispatcher = MetricsExecutor(config, None)

        metric_result, _ = dispatcher.update_metrics_on_batch(range(len(annotations)), annotations, predictions)
        assert metric_result[0][0].result == 2
        assert metric_result[1][0].result == 4

    def test_update_mse_on_interval_metric(self):
        config = [{'type': 'mse_on_interval', 'intervals': [0.0, 2.0, 4.0]}]
        annotations = [RegressionAnnotation('identifier', 3), RegressionAnnotation('identifier2', 1)]
        predictions = [RegressionPrediction('identifier', 5), RegressionPrediction('identifier2', 5)]
        dispatcher = MetricsExecutor(config, None)

        metric_result, _ = dispatcher.update_metrics_on_batch(range(len(annotations)), annotations, predictions)
        assert metric_result[0][0].result == 4
        assert metric_result[1][0].result == 16

    def test_update_rmse_on_interval_metric(self):
        config = [{'type': 'rmse_on_interval', 'intervals': [0.0, 2.0, 4.0]}]
        annotations = [RegressionAnnotation('identifier', 3), RegressionAnnotation('identifier2', 1)]
        predictions = [RegressionPrediction('identifier', 5), RegressionPrediction('identifier2', 5)]
        dispatcher = MetricsExecutor(config, None)

        metric_result, _ = dispatcher.update_metrics_on_batch(range(len(annotations)), annotations, predictions)
        assert metric_result[0][0].result == 2
        assert metric_result[1][0].result == 4

    def test_update_per_point_normed_error(self):
        config = [{'type': 'per_point_normed_error'}]
        annotations = [FacialLandmarksAnnotation('identifier', np.array([1, 1, 1, 1, 1]), np.array([1, 1, 1, 1, 1]))]
        annotations[0].metadata.update({'left_eye': 0, 'right_eye': 1})
        predictions = [FacialLandmarksPrediction('identifier', np.array([1, 1, 1, 1, 1]), np.array([1, 1, 1, 1, 1]))]
        dispatcher = MetricsExecutor(config, None)

        metric_result, _ = dispatcher.update_metrics_on_batch(range(len(annotations)), annotations, predictions)
        assert np.equal(metric_result[0][0].result.all(), np.zeros(5).all())

    def test_update_normed_error(self):
        config = [{'type': 'normed_error'}]
        annotations = [FacialLandmarksAnnotation('identifier', np.array([1, 1, 1, 1, 1]), np.array([1, 1, 1, 1, 1]))]
        annotations[0].metadata.update({'left_eye': 0, 'right_eye': 1})
        predictions = [FacialLandmarksPrediction('identifier', np.array([1, 1, 1, 1, 1]), np.array([1, 1, 1, 1, 1]))]
        dispatcher = MetricsExecutor(config, None)

        metric_result, _ = dispatcher.update_metrics_on_batch(range(len(annotations)), annotations, predictions)
        assert metric_result[0][0].result == 0
