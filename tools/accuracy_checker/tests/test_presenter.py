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
import pytest
from unittest.mock import MagicMock, call # noqa: F401
from accuracy_checker.metrics import MetricsExecutor
from accuracy_checker.presenters import ScalarPrintPresenter, VectorPrintPresenter, EvaluationResult
from accuracy_checker.representation import ClassificationAnnotation, ClassificationPrediction


class TestPresenter:
    def test_config_default_presenter(self):
        annotations = [ClassificationAnnotation('identifier', 3)]
        predictions = [ClassificationPrediction('identifier', [1.0, 1.0, 1.0, 4.0])]
        config = [{'type': 'accuracy', 'top_k': 1}]
        dispatcher = MetricsExecutor(config, None)
        dispatcher.update_metrics_on_batch(range(len(annotations)), annotations, predictions)

        for presenter, _ in dispatcher.iterate_metrics(annotations, predictions):
            assert isinstance(presenter, ScalarPrintPresenter)

    def test_config_scalar_presenter(self):
        annotations = [ClassificationAnnotation('identifier', 3)]
        predictions = [ClassificationPrediction('identifier', [1.0, 1.0, 1.0, 4.0])]
        config = [{'type': 'accuracy', 'top_k': 1, 'presenter': 'print_scalar'}]
        dispatcher = MetricsExecutor(config, None)
        dispatcher.update_metrics_on_batch(range(len(annotations)), annotations, predictions)

        for presenter, _ in dispatcher.iterate_metrics(annotations, predictions):
            assert isinstance(presenter, ScalarPrintPresenter)

    def test_config_vector_presenter(self):
        annotations = [ClassificationAnnotation('identifier', 3)]
        predictions = [ClassificationPrediction('identifier', [1.0, 1.0, 1.0, 4.0])]
        config = [{'type': 'accuracy', 'top_k': 1, 'presenter': 'print_vector'}]
        dispatcher = MetricsExecutor(config, None)
        dispatcher.update_metrics_on_batch(range(len(annotations)), annotations, predictions)

        for presenter, _ in dispatcher.iterate_metrics(annotations, predictions):
            assert isinstance(presenter, VectorPrintPresenter)

    def test_config_unknown_presenter(self):
        config = [{'type': 'accuracy', 'top_k': 1, 'presenter': 'print_somehow'}]
        with pytest.raises(ValueError):
            MetricsExecutor(config, None)

    def test_scalar_presenter_with_scalar_data(self, mocker):
        mock_write_scalar_result = mocker.patch('accuracy_checker.presenters.write_scalar_result')  # type: MagicMock
        result = EvaluationResult(
            name='scalar_metric',
            metric_type='metric',
            evaluated_value=0.1,
            reference_value=None,
            abs_threshold=None,
            rel_threshold=None,
            meta={},
            profiling_file=None
        )
        presenter = ScalarPrintPresenter()
        presenter.write_result(result)
        mock_write_scalar_result.assert_called_once_with(
            result.evaluated_value,
            result.name,
            result.abs_threshold,
            result.rel_threshold,
            None,
            postfix='%',
            scale=100,
            result_format='{:.2f}'
        )

    def test_scalar_presenter_with_vector_data(self, mocker):
        mock_write_scalar_res = mocker.patch('accuracy_checker.presenters.write_scalar_result')  # type: MagicMock
        result = EvaluationResult(
            name='vector_metric',
            metric_type='metric',
            evaluated_value=[0.4, 0.6],
            reference_value=None,
            abs_threshold=None,
            rel_threshold=None,
            meta={},
            profiling_file=None
        )
        presenter = ScalarPrintPresenter()
        presenter.write_result(result)
        mock_write_scalar_res.assert_called_once_with(
            np.mean(result.evaluated_value),
            result.name,
            result.abs_threshold,
            result.rel_threshold,
            None,
            postfix='%',
            scale=100,
            result_format='{:.2f}'
        )

    def test_default_format_for_scalar_presenter_with_ignore_formatting(self, mocker):
        mock_write_scalar_res = mocker.patch('accuracy_checker.presenters.write_scalar_result')  # type: MagicMock
        result = EvaluationResult(
            name='vector_metric',
            metric_type='metric',
            evaluated_value=[0.456],
            reference_value=None,
            abs_threshold=None,
            rel_threshold=None,
            meta={},
            profiling_file=None
        )
        presenter = ScalarPrintPresenter()
        presenter.write_result(result, ignore_results_formatting=True)
        mock_write_scalar_res.assert_called_once_with(
            np.mean(result.evaluated_value),
            result.name,
            result.abs_threshold,
            result.rel_threshold,
            None,
            postfix=' ',
            scale=1,
            result_format='{}'
        )

    def test_reference_value_for_scalar_presenter(self, mocker):
        mock_write_scalar_res = mocker.patch('accuracy_checker.presenters.write_scalar_result')  # type: MagicMock
        result = EvaluationResult(
            name='vector_metric',
            metric_type='metric',
            evaluated_value=[0.456],
            reference_value=0.456,
            abs_threshold=None,
            rel_threshold=None,
            meta={},
            profiling_file=None
        )
        presenter = ScalarPrintPresenter()
        presenter.write_result(result)
        mock_write_scalar_res.assert_called_once_with(
            np.mean(result.evaluated_value),
            result.name,
            result.abs_threshold,
            result.rel_threshold,
            (0.0, 0.0),
            postfix='%',
            scale=100,
            result_format='{:.2f}'
        )

    def test_reference_value_for_scalar_presenter_with_ignore_results_formatting(self, mocker):
        mock_write_scalar_res = mocker.patch('accuracy_checker.presenters.write_scalar_result')  # type: MagicMock
        result = EvaluationResult(
            name='vector_metric',
            metric_type='metric',
            evaluated_value=[0.456],
            reference_value=0.456,
            abs_threshold=None,
            rel_threshold=None,
            meta={},
            profiling_file=None
        )
        presenter = ScalarPrintPresenter()
        presenter.write_result(result, ignore_results_formatting=True)
        mock_write_scalar_res.assert_called_once_with(
            np.mean(result.evaluated_value),
            result.name,
            result.abs_threshold,
            result.rel_threshold,
            (0.0, 0.0),
            postfix=' ',
            scale=1,
            result_format='{}'
        )

    def test_reference_value_for_scalar_presenter_with_ref_values_dict(self, mocker):
        mock_write_scalar_res = mocker.patch('accuracy_checker.presenters.write_scalar_result')  # type: MagicMock
        result = EvaluationResult(
            name='vector_metric',
            metric_type='metric',
            evaluated_value=[0.456],
            reference_value={'vector_metric': 0.456},
            abs_threshold=None,
            rel_threshold=None,
            meta={},
            profiling_file=None
        )
        presenter = ScalarPrintPresenter()
        presenter.write_result(result)
        mock_write_scalar_res.assert_called_once_with(
            np.mean(result.evaluated_value),
            result.name,
            result.abs_threshold,
            result.rel_threshold,
            (0.0, 0.0),
            postfix='%',
            scale=100,
            result_format='{:.2f}'
        )

    def test_reference_value_for_scalar_presenter_with_ref_values_dict_no_value(self, mocker):
        mock_write_scalar_res = mocker.patch('accuracy_checker.presenters.write_scalar_result')  # type: MagicMock
        result = EvaluationResult(
            name='vector_metric',
            metric_type='metric',
            evaluated_value=[0.456],
            reference_value={'other_metric': 0.456},
            abs_threshold=None,
            rel_threshold=None,
            meta={},
            profiling_file=None
        )
        presenter = ScalarPrintPresenter()
        presenter.write_result(result)
        mock_write_scalar_res.assert_called_once_with(
            np.mean(result.evaluated_value),
            result.name,
            result.abs_threshold,
            result.rel_threshold,
            None,
            postfix='%',
            scale=100,
            result_format='{:.2f}'
        )

    def test_specific_format_for_scalar_presenter(self, mocker):
        mock_write_scalar_res = mocker.patch('accuracy_checker.presenters.write_scalar_result')  # type: MagicMock
        result = EvaluationResult(
            name='vector_metric',
            metric_type='metric',
            evaluated_value=[0.456],
            reference_value=None,
            abs_threshold=None,
            rel_threshold=None,
            meta={'scale': 0.5, 'postfix': 'km/h', 'data_format': '{:.4f}'},
            profiling_file=None
        )
        presenter = ScalarPrintPresenter()
        presenter.write_result(result)
        mock_write_scalar_res.assert_called_once_with(
            np.mean(result.evaluated_value),
            result.name,
            result.reference_value,
            result.abs_threshold,
            result.rel_threshold,
            postfix='km/h',
            scale=0.5,
            result_format='{:.4f}'
        )

    def test_specific_format_for_scalar_presenter_with_ignore_formatting(self, mocker):
        mock_write_scalar_res = mocker.patch('accuracy_checker.presenters.write_scalar_result')  # type: MagicMock
        result = EvaluationResult(
            name='vector_metric',
            metric_type='metric',
            evaluated_value=[0.456],
            reference_value=None,
            abs_threshold=None,
            rel_threshold=None,
            meta={'scale': 0.5, 'postfix': 'km/h', 'data_format': '{:.4f}'},
            profiling_file=None
        )
        presenter = ScalarPrintPresenter()
        presenter.write_result(result, ignore_results_formatting=True)
        mock_write_scalar_res.assert_called_once_with(
            np.mean(result.evaluated_value),
            result.name,
            result.reference_value,
            result.abs_threshold,
            result.rel_threshold,
            postfix=' ',
            scale=1,
            result_format='{}'
        )

    def test_vector_presenter_with_scaler_data(self, mocker):
        mock_write_scalar_res = mocker.patch('accuracy_checker.presenters.write_scalar_result')  # type: MagicMock
        result = EvaluationResult(
            name='scalar_metric',
            metric_type='metric',
            evaluated_value=0.4,
            reference_value=None,
            abs_threshold=None,
            rel_threshold=None,
            meta={},
            profiling_file=None
        )
        presenter = VectorPrintPresenter()
        presenter.write_result(result)
        mock_write_scalar_res.assert_called_once_with(
            result.evaluated_value,
            result.name,
            None,
            result.abs_threshold,
            result.rel_threshold,
            postfix='%',
            scale=100,
            value_name=None,
            result_format='{:.2f}'
        )

    def test_vector_presenter_with_scaler_data_compare_with_reference(self, mocker):
        mock_write_scalar_res = mocker.patch('accuracy_checker.presenters.write_scalar_result')  # type: MagicMock
        result = EvaluationResult(
            name='scalar_metric',
            metric_type='metric',
            evaluated_value=40,
            reference_value=42,
            abs_threshold=None,
            rel_threshold=None,
            meta={},
            profiling_file=None
        )
        presenter = VectorPrintPresenter()
        presenter.write_result(result)
        mock_write_scalar_res.assert_called_once_with(
            result.evaluated_value,
            result.name,
            result.abs_threshold,
            result.rel_threshold,
            (2.0, 0.047619047619047616),
            postfix='%',
            scale=100,
            value_name=None,
            result_format='{:.2f}'
        )

    def test_vector_presenter_with_scaler_data_compare_with_reference_ignore_formatting(self, mocker):
        mock_write_scalar_res = mocker.patch('accuracy_checker.presenters.write_scalar_result')  # type: MagicMock
        result = EvaluationResult(
            name='scalar_metric',
            metric_type='metric',
            evaluated_value=40,
            reference_value=42,
            abs_threshold=None,
            rel_threshold=None,
            meta={},
            profiling_file=None
        )
        presenter = VectorPrintPresenter()
        presenter.write_result(result, ignore_results_formatting=True)
        mock_write_scalar_res.assert_called_once_with(
            result.evaluated_value,
            result.name,
            result.abs_threshold,
            result.rel_threshold,
            (2.0, 0.047619047619047616),
            postfix=' ',
            scale=1,
            value_name=None,
            result_format='{}'
        )

    def test_vector_presenter_with_vector_data_contain_one_element(self, mocker):
        mock_write_scalar_res = mocker.patch('accuracy_checker.presenters.write_scalar_result')  # type: MagicMock
        result = EvaluationResult(
            name='scalar_metric',
            metric_type='metric',
            evaluated_value=[0.4],
            reference_value=None,
            abs_threshold=None,
            rel_threshold=None,
            meta={'names': ['prediction']},
            profiling_file=None
        )
        presenter = VectorPrintPresenter()
        presenter.write_result(result)
        mock_write_scalar_res.assert_called_once_with(
            result.evaluated_value[0],
            result.name,
            None,
            result.abs_threshold,
            result.rel_threshold,
            postfix='%',
            scale=100,
            value_name=result.meta['names'][0],
            result_format='{:.2f}'
        )

    def test_vector_presenter_with_vector_data_contain_one_element_compare_with_reference(self, mocker):
        mock_write_scalar_res = mocker.patch('accuracy_checker.presenters.write_scalar_result')  # type: MagicMock
        result = EvaluationResult(
            name='scalar_metric',
            metric_type='metric',
            evaluated_value=[40],
            reference_value=42,
            abs_threshold=None,
            rel_threshold=None,
            meta={},
            profiling_file=None
        )
        presenter = VectorPrintPresenter()
        presenter.write_result(result)
        mock_write_scalar_res.assert_called_once_with(
            result.evaluated_value[0],
            result.name,
            result.abs_threshold,
            result.rel_threshold,
            (2.0, 0.047619047619047616),
            postfix='%',
            scale=100,
            value_name=None,
            result_format='{:.2f}'
        )

    def test_vector_presenter_with_vector_data_contain_one_element_compare_with_reference_ignore_formatting(self, mocker):
        mock_write_scalar_res = mocker.patch('accuracy_checker.presenters.write_scalar_result')  # type: MagicMock
        result = EvaluationResult(
            name='vector_metric',
            metric_type='metric',
            evaluated_value=[40],
            reference_value=42,
            abs_threshold=None,
            rel_threshold=None,
            meta={},
            profiling_file=None
        )
        presenter = VectorPrintPresenter()
        presenter.write_result(result, ignore_results_formatting=True)
        mock_write_scalar_res.assert_called_once_with(
            result.evaluated_value[0],
            result.name,
            result.abs_threshold,
            result.rel_threshold,
            (2.0, 0.047619047619047616),
            postfix=' ',
            scale=1,
            value_name=None,
            result_format='{}'
        )

    def test_vector_presenter_with_vector_data_with_default_postfix_and_scale(self, mocker):
        mock_write_scalar_res = mocker.patch('accuracy_checker.presenters.write_scalar_result')  # type: MagicMock
        result = EvaluationResult(
            name='vector_metric',
            metric_type='metric',
            evaluated_value=[0.4, 0.6],
            reference_value=None,
            abs_threshold=None,
            rel_threshold=None,
            meta={'names': ['class1', 'class2']},
            profiling_file=None
        )
        presenter = VectorPrintPresenter()
        presenter.write_result(result)
        calls = [
            call(
                result.evaluated_value[0], result.name, None, None, None,
                postfix='%', scale=100, value_name=result.meta['names'][0], result_format='{:.2f}'
            ),
            call(
                result.evaluated_value[1], result.name, None, None, None,
                postfix='%', scale=100, value_name=result.meta['names'][1], result_format='{:.2f}'
            ),
            call(
                np.mean(result.evaluated_value), result.name, result.abs_threshold, result.rel_threshold,
                None, value_name='mean', postfix='%', scale=100, result_format='{:.2f}'
            )
        ]
        mock_write_scalar_res.assert_has_calls(calls)

    def test_vector_presenter_with_vector_data_has_default_format_with_ignore_formatting(self, mocker):
        mock_write_scalar_res = mocker.patch('accuracy_checker.presenters.write_scalar_result')  # type: MagicMock
        result = EvaluationResult(
            name='vector_metric',
            metric_type='metric',
            evaluated_value=[0.4, 0.6],
            reference_value=None,
            abs_threshold=None,
            rel_threshold=None,
            meta={'names': ['class1', 'class2']},
            profiling_file=None
        )
        presenter = VectorPrintPresenter()
        presenter.write_result(result, ignore_results_formatting=True)
        calls = [
            call(
                result.evaluated_value[0], result.name, None, None, None,
                postfix=' ', scale=1, value_name=result.meta['names'][0], result_format='{}'
            ),
            call(
                result.evaluated_value[1], result.name, None, None, None,
                postfix=' ', scale=1, value_name=result.meta['names'][1], result_format='{}'
            ),
            call(
                np.mean(result.evaluated_value), result.name, result.abs_threshold, result.rel_threshold, None,
                value_name='mean', postfix=' ', scale=1, result_format='{}'
            )
        ]
        mock_write_scalar_res.assert_has_calls(calls)

    def test_vector_presenter_with_vector_data_with_default_formatting_compare_with_ref(self, mocker):
        mock_write_scalar_res = mocker.patch('accuracy_checker.presenters.write_scalar_result')  # type: MagicMock
        result = EvaluationResult(
            name='vector_metric',
            metric_type='metric',
            evaluated_value=[40, 60],
            reference_value=49,
            abs_threshold=None,
            rel_threshold=None,
            meta={'names': ['class1', 'class2']},
            profiling_file=None
        )
        presenter = VectorPrintPresenter()
        presenter.write_result(result)
        calls = [
            call(
                result.evaluated_value[0], result.name, None, None, None,
                postfix='%', scale=100, value_name=result.meta['names'][0], result_format='{:.2f}'
            ),
            call(
                result.evaluated_value[1], result.name, None, None, None,
                postfix='%', scale=100, value_name=result.meta['names'][1], result_format='{:.2f}'
            ),
            call(
                np.mean(result.evaluated_value), result.name, result.abs_threshold, result.rel_threshold,
                (1.0, 0.02040816326530612), value_name='mean', postfix='%', scale=100, result_format='{:.2f}'
            )
        ]
        mock_write_scalar_res.assert_has_calls(calls)

    def test_vector_presenter_with_vector_data_has_default_format_with_ignore_formatting_compare_with_ref(self, mocker):
        mock_write_scalar_res = mocker.patch('accuracy_checker.presenters.write_scalar_result')  # type: MagicMock
        result = EvaluationResult(
            name='vector_metric',
            metric_type='metric',
            evaluated_value=[40, 60],
            reference_value=49,
            abs_threshold=None,
            rel_threshold=None,
            meta={'names': ['class1', 'class2']},
            profiling_file=None
        )
        presenter = VectorPrintPresenter()
        presenter.write_result(result, ignore_results_formatting=True)
        calls = [
            call(
                result.evaluated_value[0], result.name, None, None, None,
                postfix=' ', scale=1, value_name=result.meta['names'][0], result_format='{}'
            ),
            call(
                result.evaluated_value[1], result.name, None, None, None,
                postfix=' ', scale=1, value_name=result.meta['names'][1], result_format='{}'
            ),
            call(
                np.mean(result.evaluated_value), result.name, result.abs_threshold, result.rel_threshold,
                (1.0, 0.02040816326530612),
                value_name='mean', postfix=' ', scale=1, result_format='{}'
            )
        ]
        mock_write_scalar_res.assert_has_calls(calls)

    def test_vector_presenter_with_vector_data_has_specific_format_with_ignore_formatting(self, mocker):
        mock_write_scalar_res = mocker.patch('accuracy_checker.presenters.write_scalar_result')  # type: MagicMock
        result = EvaluationResult(
            name='scalar_metric',
            metric_type='metric',
            evaluated_value=[0.4, 0.6],
            reference_value=None,
            abs_threshold=None,
            rel_threshold=None,
            meta={'names': ['class1', 'class2'], 'scale': 0.5, 'postfix': 'km/h', 'data_format': '{:.4f}'},
            profiling_file = None
        )
        presenter = VectorPrintPresenter()
        presenter.write_result(result, ignore_results_formatting=True)
        calls = [
            call(
                result.evaluated_value[0], result.name, None, None, None,
                postfix=' ', scale=1, value_name=result.meta['names'][0], result_format='{}'
            ),
            call(
                result.evaluated_value[1], result.name, None, None, None,
                postfix=' ', scale=1, value_name=result.meta['names'][1], result_format='{}'
            ),
            call(
                np.mean(result.evaluated_value), result.name, result.reference_value,
                result.abs_threshold, result.rel_threshold,
                value_name='mean', postfix=' ', scale=1, result_format='{}'
            )
        ]
        mock_write_scalar_res.assert_has_calls(calls)

    def test_vector_presenter_with_vector_data_with_scalar_postfix(self, mocker):
        mock_write_scalar_res = mocker.patch('accuracy_checker.presenters.write_scalar_result')  # type: MagicMock
        result = EvaluationResult(
            name='scalar_metric',
            metric_type='metric',
            evaluated_value=[0.4, 0.6],
            reference_value=None,
            abs_threshold=None,
            rel_threshold=None,
            meta={'names': ['class1', 'class2'], 'postfix': '_'},
            profiling_file=None
        )
        presenter = VectorPrintPresenter()
        presenter.write_result(result)
        calls = [
            call(result.evaluated_value[0], result.name, None, None, None,
                 postfix=result.meta['postfix'], scale=100, value_name=result.meta['names'][0], result_format='{:.2f}'
                 ),
            call(
                result.evaluated_value[1], result.name, None, None, None,
                postfix=result.meta['postfix'], scale=100, value_name=result.meta['names'][1], result_format='{:.2f}'
            ),
            call(
                np.mean(result.evaluated_value), result.name,
                result.abs_threshold, result.rel_threshold, None, value_name='mean', postfix=result.meta['postfix'], scale=100, result_format='{:.2f}'
            )
        ]
        mock_write_scalar_res.assert_has_calls(calls)

    def test_vector_presenter_with_vector_data_with_scalar_scale(self, mocker):
        mock_write_scalar_res = mocker.patch('accuracy_checker.presenters.write_scalar_result')  # type: MagicMock
        result = EvaluationResult(
            name='scalar_metric',
            metric_type='metric',
            evaluated_value=[0.4, 0.6],
            reference_value=None,
            abs_threshold=None,
            rel_threshold=None,
            meta={'names': ['class1', 'class2'], 'scale': 10},
            profiling_file=None
        )
        presenter = VectorPrintPresenter()
        presenter.write_result(result)
        calls = [
            call(
                result.evaluated_value[0], result.name, None, None, None,
                postfix='%', scale=result.meta['scale'], value_name=result.meta['names'][0], result_format='{:.2f}'
            ),
            call(
                result.evaluated_value[1], result.name, None, None, None,
                postfix='%', scale=result.meta['scale'], value_name=result.meta['names'][1], result_format='{:.2f}'
            ),
            call(
                np.mean(result.evaluated_value), result.name, None,
                result.abs_threshold, result.rel_threshold,
                value_name='mean', postfix='%', scale=10, result_format='{:.2f}'
            )
        ]
        mock_write_scalar_res.assert_has_calls(calls)

    def test_vector_presenter_with_vector_data_with_vector_scale(self, mocker):
        mock_write_scalar_res = mocker.patch('accuracy_checker.presenters.write_scalar_result')  # type: MagicMock
        result = EvaluationResult(
            name='scalar_metric',
            metric_type='metric',
            evaluated_value=[0.4, 0.6],
            reference_value=None,
            abs_threshold=None,
            rel_threshold=None,
            meta={'names': ['class1', 'class2'], 'scale': [1, 1]},
            profiling_file=None
        )
        presenter = VectorPrintPresenter()
        presenter.write_result(result)
        calls = [
            call(
                result.evaluated_value[0], result.name, None, None, None,
                postfix='%', scale=result.meta['scale'][0], result_format='{:.2f}', value_name=result.meta['names'][0]
            ),
            call(
                result.evaluated_value[1], result.name, None, None, None, postfix='%',
                scale=result.meta['scale'][1], result_format='{:.2f}', value_name=result.meta['names'][1]
            ),
            call(
                np.mean(result.evaluated_value), result.name,
                result.abs_threshold, result.rel_threshold,
                None, result_format='{:.2f}', value_name='mean', postfix='%', scale=1
            )
        ]
        mock_write_scalar_res.assert_has_calls(calls)

    def test_vector_presenter_with_vector_data_with_dict_ref(self, mocker):
        mock_write_scalar_res = mocker.patch('accuracy_checker.presenters.write_scalar_result')  # type: MagicMock
        result = EvaluationResult(
            name='scalar_metric',
            metric_type='metric',
            evaluated_value=[4, 6],
            reference_value={'class1': 4, 'class2': 5},
            abs_threshold=None,
            rel_threshold=None,
            meta={'names': ['class1', 'class2'], 'scale': [1, 2]},
            profiling_file=None
        )
        presenter = VectorPrintPresenter()
        presenter.write_result(result)
        calls = [
            call(
                result.evaluated_value[0], result.name, None, None, (0, 0),
                postfix='%', scale=result.meta['scale'][0], result_format='{:.2f}', value_name=result.meta['names'][0]
            ),
            call(
                result.evaluated_value[1], result.name, None, None, (1, 0.2), postfix='%',
                scale=result.meta['scale'][1], result_format='{:.2f}', value_name=result.meta['names'][1]
            )
        ]
        mock_write_scalar_res.assert_has_calls(calls)

    def test_vector_presenter_with_vector_data_with_dict_ref_without_represented_classes(self, mocker):
        mock_write_scalar_res = mocker.patch('accuracy_checker.presenters.write_scalar_result')  # type: MagicMock
        result = EvaluationResult(
            name='scalar_metric',
            metric_type='metric',
            evaluated_value=[0.4, 0.6],
            reference_value={'class3': 0.4, 'class4': 0.5},
            abs_threshold=None,
            rel_threshold=None,
            meta={'names': ['class1', 'class2'], 'scale': [1, 1]},
            profiling_file=None
        )
        presenter = VectorPrintPresenter()
        presenter.write_result(result)
        calls = [
            call(
                result.evaluated_value[0], result.name, None, None, None,
                postfix='%', scale=result.meta['scale'][0], result_format='{:.2f}', value_name=result.meta['names'][0]
            ),
            call(
                result.evaluated_value[1], result.name, None, None, None, postfix='%',
                scale=result.meta['scale'][1], result_format='{:.2f}', value_name=result.meta['names'][1]
            ),
            call(
                np.mean(result.evaluated_value), result.name,
                result.abs_threshold, result.rel_threshold,
                None, result_format='{:.2f}', value_name='mean', postfix='%', scale=1
            )
        ]
        mock_write_scalar_res.assert_has_calls(calls)
