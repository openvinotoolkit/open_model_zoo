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
import pytest
from unittest.mock import MagicMock, call
from accuracy_checker.metrics import MetricsExecutor
from accuracy_checker.presenters import ScalarPrintPresenter, VectorPrintPresenter, EvaluationResult
from accuracy_checker.representation import ClassificationAnnotation, ClassificationPrediction



class TestPresenter:
    def test_config_default_presenter(self):
        annotations = [ClassificationAnnotation('identifier', 3)]
        predictions = [ClassificationPrediction('identifier', [1.0, 1.0, 1.0, 4.0])]
        config = {'annotation': 'mocked', 'metrics': [{'type': 'accuracy', 'top_k': 1}]}
        dispatcher = MetricsExecutor(config, None)
        dispatcher.update_metrics_on_batch(annotations, predictions)

        for presenter, _ in dispatcher.iterate_metrics(annotations, predictions):
            assert isinstance(presenter, ScalarPrintPresenter)

    def test_config_scalar_presenter(self):
        annotations = [ClassificationAnnotation('identifier', 3)]
        predictions = [ClassificationPrediction('identifier', [1.0, 1.0, 1.0, 4.0])]
        config = {'annotation': 'mocked', 'metrics': [{'type': 'accuracy', 'top_k': 1, 'presenter': 'print_scalar'}]}
        dispatcher = MetricsExecutor(config, None)
        dispatcher.update_metrics_on_batch(annotations, predictions)

        for presenter,_ in dispatcher.iterate_metrics(annotations, predictions):
            assert isinstance(presenter, ScalarPrintPresenter)

    def test_config_vactor_presenter(self):
        annotations = [ClassificationAnnotation('identifier', 3)]
        predictions = [ClassificationPrediction('identifier', [1.0, 1.0, 1.0, 4.0])]
        config = {'annotation': 'mocked', 'metrics': [{'type': 'accuracy', 'top_k': 1, 'presenter': 'print_vector'}]}
        dispatcher = MetricsExecutor(config, None)
        dispatcher.update_metrics_on_batch(annotations, predictions)

        for presenter,_ in dispatcher.iterate_metrics(annotations, predictions):
            assert isinstance(presenter, VectorPrintPresenter)

    def test_config_unknown_presenter(self):
        config = {'annotation': 'mocked', 'metrics': [{'type': 'accuracy', 'top_k': 1, 'presenter': 'print_somehow'}]}
        with pytest.raises(ValueError):
            MetricsExecutor(config, None)

    def test_scalar_presenter_with_scalar_data(self, mocker):
        mock_write_scalar_res = mocker.patch('accuracy_checker.presenters.write_scalar_result')  # type: MagicMock
        res = EvaluationResult(
            name='scalar_metric',
            evaluated_value=0.1,
            reference_value=None,
            threshold=None,
            meta={},
        )
        presenter = ScalarPrintPresenter()
        presenter.write_result(res)
        mock_write_scalar_res.assert_called_once_with(res.evaluated_value, res.name, res.reference_value, res.threshold, postfix='%', scale=100)


    def test_scalar_presenter_with_vector_data(self, mocker):
        mock_write_scalar_res = mocker.patch('accuracy_checker.presenters.write_scalar_result')  # type: MagicMock
        res = EvaluationResult(
            name='vector_metric',
            evaluated_value=[0.4, 0.6],
            reference_value=None,
            threshold=None,
            meta={},
        )
        presenter = ScalarPrintPresenter()
        presenter.write_result(res)
        mock_write_scalar_res.assert_called_once_with(np.mean(res.evaluated_value), res.name, res.reference_value, res.threshold, postfix='%', scale=100)

    def test_vector_presenter_with_scaler_data(self, mocker):
        mock_write_scalar_res = mocker.patch('accuracy_checker.presenters.write_scalar_result')  # type: MagicMock
        res = EvaluationResult(
            name='scalar_metric',
            evaluated_value=0.4,
            reference_value=None,
            threshold=None,
            meta={},
        )
        presenter = VectorPrintPresenter()
        presenter.write_result(res)
        mock_write_scalar_res.assert_called_once_with(res.evaluated_value, res.name, res.reference_value, res.threshold,
                                                      postfix='%', scale=100, value_name=None)

    def test_vector_presenter_with_vector_data_with_one_element(self, mocker):
        mock_write_scalar_res = mocker.patch('accuracy_checker.presenters.write_scalar_result')  # type: MagicMock
        res = EvaluationResult(
            name='scalar_metric',
            evaluated_value=[0.4],
            reference_value=None,
            threshold=None,
            meta={'names': ['prediction']}
        )
        presenter = VectorPrintPresenter()
        presenter.write_result(res)
        mock_write_scalar_res.assert_called_once_with(res.evaluated_value, res.name, res.reference_value, res.threshold,
                                                      postfix='%', scale=100, value_name=res.meta['names'][0])

    def test_vector_presenter_with_vector_data_with_default_postfix_and_scale(self, mocker):
        mock_write_scalar_res = mocker.patch('accuracy_checker.presenters.write_scalar_result')  # type: MagicMock
        res = EvaluationResult(
            name='scalar_metric',
            evaluated_value=[0.4, 0.6],
            reference_value=None,
            threshold=None,
            meta={'names': ['class1', 'class2']}
        )
        presenter = VectorPrintPresenter()
        presenter.write_result(res)
        calls = [call(res.evaluated_value[0], res.name, res.reference_value, res.threshold,
                      postfix='%', scale=100, value_name=res.meta['names'][0]),
                 call(res.evaluated_value[1], res.name, res.reference_value, res.threshold,
                      postfix='%', scale=100, value_name=res.meta['names'][1]),
                 call(np.mean(np.multiply(res.evaluated_value, 100)), res.name, res.reference_value, res.threshold, value_name='mean',
                      postfix='%', scale=1)]
        mock_write_scalar_res.assert_has_calls(calls)

    def test_vector_presenter_with_vector_data_with_scalar_postfix(self, mocker):
        mock_write_scalar_res = mocker.patch('accuracy_checker.presenters.write_scalar_result')  # type: MagicMock
        res = EvaluationResult(
            name='scalar_metric',
            evaluated_value=[0.4, 0.6],
            reference_value=None,
            threshold=None,
            meta={'names': ['class1', 'class2'], 'postfix': '_'}
        )
        presenter = VectorPrintPresenter()
        presenter.write_result(res)
        calls = [call(res.evaluated_value[0], res.name, res.reference_value, res.threshold,
                      postfix=res.meta['postfix'], scale=100, value_name=res.meta['names'][0]),
                 call(res.evaluated_value[1], res.name, res.reference_value, res.threshold,
                      postfix=res.meta['postfix'], scale=100, value_name=res.meta['names'][1]),
                 call(np.mean(np.multiply(res.evaluated_value, 100)), res.name, res.reference_value, res.threshold, value_name='mean',
                      postfix=res.meta['postfix'], scale=1)]
        mock_write_scalar_res.assert_has_calls(calls)

    def test_vector_presenter_with_vector_data_with_scalar_scale(self, mocker):
        mock_write_scalar_res = mocker.patch('accuracy_checker.presenters.write_scalar_result')  # type: MagicMock
        res = EvaluationResult(
            name='scalar_metric',
            evaluated_value=[0.4, 0.6],
            reference_value=None,
            threshold=None,
            meta={'names': ['class1', 'class2'], 'scale': 10}
        )
        presenter = VectorPrintPresenter()
        presenter.write_result(res)
        calls = [call(res.evaluated_value[0], res.name, res.reference_value, res.threshold,
                      postfix='%', scale=res.meta['scale'], value_name=res.meta['names'][0]),
                 call(res.evaluated_value[1], res.name, res.reference_value, res.threshold,
                      postfix='%', scale=res.meta['scale'], value_name=res.meta['names'][1]),
                 call(np.mean(np.multiply(res.evaluated_value, res.meta['scale'])), res.name, res.reference_value, res.threshold, value_name='mean',
                      postfix='%', scale=1)]
        mock_write_scalar_res.assert_has_calls(calls)

    def test_vector_presenter_with_vector_data_with_vector_scale(self, mocker):
        mock_write_scalar_res = mocker.patch('accuracy_checker.presenters.write_scalar_result')  # type: MagicMock
        res = EvaluationResult(
            name='scalar_metric',
            evaluated_value=[0.4, 0.6],
            reference_value=None,
            threshold=None,
            meta={'names': ['class1', 'class2'], 'scale': [1, 2]}
        )
        presenter = VectorPrintPresenter()
        presenter.write_result(res)
        calls = [call(res.evaluated_value[0], res.name, res.reference_value, res.threshold,
                      postfix='%', scale=res.meta['scale'][0], value_name=res.meta['names'][0]),
                 call(res.evaluated_value[1], res.name, res.reference_value, res.threshold,
                      postfix='%', scale=res.meta['scale'][1], value_name=res.meta['names'][1]),
                 call(np.mean(np.multiply(res.evaluated_value, res.meta['scale'])), res.name, res.reference_value, res.threshold, value_name='mean',
                      postfix='%', scale=1)]
        mock_write_scalar_res.assert_has_calls(calls)
