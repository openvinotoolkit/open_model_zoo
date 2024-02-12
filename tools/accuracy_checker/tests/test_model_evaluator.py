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

from unittest.mock import Mock, MagicMock

from accuracy_checker.evaluators import ModelEvaluator
from accuracy_checker.evaluators.model_evaluator import get_config_metrics

class TestModelEvaluator:
    def setup_method(self):
        self.launcher = Mock()
        self.launcher.predict.return_value = []
        data = MagicMock(data=MagicMock(), metadata=MagicMock(), identifier=0)
        self.preprocessor = Mock()
        self.preprocessor.process = Mock(return_value=data)
        self.postprocessor = Mock()
        self.adapter = MagicMock(return_value=[])
        self.input_feeder = Mock()

        annotation_0 = MagicMock()
        annotation_0.identifier = 0
        annotation_0.metadata = {'data_source': MagicMock()}
        annotation_1 = MagicMock()
        annotation_1.identifier = 1
        annotation_1.metadata = {'data_source': MagicMock()}
        annotation_container_0 = MagicMock()
        annotation_container_0.values = MagicMock(return_value=[annotation_0])
        annotation_container_1 = MagicMock()
        annotation_container_1.values = MagicMock(return_value=([annotation_1]))
        self.annotations = [[annotation_container_0], [annotation_container_1]]

        self.dataset = MagicMock()
        self.dataset.__iter__.return_value = [
            (range(1), self.annotations[0], data, [0]),
            (range(1), self.annotations[1], data, [1])]

        self.postprocessor.process_batch = Mock(side_effect=[
            ([annotation_container_0], [annotation_container_0]), ([annotation_container_1], [annotation_container_1])
        ])
        self.postprocessor.process_dataset = Mock(return_value=(
            ([annotation_container_0], [annotation_container_0]), ([annotation_container_1], [annotation_container_1])
        ))
        self.postprocessor.full_process = Mock(return_value=(
            ([annotation_container_0], [annotation_container_0]), ([annotation_container_1], [annotation_container_1])
        ))

        self.metric = Mock()
        self.metric.update_metrics_on_batch = Mock(return_value=[{}, {}])
        self.metric.profiler = None

        self.evaluator = ModelEvaluator(
            self.launcher,
            self.input_feeder,
            self.adapter,
            self.preprocessor,
            self.postprocessor,
            self.dataset,
            self.metric,
            False,
            {}
        )
        self.evaluator.store_predictions = Mock()
        self.evaluator.load = Mock(return_value=(
            ([annotation_container_0], [annotation_container_0]), ([annotation_container_1], [annotation_container_1])
        ))

    def test_process_dataset_without_storing_predictions_and_dataset_processors(self):
        self.postprocessor.has_dataset_processors = False

        self.evaluator.process_dataset(None, None)

        assert not self.evaluator.store_predictions.called
        assert not self.evaluator.load.called
        assert self.launcher.predict.called
        assert self.postprocessor.process_batch.called
        assert self.metric.update_metrics_on_batch.call_count == len(self.annotations)
        assert not self.postprocessor.process_dataset.called
        assert not self.postprocessor.full_process.called

    def test_process_dataset_with_storing_predictions_and_without_dataset_processors(self):
        self.postprocessor.has_dataset_processors = False

        self.evaluator.process_dataset('path', None)

        assert self.evaluator.store_predictions.called
        assert not self.evaluator.load.called
        assert self.launcher.predict.called
        assert self.postprocessor.process_batch.called
        assert self.metric.update_metrics_on_batch.call_count == len(self.annotations)
        assert not self.postprocessor.process_dataset.called
        assert not self.postprocessor.full_process.called

    def test_process_dataset_store_only(self):
        self.postprocessor.has_dataset_processors = False

        self.evaluator.process_dataset('path', None, store_only=True)

        assert self.evaluator.store_predictions.called
        assert not self.evaluator.load.called
        assert self.launcher.predict.called
        assert not self.postprocessor.process_batch.called
        assert not self.metric.update_metrics_on_batch.called
        assert not self.postprocessor.process_dataset.called
        assert not self.postprocessor.full_process.called

    def test_process_dataset_with_loading_predictions_and_without_dataset_processors(self, mocker):
        mocker.patch('accuracy_checker.evaluators.model_evaluator.get_path')
        self.postprocessor.has_dataset_processors = False

        self.evaluator.process_dataset('path', None)

        assert self.evaluator.load.called
        assert not self.launcher.predict.called
        assert not self.postprocessor.process_batch.called
        assert self.metric.update_metrics_on_batch.call_count == 1
        assert not self.postprocessor.process_dataset.called
        assert self.postprocessor.full_process.called

    def test_process_dataset_with_loading_predictions_and_with_dataset_processors(self, mocker):
        mocker.patch('accuracy_checker.evaluators.model_evaluator.get_path')
        self.postprocessor.has_dataset_processors = True

        self.evaluator.process_dataset('path', None)

        assert not self.evaluator.store_predictions.called
        assert self.evaluator.load.called
        assert not self.launcher.predict.called
        assert not self.postprocessor.process_batch.called
        assert self.metric.update_metrics_on_batch.call_count == 1
        assert not self.postprocessor.process_dataset.called
        assert self.postprocessor.full_process.called

    def test_model_evaluator_get_config_metrics(self, mocker):
        dataset_config = {
            'metrics': [{'type': 'accuracy', 'top_k': 1, 'reference': 0.78}],
            'subset_metrics': [{'subset_size': '20%',
                'metrics': [{'type': 'accuracy', 'top_k': 5, 'reference': 0.65}]}]
        }
        metric = {'type': 'accuracy', 'top_k': 1, 'reference': 0.78}
        selected_metric = get_config_metrics(dataset_config)[0]

        assert metric['reference'] == selected_metric['reference']
        assert metric['top_k'] == selected_metric['top_k']

    def test_model_evaluator_get_config_metrics_is_first_subset_metrics(self, mocker):
        dataset_config_sub_evaluation = { 'sub_evaluation' : 'True',
            'metrics': [{'type': 'accuracy', 'top_k': 1, 'reference': 0.78}],
            'subset_metrics': [
                {'subset_size': '10%', 'metrics': [{'type': 'accuracy', 'top_k': 5, 'reference': 0.65}]},
                {'subset_size': '20%', 'metrics': [{'type': 'accuracy', 'top_k': 5, 'reference': 0.72}]}]
        }
        subset_metric = {'type': 'accuracy', 'top_k': 5, 'reference': 0.65}
        selected_metric = get_config_metrics(dataset_config_sub_evaluation)[0]

        assert subset_metric['reference'] == selected_metric['reference']
        assert subset_metric['top_k'] == selected_metric['top_k']

    def test_model_evaluator_get_config_metrics_with_subsample_size_from_subset_metrics(self, mocker):
        dataset_config_sub_evaluation = { 'sub_evaluation' : 'True', 'subsample_size': '20%',
            'metrics': [{'type': 'accuracy', 'top_k': 1, 'reference': 0.78}],
            'subset_metrics': [
                {'subset_size': '10%', 'metrics': [{'type': 'accuracy', 'top_k': 5, 'reference': 0.65}]},
                {'subset_size': '20%', 'metrics': [{'type': 'accuracy', 'top_k': 5, 'reference': 0.72}]}]
        }
        subset_metric = {'type': 'accuracy', 'top_k': 5, 'reference': 0.72}
        selected_metric = get_config_metrics(dataset_config_sub_evaluation)[0]

        assert subset_metric['reference'] == selected_metric['reference']
        assert subset_metric['top_k'] == selected_metric['top_k']


    def test_model_evaluator_get_config_metrics_from_subset_metrics(self, mocker):
        dataset_config_sub_evaluation = { 'sub_evaluation' : 'True',
            'metrics': [{'type': 'accuracy', 'top_k': 1, 'reference': 0.78}],
            'subset_metrics': [{'subset_size': '20%',
                'metrics': [{'type': 'accuracy', 'top_k': 5, 'reference': 0.65}]}]
        }
        subset_metric = {'type': 'accuracy', 'top_k': 5, 'reference': 0.65}
        selected_metric = get_config_metrics(dataset_config_sub_evaluation)[0]

        assert subset_metric['reference'] == selected_metric['reference']
        assert subset_metric['top_k'] == selected_metric['top_k']



class TestModelEvaluatorAsync:
    def setup_method(self):
        self.launcher = MagicMock()
        self.launcher.get_async_requests = Mock(return_value=[])
        data = MagicMock(data=MagicMock(), metadata=MagicMock(), identifier=0)
        self.preprocessor = Mock()
        self.preprocessor.process = Mock(return_value=data)
        self.postprocessor = Mock()
        self.adapter = MagicMock(return_value=[])
        self.input_feeder = MagicMock()
        self.input_feeder.lstm_inputs = []

        annotation_0 = MagicMock()
        annotation_0.identifier = 0
        annotation_0.metadata = {'data_source': MagicMock()}
        annotation_1 = MagicMock()
        annotation_1.identifier = 1
        annotation_1.metadata = {'data_source': MagicMock()}
        annotation_container_0 = MagicMock()
        annotation_container_0.values = MagicMock(return_value=[annotation_0])
        annotation_container_1 = MagicMock()
        annotation_container_1.values = MagicMock(return_value=([annotation_1]))
        self.annotations = [[annotation_container_0], [annotation_container_1]]

        self.dataset = MagicMock()
        self.dataset.__iter__.return_value = [
            (range(1), self.annotations[0], data, [0]),
            (range(1), self.annotations[1], data, [1])]
        self.dataset.multi_infer = False

        self.postprocessor.process_batch = Mock(side_effect=[
            ([annotation_container_0], [annotation_container_0]), ([annotation_container_1], [annotation_container_1])
        ])
        self.postprocessor.process_dataset = Mock(return_value=(
            ([annotation_container_0], [annotation_container_0]), ([annotation_container_1], [annotation_container_1])
        ))
        self.postprocessor.full_process = Mock(return_value=(
            ([annotation_container_0], [annotation_container_0]), ([annotation_container_1], [annotation_container_1])
        ))

        self.metric = Mock()
        self.metric.update_metrics_on_batch = Mock(return_value=[{}, {}])
        self.metric.profiler = None

        self.evaluator = ModelEvaluator(
            self.launcher,
            self.input_feeder,
            self.adapter,
            self.preprocessor,
            self.postprocessor,
            self.dataset,
            self.metric,
            True,
            {}
        )
        self.evaluator.store_predictions = Mock()
        self.evaluator.load = Mock(return_value=(
            ([annotation_container_0], [annotation_container_0]), ([annotation_container_1], [annotation_container_1])
        ))

    def test_process_dataset_without_storing_predictions_and_dataset_processors(self):
        self.postprocessor.has_dataset_processors = False
        self.launcher.allow_reshape_input = False
        self.preprocessor.has_multi_infer_transformations = False
        self.launcher.dyn_input_layers = False


        self.evaluator.process_dataset(None, None)

        assert not self.evaluator.store_predictions.called
        assert not self.evaluator.load.called
        assert not self.launcher.predict.called
        assert self.launcher.get_infer_queue.called

    def test_process_dataset_with_storing_predictions_and_without_dataset_processors(self):
        self.postprocessor.has_dataset_processors = False
        self.launcher.allow_reshape_input = False
        self.preprocessor.has_multi_infer_transformations = False
        self.dataset.multi_infer = False
        self.launcher.dyn_input_layers = False

        self.evaluator.process_dataset('path', None)

        assert not self.evaluator.load.called
        assert not self.launcher.predict.called
        assert self.launcher.get_infer_queue.called

    def test_process_dataset_with_loading_predictions_and_without_dataset_processors(self, mocker):
        mocker.patch('accuracy_checker.evaluators.model_evaluator.get_path')
        self.postprocessor.has_dataset_processors = False

        self.evaluator.process_dataset('path', None)

        assert self.evaluator.load.called
        assert not self.launcher.predict.called
        assert not self.launcher.predict_async.called
        assert not self.postprocessor.process_batch.called
        assert self.metric.update_metrics_on_batch.call_count == 1
        assert not self.postprocessor.process_dataset.called
        assert self.postprocessor.full_process.called

    def test_switch_to_sync_predict_if_need_reshaping(self):
        self.postprocessor.has_dataset_processors = False
        self.launcher.allow_reshape_input = True
        self.launcher.dynamic_shapes_policy = 'static'
        self.preprocessor.has_multi_infer_transformations = False

        self.evaluator.process_dataset(None, None)

        assert not self.evaluator.store_predictions.called
        assert not self.evaluator.load.called
        assert self.launcher.predict.called
        assert not self.launcher.predict_async.called
        assert self.metric.update_metrics_on_batch.call_count == len(self.annotations)

    def test_switch_to_sync_predict_if_need_multi_infer_after_preprocessing(self):
        self.postprocessor.has_dataset_processors = False
        self.launcher.allow_reshape_input = False
        self.preprocessor.has_multi_infer_transformations = True

        self.evaluator.process_dataset(None, None)

        assert not self.evaluator.store_predictions.called
        assert not self.evaluator.load.called
        assert self.launcher.predict.called
        assert not self.launcher.predict_async.called
        assert self.metric.update_metrics_on_batch.call_count == len(self.annotations)

    def test_switch_to_sync_predict_if_need_multi_infer(self):
        self.postprocessor.has_dataset_processors = False
        self.launcher.allow_reshape_input = False
        self.preprocessor.has_multi_infer_transformations = False
        self.dataset.multi_infer = True

        self.evaluator.process_dataset(None, None)

        assert not self.evaluator.store_predictions.called
        assert not self.evaluator.load.called
        assert self.launcher.predict.called
        assert not self.launcher.predict_async.called
        assert self.metric.update_metrics_on_batch.call_count == len(self.annotations)
