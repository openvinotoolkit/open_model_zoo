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

import copy
import pickle

from ..utils import get_path, extract_image_representations
from ..dataset import Dataset
from ..launcher import create_launcher, DummyLauncher, InputFeeder
from ..launcher.loaders import PickleLoader
from ..logging import print_info, warning
from ..metrics import MetricsExecutor
from ..postprocessor import PostprocessingExecutor
from ..preprocessor import PreprocessingExecutor
from ..adapters import create_adapter
from ..config import ConfigError
from ..data_readers import BaseReader, REQUIRES_ANNOTATIONS
from .base_evaluator import BaseEvaluator


# pylint: disable=W0223
class ModelEvaluator(BaseEvaluator):
    def __init__(
            self, launcher, input_feeder, adapter, reader, preprocessor, postprocessor, dataset, metric, async_mode
    ):
        self.launcher = launcher
        self.input_feeder = input_feeder
        self.adapter = adapter
        self.reader = reader
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.dataset = dataset
        self.metric_executor = metric
        self.process_dataset = self.process_dataset_sync if not async_mode else self.process_dataset_async

        self._annotations = []
        self._predictions = []
        self._metrics_results = []

    @classmethod
    def from_configs(cls, model_config):
        model_name = model_config['name']
        launcher_config = model_config['launchers'][0]
        dataset_config = model_config['datasets'][0]
        dataset_name = dataset_config['name']
        data_reader_config = dataset_config.get('reader', 'opencv_imread')
        data_source = dataset_config.get('data_source')

        dataset = Dataset(dataset_config)
        if isinstance(data_reader_config, str):
            data_reader_type = data_reader_config
            data_reader_config = None
        elif isinstance(data_reader_config, dict):
            data_reader_type = data_reader_config['type']
        else:
            raise ConfigError('reader should be dict or string')
        if data_reader_type in REQUIRES_ANNOTATIONS:
            data_source = dataset.annotation
        data_reader = BaseReader.provide(data_reader_type, data_source, data_reader_config)
        launcher_kwargs = {}
        enable_ie_preprocessing = (
            dataset_config.get('_ie_preprocessing', False)
            if launcher_config['framework'] == 'dlsdk' else False
        )
        preprocessor = PreprocessingExecutor(
            dataset_config.get('preprocessing'), dataset_name, dataset.metadata,
            enable_ie_preprocessing=enable_ie_preprocessing
        )
        if enable_ie_preprocessing:
            launcher_config['preprocessor'] = preprocessor
        if launcher_config['framework'] == 'dummy' and launcher_config.get('provide_identifiers', False):
            launcher_kwargs = {'identifiers': dataset.identifiers}
        launcher = create_launcher(launcher_config, model_name, **launcher_kwargs)
        async_mode = launcher.async_mode if hasattr(launcher, 'async_mode') else False
        config_adapter = launcher_config.get('adapter')
        adapter = None if not config_adapter else create_adapter(config_adapter, launcher, dataset)
        input_feeder = InputFeeder(
            launcher.config.get('inputs', []), launcher.inputs, launcher.fit_to_input, launcher.default_layout,
            launcher_config['framework'] == 'dummy'
        )
        preprocessor.input_shapes = launcher.inputs_info_for_meta()
        postprocessor = PostprocessingExecutor(dataset_config.get('postprocessing'), dataset_name, dataset.metadata)
        metric_dispatcher = MetricsExecutor(dataset_config.get('metrics', []), dataset)

        return cls(
            launcher, input_feeder, adapter, data_reader,
            preprocessor, postprocessor, dataset, metric_dispatcher, async_mode
        )

    @staticmethod
    def get_processing_info(config):
        launcher_config = config['launchers'][0]
        dataset_config = config['datasets'][0]

        return (
            config['name'],
            launcher_config['framework'], launcher_config.get('device', ''), launcher_config.get('tags'),
            dataset_config['name']
        )

    def _get_batch_input(self, batch_annotation):
        batch_identifiers = [annotation.identifier for annotation in batch_annotation]
        batch_input = [self.reader(identifier=identifier) for identifier in batch_identifiers]
        for annotation, input_data in zip(batch_annotation, batch_input):
            self.dataset.set_annotation_metadata(annotation, input_data, self.reader.data_source)
        batch_input = self.preprocessor.process(batch_input, batch_annotation)
        _, batch_meta = extract_image_representations(batch_input)
        filled_inputs = self.input_feeder.fill_inputs(batch_input)

        return filled_inputs, batch_meta, batch_identifiers

    def process_dataset_async(self, stored_predictions, progress_reporter, *args, **kwargs):
        def _process_ready_predictions(batch_predictions, batch_identifiers, batch_meta, adapter):
            if adapter:
                batch_predictions = self.adapter.process(batch_predictions, batch_identifiers, batch_meta)

            return batch_predictions

        def completion_callback(status_code, request_id):
            if status_code:
                warning('Request {} failed with status code {}'.format(request_id, status_code))
            queued_irs.remove(request_id)
            ready_irs.append(request_id)

        def prepare_dataset():
            if self.dataset.batch is None:
                self.dataset.batch = self.launcher.batch

        prepare_dataset()

        if (
                self.launcher.allow_reshape_input or self.input_feeder.lstm_inputs or
                self.preprocessor.has_multi_infer_transformations or
                getattr(self.reader, 'multi_infer', False)
        ):
            warning('Model can not to be processed in async mode. Switched to sync.')
            return self.process_dataset_sync(stored_predictions, progress_reporter, *args, **kwargs)

        if self._is_stored(stored_predictions) or isinstance(self.launcher, DummyLauncher):
            self._annotations, self._predictions = self._load_stored_predictions(stored_predictions, progress_reporter)

        if progress_reporter:
            progress_reporter.reset(self.dataset.size)

        predictions_to_store = []
        dataset_iterator = iter(enumerate(self.dataset))
        infer_requests_pool = {ir.request_id: ir for ir in self.launcher.get_async_requests()}
        free_irs = list(infer_requests_pool)
        queued_irs, ready_irs = [], []
        for _, async_request in infer_requests_pool.items():
            async_request.set_completion_callback(completion_callback)

        while free_irs or queued_irs or ready_irs:
            self._fill_free_irs(free_irs, queued_irs, infer_requests_pool, dataset_iterator)
            free_irs[:] = []

            if ready_irs:
                while ready_irs:
                    ready_ir_id = ready_irs.pop(0)
                    ready_data = infer_requests_pool[ready_ir_id].get_result()
                    (batch_id, batch_input_ids, batch_annotation), batch_meta, batch_raw_predictions = ready_data
                    batch_identifiers = [annotation.identifier for annotation in batch_annotation]
                    batch_predictions = _process_ready_predictions(
                        batch_raw_predictions, batch_identifiers, batch_meta, self.adapter
                    )
                    free_irs.append(ready_ir_id)
                    if stored_predictions:
                        predictions_to_store.extend(copy.deepcopy(batch_predictions))
                    annotations, predictions = self.postprocessor.process_batch(
                        batch_annotation, batch_predictions, batch_meta
                    )
                    self.metric_executor.update_metrics_on_batch(batch_input_ids, annotations, predictions)

                    if self.metric_executor.need_store_predictions:
                        self._annotations.extend(annotations)
                        self._predictions.extend(predictions)

                    if progress_reporter:
                        progress_reporter.update(batch_id, len(batch_predictions))

        if progress_reporter:
            progress_reporter.finish()

        if stored_predictions:
            self.store_predictions(stored_predictions, predictions_to_store)

    def process_dataset_sync(self, stored_predictions, progress_reporter, *args, **kwargs):
        if progress_reporter:
            progress_reporter.reset(self.dataset.size)
        if self._is_stored(stored_predictions) or isinstance(self.launcher, DummyLauncher):
            self._annotations, self._predictions = self.load(stored_predictions, progress_reporter)
            self._annotations, self._predictions = self.postprocessor.full_process(self._annotations, self._predictions)

            self.metric_executor.update_metrics_on_batch(
                range(len(self._annotations)), self._annotations, self._predictions
            )
            return self._annotations, self._predictions

        if self.dataset.batch is None:
            self.dataset.batch = self.launcher.batch
        output_callback = kwargs.get('output_callback')
        predictions_to_store = []
        for batch_id, (batch_input_ids, batch_annotation) in enumerate(self.dataset):
            filled_inputs, batch_meta, batch_identifiers = self._get_batch_input(batch_annotation)
            batch_predictions = self.launcher.predict(filled_inputs, batch_meta, **kwargs)

            if self.adapter:
                self.adapter.output_blob = self.adapter.output_blob or self.launcher.output_blob
                batch_predictions = self.adapter.process(batch_predictions, batch_identifiers, batch_meta)

            if stored_predictions:
                predictions_to_store.extend(copy.deepcopy(batch_predictions))

            annotations, predictions = self.postprocessor.process_batch(batch_annotation, batch_predictions, batch_meta)
            self.metric_executor.update_metrics_on_batch(batch_input_ids, annotations, predictions)
            if output_callback:
                output_callback(annotations, predictions)

            if self.metric_executor.need_store_predictions:
                self._annotations.extend(annotations)
                self._predictions.extend(predictions)

            if progress_reporter:
                progress_reporter.update(batch_id, len(batch_predictions))

        if progress_reporter:
            progress_reporter.finish()

        if stored_predictions:
            self.store_predictions(stored_predictions, predictions_to_store)
        return self._annotations, self._predictions

    @staticmethod
    def _is_stored(stored_predictions=None):
        if not stored_predictions:
            return False

        try:
            get_path(stored_predictions)
            return True
        except OSError:
            return False

    def _load_stored_predictions(self, stored_predictions, progress_reporter):
        self._annotations, self._predictions = self.load(stored_predictions, progress_reporter)
        self._annotations, self._predictions = self.postprocessor.full_process(self._annotations, self._predictions)
        self.metric_executor.update_metrics_on_batch(
            range(len(self._annotations)), self._annotations, self._predictions
        )

        return self._annotations, self._predictions

    def _fill_free_irs(self, free_irs, queued_irs, infer_requests_pool, dataset_iterator):
        for ir_id in free_irs:
            try:
                batch_id, (batch_input_ids, batch_annotation) = next(dataset_iterator)
            except StopIteration:
                break

            batch_input, batch_meta, _ = self._get_batch_input(batch_annotation)
            self.launcher.predict_async(infer_requests_pool[ir_id], batch_input, batch_meta,
                                        context=tuple([batch_id, batch_input_ids, batch_annotation]))
            queued_irs.append(ir_id)

        return free_irs, queued_irs

    def compute_metrics(self, print_results=True, ignore_results_formatting=False):
        if self._metrics_results:
            del self._metrics_results
            self._metrics_results = []

        for result_presenter, evaluated_metric in self.metric_executor.iterate_metrics(
                self._annotations, self._predictions):
            self._metrics_results.append(evaluated_metric)
            if print_results:
                result_presenter.write_result(evaluated_metric, ignore_results_formatting)
        return self._metrics_results

    def extract_metrics_results(self, print_results=True, ignore_results_formatting=False):
        if not self._metrics_results:
            self.compute_metrics(False, ignore_results_formatting)

        result_presenters = self.metric_executor.get_metric_presenters()
        extracted_results, extracted_meta = [], []
        for presenter, metric_result in zip(result_presenters, self._metrics_results):
            result, metadata = presenter.extract_result(metric_result)
            if isinstance(result, list):
                extracted_results.extend(result)
                extracted_meta.extend(metadata)
            else:
                extracted_results.append(result)
                extracted_meta.append(metadata)
            if print_results:
                presenter.write_result(metric_result, ignore_results_formatting)

        return extracted_results, extracted_meta

    def print_metrics_results(self, ignore_results_formatting=False):
        if not self._metrics_results:
            self.compute_metrics(True, ignore_results_formatting)
            return
        result_presenters = self.metric_executor.get_metric_presenters()
        for presenter, metric_result in zip(result_presenters, self._metrics_results):
            presenter.write_result(metric_result, ignore_results_formatting)

    def load(self, stored_predictions, progress_reporter):
        annotations = self.dataset.annotation
        if self.postprocessor.has_processors:
            self.dataset.provide_data_info(self.reader, annotations)
        launcher = self.launcher
        if not isinstance(launcher, DummyLauncher):
            launcher = DummyLauncher({
                'framework': 'dummy',
                'loader': PickleLoader.__provider__,
                'data_path': stored_predictions
            }, adapter=None)

        identifiers = self.dataset.identifiers
        predictions = launcher.predict(identifiers)
        if self.adapter is not None:
            predictions = self.adapter.process(predictions, identifiers, [])

        if progress_reporter:
            progress_reporter.finish(False)

        return annotations, predictions

    @property
    def metrics_results(self):
        if not self.metrics_results:
            self.compute_metrics(print_results=False)
        computed_metrics = copy.deepcopy(self._metrics_results)
        return computed_metrics

    @staticmethod
    def store_predictions(stored_predictions, predictions):
        # since at the first time file does not exist and then created we can not use it as a pathlib.Path object
        with open(stored_predictions, "wb") as content:
            pickle.dump(predictions, content)
            print_info("prediction objects are save to {}".format(stored_predictions))

    def reset_progress(self, progress_reporter):
        progress_reporter.reset(self.dataset.size)

    def reset(self):
        self.metric_executor.reset()
        del self._annotations
        del self._predictions
        del self._metrics_results
        if hasattr(self, 'infer_requests_pool'):
            del self.infer_requests_pool
        self._annotations = []
        self._predictions = []
        self._metrics_results = []
        self.dataset.reset(self.postprocessor.has_processors)
        self.reader.reset()

    def release(self):
        self.input_feeder.release()
        self.launcher.release()
