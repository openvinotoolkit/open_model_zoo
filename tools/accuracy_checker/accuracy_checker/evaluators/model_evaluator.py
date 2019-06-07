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

import time
import copy
import pickle

from ..utils import get_path, set_image_metadata, extract_image_representations
from ..dataset import Dataset
from ..launcher import create_launcher, DummyLauncher, InputFeeder
from ..launcher.loaders import PickleLoader
from ..logging import print_info
from ..metrics import MetricsExecutor
from ..postprocessor import PostprocessingExecutor
from ..preprocessor import PreprocessingExecutor
from ..adapters import create_adapter
from ..config import ConfigError
from ..data_readers import BaseReader


class ModelEvaluator:
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
        self.dataset_processor = self.process_dataset if not async_mode else self.process_dataset_async

        self._annotations = []
        self._predictions = []

    @classmethod
    def from_configs(cls, launcher_config, dataset_config):
        dataset_name = dataset_config['name']
        data_reader_config = dataset_config.get('reader', 'opencv_imread')
        data_source = dataset_config.get('data_source')

        if isinstance(data_reader_config, str):
            data_reader = BaseReader.provide(data_reader_config, data_source)
        elif isinstance(data_reader_config, dict):
            data_reader = BaseReader.provide(data_reader_config['type'], data_source, data_reader_config)
        else:
            raise ConfigError('reader should be dict or string')

        dataset = Dataset(dataset_config)
        async_mode = launcher_config.get('async_mode', False)
        launcher = create_launcher(launcher_config)
        config_adapter = launcher_config.get('adapter')
        adapter = None if not config_adapter else create_adapter(config_adapter, launcher, dataset)
        input_feeder = InputFeeder(
            launcher.config.get('inputs', []), launcher.inputs, launcher.fit_to_input, launcher.default_layout
        )
        preprocessor = PreprocessingExecutor(
            dataset_config.get('preprocessing'), dataset_name, dataset.metadata, launcher.inputs_info_for_meta()
        )
        postprocessor = PostprocessingExecutor(dataset_config.get('postprocessing'), dataset_name, dataset.metadata)
        metric_dispatcher = MetricsExecutor(dataset_config.get('metrics', []), dataset)

        return cls(
            launcher, input_feeder, adapter, data_reader,
            preprocessor, postprocessor, dataset, metric_dispatcher, async_mode
        )

    def _get_batch_input(self, batch_annotation):
        batch_identifiers = [annotation.identifier for annotation in batch_annotation]
        batch_input = [self.reader(identifier=identifier) for identifier in batch_identifiers]
        for annotation, input_data in zip(batch_annotation, batch_input):
            set_image_metadata(annotation, input_data)
            annotation.metadata['data_source'] = self.reader.data_source
        batch_input = self.preprocessor.process(batch_input, batch_annotation)
        _, batch_meta = extract_image_representations(batch_input)
        filled_inputs = self.input_feeder.fill_inputs(batch_input)

        return filled_inputs, batch_meta, batch_identifiers

    def process_dataset_async(self, stored_predictions, progress_reporter, *args, **kwargs):
        def _process_ready_predictions(batch_predictions, batch_identifiers, batch_meta, adapter, raw_outputs_callback):
            if raw_outputs_callback:
                raw_outputs_callback([batch_predictions])
            if adapter:
                batch_predictions = self.adapter.process(batch_predictions, batch_identifiers, batch_meta)

            return batch_predictions

        if self._is_stored(stored_predictions) or isinstance(self.launcher, DummyLauncher):
            self._annotations, self._predictions = self._load_stored_predictions(stored_predictions, progress_reporter)

        self.dataset.batch = self.launcher.batch
        predictions_to_store = []
        dataset_iterator = iter(enumerate(self.dataset))
        free_irs = self.launcher.infer_requests
        queued_irs = []
        wait_time = 0.01

        while free_irs or queued_irs:
            self._fill_free_irs(free_irs, queued_irs, dataset_iterator)
            free_irs[:] = []

            ready_irs, queued_irs = self._wait_for_any(queued_irs)
            if ready_irs:
                wait_time = 0.01
                for batch_id, batch_annotation, batch_meta, batch_predictions, ir in ready_irs:
                    batch_identifiers = [annotation.identifier for annotation in batch_annotation]
                    batch_predictions = _process_ready_predictions(
                        batch_predictions, batch_identifiers, batch_meta, self.adapter, kwargs.get('output_callback')
                    )
                    free_irs.append(ir)
                    if stored_predictions:
                        predictions_to_store.extend(copy.deepcopy(batch_predictions))
                    annotations, predictions = self.postprocessor.process_batch(batch_annotation, batch_predictions)

                    if not self.postprocessor.has_dataset_processors:
                        self.metric_executor.update_metrics_on_batch(annotations, predictions)

                    self._annotations.extend(annotations)
                    self._predictions.extend(predictions)

                    if progress_reporter:
                        progress_reporter.update(batch_id, len(batch_predictions))

            else:
                time.sleep(wait_time)
                wait_time = max(wait_time * 2, .16)

        if progress_reporter:
            progress_reporter.finish()

        if stored_predictions:
            self.store_predictions(stored_predictions, predictions_to_store)

        if self.postprocessor.has_dataset_processors:
            self.metric_executor.update_metrics_on_batch(self._annotations, self._predictions)

        return self.postprocessor.process_dataset(self._annotations, self._predictions)

    def process_dataset(self, stored_predictions, progress_reporter, *args, **kwargs):
        if self._is_stored(stored_predictions) or isinstance(self.launcher, DummyLauncher):
            self._annotations, self._predictions = self.load(stored_predictions, progress_reporter)
            self._annotations, self._predictions = self.postprocessor.full_process(self._annotations, self._predictions)

            self.metric_executor.update_metrics_on_batch(self._annotations, self._predictions)
            return self._annotations, self._predictions

        self.dataset.batch = self.launcher.batch
        predictions_to_store = []
        for batch_id, batch_annotation in enumerate(self.dataset):
            filled_inputs, batch_meta, batch_identifiers = self._get_batch_input(batch_annotation)
            batch_predictions = self.launcher.predict(filled_inputs, batch_meta, *args, **kwargs)
            if self.adapter:
                self.adapter.output_blob = self.adapter.output_blob or self.launcher.output_blob
                batch_predictions = self.adapter.process(batch_predictions, batch_identifiers, batch_meta)

            if stored_predictions:
                predictions_to_store.extend(copy.deepcopy(batch_predictions))

            annotations, predictions = self.postprocessor.process_batch(batch_annotation, batch_predictions, batch_meta)
            if not self.postprocessor.has_dataset_processors:
                self.metric_executor.update_metrics_on_batch(annotations, predictions)

            self._annotations.extend(annotations)
            self._predictions.extend(predictions)

            if progress_reporter:
                progress_reporter.update(batch_id, len(batch_predictions))

        if progress_reporter:
            progress_reporter.finish()

        if stored_predictions:
            self.store_predictions(stored_predictions, predictions_to_store)

        if self.postprocessor.has_dataset_processors:
            self.metric_executor.update_metrics_on_batch(self._annotations, self._predictions)

        return self.postprocessor.process_dataset(self._annotations, self._predictions)

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
        self.metric_executor.update_metrics_on_batch(self._annotations, self._predictions)

        return self._annotations, self._predictions

    @staticmethod
    def _wait_for_any(irs):
        if not irs:
            return [], []

        result = []
        free_indexes = []
        for ir_id, (batch_id, batch_annotation, batch_meta, ir) in enumerate(irs):
            if ir.wait(0) == 0:
                result.append((batch_id, batch_annotation, batch_meta, [ir.outputs], ir))
                free_indexes.append(ir_id)
        irs = [ir for ir_id, ir in enumerate(irs) if ir_id not in free_indexes]
        return result, irs

    def _fill_free_irs(self, free_irs, queued_irs, dataset_iterator):
        for ir in free_irs:
            try:
                batch_id, batch_annotation = next(dataset_iterator)
            except StopIteration:
                break

            batch_input, batch_meta, _ = self._get_batch_input(batch_annotation)
            self.launcher.predict_async(ir, batch_input, batch_meta)
            queued_irs.append((batch_id, batch_annotation, batch_meta, ir))

        return free_irs, queued_irs

    def compute_metrics(self, output_callback=None, ignore_results_formatting=False):
        for result_presenter, evaluated_metric in self.metric_executor.iterate_metrics(
                self._annotations, self._predictions):
            result_presenter.write_result(evaluated_metric, output_callback, ignore_results_formatting)

    def load(self, stored_predictions, progress_reporter):
        self._annotations = self.dataset.annotation
        launcher = self.launcher
        if not isinstance(launcher, DummyLauncher):
            launcher = DummyLauncher({
                'framework': 'dummy',
                'loader': PickleLoader.__provider__,
                'data_path': stored_predictions
            }, adapter=None)

        predictions = launcher.predict([annotation.identifier for annotation in self._annotations])

        if progress_reporter:
            progress_reporter.finish(False)

        return self._annotations, predictions

    @staticmethod
    def store_predictions(stored_predictions, predictions):
        # since at the first time file does not exist and then created we can not use it as a pathlib.Path object
        with open(stored_predictions, "wb") as content:
            pickle.dump(predictions, content)
            print_info("prediction objects are save to {}".format(stored_predictions))

    def release(self):
        self.launcher.release()
