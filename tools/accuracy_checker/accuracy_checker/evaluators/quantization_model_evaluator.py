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

from ..utils import extract_image_representations, contains_any
from ..dataset import Dataset, DatasetWrapper
from ..launcher import create_launcher, InputFeeder
from ..metrics import MetricsExecutor
from ..postprocessor import PostprocessingExecutor
from ..preprocessor import PreprocessingExecutor
from ..adapters import create_adapter
from ..config import ConfigError
from ..data_readers import BaseReader
from ..statistics_collector import StatisticsCollector
from ..progress_reporters import ProgressReporter


class ModelEvaluator:
    def __init__(
            self, launcher, input_feeder, adapter, dataset, preprocessor, postprocessor, metric
    ):
        self.launcher = launcher
        self.input_feeder = input_feeder
        self.adapter = adapter
        self.dataset = dataset
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.metric_executor = metric
        self.stat_collector = None

        self._annotations = []
        self._predictions = []
        self._metrics_results = []

    @classmethod
    def from_configs(cls, config):
        model_config = config['models'][0]
        dataset_config = model_config['datasets'][0]
        launcher_config = model_config['launchers'][0]
        dataset_name = dataset_config['name']
        data_reader_config = dataset_config.get('reader', 'opencv_imread')
        data_source = dataset_config.get('data_source')

        if isinstance(data_reader_config, str):
            data_reader = BaseReader.provide(data_reader_config, data_source)
        elif isinstance(data_reader_config, dict):
            data_reader = BaseReader.provide(data_reader_config['type'], data_source, data_reader_config)
        else:
            raise ConfigError('reader should be dict or string')
        annotation_reader = None
        dataset_meta = {}
        metric_dispatcher = None
        if contains_any(dataset_config, ['annotation', 'annotation_conversion']):
            annotation_reader = Dataset(dataset_config)
            dataset_meta = annotation_reader.metadata
        dataset = DatasetWrapper(data_reader, annotation_reader)
        launcher = create_launcher(launcher_config, delayed_model_loading=True)
        config_adapter = launcher_config.get('adapter')
        adapter = None if not config_adapter else create_adapter(config_adapter, None, annotation_reader)
        preprocessor = PreprocessingExecutor(
            dataset_config.get('preprocessing'), dataset_name, dataset_meta
        )
        postprocessor = PostprocessingExecutor(dataset_config.get('postprocessing'), dataset_name, dataset_meta)
        if 'metrics' in dataset_config:
            metric_dispatcher = MetricsExecutor(dataset_config.get('metrics', []), annotation_reader)

        return cls(
            launcher, None, adapter, dataset,
            preprocessor, postprocessor, metric_dispatcher
        )

    def _get_batch_input(self, batch_input, batch_annotation):
        batch_input = self.preprocessor.process(batch_input, batch_annotation)
        _, batch_meta = extract_image_representations(batch_input)
        filled_inputs = self.input_feeder.fill_inputs(batch_input)

        return filled_inputs, batch_meta

    def process_dataset_async(
            self,
            nreq=2,
            statistics_functors_maping=None,
            subset=None,
            num_images=None,
            check_progress=False,
            **kwargs
    ):
        def _process_ready_predictions(batch_predictions, batch_identifiers, batch_meta, adapter, raw_outputs_callback):
            if self.stat_collector:
                self.stat_collector.process_batch(batch_predictions)
            if raw_outputs_callback:
                raw_outputs_callback(batch_predictions)
            if adapter:
                batch_predictions = self.adapter.process(batch_predictions, batch_identifiers, batch_meta)

            return batch_predictions

        def _create_subset(subset, num_images):
            if subset is not None:
                self.dataset.make_subset(ids=subset)
            elif num_images is not None:
                self.dataset.make_subset(end=num_images)

        self.dataset.batch = self.launcher.batch
        self.stat_collector = None
        progress_reporter = None

        _create_subset(subset, num_images)

        if check_progress:
            progress_reporter = ProgressReporter.provide('print', self.dataset.size)

        dataset_iterator = iter(enumerate(self.dataset))
        if self.launcher.num_requests != nreq:
            self.launcher.num_requests = nreq

        if statistics_functors_maping:
            self.stat_collector = StatisticsCollector(statistics_functors_maping, self.launcher.batch)
        free_irs = self.launcher.infer_requests
        queued_irs = []
        wait_time = 0.01

        while free_irs or queued_irs:
            self._fill_free_irs(free_irs, queued_irs, dataset_iterator, **kwargs)
            free_irs[:] = []

            ready_irs, queued_irs = self._wait_for_any(queued_irs)
            if ready_irs:
                wait_time = 0.01
                for batch_id, batch_annotation, batch_identifiers, batch_meta, batch_predictions, ir in ready_irs:
                    batch_predictions = _process_ready_predictions(
                        batch_predictions, batch_identifiers, batch_meta, self.adapter, kwargs.get('output_callback')
                    )
                    free_irs.append(ir)
                    annotations, predictions = self.postprocessor.process_batch(batch_annotation, batch_predictions)

                    if not self.postprocessor.has_dataset_processors and self.metric_executor:
                        self.metric_executor.update_metrics_on_batch(annotations, predictions)

                    self._annotations.extend(annotations)
                    self._predictions.extend(predictions)

                    if progress_reporter:
                        progress_reporter.update(batch_id, len(batch_predictions))
            else:
                time.sleep(wait_time)
                wait_time = max(wait_time * 2, .16)

        if self.postprocessor.has_dataset_processors:
            self.metric_executor.update_metrics_on_batch(self._annotations, self._predictions)

        if progress_reporter:
            progress_reporter.finish()

        return self.postprocessor.process_dataset(self._annotations, self._predictions)

    def process_dataset(
            self,
            statistics_functors_maping=None,
            subset=None,
            num_images=None,
            check_progress=False,
            **kwargs
    ):
        self.dataset.batch = self.launcher.batch
        progress_reporter = None

        if statistics_functors_maping:
            self.stat_collector = StatisticsCollector(statistics_functors_maping, self.launcher.batch)

        if subset is not None:
            self.dataset.make_subset(ids=subset)

        elif num_images is not None:
            self.dataset.make_subset(end=num_images)

        if check_progress:
            progress_reporter = ProgressReporter.provide('print', self.dataset.size)

        for batch_id, (batch_annotation, batch_inputs, batch_identifiers) in enumerate(self.dataset):
            filled_inputs, batch_meta = self._get_batch_input(batch_inputs, batch_annotation)
            batch_predictions = self.launcher.predict(filled_inputs, batch_meta, **kwargs)
            if self.stat_collector:
                self.stat_collector.process_batch(batch_predictions)
            if self.adapter:
                self.adapter.output_blob = self.adapter.output_blob or self.launcher.output_blob
                batch_predictions = self.adapter.process(batch_predictions, batch_identifiers, batch_meta)

            annotations, predictions = self.postprocessor.process_batch(batch_annotation, batch_predictions, batch_meta)
            if not self.postprocessor.has_dataset_processors and self.metric_executor:
                self.metric_executor.update_metrics_on_batch(annotations, predictions)

            self._annotations.extend(annotations)
            self._predictions.extend(predictions)

            if progress_reporter:
                progress_reporter.update(batch_id, len(batch_predictions))

        if self.postprocessor.has_dataset_processors and self.metric_executor:
            self.metric_executor.update_metrics_on_batch(self._annotations, self._predictions)

        if progress_reporter:
            progress_reporter.finish()

        return self.postprocessor.process_dataset(self._annotations, self._predictions)

    @staticmethod
    def _wait_for_any(irs):
        if not irs:
            return [], []

        result = []
        free_indexes = []
        for ir_id, (batch_id, batch_annotation, batch_identifiers, batch_meta, ir) in enumerate(irs):
            if ir.wait(0) == 0:
                result.append((batch_id, batch_annotation, batch_identifiers, batch_meta, ir.outputs, ir))
                free_indexes.append(ir_id)
        irs = [ir for ir_id, ir in enumerate(irs) if ir_id not in free_indexes]
        return result, irs

    def _fill_free_irs(self, free_irs, queued_irs, dataset_iterator, **kwargs):
        for ir in free_irs:
            try:
                batch_id, (batch_annotation, batch_inputs, batch_identifiers) = next(dataset_iterator)
            except StopIteration:
                break

            batch_input, batch_meta = self._get_batch_input(batch_inputs, batch_annotation)
            self.launcher.predict_async(ir, batch_input, batch_meta, **kwargs)
            queued_irs.append((batch_id, batch_annotation, batch_identifiers, batch_meta, ir))

        return free_irs, queued_irs

    def compute_metrics(self, print_results=True, ignore_results_formatting=False):
        if not self.metric_executor:
            return []
        if self._metrics_results:
            del self._metrics_results
            self._metrics_results = []

        for result_presenter, evaluated_metric in self.metric_executor.iterate_metrics(
                self._annotations, self._predictions):
            self._metrics_results.append(evaluated_metric)
            if print_results:
                result_presenter.write_result(evaluated_metric, ignore_results_formatting=ignore_results_formatting)
        return self._metrics_results

    def print_metrics_results(self, ignore_results_formatting=False):
        if not self._metrics_results:
            self.compute_metrics(True, ignore_results_formatting)
            return
        result_presenters = self.metric_executor.get_metric_presenters()
        for presenter, metric_result in zip(result_presenters, self._metrics_results):
            presenter.write_results(metric_result, ignore_results_formatting)

    @property
    def metrics_results(self):
        if not self.metrics_results:
            self.compute_metrics(print_results=False)
        computed_metrics = copy.deepcopy(self._metrics_results)
        return computed_metrics

    def load_network(self, network=None):
        self.launcher.load_network(network)
        self.input_feeder = InputFeeder(
            self.launcher.config.get('inputs', []), self.launcher.inputs,
            self.launcher.fit_to_input, self.launcher.default_layout
        )
        if self.adapter:
            self.adapter.output_blob = self.launcher.output_blob

    def load_network_from_ir(self, xml_path, bin_path):
        self.launcher.load_ir(xml_path, bin_path)
        self.input_feeder = InputFeeder(
            self.launcher.config.get('inputs', []), self.launcher.inputs,
            self.launcher.fit_to_input, self.launcher.default_layout
        )
        if self.adapter:
            self.adapter.output_blob = self.launcher.output_blob

    def get_network(self):
        return self.launcher.network

    def get_statistics(self):
        if not self.stat_collector:
            return None
        return self.stat_collector.get_statistics()

    def reset(self):
        if self.metric_executor:
            self.metric_executor.reset()
        del self._annotations
        del self._predictions
        del self._metrics_results
        self._annotations = []
        self._predictions = []
        self._metrics_results = []
        self.dataset.reset()

    def release(self):
        self.launcher.release()
