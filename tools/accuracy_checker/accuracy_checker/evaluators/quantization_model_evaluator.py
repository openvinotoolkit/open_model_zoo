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
import numpy as np

from ..utils import extract_image_representations, contains_any
from ..dataset import Dataset, DatasetWrapper
from ..launcher import create_launcher, InputFeeder
from ..logging import warning
from ..metrics import MetricsExecutor
from ..postprocessor import PostprocessingExecutor
from ..preprocessor import PreprocessingExecutor
from ..adapters import create_adapter
from ..config import ConfigError
from ..data_readers import BaseReader, REQUIRES_ANNOTATIONS
from ..progress_reporters import ProgressReporter


class ModelEvaluator:
    def __init__(
            self, launcher, adapter, dataset_config
    ):
        self.launcher = launcher
        self.input_feeder = None
        self.adapter = adapter
        self.dataset_config = dataset_config
        self.preprocessor = None
        self.dataset = None
        self.postprocessor = None
        self.metric_executor = None

        self._annotations = []
        self._predictions = []
        self._input_ids = []
        self._metrics_results = []

    @classmethod
    def from_configs(cls, config):
        model_config = config['models'][0]
        dataset_config = model_config['datasets']
        launcher_config = model_config['launchers'][0]
        launcher = create_launcher(launcher_config, delayed_model_loading=True)
        config_adapter = launcher_config.get('adapter')
        adapter = None if not config_adapter else create_adapter(config_adapter, None, None)

        return cls(
            launcher, adapter, dataset_config
        )

    def _get_batch_input(self, batch_input, batch_annotation):
        batch_input = self.preprocessor.process(batch_input, batch_annotation)
        _, batch_meta = extract_image_representations(batch_input)
        filled_inputs = self.input_feeder.fill_inputs(batch_input)

        return filled_inputs, batch_meta

    def process_dataset_async(
            self,
            nreq=None,
            subset=None,
            num_images=None,
            check_progress=False,
            dataset_tag='',
            output_callback=None,
            allow_pairwise_subset=False,
            **kwargs
    ):

        def _process_ready_predictions(batch_raw_predictions, batch_identifiers, batch_meta, adapter):
            if adapter:
                return self.adapter.process(batch_raw_predictions, batch_identifiers, batch_meta)

            return batch_raw_predictions

        def _create_subset(subset, num_images):
            if subset is not None:
                self.dataset.make_subset(ids=subset, accept_pairs=allow_pairwise_subset)
            elif num_images is not None:
                self.dataset.make_subset(end=num_images, accept_pairs=allow_pairwise_subset)

        def _set_number_infer_requests(nreq):
            if nreq is None:
                nreq = self.launcher.auto_num_requests()
            if self.launcher.num_requests != nreq:
                self.launcher.num_requests = nreq

        if self.dataset is None or (dataset_tag and self.dataset.tag != dataset_tag):
            self.select_dataset(dataset_tag)

        if self.launcher.allow_reshape_input or self.preprocessor.has_multi_infer_transformations:
            warning('Model can not to be processed in async mode. Switched to sync.')
            return self.process_dataset(
                subset, num_images, check_progress, dataset_tag, output_callback, allow_pairwise_subset, **kwargs
            )
        _set_number_infer_requests(nreq)

        self.dataset.batch = self.launcher.batch
        self.preprocessor.input_shapes = self.launcher.inputs_info_for_meta()
        progress_reporter = None

        _create_subset(subset, num_images)

        if check_progress:
            progress_reporter = self._create_progress_reporter(check_progress, self.dataset.size)

        dataset_iterator = iter(enumerate(self.dataset))

        free_irs = self.launcher.infer_requests
        queued_irs = []
        wait_time = 0.01

        while free_irs or queued_irs:
            self._fill_free_irs(free_irs, queued_irs, dataset_iterator, **kwargs)
            free_irs[:] = []

            ready_irs, queued_irs = self._wait_for_any(queued_irs)
            if ready_irs:
                wait_time = 0.01
                while ready_irs:
                    ready_data = ready_irs.pop(0)
                    (
                        batch_id,
                        batch_input_ids,
                        batch_annotation,
                        batch_identifiers,
                        batch_meta,
                        batch_raw_predictions,
                        ir
                    ) = ready_data
                    batch_predictions = _process_ready_predictions(
                        batch_raw_predictions, batch_identifiers, batch_meta, self.adapter
                    )
                    free_irs.append(ir)
                    annotations, predictions = self.postprocessor.process_batch(
                        batch_annotation, batch_predictions, batch_meta
                    )

                    metrics_result = None
                    if self.metric_executor:
                        metrics_result = self.metric_executor.update_metrics_on_batch(
                            batch_input_ids, annotations, predictions
                        )
                        if self.metric_executor.need_store_predictions:
                            self._annotations.extend(annotations)
                            self._predictions.extend(predictions)

                    if output_callback:
                        output_callback(
                            batch_raw_predictions,
                            metrics_result=metrics_result,
                            element_identifiers=batch_identifiers,
                            dataset_indices=batch_input_ids
                        )

                    if progress_reporter:
                        progress_reporter.update(batch_id, len(batch_predictions))
            else:
                time.sleep(wait_time)
                wait_time = max(wait_time * 2, .16)

        if progress_reporter:
            progress_reporter.finish()

    def select_dataset(self, dataset_tag):
        if self.dataset is not None and isinstance(self.dataset_config, list):
            return
        dataset_attributes = create_dataset_attributes(self.dataset_config, dataset_tag)
        self.dataset, self.metric_executor, self.preprocessor, self.postprocessor = dataset_attributes
        if self.dataset.annotation_reader and self.dataset.annotation_reader.metadata:
            self.adapter.label_map = self.dataset.annotation_reader.metadata.get('label_map')

    def process_dataset(
            self,
            subset=None,
            num_images=None,
            check_progress=False,
            dataset_tag='',
            output_callback=None,
            allow_pairwise_subset=False,
            **kwargs
    ):
        def _create_subset(subset, num_images):
            if subset is not None:
                self.dataset.make_subset(ids=subset, accept_pairs=allow_pairwise_subset)
            elif num_images is not None:
                self.dataset.make_subset(end=num_images, accept_pairs=allow_pairwise_subset)

        if self.dataset is None or (dataset_tag and self.dataset.tag != dataset_tag):
            self.select_dataset(dataset_tag)
        self.dataset.batch = self.launcher.batch
        self.preprocessor.input_shapes = self.launcher.inputs_info_for_meta()
        progress_reporter = None

        _create_subset(subset, num_images)

        if check_progress:
            progress_reporter = self._create_progress_reporter(check_progress, self.dataset.size)

        for batch_id, (batch_input_ids, batch_annotation, batch_inputs, batch_identifiers) in enumerate(self.dataset):
            filled_inputs, batch_meta = self._get_batch_input(batch_inputs, batch_annotation)
            batch_raw_predictions = self.launcher.predict(filled_inputs, batch_meta, **kwargs)
            if self.adapter:
                self.adapter.output_blob = self.adapter.output_blob or self.launcher.output_blob
                batch_predictions = self.adapter.process(batch_raw_predictions, batch_identifiers, batch_meta)
            else:
                batch_predictions = batch_raw_predictions

            annotations, predictions = self.postprocessor.process_batch(batch_annotation, batch_predictions, batch_meta)
            metrics_result = None
            if self.metric_executor:
                metrics_result = self.metric_executor.update_metrics_on_batch(batch_input_ids, annotations, predictions)
                if self.metric_executor.need_store_predictions:
                    self._annotations.extend(annotations)
                    self._predictions.extend(predictions)

            if output_callback:
                if isinstance(batch_raw_predictions, list) and len(batch_raw_predictions) == 1:
                    batch_raw_predictions = batch_raw_predictions[0]
                output_callback(
                    batch_raw_predictions,
                    metrics_result=metrics_result,
                    element_identifiers=batch_identifiers,
                    dataset_indices=batch_input_ids
                )

            if progress_reporter:
                progress_reporter.update(batch_id, len(batch_predictions))

        if progress_reporter:
            progress_reporter.finish()

    @staticmethod
    def _wait_for_any(irs):
        if not irs:
            return [], []

        free_indexes = []
        for ir_id, (_, _, _, _, _, ir) in enumerate(irs):
            if ir.wait(0) == 0:
                free_indexes.append(ir_id)
        result = []
        free_indexes.sort(reverse=True)
        for idx in free_indexes:
            batch_id, batch_input_ids, batch_annotation, batch_identifiers, batch_meta, ir = irs.pop(idx)
            result.append((batch_id, batch_input_ids, batch_annotation, batch_identifiers, batch_meta, ir.outputs, ir))

        return result, irs

    def _fill_free_irs(self, free_irs, queued_irs, dataset_iterator, **kwargs):
        for ir in free_irs:
            try:
                batch_id, (batch_input_ids, batch_annotation, batch_inputs, batch_identifiers) = next(dataset_iterator)
            except StopIteration:
                break

            batch_input, batch_meta = self._get_batch_input(batch_inputs, batch_annotation)
            self.launcher.predict_async(ir, batch_input, batch_meta, **kwargs)
            queued_irs.append((batch_id, batch_input_ids, batch_annotation, batch_identifiers, batch_meta, ir))

        return free_irs, queued_irs

    @staticmethod
    def _create_progress_reporter(check_progress, dataset_size):
        pr_kwargs = {}
        if isinstance(check_progress, int) and not isinstance(check_progress, bool):
            pr_kwargs = {"print_interval": check_progress}

        return ProgressReporter.provide('print', dataset_size, **pr_kwargs)

    def compute_metrics(self, print_results=True, ignore_results_formatting=False):
        if not self.metric_executor:
            return []
        if self._metrics_results:
            del self._metrics_results
            self._metrics_results = []
        if self._input_ids:
            indexes = np.argsort(self._input_ids)
            annotations = [self._annotations[idx] for idx in indexes]
            predictions = [self._predictions[idx] for idx in indexes]
            self._annotations = annotations
            self._predictions = predictions

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
            presenter.write_result(metric_result, ignore_results_formatting)

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

    def get_metrics_attributes(self):
        if not self.metric_executor:
            return {}
        return self.metric_executor.get_metrics_attributes()

    def register_metric(self, metric_config):
        if isinstance(metric_config, str):
            self.metric_executor.register_metric({'type': metric_config})
        elif isinstance(metric_config, dict):
            self.metric_executor.register_metric(metric_config)
        else:
            raise ValueError('Unsupported metric configuration type {}'.format(type(metric_config)))

    def register_postprocessor(self, postprocessing_config):
        if isinstance(postprocessing_config, str):
            self.postprocessor.register_postprocessor({'type': postprocessing_config})
        elif isinstance(postprocessing_config, dict):
            self.postprocessor.register_postprocessor(postprocessing_config)
        else:
            raise ValueError('Unsupported post-processor configuration type {}'.format(type(postprocessing_config)))

    def reset(self):
        if self.metric_executor:
            self.metric_executor.reset()
        del self._annotations
        del self._predictions
        del self._input_ids
        del self._metrics_results
        self._annotations = []
        self._predictions = []
        self._input_ids = []
        self._metrics_results = []
        if self.dataset:
            self.dataset.reset(self.postprocessor.has_processors)

    def release(self):
        self.launcher.release()


def create_dataset_attributes(config, tag):
    if isinstance(config, list):
        dataset_config = config[0]
    elif isinstance(config, dict):
        dataset_config = config.get(tag)
        if not dataset_config:
            raise ConfigError('suitable dataset for *{}* not found'.format(tag))
    else:
        raise TypeError('unknown type for config, dictionary or list must be')

    dataset_name = dataset_config['name']
    data_reader_config = dataset_config.get('reader', 'opencv_imread')
    data_source = dataset_config.get('data_source')
    annotation_reader = None
    dataset_meta = {}
    if contains_any(dataset_config, ['annotation', 'annotation_conversion']):
        annotation_reader = Dataset(dataset_config)
        dataset_meta = annotation_reader.metadata
    if isinstance(data_reader_config, str):
        data_reader_type = data_reader_config
        data_reader_config = None
    elif isinstance(data_reader_config, dict):
        data_reader_type = data_reader_config['type']
    else:
        raise ConfigError('reader should be dict or string')
    if data_reader_type in REQUIRES_ANNOTATIONS:
        if annotation_reader is None:
            raise ConfigError('data reader *{}* requires annotation'.format(data_reader_type))
        data_source = annotation_reader.annotation
    data_reader = BaseReader.provide(data_reader_type, data_source, data_reader_config)

    metric_dispatcher = None
    dataset = DatasetWrapper(data_reader, annotation_reader)
    preprocessor = PreprocessingExecutor(
        dataset_config.get('preprocessing'), dataset_name, dataset_meta
    )
    postprocessor = PostprocessingExecutor(dataset_config.get('postprocessing'), dataset_name, dataset_meta)
    if 'metrics' in dataset_config:
        metric_dispatcher = MetricsExecutor(dataset_config.get('metrics', []), annotation_reader)

    return dataset, metric_dispatcher, preprocessor, postprocessor
