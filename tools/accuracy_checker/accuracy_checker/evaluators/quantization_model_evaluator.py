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

import copy
import numpy as np

from ..utils import extract_image_representations, contains_any
from ..dataset import Dataset, DataProvider, AnnotationProvider
from ..launcher import create_launcher, InputFeeder
from ..logging import warning
from ..metrics import MetricsExecutor
from ..postprocessor import PostprocessingExecutor
from ..preprocessor import PreprocessingExecutor
from ..adapters import create_adapter
from ..config import ConfigError
from ..data_readers import BaseReader, REQUIRES_ANNOTATIONS
from ..progress_reporters import ProgressReporter
from .module_evaluator import ModuleEvaluator


def create_model_evaluator(config):
    cascade = 'evaluations' in config
    if not cascade:
        return ModelEvaluator.from_configs(config)

    return ModuleEvaluator.from_configs(config['evaluations'][0], delayed_model_loading=True)


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
        self._dumped_annotations = []

    @classmethod
    def from_configs(cls, config):
        model_config = config['models'][0]
        model_name = model_config['name']
        dataset_config = model_config['datasets']
        launcher_config = model_config['launchers'][0]
        launcher = create_launcher(launcher_config, model_name, delayed_model_loading=True)
        config_adapter = launcher_config.get('adapter')
        adapter = None if not config_adapter else create_adapter(
            config_adapter, launcher, None, delayed_model_loading=True
        )

        return cls(
            launcher, adapter, dataset_config
        )

    def _get_batch_input(self, batch_input, batch_annotation):
        batch_input = self.preprocessor.process(batch_input, batch_annotation)
        batch_meta = extract_image_representations(batch_input, meta_only=True)
        filled_inputs = self.input_feeder.fill_inputs(batch_input)

        return filled_inputs, batch_meta

    # pylint: disable=R0912,R1702
    def process_dataset_async(
            self,
            nreq=None,
            subset=None,
            num_images=None,
            check_progress=False,
            dataset_tag='',
            output_callback=None,
            allow_pairwise_subset=False,
            dump_prediction_to_annotation=False,
            calculate_metrics=True,
            **kwargs
    ):

        def _process_ready_predictions(batch_raw_predictions, batch_identifiers, batch_meta, calculate_metrics=True):
            if self.adapter and calculate_metrics:
                return self.adapter.process(batch_raw_predictions, batch_identifiers, batch_meta)

            return batch_raw_predictions

        def completion_callback(status_code, request_id):
            if status_code:
                warning('Request {} failed with status code {}'.format(request_id, status_code))
            queued_irs.remove(request_id)
            ready_irs.append(request_id)

        self._prepare_to_evaluation(dataset_tag, dump_prediction_to_annotation)

        if (
                self.launcher.allow_reshape_input or self.input_feeder.lstm_inputs or
                self.preprocessor.has_multi_infer_transformations or
                self.dataset.multi_infer
        ):
            warning('Model can not to be processed in async mode. Switched to sync.')
            return self.process_dataset(
                subset,
                num_images,
                check_progress,
                dataset_tag,
                output_callback,
                allow_pairwise_subset,
                dump_prediction_to_annotation,
                calculate_metrics,
                **kwargs
            )

        self._set_number_infer_requests(nreq)
        self._create_subset(subset, num_images, allow_pairwise_subset)

        progress_reporter = None if not check_progress else self._create_progress_reporter(
            check_progress, self.dataset.size
        )
        dataset_iterator = iter(enumerate(self.dataset))
        free_irs, queued_irs, ready_irs = [], [], []
        infer_requests_pool = self._prepare_requests_pool(completion_callback)
        free_irs = list(infer_requests_pool)
        while free_irs or queued_irs or ready_irs:
            self._fill_free_irs(free_irs, queued_irs, infer_requests_pool, dataset_iterator, **kwargs)
            free_irs[:] = []

            if ready_irs:
                while ready_irs:
                    ready_ir_id = ready_irs.pop(0)
                    ready_data = infer_requests_pool[ready_ir_id].get_result()
                    (
                        (batch_id, batch_input_ids, batch_annotation, batch_identifiers),
                        batch_meta,
                        batch_raw_predictions,
                    ) = ready_data
                    batch_predictions = _process_ready_predictions(
                        batch_raw_predictions, batch_identifiers, batch_meta,
                        calculate_metrics or dump_prediction_to_annotation
                    )
                    free_irs.append(ready_ir_id)

                    metrics_result = None
                    if calculate_metrics:
                        annotations, predictions = self.postprocessor.process_batch(
                            batch_annotation, batch_predictions, batch_meta, dump_prediction_to_annotation
                        )
                        if dump_prediction_to_annotation:
                            threshold = kwargs.get('annotation_conf_threshold', 0.0)
                            annotations = []
                            for prediction in predictions:
                                generated_annotation = prediction.to_annotation(threshold=threshold)
                                if generated_annotation:
                                    annotations.append(generated_annotation)
                            self._dumped_annotations.extend(annotations)
                        if self.metric_executor:
                            metrics_result, _ = self.metric_executor.update_metrics_on_batch(
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

        if dump_prediction_to_annotation:
            self.register_dumped_annotations()

        if progress_reporter:
            progress_reporter.finish()

    def register_dumped_annotations(self):
        if not self._dumped_annotations:
            return
        meta = Dataset.load_meta(self.dataset.dataset_config)
        self.dataset.set_annotation(self._dumped_annotations, meta)

    def select_dataset(self, dataset_tag):
        if self.dataset is not None and isinstance(self.dataset_config, list):
            return
        dataset_attributes = create_dataset_attributes(self.dataset_config, dataset_tag, self._dumped_annotations)
        self.dataset, self.metric_executor, self.preprocessor, self.postprocessor = dataset_attributes
        if self.dataset.annotation_provider and self.dataset.annotation_provider.metadata:
            self.adapter.label_map = self.dataset.annotation_provider.metadata.get('label_map')

    def _create_subset(self, subset=None, num_images=None, allow_pairwise=False):
        if subset is not None:
            self.dataset.make_subset(ids=subset, accept_pairs=allow_pairwise)
        elif num_images is not None:
            self.dataset.make_subset(end=num_images, accept_pairs=allow_pairwise)

    def _set_number_infer_requests(self, nreq=None):
        if nreq is None:
            nreq = self.launcher.auto_num_requests()
        if self.launcher.num_requests != nreq:
            self.launcher.num_requests = nreq

    def _prepare_to_evaluation(self, dataset_tag='', dump_prediction_to_annotation=False):
        if self.dataset is None or (dataset_tag and self.dataset.tag != dataset_tag):
            self.select_dataset(dataset_tag)
        if dump_prediction_to_annotation:
            self._dumped_annotations = []

        if self.dataset.batch is None:
            self.dataset.batch = self.launcher.batch
        self.preprocessor.input_shapes = self.launcher.inputs_info_for_meta()

    # pylint: disable=R0912
    def process_dataset(
            self,
            subset=None,
            num_images=None,
            check_progress=False,
            dataset_tag='',
            output_callback=None,
            allow_pairwise_subset=False,
            dump_prediction_to_annotation=False,
            calculate_metrics=True,
            **kwargs
    ):

        self._prepare_to_evaluation(dataset_tag, dump_prediction_to_annotation)
        self._create_subset(subset, num_images, allow_pairwise_subset)

        progress_reporter = None if not check_progress else self._create_progress_reporter(
            check_progress, self.dataset.size
        )

        for batch_id, (batch_input_ids, batch_annotation, batch_inputs, batch_identifiers) in enumerate(self.dataset):
            filled_inputs, batch_meta = self._get_batch_input(batch_inputs, batch_annotation)
            batch_raw_predictions = self.launcher.predict(filled_inputs, batch_meta, **kwargs)
            if self.adapter and (calculate_metrics or dump_prediction_to_annotation):
                self.adapter.output_blob = self.adapter.output_blob or self.launcher.output_blob
                batch_predictions = self.adapter.process(batch_raw_predictions, batch_identifiers, batch_meta)
            else:
                batch_predictions = batch_raw_predictions

            metrics_result = None
            if calculate_metrics:
                annotations, predictions = self.postprocessor.process_batch(
                    batch_annotation, batch_predictions, batch_meta, dump_prediction_to_annotation
                )
                if dump_prediction_to_annotation:
                    threshold = kwargs.get('annotation_conf_threshold', 0.0)
                    annotations = []
                    for prediction in predictions:
                        generated_annotation = prediction.to_annotation(threshold=threshold)
                        if generated_annotation:
                            annotations.append(generated_annotation)
                    self._dumped_annotations.extend(annotations)

                if self.metric_executor:
                    metrics_result, _ = self.metric_executor.update_metrics_on_batch(
                        batch_input_ids, annotations, predictions
                    )
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

        if dump_prediction_to_annotation:
            self.register_dumped_annotations()

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

    def _fill_free_irs(self, free_irs, queued_irs, infer_requests_pool, dataset_iterator, **kwargs):
        for ir_id in free_irs:
            try:
                batch_id, (batch_input_ids, batch_annotation, batch_inputs, batch_identifiers) = next(dataset_iterator)
            except StopIteration:
                break

            batch_input, batch_meta = self._get_batch_input(batch_inputs, batch_annotation)
            self.launcher.predict_async(infer_requests_pool[ir_id], batch_input, batch_meta,
                                        context=(batch_id, batch_input_ids, batch_annotation, batch_identifiers))
            queued_irs.append(ir_id)

        return free_irs, queued_irs

    @staticmethod
    def _create_progress_reporter(check_progress, dataset_size):
        pr_kwargs = {}
        if isinstance(check_progress, int) and not isinstance(check_progress, bool):
            pr_kwargs = {"print_interval": check_progress}

        return ProgressReporter.provide('print', dataset_size, **pr_kwargs)

    def _prepare_requests_pool(self, completion_callback):
        infer_requests_pool = {ir.request_id: ir for ir in self.launcher.get_async_requests()}
        for _, async_request in infer_requests_pool.items():
            async_request.set_completion_callback(completion_callback)

        return infer_requests_pool

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

    def load_network(self, network_list=None):
        network = next(iter(network_list))['model'] if network_list is not None else None
        self.launcher.load_network(network)
        self.input_feeder = InputFeeder(
            self.launcher.config.get('inputs', []), self.launcher.inputs,
            self.launcher.fit_to_input, self.launcher.default_layout
        )
        if self.adapter:
            self.adapter.output_blob = self.adapter.output_blob or self.launcher.output_blob

    def load_network_from_ir(self, models_list):
        model_paths = next(iter(models_list))
        xml_path, bin_path = model_paths['model'], model_paths['weights']
        self.launcher.load_ir(xml_path, bin_path)
        self.input_feeder = InputFeeder(
            self.launcher.config.get('inputs', []), self.launcher.inputs,
            self.launcher.fit_to_input, self.launcher.default_layout
        )
        if self.adapter:
            self.adapter.output_blob = self.adapter.output_blob or self.launcher.output_blob

    def get_network(self):
        return [{'model': self.launcher.network}]

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
        if self.adapter:
            self.adapter.reset()

    def release(self):
        self.launcher.release()
        self.input_feeder.release()
        if self.adapter:
            self.adapter.release()


def create_dataset_attributes(config, tag, dumped_annotations=None):
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
    if contains_any(dataset_config, ['annotation', 'annotation_conversion']) or dumped_annotations:
        annotation, meta = Dataset.load_annotation(dataset_config)
        annotation_reader = AnnotationProvider(
            annotation if not dumped_annotations else dumped_annotations, meta, dataset_name, dataset_config
        )
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
        data_source = annotation_reader
    data_reader = BaseReader.provide(data_reader_type, data_source, data_reader_config)

    metric_dispatcher = None
    dataset = DataProvider(data_reader, annotation_reader, tag=tag, dataset_config=dataset_config)
    preprocessor = PreprocessingExecutor(
        dataset_config.get('preprocessing'), dataset_name, dataset_meta
    )
    postprocessor = PostprocessingExecutor(dataset_config.get('postprocessing'), dataset_name, dataset_meta)
    if 'metrics' in dataset_config:
        metric_dispatcher = MetricsExecutor(dataset_config.get('metrics', []), annotation_reader)

    return dataset, metric_dispatcher, preprocessor, postprocessor
