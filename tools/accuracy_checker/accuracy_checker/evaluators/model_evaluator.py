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
import pickle
import platform

from ..utils import get_path, extract_image_representations, is_path
from ..dataset import Dataset
from ..launcher import create_launcher, DummyLauncher, InputFeeder, Launcher
from ..launcher.loaders import StoredPredictionBatch
from ..logging import print_info, warning
from ..metrics import MetricsExecutor
from ..postprocessor import PostprocessingExecutor
from ..preprocessor import PreprocessingExecutor
from ..adapters import create_adapter, Adapter
from ..config import ConfigError, StringField
from ..data_readers import BaseReader, DataRepresentation
from .base_evaluator import BaseEvaluator
from .quantization_model_evaluator import create_dataset_attributes


# pylint: disable=W0223,R0904
class ModelEvaluator(BaseEvaluator):
    def __init__(
            self, launcher, input_feeder, adapter, preprocessor, postprocessor, dataset, metric, async_mode, config
    ):
        self.config = config
        self.launcher = launcher
        self.input_feeder = input_feeder
        self.adapter = adapter
        self.preprocessor = preprocessor
        self.postprocessor = postprocessor
        self.dataset = dataset
        self.metric_executor = metric
        self.process_dataset = self.process_dataset_sync if not async_mode else self.process_dataset_async
        self.async_mode = async_mode

        self._annotations = []
        self._predictions = []
        self._metrics_results = []

    @classmethod
    def from_configs(cls, model_config, delayed_annotation_loading=False):
        model_name = model_config['name']
        launcher_config = model_config['launchers'][0]
        dataset_config = model_config['datasets'][0]
        dataset_name = dataset_config['name']

        postpone_model_loading = (
            not model_config.get('_store_only', False) and cls._is_stored(model_config.get('_stored_data'))
        )

        dataset = Dataset(dataset_config) if not delayed_annotation_loading else None
        dataset_metadata = dataset.metadata if dataset is not None else {}
        launcher_kwargs = {'delayed_model_loading': postpone_model_loading}
        enable_ie_preprocessing = (
            dataset_config.get('_ie_preprocessing', False)
            if launcher_config['framework'] == 'dlsdk' else False
        )
        preprocessor = PreprocessingExecutor(
            dataset_config.get('preprocessing'), dataset_name, dataset_metadata,
            enable_ie_preprocessing=enable_ie_preprocessing
        )
        input_precision = launcher_config.get('_input_precision', [])
        if enable_ie_preprocessing:
            launcher_kwargs['preprocessor'] = preprocessor
        if launcher_config['framework'] == 'dummy' and launcher_config.get('provide_identifiers', False):
            launcher_kwargs = {'identifiers': dataset.identifiers if dataset is not None else []}
        if input_precision:
            launcher_kwargs['postpone_inputs_configuration'] = True
        launcher = create_launcher(launcher_config, model_name, **launcher_kwargs)
        async_mode = launcher.async_mode if hasattr(launcher, 'async_mode') else False
        config_adapter = launcher_config.get('adapter')
        adapter = None if not config_adapter else create_adapter(config_adapter, launcher, dataset)
        launcher_inputs = launcher.inputs if not postpone_model_loading else {}
        input_feeder = InputFeeder(
            launcher.config.get('inputs', []), launcher_inputs, launcher.fit_to_input, launcher.default_layout,
            launcher_config['framework'] == 'dummy' or postpone_model_loading, input_precision
        )
        if not postpone_model_loading:
            if input_precision:
                launcher.update_input_configuration(input_feeder.inputs_config)
            preprocessor.input_shapes = launcher.inputs_info_for_meta()
        postprocessor = PostprocessingExecutor(dataset_config.get('postprocessing'), dataset_name, dataset_metadata)
        metric_dispatcher = None
        if not delayed_annotation_loading:
            metric_dispatcher = MetricsExecutor(dataset_config.get('metrics', []), dataset)
            if metric_dispatcher.profile_metrics:
                metric_dispatcher.set_processing_info(ModelEvaluator.get_processing_info(model_config))

        return cls(
            launcher, input_feeder, adapter,
            preprocessor, postprocessor, dataset, metric_dispatcher, async_mode, model_config
        )

    @classmethod
    def validate_config(cls, model_config, delayed_annotation_loading=False):
        uri_prefix = ''
        if 'models' in model_config:
            model_config = model_config['models'][0]
            uri_prefix = 'models'
        config_errors = []
        if 'launchers' not in model_config or not model_config['launchers']:
            config_errors.append(
                ConfigError(
                    'launchers section is not provided', model_config.get('launchers', []),
                    '{}.launchers'.format(uri_prefix) if uri_prefix else 'launchers', validation_scheme=Launcher
                )
            )
        else:
            for launcher_id, launcher_config in enumerate(model_config['launchers']):
                adapter_config = launcher_config.get('adapter')
                launcher_uri_prefix = '{}.launchers.{}'.format(
                    uri_prefix, launcher_id) if uri_prefix else 'launchers.{}'.format(launcher_id)
                config_errors.extend(
                    Launcher.validate_config(launcher_config, fetch_only=True, uri_prefix=launcher_uri_prefix))
                if adapter_config:
                    adapter_uri = '{}.adapter'.format(launcher_uri_prefix)
                    config_errors.extend(
                        Adapter.validate_config(adapter_config, fetch_only=True, uri_prefix=adapter_uri))

        datasets_uri = '{}.datasets'.format(uri_prefix) if uri_prefix else 'datasets'
        if 'datasets' not in model_config or (not delayed_annotation_loading and not model_config['datasets']):
            config_errors.append(
                ConfigError(
                    'datasets section is not provided', model_config.get('datasets', []),
                    datasets_uri, validation_scheme=Dataset
                )
            )
        else:
            for dataset_id, dataset_config in enumerate(model_config['datasets']):
                data_reader_config = dataset_config.get('reader', 'opencv_imread')
                current_dataset_uri = '{}.{}'.format(datasets_uri, dataset_id)
                if not delayed_annotation_loading:
                    config_errors.extend(
                        Dataset.validate_config(dataset_config, fetch_only=True, uri_prefix=current_dataset_uri)
                    )
                    config_errors.extend(
                        MetricsExecutor.validate_config(
                            dataset_config.get('metrics', []), fetch_only=True,
                            uri_prefix='{}.metrics'.format(current_dataset_uri))
                    )

                config_errors.extend(
                    BaseReader.validate_config(
                        data_reader_config, data_source=dataset_config.get('data_source'), fetch_only=True,
                        uri_prefix='{}.reader'.format(current_dataset_uri),
                        check_data_source=not delayed_annotation_loading,
                        check_reader_type=delayed_annotation_loading
                    )
                )

                config_errors.extend(
                    PreprocessingExecutor.validate_config(
                        dataset_config.get('preprocessing'), fetch_only=True,
                        uri_prefix='{}.preprocessing'.format(current_dataset_uri)
                    )
                )
                config_errors.extend(
                    PostprocessingExecutor.validate_config(
                        dataset_config.get('postprocessing'), fetch_only=True,
                        uri_prefix='{}.postprocessing'.format(current_dataset_uri)
                    )
                )

        return config_errors

    @classmethod
    def validation_scheme(cls):
        return {'models': [
            {'name': StringField(description='model name'),
             'launchers': Launcher,
             'datasets': Dataset
             }]}

    @staticmethod
    def get_processing_info(config):
        launcher_config = config['launchers'][0]
        dataset_config = config['datasets'][0]

        return (
            config['name'],
            launcher_config['framework'], launcher_config.get('device', ''), launcher_config.get('tags'),
            dataset_config['name']
        )

    def send_processing_info(self, sender):
        if not sender:
            return {}
        launcher_config = self.config['launchers'][0]
        dataset_config = self.config['datasets'][0]
        framework = launcher_config['framework']
        device = launcher_config.get('device', 'CPU')
        details = {
            'platform': platform.system,
            'framework': framework if framework != 'dlsdk' else 'openvino',
            'device': device.upper(),
            'inference_model': 'sync' if not self.async_mode else 'async'
        }
        model_type = None

        if hasattr(self.launcher, 'get_model_file_type'):
            model_type = self.launcher.get_model_file_type()
        adapter = launcher_config.get('adapter')
        adapter_type = None
        if adapter:
            adapter_type = adapter if isinstance(adapter, str) else adapter.get('type')
        metrics = dataset_config.get('metrics', [])
        metric_info = [metric['type'] for metric in metrics]
        details.update({
            'metrics': metric_info,
            'model_file_type': model_type,
            'adapter': adapter_type,
        })
        details.update(self.dataset.send_annotation_info(dataset_config))
        return details

    def _get_batch_input(self, batch_annotation, batch_input):
        batch_input = self.preprocessor.process(batch_input, batch_annotation)
        batch_meta = extract_image_representations(batch_input, meta_only=True)
        filled_inputs = self.input_feeder.fill_inputs(batch_input)

        return filled_inputs, batch_meta

    def process_dataset_async(self, stored_predictions, progress_reporter, *args, **kwargs):
        def completion_callback(status_code, request_id):
            if status_code:
                warning('Request {} failed with status code {}'.format(request_id, status_code))
            queued_irs.remove(request_id)
            ready_irs.append(request_id)

        def prepare_dataset(store_only_mode):
            if self.dataset is None:
                raise ConfigError('dataset entry is not assigned for execution')

            if self.dataset.batch is None:
                self.dataset.batch = self.launcher.batch
            if progress_reporter:
                progress_reporter.reset(self.dataset.size)
            if self._is_stored(stored_predictions) and store_only_mode:
                self._reset_stored_predictions(stored_predictions)

        store_only = kwargs.get('store_only', False)
        prepare_dataset(store_only)

        if (
                self.launcher.allow_reshape_input or self.input_feeder.lstm_inputs or
                self.preprocessor.has_multi_infer_transformations or
                self.dataset.multi_infer
        ):
            warning('Model can not to be processed in async mode. Switched to sync.')
            return self.process_dataset_sync(stored_predictions, progress_reporter, *args, **kwargs)

        if (self._is_stored(stored_predictions) or isinstance(self.launcher, DummyLauncher)) and not store_only:
            return self._load_stored_predictions(stored_predictions, progress_reporter)

        output_callback = kwargs.get('output_callback')
        metric_config = self._configure_metrics(kwargs, output_callback)
        _, compute_intermediate_metric_res, metric_interval, ignore_results_formatting = metric_config
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
                    free_irs.append(ready_ir_id)
                    if stored_predictions:
                        self.prepare_prediction_to_store(
                            batch_raw_predictions, batch_identifiers, batch_meta, stored_predictions
                        )
                    if not store_only:
                        self._process_batch_results(
                            batch_raw_predictions, batch_annotation, batch_identifiers,
                            batch_input_ids, batch_meta, False, output_callback)

                    if progress_reporter:
                        progress_reporter.update(batch_id, len(batch_identifiers))
                        if compute_intermediate_metric_res and progress_reporter.current % metric_interval == 0:
                            self.compute_metrics(
                                print_results=True, ignore_results_formatting=ignore_results_formatting
                            )

        if progress_reporter:
            progress_reporter.finish()

        if stored_predictions:
            print_info("prediction objects are save to {}".format(stored_predictions))

    def process_dataset_sync(self, stored_predictions, progress_reporter, *args, **kwargs):
        if self.dataset is None:
            raise ConfigError('dataset entry is not assigned for evaluation')

        if progress_reporter:
            progress_reporter.reset(self.dataset.size)
        store_only = kwargs.get('store_only', False)

        if (self._is_stored(stored_predictions) or isinstance(self.launcher, DummyLauncher)) and not store_only:
            self._load_stored_predictions(stored_predictions, progress_reporter)
            return self._annotations, self._predictions

        if self.dataset.batch is None:
            self.dataset.batch = self.launcher.batch
        if store_only and self._is_stored(stored_predictions):
            self._reset_stored_predictions(stored_predictions)
        output_callback = kwargs.get('output_callback')
        metric_config = self._configure_metrics(kwargs, output_callback)
        enable_profiling, compute_intermediate_metric_res, metric_interval, ignore_results_formatting = metric_config
        for batch_id, (batch_input_ids, batch_annotation, batch_input, batch_identifiers) in enumerate(self.dataset):
            filled_inputs, batch_meta = self._get_batch_input(batch_annotation, batch_input)
            batch_predictions = self.launcher.predict(filled_inputs, batch_meta, **kwargs)
            if stored_predictions:
                self.prepare_prediction_to_store(batch_predictions, batch_identifiers, batch_meta, stored_predictions)
            if not store_only:
                self._process_batch_results(
                    batch_predictions, batch_annotation, batch_identifiers,
                    batch_input_ids, batch_meta, enable_profiling, output_callback)

            if progress_reporter:
                progress_reporter.update(batch_id, len(batch_identifiers))
                if compute_intermediate_metric_res and progress_reporter.current % metric_interval == 0:
                    self.compute_metrics(print_results=True, ignore_results_formatting=ignore_results_formatting)

        if progress_reporter:
            progress_reporter.finish()

        if stored_predictions:
            print_info("prediction objects are save to {}".format(stored_predictions))
        return self._annotations, self._predictions

    def _process_batch_results(
            self, batch_predictions, batch_annotations, batch_identifiers, batch_input_ids, batch_meta,
            enable_profiling=False, output_callback=None):
        if self.adapter:
            self.adapter.output_blob = self.adapter.output_blob or self.launcher.output_blob
            batch_predictions = self.adapter.process(batch_predictions, batch_identifiers, batch_meta)

        annotations, predictions = self.postprocessor.process_batch(
            batch_annotations, batch_predictions, batch_meta
        )
        _, profile_result = self.metric_executor.update_metrics_on_batch(
            batch_input_ids, annotations, predictions, enable_profiling
        )
        if output_callback:
            callback_kwargs = {'profiling_result': profile_result} if enable_profiling else {}
            output_callback(annotations, predictions, **callback_kwargs)

        if self.metric_executor.need_store_predictions:
            self._annotations.extend(annotations)
            self._predictions.extend(predictions)

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
        predictions = self.load(stored_predictions, progress_reporter)
        annotations = self.dataset.annotation
        if self.postprocessor.has_processors:
            print_info("Postprocess results:")
            self.dataset.provide_data_info(annotations, progress_reporter)
            annotations, predictions = self.postprocessor.full_process(annotations, predictions)
        self.metric_executor.update_metrics_on_batch(
            range(len(annotations)), annotations, predictions
        )
        if self.metric_executor.need_store_predictions:
            self._annotations = annotations
            self._predictions = predictions

        return annotations, predictions

    def _fill_free_irs(self, free_irs, queued_irs, infer_requests_pool, dataset_iterator):
        for ir_id in free_irs:
            try:
                batch_id, (batch_input_ids, batch_annotation, batch_input, _) = next(dataset_iterator)
            except StopIteration:
                break

            batch_input, batch_meta = self._get_batch_input(batch_annotation, batch_input)
            self.launcher.predict_async(infer_requests_pool[ir_id], batch_input, batch_meta,
                                        context=(batch_id, batch_input_ids, batch_annotation))
            queued_irs.append(ir_id)

        return free_irs, queued_irs

    def process_single_image(self, image):
        if self.dataset is None and not hasattr(self, '_reader'):
            data_reader_config = self.config['datasets'][0].get('reader', 'opencv_imread')
            data_source = None
            if isinstance(data_reader_config, str):
                data_reader_type = data_reader_config
                data_reader_config = None
            elif isinstance(data_reader_config, dict):
                data_reader_type = data_reader_config['type']
            else:
                raise ConfigError('reader should be dict or string')
            self._reader = BaseReader.provide(
                data_reader_type, data_source, data_reader_config, postpone_data_source=True
            )
        elif not hasattr(self, '_reader'):
            self._reader = self.dataset.data_provider.data_reader
        input_data = self._prepare_data_for_single_inference(image)
        batch_input = self.preprocessor.process(input_data)
        batch_meta = extract_image_representations(batch_input, meta_only=True)
        filled_inputs = self.input_feeder.fill_inputs(batch_input)
        batch_predictions = self.launcher.predict(filled_inputs, batch_meta)

        if self.adapter:
            self.adapter.output_blob = self.adapter.output_blob or self.launcher.output_blob
            batch_predictions = self.adapter.process(batch_predictions, [image], batch_meta)

        _, predictions = self.postprocessor.process_batch(
            None, batch_predictions, batch_meta, allow_empty_annotation=True
        )
        self.input_feeder.ordered_inputs = False
        return predictions[0]

    def _prepare_data_for_single_inference(self, data):
        def get_data(image, create_representation=True):
            if is_path(image):
                return [
                    DataRepresentation(self._reader.read_dispatcher(image), identifier=image)
                    if create_representation else self._reader.read_dispatcher(image)]
            return [DataRepresentation(image, identifier='image') if create_representation else image]

        if not isinstance(data, list):
            return get_data(data)

        input_data = []
        for item in data:
            input_data.extend(get_data(item, False))
        self.input_feeder.ordered_inputs = True

        return [DataRepresentation(input_data, identifier=list(range(len(data))))]

    def select_dataset(self, dataset_config):
        dataset_attributes = create_dataset_attributes(dataset_config, '', False)
        self.dataset, self.metric_executor, self.preprocessor, self.postprocessor = dataset_attributes
        if self.dataset.annotation_provider and self.dataset.annotation_provider.metadata:
            self.adapter.label_map = self.dataset.annotation_provider.metadata.get('label_map')

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
        launcher = self.launcher
        identifiers = self.dataset.identifiers
        if not isinstance(launcher, DummyLauncher):
            launcher = DummyLauncher({
                'framework': 'dummy',
                'loader': 'pickle',
                'data_path': stored_predictions,
            }, adapter=self.adapter, identifiers=identifiers, progress=progress_reporter)

        predictions = launcher.predict(identifiers)
        if progress_reporter:
            progress_reporter.finish(False)

        return predictions

    def prepare_prediction_to_store(self, batch_predictions, batch_identifiers, batch_meta, stored_predictions):
        prediction_to_store = StoredPredictionBatch(batch_predictions, batch_identifiers, batch_meta)
        self.store_predictions(stored_predictions, prediction_to_store)

    @property
    def metrics_results(self):
        if not self.metrics_results:
            self.compute_metrics(print_results=False)
        computed_metrics = copy.deepcopy(self._metrics_results)
        return computed_metrics

    def set_profiling_dir(self, profiler_dir):
        self.metric_executor.set_profiling_dir(profiler_dir)

    def _configure_metrics(self, config, output_callback):
        store_only = config.get('store_only', False)
        enable_profiling = config.get('profile', False)
        profile_type = 'json' if output_callback and enable_profiling else config.get('profile_report_type')
        if enable_profiling:
            if not store_only:
                self.metric_executor.enable_profiling(self.dataset, profile_type)
            else:
                warning("Metric profiling disabled for prediction storing mode")
                enable_profiling = False
        compute_intermediate_metric_res = config.get('intermediate_metrics_results', False)
        if compute_intermediate_metric_res and store_only:
            warning("Metric calculation disabled for prediction storing mode")
            compute_intermediate_metric_res = False
        metric_interval, ignore_results_formatting = None, None
        if compute_intermediate_metric_res:
            metric_interval = config.get('metrics_interval', 1000)
            ignore_results_formatting = config.get('ignore_results_formatting', False)
        return enable_profiling, compute_intermediate_metric_res, metric_interval, ignore_results_formatting

    @staticmethod
    def store_predictions(stored_predictions, predictions):
        # since at the first time file does not exist and then created we can not use it as a pathlib.Path object
        with open(stored_predictions, "ab") as content:
            pickle.dump(predictions, content)

    @staticmethod
    def _reset_stored_predictions(stored_predictions):
        with open(stored_predictions, 'wb'):
            print_info("File {} will be cleared for storing predictions".format(stored_predictions))

    @property
    def dataset_size(self):
        return self.dataset.size

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
        if self.adapter:
            self.adapter.reset()

    def release(self):
        self.input_feeder.release()
        self.launcher.release()
        if self.adapter:
            self.adapter.release()
