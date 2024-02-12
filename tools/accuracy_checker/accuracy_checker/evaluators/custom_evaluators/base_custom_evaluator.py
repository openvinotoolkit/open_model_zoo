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

from ..base_evaluator import BaseEvaluator
from ...progress_reporters import ProgressReporter
from ..quantization_model_evaluator import create_dataset_attributes
from ...launcher import create_launcher


# base class for custom evaluators
class BaseCustomEvaluator(BaseEvaluator):
    def __init__(self, dataset_config, launcher, orig_config):
        self.dataset_config = dataset_config
        self.dataset = None
        self.preprocessing_executor = None
        self.preprocessor = None
        self.postprocessor = None
        self.metric_executor = None
        self.launcher = launcher
        self._metrics_results = []
        self.config = orig_config
        self._annotations = []
        self._predictions = []
        self.adapter_type = None
        self.model = None
        self._part_by_name = None

    @staticmethod
    def get_dataset_and_launcher_info(config):
        dataset_config = config['datasets']
        launcher_config = config['launchers'][0]
        if launcher_config['framework'] == 'dlsdk' and 'device' not in launcher_config:
            launcher_config['device'] = 'CPU'
        launcher = create_launcher(launcher_config, delayed_model_loading=True)
        return dataset_config, launcher, launcher_config

    def process_dataset(self, subset=None, num_images=None, check_progress=False, dataset_tag='',
                        output_callback=None, allow_pairwise_subset=False, dump_prediction_to_annotation=False,
                        calculate_metrics=True, **kwargs):
        self._prepare_dataset(dataset_tag)
        self._create_subset(subset, num_images, allow_pairwise_subset)
        metric_config = self.configure_intermediate_metrics_results(kwargs)

        if 'progress_reporter' in kwargs:
            _progress_reporter = kwargs['progress_reporter']
            if _progress_reporter is not None:
                _progress_reporter.reset(self.dataset.size)
        else:
            _progress_reporter = None if not check_progress else self._create_progress_reporter(
                check_progress, self.dataset.size
            )

        self._process(output_callback, calculate_metrics, _progress_reporter, metric_config, kwargs.get('csv_result'))

        if _progress_reporter:
            _progress_reporter.finish()

    def _prepare_dataset(self, dataset_tag=''):
        if self.dataset is None or (dataset_tag and self.dataset.tag != dataset_tag):
            self.select_dataset(dataset_tag)

        if self.dataset.batch is None:
            self.dataset.batch = 1

    def select_dataset(self, dataset_tag):
        if self.dataset is not None and isinstance(self.dataset_config, list):
            return
        dataset_attributes = create_dataset_attributes(self.dataset_config, dataset_tag)
        self.dataset, self.metric_executor, self.preprocessor, self.postprocessor = dataset_attributes

    def _create_subset(self, subset=None, num_images=None, allow_pairwise=False):
        if subset is not None:
            self.dataset.make_subset(ids=subset, accept_pairs=allow_pairwise)
        elif num_images is not None:
            self.dataset.make_subset(end=num_images, accept_pairs=allow_pairwise)

    @staticmethod
    def configure_intermediate_metrics_results(config):
        compute_intermediate_metric_res = config.get('intermediate_metrics_results', False)
        metric_interval, ignore_results_formatting, ignore_metric_reference = None, None, None
        if compute_intermediate_metric_res:
            metric_interval = config.get('metrics_interval', 1000)
            ignore_results_formatting = config.get('ignore_results_formatting', False)
            ignore_metric_reference = config.get('ignore_metric_reference', False)
        return compute_intermediate_metric_res, metric_interval, ignore_results_formatting, ignore_metric_reference

    def _process(self, output_callback, calculate_metrics, progress_reporter, metric_config, csv_file):
        pass

    def _get_metrics_result(self, input_ids, annotation, prediction, calculate_metrics):
        metrics_result = None
        if self.metric_executor and calculate_metrics:
            metrics_result, _ = self.metric_executor.update_metrics_on_batch(input_ids, annotation, prediction)
            if self.metric_executor.need_store_predictions:
                self._annotations.extend(annotation)
                self._predictions.extend(prediction)
        return metrics_result

    def _update_progress(self, progress_reporter, metric_config, batch_id, prediction_size, csv_file):
        (compute_intermediate_metric_res, metric_interval, ignore_results_formatting,
         ignore_metric_reference) = metric_config
        if progress_reporter:
            progress_reporter.update(batch_id, prediction_size)
            if compute_intermediate_metric_res and progress_reporter.current % metric_interval == 0:
                self.compute_metrics(
                    print_results=True, ignore_results_formatting=ignore_results_formatting,
                    ignore_metric_reference=ignore_metric_reference
                )
                self.write_results_to_csv(csv_file, ignore_results_formatting, metric_interval)

    def compute_metrics(self, print_results=True, ignore_results_formatting=False, ignore_metric_reference=False):
        if self._metrics_results:
            del self._metrics_results
            self._metrics_results = []
        for result_presenter, evaluated_metric in self.metric_executor.iterate_metrics(
                self._annotations, self._predictions):
            self._metrics_results.append(evaluated_metric)
            if print_results:
                result_presenter.write_result(evaluated_metric, ignore_results_formatting, ignore_metric_reference)
        return self._metrics_results

    def print_metrics_results(self, ignore_results_formatting=False, ignore_metric_reference=False):
        if not self._metrics_results:
            self.compute_metrics(True, ignore_results_formatting, ignore_metric_reference)
            return
        result_presenters = self.metric_executor.get_metric_presenters()
        for presenter, metric_result in zip(result_presenters, self._metrics_results):
            presenter.write_result(metric_result, ignore_results_formatting, ignore_metric_reference)

    def extract_metrics_results(self, print_results=True, ignore_results_formatting=False,
                                ignore_metric_reference=False):
        if not self._metrics_results:
            self.compute_metrics(False, ignore_results_formatting, ignore_metric_reference)
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
                presenter.write_result(metric_result, ignore_results_formatting, ignore_metric_reference)
        return extracted_results, extracted_meta

    def register_metric(self, metric_config):
        if isinstance(metric_config, str):
            self.metric_executor.register_metric({'type': metric_config})
        elif isinstance(metric_config, dict):
            self.metric_executor.register_metric(metric_config)
        else:
            raise ValueError('Unsupported metric configuration type {}'.format(type(metric_config)))

    def get_metrics_attributes(self):
        if not self.metric_executor:
            return {}
        return self.metric_executor.get_metrics_attributes()

    def set_profiling_dir(self, profiler_dir):
        self.metric_executor.set_profiling_dir(profiler_dir)

    @property
    def dataset_size(self):
        return self.dataset.size

    @staticmethod
    def _create_progress_reporter(check_progress, dataset_size):
        pr_kwargs = {}
        if isinstance(check_progress, int) and not isinstance(check_progress, bool):
            pr_kwargs = {"print_interval": check_progress}
        return ProgressReporter.provide('print', dataset_size, **pr_kwargs)

    def send_processing_info(self, sender):
        if not sender:
            return {}
        model_type = None
        details = {}
        metrics = self.dataset_config[0].get('metrics', [])
        metric_info = [metric['type'] for metric in metrics]
        details.update({
            'metrics': metric_info,
            'model_file_type': model_type,
            'adapter': self.adapter_type,
        })
        if self.dataset is None:
            self.select_dataset('')
        details.update(self.dataset.send_annotation_info(self.dataset_config[0]))
        return details

    @staticmethod
    def get_processing_info(config):
        module_specific_params = config.get('module_config')
        model_name = config['name']
        dataset_config = module_specific_params['datasets'][0]
        launcher_config = module_specific_params['launchers'][0]
        return (
            model_name, launcher_config['framework'], launcher_config.get('device', 'CPU'), launcher_config.get('tags'),
            dataset_config['name']
        )

    def reset(self):
        if self.metric_executor:
            self.metric_executor.reset()
        if hasattr(self, '_annotations'):
            del self._annotations
            del self._predictions
        del self._metrics_results
        self._annotations = []
        self._predictions = []
        self._metrics_results = []
        if self.dataset:
            self.dataset.reset(self.postprocessor.has_processors)

    def release(self):
        self._release_model()
        self.launcher.release()

    def _release_model(self):
        if self.model:
            self.model.release()
        if self._part_by_name:
            for model in self._part_by_name.values():
                model.release()

    def set_launcher_property(self, property_dict):
        self.launcher.ie_core.set_property(property_dict)

    def register_postprocessor(self, postprocessing_config):
        pass

    def register_dumped_annotations(self):
        pass

    def load_network(self, network=None):
        if self.model:
            self.model.load_network(network, self.launcher)

    def load_network_from_ir(self, models_list):
        if self.model:
            self.model.load_model(models_list, self.launcher)

    def get_network(self):
        if self.model:
            return self.model.get_network()
        return []
