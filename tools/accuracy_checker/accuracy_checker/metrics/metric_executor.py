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

import warnings
from collections import namedtuple, OrderedDict

from ..config import ConfigValidator, ConfigError, StringField
from ..presenters import BasePresenter, EvaluationResult
from .metric import Metric, FullDatasetEvaluationMetric
from .metric_profiler import ProfilingExecutor

MetricInstance = namedtuple(
    'MetricInstance', ['name', 'metric_type', 'metric_fn', 'reference', 'abs_threshold', 'rel_threshold', 'presenter']
)


class MetricsExecutor:
    """
    Class for evaluating metrics according to dataset configuration entry.
    """

    def __init__(self, metrics_config, dataset=None, state=None, ignore_dataset_meta=True, postpone_metrics=False):
        self.state = state or {}
        dataset_name = dataset.name if dataset else ''
        self._dataset = dataset
        self.profile_metrics = False if dataset is None else dataset.config.get('_profile', False)
        self.profiler_dir = None
        self.profiler = None
        if not postpone_metrics:
            if self.profile_metrics:
                profiler_type = dataset.config.get('_report_type', 'csv')
                self.profiler = ProfilingExecutor(profile_report_type=profiler_type)
                self.profiler_dir = dataset.config.get('_profiler_log_dir')
                self.profiler.set_dataset_meta(self._dataset.metadata)

            self.metrics = []
            self.need_store_predictions = False
            self._metric_names = set()
            if not metrics_config:
                raise ConfigError('{} dataset config must specify "{}"'.format(dataset_name, 'metrics'))

            for metric_config_entry in metrics_config:
                self.register_metric(metric_config_entry, ignore_dataset_meta)

    @classmethod
    def parameters(cls):
        return {
            'type': StringField(
                choices=Metric.providers, description="Metric providers: {}".format(", ".join(Metric.providers))
            )
        }

    @property
    def dataset(self):
        return self._dataset

    @dataset.setter
    def _set_dataset(self, dataset):
        self._dataset = dataset

    def update_metrics_on_object(self, annotation, prediction):
        """
        Updates metric value corresponding given annotation and prediction objects.
        """

        metric_results = []

        for metric in self.metrics:
            metric_results.append(metric.metric_fn.submit(annotation, prediction))

        return metric_results

    def update_metrics_on_batch(self, batch_ids, annotation, prediction,
                                profile=False, deprocessed_annotation=None, deprocessed_prediction=None):
        """
        Updates metric value corresponding given batch.

        Args:
            annotation: list of batch number of annotation objects.
            prediction: list of batch number of prediction objects.
        """

        results = OrderedDict()
        profile_results = OrderedDict()

        for idx, (input_id, single_annotation, single_prediction) in enumerate(zip(batch_ids, annotation, prediction)):
            if profile:
                if deprocessed_annotation is not None and deprocessed_prediction is not None:
                    self.profiler.update_annotation_and_prediction(
                        deprocessed_annotation[idx], deprocessed_prediction[idx])
            results[input_id] = self.update_metrics_on_object(single_annotation, single_prediction)
            if profile:
                profile_results[input_id] = self.profiler.get_last_report()

        return results, profile_results

    def iterate_metrics(self, annotations, predictions):
        for name, metric_type, functor, reference, abs_threshold, rel_threshold, presenter in self.metrics:
            profiling_file = None if functor.profiler is None else functor.profiler.report_file
            yield presenter, EvaluationResult(
                name=name,
                metric_type=metric_type,
                evaluated_value=functor(annotations, predictions),
                reference_value=reference,
                abs_threshold=abs_threshold,
                rel_threshold=rel_threshold,
                meta=functor.meta,
                profiling_file=profiling_file
            )

    @staticmethod
    def get_metric_result_template(metrics, ignore_refs):
        type_ = 'type'
        identifier = 'name'
        reference = 'reference'
        abs_threshold = 'abs_threshold'
        rel_threshold = 'rel_threshold'
        presenter = 'presenter'
        for metric_config in metrics:
            metric_type = metric_config.get(type_)
            metric_cls = Metric.resolve(metric_type)
            metric_meta = metric_cls.get_common_meta()

            metric_identifier = metric_config.get(identifier, metric_type)
            metric_presenter = BasePresenter.provide(metric_config.get(presenter, 'print_scalar'))
            abs_threshold_v = metric_config.get(abs_threshold)
            rel_threshold_v = metric_config.get(rel_threshold)
            reference_v = metric_config.get(reference)
            if reference_v is not None and not isinstance(reference_v, (int, float, dict)):
                raise ConfigError(
                    'reference value should be represented as number or dictionary with numbers for each submetric'
                )
            profiling_file = None
            values = None
            if reference_v is not None and isinstance(reference_v, dict):
                num_results = len(reference_v) - 1 if metric_meta.get('calculate_mean', True) else len(reference_v)
                values = [None] * num_results
            yield metric_presenter, EvaluationResult(
                name=metric_identifier,
                metric_type=metric_type,
                evaluated_value=values,
                reference_value=reference_v if not ignore_refs else None,
                abs_threshold=abs_threshold_v,
                rel_threshold=rel_threshold_v,
                meta=metric_meta,
                profiling_file=profiling_file
            )

    def register_metric(self, metric_config_entry, ignore_dataset_meta=False):
        type_ = 'type'
        identifier = 'name'
        reference = 'reference'
        abs_threshold = 'abs_threshold'
        threshold = 'threshold'
        rel_threshold = 'rel_threshold'
        presenter = 'presenter'
        metric_config_validator = ConfigValidator(
            "metrics", on_extra_argument=ConfigValidator.IGNORE_ON_EXTRA_ARGUMENT,
            fields=self.parameters()
        )
        metric_type = metric_config_entry.get(type_)
        metric_config_validator.validate(metric_config_entry, type_)

        metric_identifier = metric_config_entry.get(identifier, metric_type)
        if metric_identifier in self._metric_names:
            raise ConfigError(
                'non-unique metric identifier {}, please define metric name field with unique value'.format(
                    metric_identifier)
            )
        self._metric_names.add(metric_identifier)
        annotation_source = metric_config_entry.get('annotation_source', '')
        prediction_source = metric_config_entry.get('prediction_source', '')
        metric_kwargs = {}
        if self.profile_metrics:
            profiler = self.profiler.register_profiler_for_metric(
                metric_type, metric_identifier, annotation_source, prediction_source
            )
            metric_kwargs['profiler'] = profiler

        metric_fn = Metric.provide(
            metric_type, metric_config_entry, self.dataset, metric_identifier, state=self.state, **metric_kwargs
        )
        metric_presenter = BasePresenter.provide(metric_config_entry.get(presenter, 'print_scalar'))
        threshold_v = metric_fn.config.get(threshold)
        abs_threshold_v = metric_fn.config.get(abs_threshold)
        reference_v = metric_fn.config.get(reference)
        if reference_v is not None and not isinstance(reference_v, (int, float, dict)):
            raise ConfigError(
                'reference value should be represented as number or dictionary with numbers for each submetric'
            )
        if threshold_v is not None and abs_threshold_v is not None:
            warnings.warn(
                f'both threshold and abs_threshold are provided for metric {metric_identifier}. '
                f'threshold will be ignored'
            )
        if abs_threshold_v is None:
            abs_threshold_v = threshold_v
        if threshold_v is not None:
            warnings.warn('threshold option is deprecated. Please use abs_threshold instead', DeprecationWarning)

        self.metrics.append(MetricInstance(
            metric_identifier,
            metric_type,
            metric_fn,
            reference_v,
            abs_threshold_v,
            metric_config_entry.get(rel_threshold),
            metric_presenter
        ))
        if isinstance(metric_fn, FullDatasetEvaluationMetric):
            self.need_store_predictions = True

    def enable_profiling(self, dataset, report_type=None):
        profiler_type = dataset.config.get('_report_type', 'csv') if report_type is None else report_type
        self.profiler = ProfilingExecutor(profile_report_type=profiler_type)
        self.profiler.set_dataset_meta(self._dataset.metadata)
        for metric in self.metrics:
            annotation_source = metric.metric_fn.config.get('annotation_source', '')
            prediction_source = metric.metric_fn.config.get('prediction_source', '')
            profiler = self.profiler.register_profiler_for_metric(
                metric.metric_type, metric.name, annotation_source, prediction_source
            )
            metric.metric_fn.set_profiler(profiler)

    def get_metric_presenters(self):
        return [metric.presenter for metric in self.metrics]

    def get_metrics_direction(self):
        return {metric.name: metric.metric_fn.meta.get('target', 'higher-better') for metric in self.metrics}

    def get_metrics_attributes(self):
        return {
            metric.name: {
                'direction': metric.metric_fn.meta.get('target', 'higher-better'),
                'type': metric.metric_type
            } for metric in self.metrics
        }

    def set_profiling_dir(self, profiler_dir):
        if self.profiler:
            self.profiler.set_profiling_dir(profiler_dir)
        self.profiler_dir = profiler_dir

    def set_processing_info(self, processing_info):
        if self.profiler:
            self.profiler.set_executing_info(processing_info)

    def reset(self):
        for metric in self.metrics:
            metric.metric_fn.reset()

    @classmethod
    def validate_config(cls, metrics, fetch_only=False, uri_prefix=''):
        metrics_uri = uri_prefix or 'metrics'
        if not metrics:
            if fetch_only:
                upper_level_uri = (
                    metrics_uri.replace('.metrics', '') if metrics_uri.endswith('.metrics') else metrics_uri
                )
                return [ConfigError("Metrics are not provided", metrics, upper_level_uri)]
        errors = []
        metric_ids = set()
        for metric_id, metric in enumerate(metrics):
            metric_uri = '{}.{}'.format(metrics_uri, metric_id)
            errors.extend(Metric.validate_config(metric, fetch_only=fetch_only, uri_prefix=metric_uri))
            if 'type' not in metric:
                continue
            metric_name = metric.get('name', metric['type'])
            if metric_name in metric_ids:
                error = ConfigError(f'non-unique metric identifier {metric_name}, '
                                    f'please define metric name field with unique value', metric, metric_uri
                                    )
                if not fetch_only:
                    raise error
                errors.append(error)
            metric_ids.add(metric_name)

        return errors
