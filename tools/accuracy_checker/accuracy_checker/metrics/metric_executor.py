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

    def __init__(self, metrics_config, dataset=None, state=None):
        self.state = state or {}
        dataset_name = dataset.name if dataset else ''
        if not metrics_config:
            raise ConfigError('{} dataset config must specify "{}"'.format(dataset_name, 'metrics'))

        self._dataset = dataset
        self.profile_metrics = False if dataset is None else dataset.config.get('_profile', False)
        if self.profile_metrics:
            profiler_type = dataset.config.get('_report_type', 'csv')
            self.profiler = ProfilingExecutor(profile_report_type=profiler_type)
            self.profiler.set_dataset_meta(self._dataset.metadata)

        self.metrics = []
        self.need_store_predictions = False
        for metric_config_entry in metrics_config:
            self.register_metric(metric_config_entry)

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

    def update_metrics_on_batch(self, batch_ids, annotation, prediction, profile=False):
        """
        Updates metric value corresponding given batch.

        Args:
            annotation: list of batch number of annotation objects.
            prediction: list of batch number of prediction objects.
        """

        results = OrderedDict()
        profile_results = OrderedDict()

        for input_id, single_annotation, single_prediction in zip(batch_ids, annotation, prediction):
            results[input_id] = self.update_metrics_on_object(single_annotation, single_prediction)
            if profile:
                profile_results[input_id] = self.profiler.get_last_report()

        return results, profile_results

    def iterate_metrics(self, annotations, predictions):
        for name, metric_type, functor, reference, abs_threshold, rel_threshold, presenter in self.metrics:
            yield presenter, EvaluationResult(
                name=name,
                metric_type=metric_type,
                evaluated_value=functor(annotations, predictions),
                reference_value=reference,
                abs_threshold=abs_threshold,
                rel_threshold=rel_threshold,
                meta=functor.meta,
            )

    def register_metric(self, metric_config_entry):
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
        threshold_v = metric_config_entry.get(threshold)
        abs_threshold_v = metric_config_entry.get(abs_threshold)
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
            metric_config_entry.get(reference),
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
        self.profiler.set_profiling_dir(profiler_dir)

    def set_processing_info(self, processing_info):
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
        for metric_id, metric in enumerate(metrics):
            metric_uri = '{}.{}'.format(metrics_uri, metric_id)
            errors.extend(Metric.validate_config(metric, fetch_only=fetch_only, uri_prefix=metric_uri))

        return errors
