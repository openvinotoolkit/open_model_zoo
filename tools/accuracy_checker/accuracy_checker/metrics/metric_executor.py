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

from collections import namedtuple, OrderedDict

from ..presenters import BasePresenter, EvaluationResult
from ..config import StringField
from .metric import Metric, FullDatasetEvaluationMetric
from ..config import ConfigValidator, ConfigError

MetricInstance = namedtuple(
    'MetricInstance', ['name', 'metric_type', 'metric_fn', 'reference', 'threshold', 'presenter']
)


class MetricsExecutor:
    """
    Class for evaluating metrics according to dataset configuration entry.
    """

    def __init__(self, metrics_config, dataset=None, state=None):
        self.state = state or {}
        dataset_name = dataset.name if dataset else ''
        message_prefix = '{}'.format(dataset_name)
        if not metrics_config:
            raise ConfigError('{} dataset config must specify "{}"'.format(message_prefix, 'metrics'))

        self._dataset = dataset

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
        for metric in self.metrics:
            metric.metric_fn.dataset = dataset

    def __call__(self, context, *args, **kwargs):
        self.update_metrics_on_batch(
            context.input_ids_batch, context.annotation_batch, context.prediction_batch
        )
        context.annotations.extend(context.annotation_batch)
        context.predictions.extend(context.prediction_batch)

    def update_metrics_on_object(self, annotation, prediction):
        """
        Updates metric value corresponding given annotation and prediction objects.
        """

        metric_results = []

        for metric in self.metrics:
            metric_results.append(metric.metric_fn.submit(annotation, prediction))

        return metric_results

    def update_metrics_on_batch(self, batch_ids, annotation, prediction):
        """
        Updates metric value corresponding given batch.

        Args:
            annotation: list of batch number of annotation objects.
            prediction: list of batch number of prediction objects.
        """

        results = OrderedDict()

        for input_id, single_annotation, single_prediction in zip(batch_ids, annotation, prediction):
            results[input_id] = self.update_metrics_on_object(single_annotation, single_prediction)

        return results

    def iterate_metrics(self, annotations, predictions):
        for name, metric_type, functor, reference, threshold, presenter in self.metrics:
            yield presenter, EvaluationResult(
                name=name,
                metric_type=metric_type,
                evaluated_value=functor(annotations, predictions),
                reference_value=reference,
                threshold=threshold,
                meta=functor.meta,
            )

    def register_metric(self, metric_config_entry):
        type_ = 'type'
        identifier = 'name'
        reference = 'reference'
        threshold = 'threshold'
        presenter = 'presenter'
        metric_config_validator = ConfigValidator(
            "metrics", on_extra_argument=ConfigValidator.IGNORE_ON_EXTRA_ARGUMENT,
            fields=self.parameters()
        )
        metric_type = metric_config_entry.get(type_)
        metric_config_validator.validate(metric_config_entry, type_)

        metric_identifier = metric_config_entry.get(identifier, metric_type)

        metric_fn = Metric.provide(
            metric_type, metric_config_entry, self.dataset, metric_identifier, state=self.state
        )
        metric_presenter = BasePresenter.provide(metric_config_entry.get(presenter, 'print_scalar'))

        self.metrics.append(MetricInstance(
            metric_identifier,
            metric_type,
            metric_fn,
            metric_config_entry.get(reference),
            metric_config_entry.get(threshold),
            metric_presenter
        ))
        if isinstance(metric_fn, FullDatasetEvaluationMetric):
            self.need_store_predictions = True

    def get_metric_presenters(self):
        return [metric.presenter for metric in self.metrics]

    def get_metrics_direction(self):
        return {metric.name: metric.metric_fn.meta.get('target', 'higher-better') for metric in self.metrics}

    def get_metrics_attributes(self):
        return {
            metric.name: {
                'direction':  metric.metric_fn.meta.get('target', 'higher-better'),
                'type': metric.metric_type
            } for metric in self.metrics
        }

    def reset(self):
        for metric in self.metrics:
            metric.metric_fn.reset()
