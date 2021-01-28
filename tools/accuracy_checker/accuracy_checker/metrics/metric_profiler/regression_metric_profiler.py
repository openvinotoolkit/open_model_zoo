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

import numpy as np
from .base_profiler import MetricProfiler


class RegressionMetricProfiler(MetricProfiler):
    __provider__ = 'regression'
    fields = ['identifier', 'annotation_value', 'prediction_value', 'diff']

    def generate_profiling_data(self, identifier, metric_name, diff, annotation_value=None, prediction_value=None):
        if self._last_profile and self._last_profile['identifier'] == identifier:
            self._last_profile['{}_result'.format(metric_name)] = diff
            return self._last_profile
        return {
            'identifier': identifier,
            'annotation_value': annotation_value,
            'prediction_value': prediction_value,
            '{}_result'.format(metric_name): diff
        }


class ComplexRegressionMetricProfiler(MetricProfiler):
    __provider__ = 'complex_regression'
    fields = ['identifier', 'diff']

    def generate_profiling_data(self, identifier, metric_name, diff, *args, **kwargs):
        if self._last_profile and self._last_profile['identifier'] == identifier:
            self._last_profile['{}_result'.format(metric_name)] = diff
            return self._last_profile
        return {
            'identifier': identifier,
            '{}_result'.format(metric_name): diff
        }


class PointRegression(MetricProfiler):
    __provider__ = 'point_regression'

    def __init__(self, metric_name, dump_iterations=100, name=None):
        self.updated_fields = False
        self.metric_names = []
        super().__init__(metric_name, dump_iterations, name=name)

    def register_metric(self, metric_name):
        self.metric_names.append(metric_name)

    def generate_profiling_data(
            self, identifier, metric_name, annotation_x, annotation_y, prediction_x, prediction_y, error
    ):
        if not self.updated_fields:
            self._construct_field_names(len(annotation_x), error)
        if self._last_profile and self._last_profile['identifier'] == identifier:
            report = self._last_profile
        else:
            report = {'identifier': identifier}
            for point_id, _ in enumerate(annotation_x):
                report.update({
                    "annotation point {} X".format(point_id): annotation_x[point_id],
                    "annotation point {} Y".format(point_id): annotation_y[point_id],
                    "prediction point {} X".format(point_id): prediction_x[point_id],
                    "prediction point {} Y".format(point_id): prediction_y[point_id],
                })
        if np.isscalar(error):
            report['{}_result'.format(metric_name)] = error
        else:
            errors = {'{} result {}'.format(metric_name, error_id): err for error_id, err in enumerate(error)}
            report.update(errors)

    def _construct_field_names(self, num_points, error):
        self.fields = ['identifier']
        for point in range(num_points):
            self.fields.extend([
                'annotation point {} X'.format(point),
                'annotation point {} Y'.format(point),
                'prediction point {} X'.format(point),
                'prediction point {} Y'.format(point)
            ])
        for metric in self.metric_names:
            if np.isscalar(error):
                self.fields.append('{}_result'.format(metric))
            else:
                self.fields.extend(['{}_result_{}'.format(metric, err_id) for err_id, _ in enumerate(error)])
        self.updated_fields = True
