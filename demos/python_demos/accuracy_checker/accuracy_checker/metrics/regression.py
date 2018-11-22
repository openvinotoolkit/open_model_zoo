"""
 Copyright (c) 2018 Intel Corporation

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
import numpy as np

from ..representation import (RegressionAnnotation, RegressionPrediction,
                              PointRegressionAnnotation, PointRegressionPrediction)
from .metric import PerImageEvaluationMetric, BaseMetricConfig
from ..config import BaseField, NumberField, BoolField, ConfigError, StringField
from ..utils import string_to_tuple, finalize_metric_result


class BaseRegressionMetric(PerImageEvaluationMetric):
    annotation_types = (RegressionAnnotation, )
    prediction_types = (RegressionPrediction, )

    def __init__(self, value_differ, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_differ = value_differ

    def configure(self):
        self.meta['names'] = ['mean', 'std']
        self.meta['scale'] = 1
        self.meta['postfix'] = ' '
        self.meta['calculate_mean'] = False
        self.magnitude = []


    def update(self, annotation, prediction):
        self.magnitude.append(self.value_differ(annotation.value, prediction.value))

    def evaluate(self, annotations, predictions):
        return np.mean(self.magnitude), np.std(self.magnitude)


class BaseIntervalRegressionMetricConfig(BaseMetricConfig):
    intervals = BaseField(optional=True)
    start = NumberField(optional=True)
    end = NumberField(optional=True)
    step = NumberField(optional=True)
    ignore_values_not_in_interval = BoolField(optional=True)


class BaseRegressionOnIntervals(PerImageEvaluationMetric):
    annotation_types = (RegressionAnnotation, )
    prediction_types = (RegressionPrediction, )

    def __init__(self, value_differ, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_differ = value_differ

    def validate_config(self):
        validator = BaseIntervalRegressionMetricConfig('regression_on_intervals_config')
        validator.validate(self.config)

    def configure(self):
        self.meta['scale'] = 1
        self.meta['postfix'] = ' '
        self.meta['calculate_mean'] = False
        self.ignore_out_of_range = self.config.get('ignore_values_not_in_interval', True)
        self.intervals = self.config.get('intervals')
        if self.intervals is None:
            stop = self.config.get('end')
            if stop is None:
                raise ConfigError('intervals or start-step-end of interval should be specified for metric')
            start = self.config.get('start', 0.0)
            step = self.config.get('step', 1.0)
            self.intervals = np.arange(start, stop+step, step)

        if not isinstance(self.intervals, (list, np.ndarray)):
            self.intervals = string_to_tuple(self.intervals)

        self.intervals = np.unique(self.intervals)
        self.magnitude = [[] for _ in range(len(self.intervals) + 1)]

        self.meta['names'] = (['mean: < ' + str(self.intervals[0]), 'std: < ' + str(self.intervals[0])]
                              if not self.ignore_out_of_range else [])
        for index in range(len(self.intervals) - 1):
            self.meta['names'].append('mean: <= ' + str(self.intervals[index]) +' < ' + str(self.intervals[index + 1]))
            self.meta['names'].append('std: <= ' + str(self.intervals[index]) + ' < ' + str(self.intervals[index + 1]))
        if not self.ignore_out_of_range:
            self.meta['names'].append('mean: > '+ str(self.intervals[-1]))
            self.meta['names'].append('std: > ' + str(self.intervals[-1]))

    def update(self, annotation, prediction):
        index = find_interval(annotation.value, self.intervals)
        self.magnitude[index].append(self.value_differ(annotation.value, prediction.value))

    def evaluate(self, annotations, predictions):
        if self.ignore_out_of_range:
            self.magnitude = self.magnitude[1:-1]
        res = np.reshape([[np.mean(values), np.std(values)] if values else [np.nan, np.nan]
                          for values in self.magnitude], -1)
        res, self.meta['names'] = finalize_metric_result(res, self.meta['names'])

        if not res:
            warnings.warn("No values in given interval")
            res.append(0)

        return res


class MeanAbsoluteError(BaseRegressionMetric):
    __provider__ = 'mae'

    def __init__(self, *args, **kwargs):
        super().__init__(mae_differ, *args, **kwargs)


class MeanSquaredError(BaseRegressionMetric):
    __provider__ = 'mse'

    def __init__(self, *args, **kwargs):
        super().__init__(mse_differ, *args, **kwargs)


class RootMeanSquaredError(BaseRegressionMetric):
    __provider__ = 'rmse'

    def __init__(self, *args, **kwargs):
        super().__init__(mse_differ, *args, **kwargs)

    def evaluate(self, annotations, predictions):
        return np.sqrt(np.mean(self.magnitude)), np.sqrt(np.std(self.magnitude))


class MeanAbsoluteErrorOnInterval(BaseRegressionOnIntervals):
    __provider__ = 'mae_on_interval'

    def __init__(self, *args, **kwargs):
        super().__init__(mae_differ, *args, **kwargs)


class MeanSquaredErrorOnInterval(BaseRegressionOnIntervals):
    __provider__ = 'mse_on_interval'

    def __init__(self, *args, **kwargs):
        super().__init__(mse_differ, *args, **kwargs)


class RootMeanSquaredErrorOnInterval(BaseRegressionOnIntervals):
    __provider__ = 'rmse_on_interval'

    def __init__(self, *args, **kwargs):
        super().__init__(mse_differ, *args, **kwargs)

    def evaluate(self, annotations, predictions):
        if self.ignore_out_of_range:
            self.magnitude = self.magnitude[1:-1]
        res = np.reshape([[np.sqrt(np.mean(values)), np.sqrt(np.std(values))] if values else [np.nan, np.nan]
                          for values in self.magnitude], -1)
        res, self.meta['names'] = finalize_metric_result(res, self.meta['names'])

        if not res:
            warnings.warn("No values in given interval")
            res.append(0)

        return res

class BasePointRegressionMetricConfig(BaseMetricConfig):
    scaling_distance = StringField(optional=True)


class PointRegression(PerImageEvaluationMetric):
    annotation_types = (PointRegressionAnnotation, )
    prediction_types = (PointRegressionPrediction, )

    def __init__(self, value_differ, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_differ = value_differ

    def validate_config(self):
        config_validator = BasePointRegressionMetricConfig('point_regression_metric')
        config_validator.validate(self.config)

    def configure(self):
        self.meta['scale'] = 1
        self.meta['postfix'] = ' '
        self.meta['calculate_mean'] = True
        self.magnitude = []
        self.distance_points = None
        distance_points = self.config.get('scaling_distance')
        if distance_points is not None:
            distance_points = string_to_tuple(distance_points, int)
            if len(distance_points) != 2:
                raise ConfigError('expected index of coordinates 2 points as scaling distance points')
            self.distance_points = distance_points

    def update(self, annotation, prediction):
        self.magnitude.append(self.value_differ(annotation.x_values, annotation.y_values,
                                                prediction.x_values, prediction.y_values))

    def evaluate(self, annotations, predictions):
        return np.mean(self.magnitude)


class PerPointRegression(PointRegression):
    __provider__ = 'per_point_regression'

    def __init__(self, *args, **kwargs):
        super().__init__(point_regression_differ, *args, **kwargs)

    def update(self, annotation, prediction):
        result = self.value_differ(annotation.x_values, annotation.y_values, prediction.x_values, prediction.y_values)
        if self.distance_points is not None:
            scale_distance = calculate_distance(annotation.x_values, annotation.y_values, self.distance_points)
            result /= scale_distance
        self.magnitude.append(result)

    def evaluate(self, annotations, predictions):
        num_points = np.shape(self.magnitude)[1]
        point_result_name_pattern = 'point_{}_rmse'
        self.meta['names'] = [point_result_name_pattern.format(point_id) for point_id in range(num_points)]
        per_point_rmse = np.mean(self.magnitude, axis=1)
        per_point_rmse, self.meta['names'] = finalize_metric_result(per_point_rmse, self.meta['names'])
        return per_point_rmse


class AveragePointError(PointRegression):
    __provider__ = 'average_point_error'

    def __init__(self, *args, **kwargs):
        super().__init__(point_regression_differ, *args, **kwargs)

    def update(self, annotation, prediction):
        per_point_result = self.value_differ(annotation.x_values, annotation.y_values,
                                             prediction.x_values, prediction.y_values)
        avg_result = np.sum(per_point_result) / len(per_point_result)
        if self.distance_points is not None:
            scale_distance = calculate_distance(annotation.x_values, annotation.y_values, self.distance_points)
            avg_result /= scale_distance
        self.magnitude.append(avg_result)

    def evaluate(self, annotations, predictions):
        return np.mean(self.magnitude)

def calculate_distance(x_coords, y_coords, selected_points):
    first_point = [x_coords[selected_points[0]], y_coords[selected_points[0]]]
    second_point = [x_coords[selected_points[1]], y_coords[selected_points[1]]]
    return np.linalg.norm(np.subtract(first_point, second_point))

def mae_differ(annotation_val, prediction_val):
    return np.abs(annotation_val - prediction_val)


def mse_differ(annotation_val, prediction_val):
    return (annotation_val - prediction_val)**2


def find_interval(value, intervals):
    for index, point in enumerate(intervals):
        if value < point:
            return index
    return len(intervals)


def point_regression_differ(annotation_val_x, annotation_val_y, prediction_val_x, prediction_val_y):
    annotation_points = [[annotation_x, annotation_y] for annotation_x, annotation_y in zip(
        annotation_val_x, annotation_val_y)]
    prediction_points = [[prediction_x, prediction_y] for prediction_x, prediction_y in zip(
        prediction_val_x, prediction_val_y)]
    loss = np.subtract(annotation_points, prediction_points)
    per_point_error = np.linalg.norm(loss, 2, axis=1)
    return per_point_error
