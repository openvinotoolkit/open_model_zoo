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
from collections import OrderedDict
from functools import singledispatch
import numpy as np

from ..representation import (
    RegressionAnnotation,
    RegressionPrediction,
    FacialLandmarksAnnotation,
    FacialLandmarksPrediction,
    FacialLandmarks3DAnnotation,
    FacialLandmarks3DPrediction,
    GazeVectorAnnotation,
    GazeVectorPrediction,
    DepthEstimationAnnotation,
    DepthEstimationPrediction,
    ImageProcessingAnnotation,
    ImageProcessingPrediction,
    FeaturesRegressionAnnotation,
    PoseEstimationAnnotation,
    PoseEstimationPrediction,
    OpticalFlowAnnotation,
    OpticalFlowPrediction,
    BackgroundMattingAnnotation,
    BackgroundMattingPrediction,
    NiftiRegressionAnnotation,
)

from .metric import PerImageEvaluationMetric
from ..config import BaseField, NumberField, BoolField, ConfigError
from ..utils import string_to_tuple, finalize_metric_result, contains_all


class BaseRegressionMetric(PerImageEvaluationMetric):
    annotation_types = (
        RegressionAnnotation, FeaturesRegressionAnnotation, DepthEstimationAnnotation, ImageProcessingAnnotation,
        BackgroundMattingAnnotation, NiftiRegressionAnnotation,
    )
    prediction_types = (
        RegressionPrediction, DepthEstimationPrediction, ImageProcessingPrediction, BackgroundMattingPrediction,
    )

    def __init__(self, value_differ, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_differ = value_differ
        self.calculate_diff = singledispatch(self._calculate_diff_regression_rep)
        self.calculate_diff.register(DepthEstimationAnnotation, self._calculate_diff_depth_estimation_rep)

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'max_error': BoolField(optional=True, default=False, description='Calculate max error in magnitude')
        })
        return params

    def configure(self):
        self.max_error = self.get_value_from_config('max_error')
        self.meta.update({
            'names': ['mean', 'std'] if not self.max_error else ['mean', 'std', 'max_error'],
            'scale': 1, 'postfix': ' ', 'calculate_mean': False, 'target': 'higher-worse'
        })
        self.magnitude = []

    def update(self, annotation, prediction):
        diff = self.calculate_diff(annotation, prediction)
        if isinstance(diff, dict):
            if not self.magnitude:
                self.magnitude = OrderedDict()
            for key, difference in diff.items():
                v_mag = self.magnitude.get(key, [])
                v_mag.append(difference)
                self.magnitude[key] = v_mag
            return np.mean(next(iter(diff.values())))

        if self.profiler:
            if isinstance(annotation, RegressionAnnotation):
                ann_value, pred_value = annotation.value, prediction.value
                self.profiler.update(annotation.identifier, self.name, diff, ann_value, pred_value)
            else:
                self.profiler.update(annotation.identifier, self.name, '', '', diff)
        self.magnitude.append(diff)
        if np.ndim(diff) > 1:
            return np.mean(diff)

        return diff

    def _calculate_diff_regression_rep(self, annotation, prediction):
        if isinstance(annotation.value, dict):
            if not isinstance(prediction.value, dict):
                if len(annotation.value) != 1:
                    raise ConfigError('both annotation and prediction should be dict-like in case of multiple outputs')
                return self.value_differ(next(iter(annotation.value.values())), prediction.value)
            diff_dict = OrderedDict()
            for key in annotation.value:
                diff = self.value_differ(annotation.value[key], prediction.value[key])
                if np.ndim(diff) > 1:
                    diff = np.mean(diff)
                diff_dict[key] = diff
            return diff_dict
        if isinstance(prediction.value, dict):
            if len(prediction.value) != 1:
                raise ConfigError('annotation for all predictions should be provided')
            diff = self.value_differ(annotation.value, next(iter(prediction.value.values())))
            if not np.isscalar(diff) and np.size(diff) > 1:
                diff = np.mean(diff)
            return diff
        diff = self.value_differ(annotation.value, prediction.value)
        if not np.isscalar(diff) and np.size(diff) > 1:
            diff = np.mean(diff)
        return diff

    def _calculate_diff_depth_estimation_rep(self, annotation, prediction):
        diff = annotation.mask * self.value_differ(annotation.depth_map, prediction.depth_map)
        ret = 0

        if np.sum(annotation.mask) > 0:
            ret = np.sum(diff) / np.sum(annotation.mask)

        return ret

    def evaluate(self, annotations, predictions):
        if self.profiler:
            self.profiler.finish()
        if isinstance(self.magnitude, dict):
            names, result = [], []
            for key, values in self.magnitude.items():
                names.extend(
                    ['{}@mean'.format(key), '{}@std'.format(key)]
                    if not self.max_error else ['{}@mean'.format(key), '{}@std'.format(key), '{}@max_errir'.format(key)]
                )
                result.extend([np.mean(values), np.std(values)])
                if self.max_error:
                    result.append(np.max(values))
            self.meta['names'] = names
            return result

        if not self.max_error:
            return np.mean(self.magnitude), np.std(self.magnitude)
        return np.mean(self.magnitude), np.std(self.magnitude), np.max(self.magnitude)

    def reset(self):
        self.magnitude = []
        if self.profiler:
            self.profiler.reset()


class BaseRegressionOnIntervals(PerImageEvaluationMetric):
    annotation_types = (RegressionAnnotation, )
    prediction_types = (RegressionPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'intervals': BaseField(optional=True, description="Comma-separated list of interval boundaries."),
            'start': NumberField(
                optional=True, default=0.0,
                description="Start value: way to generate range of intervals from start to end with length step."),
            'end': NumberField(
                optional=True,
                description="Stop value: way to generate range of intervals from start to end with length step."
            ),
            'step': NumberField(
                optional=True, default=1.0,
                description="Step value: way to generate range of intervals from start to end with length step."
            ),
            'ignore_values_not_in_interval': BoolField(
                optional=True, default=True,
                description="Allows create additional intervals for values less than minimal value "
                            "in interval and greater than maximal."
            )
        })

        return parameters

    def __init__(self, value_differ, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.value_differ = value_differ

    def configure(self):
        self.meta.update({'scale': 1, 'postfix': ' ', 'calculate_mean': False, 'target': 'higher-worse'})
        self.ignore_out_of_range = self.get_value_from_config('ignore_values_not_in_interval')

        self.intervals = self.get_value_from_config('intervals')
        if not self.intervals:
            stop = self.get_value_from_config('end')
            if not stop:
                raise ConfigError('intervals or start-step-end of interval should be specified for metric')

            start = self.get_value_from_config('start')
            step = self.get_value_from_config('step')
            self.intervals = np.arange(start, stop + step, step)

        if not isinstance(self.intervals, (list, np.ndarray)):
            self.intervals = string_to_tuple(self.intervals)

        self.intervals = np.unique(self.intervals)
        self.magnitude = [[] for _ in range(len(self.intervals) + 1)]
        self._create_meta()

    def update(self, annotation, prediction):
        index = find_interval(annotation.value, self.intervals)
        diff = self.value_differ(annotation.value, prediction.value)
        self.magnitude[index].append(diff)
        if self.profiler:
            self.profiler.update(annotation.identifier, self.name, diff, annotation.value, prediction.value)

        return diff

    def evaluate(self, annotations, predictions):
        if self.ignore_out_of_range:
            self.magnitude = self.magnitude[1:-1]

        result = [[np.mean(values), np.std(values)] if values else [np.nan, np.nan] for values in self.magnitude]
        result, self.meta['names'] = finalize_metric_result(np.reshape(result, -1), self.meta['names'])

        if not result:
            warnings.warn("No values in given interval")
            result.append(0)

        if self.profiler:
            self.profiler.finish()

        return result

    def _create_meta(self):
        self.meta['names'] = ([])
        if not self.ignore_out_of_range:
            self.meta['names'] = (['mean: < ' + str(self.intervals[0]), 'std: < ' + str(self.intervals[0])])

        for index in range(len(self.intervals) - 1):
            self.meta['names'].append('mean: <= ' + str(self.intervals[index]) + ' < ' + str(self.intervals[index + 1]))
            self.meta['names'].append('std: <= ' + str(self.intervals[index]) + ' < ' + str(self.intervals[index + 1]))

        if not self.ignore_out_of_range:
            self.meta['names'].append('mean: > ' + str(self.intervals[-1]))
            self.meta['names'].append('std: > ' + str(self.intervals[-1]))

    def reset(self):
        self.magnitude = [[] for _ in range(len(self.intervals) + 1)]
        self._create_meta()
        if self.profiler:
            self.profiler.finish()


class MeanAbsoluteError(BaseRegressionMetric):
    __provider__ = 'mae'

    def __init__(self, *args, **kwargs):
        super().__init__(mae_differ, *args, **kwargs)


class MeanSquaredError(BaseRegressionMetric):
    __provider__ = 'mse'

    def __init__(self, *args, **kwargs):
        super().__init__(mse_differ, *args, **kwargs)


class Log10Error(BaseRegressionMetric):
    __provider__ = 'log10_error'

    def __init__(self, *args, **kwargs):
        super().__init__(log10_differ, *args, **kwargs)


class MeanAbsolutePercentageError(BaseRegressionMetric):
    __provider__ = 'mape'

    def __init__(self, *args, **kwargs):
        super().__init__(mape_differ, *args, **kwargs)


class RootMeanSquaredError(BaseRegressionMetric):
    __provider__ = 'rmse'

    def __init__(self, *args, **kwargs):
        super().__init__(mse_differ, *args, **kwargs)

    def update(self, annotation, prediction):
        rmse = np.sqrt(self.calculate_diff(annotation, prediction))
        if self.profiler:
            if isinstance(annotation, RegressionAnnotation):
                ann_value, pred_value = annotation.value, prediction.value
                self.profiler.update(annotation.identifier, self.name, rmse, ann_value, pred_value)
            else:
                self.profiler.update(annotation.identifier, self.name, rmse)
        self.magnitude.append(rmse)
        return rmse


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

    def update(self, annotation, prediction):
        mse = super().update(annotation, prediction)
        return np.sqrt(mse)

    def evaluate(self, annotations, predictions):
        if self.ignore_out_of_range:
            self.magnitude = self.magnitude[1:-1]

        result = []
        for values in self.magnitude:
            error = [np.sqrt(np.mean(values)), np.sqrt(np.std(values))] if values else [np.nan, np.nan]
            result.append(error)

        result, self.meta['names'] = finalize_metric_result(np.reshape(result, -1), self.meta['names'])

        if not result:
            warnings.warn("No values in given interval")
            result.append(0)
        if self.profiler:
            self.profiler.finish()

        return result


def relative_err(target, pred):
    if len(target.shape) > 2:
        target = target.flatten()
    if len(pred.shape) > 2:
        pred = pred.flatten()
    size = min(target.size, pred.size)
    return np.linalg.norm(target[:size] - pred[:size], 2) / (np.linalg.norm(target[:size], 2) + np.finfo(float).eps)


class RelativeL2Error(BaseRegressionMetric):
    __provider__ = 'relative_l2_error'

    def __init__(self, *args, **kwargs):
        super().__init__(relative_err, *args, **kwargs)


class FacialLandmarksPerPointNormedError(PerImageEvaluationMetric):
    __provider__ = 'per_point_normed_error'

    annotation_types = (FacialLandmarksAnnotation, FacialLandmarks3DAnnotation)
    prediction_types = (FacialLandmarksPrediction, FacialLandmarks3DPrediction)

    def configure(self):
        self.meta.update({
            'scale': 1, 'postfix': ' ', 'calculate_mean': True, 'data_format': '{:.4f}', 'target': 'higher-worse'
        })
        self.magnitude = []

    def update(self, annotation, prediction):
        result = point_regression_differ(
            annotation.x_values, annotation.y_values, prediction.x_values, prediction.y_values
        )
        result /= np.maximum(annotation.interocular_distance, np.finfo(np.float64).eps)
        self.magnitude.append(result)
        if self.profiler:
            self.profiler.update(
                annotation.identifier,
                self.name,
                annotation.x_values, annotation.y_values,
                prediction.x_values, prediction.y_values,
                result
            )

        return result

    def evaluate(self, annotations, predictions):
        num_points = np.shape(self.magnitude)[1]
        point_result_name_pattern = 'point_{}_normed_error'
        self.meta['names'] = [point_result_name_pattern.format(point_id) for point_id in range(num_points)]
        per_point_rmse = np.mean(self.magnitude, axis=0)
        per_point_rmse, self.meta['names'] = finalize_metric_result(per_point_rmse, self.meta['names'])
        if self.profiler:
            self.profiler.finish()

        return per_point_rmse

    def reset(self):
        self.magnitude = []
        if self.profiler:
            self.profiler.reset()


class FacialLandmarksNormedError(PerImageEvaluationMetric):
    __provider__ = 'normed_error'

    annotation_types = (FacialLandmarksAnnotation, FacialLandmarks3DAnnotation)
    prediction_types = (FacialLandmarksPrediction, FacialLandmarks3DPrediction)

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'calculate_std': BoolField(
                optional=True, default=False, description="Allows calculation of standard deviation"
            ),
            'percentile': NumberField(
                optional=True, value_type=int, min_value=0, max_value=100,
                description="Calculate error rate for given percentile."
            )
        })

        return parameters

    def configure(self):
        self.calculate_std = self.get_value_from_config('calculate_std')
        self.percentile = self.get_value_from_config('percentile')
        self.meta.update({
            'scale': 1,
            'postfix': ' ',
            'calculate_mean': not self.calculate_std or not self.percentile,
            'data_format': '{:.4f}',
            'target': 'higher-worse'
        })
        self.magnitude = []

    def update(self, annotation, prediction):
        per_point_result = point_regression_differ(
            annotation.x_values, annotation.y_values, prediction.x_values, prediction.y_values
        )
        avg_result = np.sum(per_point_result) / len(per_point_result)
        avg_result /= np.maximum(annotation.interocular_distance, np.finfo(np.float64).eps)
        if self.profiler:
            self.profiler.update(
                annotation.identifier,
                self.name,
                annotation.x_values, annotation.y_values,
                prediction.x_values, prediction.y_values,
                avg_result
            )
        self.magnitude.append(avg_result)

        return avg_result

    def evaluate(self, annotations, predictions):
        self.meta['names'] = ['mean']
        result = [np.mean(self.magnitude)]

        if self.calculate_std:
            result.append(np.std(self.magnitude))
            self.meta['names'].append('std')

        if self.percentile:
            sorted_magnitude = np.sort(self.magnitude)
            index = len(self.magnitude) / 100 * self.percentile
            result.append(sorted_magnitude[int(index)])
            self.meta['names'].append('{}th percentile'.format(self.percentile))

        if self.profiler:
            self.profiler.finish()

        return result

    def reset(self):
        self.magnitude = []
        if self.profiler:
            self.profiler.reset()


class NormalizedMeanError(PerImageEvaluationMetric):
    __provider__ = 'nme'
    annotation_types = (FacialLandmarks3DAnnotation, )
    prediction_types = (FacialLandmarks3DPrediction, )

    @classmethod
    def parameters(cls):
        parameters = super().parameters()
        parameters.update({
            'only_2d': BoolField(
                optional=True, default=False, description="Allows metric calculation only across x and y dimensions"
            ),
        })

        return parameters

    def configure(self):
        self.meta.update({
            'scale': 1,
            'postfix': ' ',
            'data_format': '{:.4f}',
            'target': 'higher-worse'
        })
        self.only_2d = self.get_value_from_config('only_2d')
        self.magnitude = []

    def update(self, annotation, prediction):
        gt = np.array([annotation.x_values, annotation.y_values, annotation.z_values]).T
        pred = np.array([prediction.x_values, prediction.y_values, prediction.z_values]).T

        diff = np.square(gt - pred)
        dist = np.sqrt(np.sum(diff[:, 0:2], axis=1)) if self.only_2d else np.sqrt(np.sum(diff, axis=1))
        normalized_result = dist / annotation.normalization_coef(self.only_2d)
        self.magnitude.append(np.mean(normalized_result))

        return np.mean(normalized_result)

    def evaluate(self, annotations, predictions):
        self.meta['names'] = ['mean']
        return np.mean(self.magnitude)

    def reset(self):
        self.magnitude = []


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
    if len(np.shape(prediction_val_x)) == 2:
        prediction_val_x = prediction_val_x[0]
        prediction_val_y = prediction_val_y[0]
    loss = np.subtract(list(zip(annotation_val_x, annotation_val_y)), list(zip(prediction_val_x, prediction_val_y)))
    return np.linalg.norm(loss, 2, axis=1)



def angle_differ(gt_gaze_vector, predicted_gaze_vector):
    return np.arccos(
        gt_gaze_vector.dot(predicted_gaze_vector) / np.linalg.norm(gt_gaze_vector)
        / np.linalg.norm(predicted_gaze_vector)
    ) * 180 / np.pi


def log10_differ(annotation_val, prediction_val):
    return np.abs(np.log10(annotation_val) - np.log10(prediction_val))


def mape_differ(annotation_val, prediction_val):
    return np.abs(annotation_val - prediction_val) / annotation_val


class AngleError(BaseRegressionMetric):
    __provider__ = 'angle_error'

    annotation_types = (GazeVectorAnnotation, )
    prediction_types = (GazeVectorPrediction, )

    def __init__(self, *args, **kwargs):
        super().__init__(angle_differ, *args, **kwargs)


class PercentageCorrectKeypoints(PerImageEvaluationMetric):
    __provider__ = 'pckh'
    annotation_types = (PoseEstimationAnnotation, )
    prediction_types = (PoseEstimationPrediction, )

    @classmethod
    def parameters(cls):
        params = super().parameters()
        params.update({
            'threshold': NumberField(optional=True, default=0.5),
            'score_bias': NumberField(optional=True, default=0.6),
            'num_joints': NumberField(optional=True, default=16, value_type=int)
        })
        return params

    def configure(self):
        if not self.dataset.metadata or 'joints' not in self.dataset.metadata:
            raise ConfigError('PCKh metrics require joints providing in dataset_meta'
                              'Please provide dataset meta file or regenerate annotation')
        self.joints = self.dataset.metadata['joints']
        self.num_joints = self.get_value_from_config('num_joints')
        self.jnt_count = np.zeros(self.num_joints)
        self.pck = np.zeros(self.num_joints)
        self.threshold = self.get_value_from_config('threshold')
        self.score_bias = self.get_value_from_config('score_bias')
        self.meta.update({
            'names': ['mean', 'head', 'shoulder', 'elbow', 'wrist', 'hip', 'knee', 'ankle', 'mean'],
            'calculate_mean': False
        })
        if not contains_all(
                self.joints, ['head', 'lsho', 'rsho', 'lwri', 'rwri', 'lhip', 'rhip', 'lkne', 'rkne', 'lank', 'rank']
        ):
            raise ConfigError('not all important joints are provided')

    def update(self, annotation, prediction):
        jnt_visible = annotation.visibility
        pos_pred = np.array([[x, y] for x, y in zip(prediction.x_values, prediction.y_values)])
        pos_gt = np.array([[x, y] for x, y in zip(annotation.x_values, annotation.y_values)])
        uv_error = pos_pred - pos_gt
        uv_err = np.linalg.norm(uv_error, axis=1)
        headbox = np.array(annotation.metadata['headbox'])
        headsizes = headbox[1] - headbox[0]
        headsizes = np.linalg.norm(headsizes, axis=0)
        headsizes *= self.score_bias
        scale = headsizes
        scaled_uv_err = np.divide(uv_err, scale)
        scaled_uv_err = np.multiply(scaled_uv_err, jnt_visible)
        self.jnt_count += jnt_visible
        less_than_threshold = np.multiply((scaled_uv_err < self.threshold), jnt_visible)
        self.pck += less_than_threshold
        return np.mean(np.divide(
            less_than_threshold.astype(float),
            jnt_visible.astype(float),
            out=np.zeros_like(less_than_threshold, dtype=float),
            where=jnt_visible != 0
        ))

    def evaluate(self, annotations, predictions):
        full_score = np.divide(self.pck, self.jnt_count, out=np.zeros_like(self.jnt_count), where=self.jnt_count != 0)
        full_score = np.ma.array(full_score, mask=False)
        full_score[6:8].mask = True
        return [
            np.mean(full_score),
            full_score[self.joints['head']],
            0.5 * (full_score[self.joints['lsho']] + full_score[self.joints['rsho']]),
            0.5 * (full_score[self.joints['lelb']] + full_score[self.joints['relb']]),
            0.5 * (full_score[self.joints['lwri']] + full_score[self.joints['rwri']]),
            0.5 * (full_score[self.joints['lhip']] + full_score[self.joints['rhip']]),
            0.5 * (full_score[self.joints['lkne']] + full_score[self.joints['rkne']]),
            0.5 * (full_score[self.joints['lank']] + full_score[self.joints['rank']]),
        ]

    def reset(self):
        self.jnt_count = np.zeros(self.num_joints)
        self.pck = np.zeros(self.num_joints)


class EndPointError(BaseRegressionMetric):
    __provider__ = 'epe'
    annotation_types = (OpticalFlowAnnotation, )
    prediction_types = (OpticalFlowPrediction, )

    def __init__(self, *args, **kwargs):
        def l2_diff(ann_value, pred_value):
            return np.mean(np.linalg.norm(ann_value - pred_value, ord=2, axis=2))
        super().__init__(l2_diff, *args, **kwargs)
