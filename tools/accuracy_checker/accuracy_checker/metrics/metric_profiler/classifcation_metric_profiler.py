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


def preprocess_prediction_list(prediction_label, data_type=int):
    if np.isscalar(prediction_label):
        pred_label = data_type(prediction_label)
    else:
        if np.shape(prediction_label):
            pred_label = (
                prediction_label.astype(data_type).tolist()
                if np.size(prediction_label) > 1 else data_type(prediction_label[0])
            )
        else:
            pred_label = prediction_label.astype(data_type)
            pred_label = pred_label.tolist() if isinstance(prediction_label, type(np.array(0))) else ''
    return pred_label


class ClassificationMetricProfiler(MetricProfiler):
    __provider__ = 'classification'
    fields = ['identifier', 'annotation_label', 'prediction_label']

    def generate_profiling_data(
            self, identifier, annotation_label, prediction_label, metric_name, metric_result, prediction_scores=None
    ):
        if self._last_profile and self._last_profile['identifier'] == identifier:
            self._last_profile['{}_result'.format(metric_name)] = metric_result.tolist()
            return self._last_profile
        if 'prediction_scores' not in self.fields and self.report_type == 'json' and prediction_scores is not None:
            self.fields.append('prediction_scores')
        result = {
            'identifier': identifier,
            'annotation_label': int(annotation_label),
            'prediction_label': preprocess_prediction_list(prediction_label),
            '{}_result'.format(metric_name): preprocess_prediction_list(metric_result, float)
        }
        if self.report_type == 'json':
            result['prediction_scores'] = preprocess_prediction_list(prediction_scores, float)
        return result


class CharRecognitionMetricProfiler(MetricProfiler):
    __provider__ = 'char_classification'
    fields = ['identifier', 'annotation_label', 'prediction_label', 'result']

    def generate_profiling_data(self, identifier, annotation_label, prediction_label, metric_name, metric_result):
        if self._last_profile and self._last_profile['identifier'] == identifier:
            self._last_profile['{}_result'.format(metric_name)] = metric_result.tolist()
            return self._last_profile
        return {
            'identifier': identifier,
            'annotation_label': int(annotation_label),
            'prediction_label': preprocess_prediction_list(prediction_label),
            '{}_result'.format(metric_name): preprocess_prediction_list(metric_result, float)
        }


class ClipAccuracyProfiler(MetricProfiler):
    __provider__ = 'clip_classification'
    fields = [
        'identifier', 'annotation_label', 'prediction_label', 'clip_accuracy', 'video_average', 'video_average_accuracy'
    ]

    def generate_profiling_data(self, identifier, annotation_label, prediction_label, metric_name, metric_result):
        if self._last_profile and self._last_profile['identifier'] == identifier:
            self._last_profile['{}_result'.format(metric_name)] = metric_result.tolist()
            return self._last_profile
        return {
            'identifier': identifier,
            'annotation_label': annotation_label,
            'prediction_label': preprocess_prediction_list(prediction_label),
            '{}_result'.format(metric_name): metric_result
        }


class BinaryClassificationProfiler(MetricProfiler):
    __provider__ = 'binary_classification'
    fields = ['identifier', 'annotation_label', 'prediction_label', 'TP', 'TN', 'FP', 'FN', 'result']

    def generate_profiling_data(
            self, identifier, annotation_label, prediction_label, tp, tn, fp, fn, metric_name, metric_result):
        if self._last_profile and self._last_profile['identifier'] == identifier:
            self._last_profile['{}_result'.format(metric_name)] = metric_result
            return self._last_profile
        return {
            'identifier': identifier,
            'annotation_label': annotation_label,
            'prediction_label': preprocess_prediction_list(prediction_label),
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            '{}_result'.format(metric_name): metric_result
        }
