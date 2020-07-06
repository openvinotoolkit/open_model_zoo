import numpy as np
from .base_profiler import MetricProfiler


def preprocess_prediction_label(prediction_label):
    if np.isscalar(prediction_label):
        pred_label = int(prediction_label)
    else:
        if np.shape(prediction_label):
            pred_label = prediction_label.astype(int).tolist() if len(np.shape(prediction_label)) > 1 else int(prediction_label[0])
        else:
            pred_label = prediction_label.astype(int)
            pred_label = pred_label.tolist() if isinstance(prediction_label, np.array) else ''
    return pred_label


class ClassificationMetricProfiler(MetricProfiler):
    __provider__ = 'classification'
    fields = ['identifier', 'annotation_label', 'prediction_label']

    def generate_profiling_data(self, identifier, annotation_label, prediction_label, metric_name, metric_result):
        if self._last_profile and self._last_profile['identifier'] == identifier:
            self._last_profile['{}_result'.format(metric_name)] = metric_result.tolist()
            return self._last_profile
        return {
            'identifier': identifier,
            'annotation_label': int(annotation_label),
            'prediction_label': preprocess_prediction_label(prediction_label),
            '{}_result'.format(metric_name): metric_result.tolist()
        }


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
            'prediction_label': preprocess_prediction_label(prediction_label),
            '{}_result'.format(metric_name): metric_result.tolist()
        }


class ClipAccuracyProfiler(MetricProfiler):
    __provider__ = 'clip_accuracy'
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
            'prediction_label': preprocess_prediction_label(prediction_label),
            '{}_result'.format(metric_name): metric_result
        }


class BinaryClassificationProfiler(MetricProfiler):
    __provider__ = 'binary_classification'
    fields = ['identifier', 'annotation_label', 'prediction_label', 'TP', 'TN', 'FP', 'FN', 'result']

    def generate_profiling_data(
            self, identifier, annotation_label, prediction_label, tp, tn, fp, fn,metric_name, metric_result):
        if self._last_profile and self._last_profile['identifier'] == identifier:
            self._last_profile['{}_result'.format(metric_name)] = metric_result
            return self._last_profile
        return {
            'identifier': identifier,
            'annotation_label': annotation_label,
            'prediction_label': preprocess_prediction_label(prediction_label),
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            '{}_result'.format(metric_name): metric_result
        }
