from .base_profiler import MetricProfiler


class ClassificationMetricProfiler(MetricProfiler):
    __provider__ = 'classification'
    fields = ['identifier', 'annotation_label', 'prediction_label', 'result']

    def generate_profiling_data(self, identifier, annotation_label, prediction_label, metric_result):
        return {
            'identifier': identifier,
            'annotation_label': annotation_label,
            'prediction_label': prediction_label,
            'result': metric_result
        }


class ClipAccuracyProfiler(MetricProfiler):
    __provider__ = 'clip_accuracy'
    fields = [
        'identifier', 'annotation_label', 'prediction_label', 'clip_accuracy', 'video_average', 'video_average_accuracy'
    ]

    def generate_profiling_data(self, identifier, annotation_label, prediction_label, clip_accuracy, average_video):
        return {
            'identifier': identifier,
            'annotation_label': annotation_label,
            'prediction_label': prediction_label,
            'clip_accuracy': clip_accuracy,
            'video_average_accuracy': average_video
        }


class BinaryClassificationProfiler(MetricProfiler):
    __provider__ = 'binary_classification'
    fields = ['identifier', 'annotation_label', 'prediction_label', 'TP', 'TN', 'FP', 'FN', 'result']

    def generate_profiling_data(self, identifier, annotation_label, prediction_label, tp, tn, fp, fn, metric_result):
        return {
            'identifier': identifier,
            'annotation_label': annotation_label,
            'prediction_label': prediction_label,
            'TP': tp,
            'TN': tn,
            'FP': fp,
            'FN': fn,
            'result': metric_result
        }
