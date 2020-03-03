from .base_profiler import MetricProfiler


class RegressionMetricProfiler(MetricProfiler):
    __provider__ = 'regression'
    fields = ['identifier', 'annotation_value', 'prediction_value', 'diff']

    def generate_profiling_data(self, identifier, annotation_value, prediction_value, diff):
        return {
            'identifier': identifier,
            'annotation_value': annotation_value,
            'prediction_value': prediction_value,
            'diff': diff
        }
