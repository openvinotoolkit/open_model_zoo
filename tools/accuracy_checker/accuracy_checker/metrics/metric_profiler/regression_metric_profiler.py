import numpy as np
from .base_profiler import MetricProfiler


class RegressionMetricProfiler(MetricProfiler):
    __provider__ = 'regression'
    fields = ['identifier', 'annotation_value', 'prediction_value', 'diff']

    def generate_profiling_data(self, identifier, diff, annotation_value=None, prediction_value=None):
        return {
            'identifier': identifier,
            'annotation_value': annotation_value,
            'prediction_value': prediction_value,
            'diff': diff
        }


class ComplexRegressionMetricProfiler(MetricProfiler):
    __provider__ = 'complex_regression'
    fields = ['identifier', 'diff']

    def generate_profiling_data(self, identifier, diff, *args, **kwargs):
        return {
            'identifier': identifier,
            'diff': diff
        }


class PointRegression(MetricProfiler):
    __provider__ = 'point_regression'

    def __init__(self, metric_name, dump_iterations=100):
        self.updated_fields = False
        super().__init__(metric_name, dump_iterations)

    def generate_profiling_data(
            self, identifier, annotation_x, annotation_y, prediction_x, prediction_y, error
    ):
        if not self.updated_fields:
            self._construct_field_names(len(annotation_x), error)
        report = {'identifier': identifier}
        for point_id, _ in enumerate(annotation_x):
            report.update({
                "annotation point {} X".format(point_id): annotation_x[point_id],
                "annotation point {} Y".format(point_id): annotation_y[point_id],
                "prediction point {} X".format(point_id): prediction_x[point_id],
                "prediction point {} Y".format(point_id): prediction_y[point_id],
            })
            if np.isscalar(error):
                report['diff'] = error
            else:
                errors = {'diff {}'.format(error_id): err for error_id, err in enumerate(error)}
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
        if np.isscalar(error):
            self.fields.append('diff')
        else:
            self.fields.extend(['diff {}'.format(err_id) for err_id, _ in enumerate(error)])
