import numpy as np
from .base_profiler import MetricProfiler


class SegmentationMetricProfiler(MetricProfiler):
    __provider__ = 'segmentation'
    fields = ['identifier']

    def __init__(self, metric_name, dump_iterations=100):
        self.updated_fields = False
        self.names = []

        super().__init__(metric_name, dump_iterations)

    def generate_profiling_data(self, identifier, metric_result, predicted_mask):
        dumping_dir = self.out_dir / 'dumped'
        if not dumping_dir.exists():
            dumping_dir.mkdir(parents=True)
        if not self.updated_fields:
            self._create_fields(metric_result)
        report = {'identifier': identifier}
        if np.isscalar(metric_result) or np.size(metric_result) == 1:
            report['result'] = np.mean(metric_result)
            return report
        if not self.names:
            metrics_results = {'class {}'.format(class_id): result for class_id, result in enumerate(metric_result)}
        else:
            metrics_results = dict(zip(self.names, metric_result))
        report.update(metrics_results)
        dumped_file_name = identifier.split('.')[0] + '.npy'
        predicted_mask.dump(str(dumping_dir / dumped_file_name))

        return report

    def _create_fields(self, metric_result):
        self.fields = ['identifier']
        if np.isscalar(metric_result) or np.size(metric_result) == 1:
            self.fields.append('result')
        else:
            if self.names:
                self.fields.extend(self.names)
            else:
                self.fields.extend(['class {}'.format(class_id) for class_id, _ in enumerate(metric_result)])
        self.updated_fields = True
