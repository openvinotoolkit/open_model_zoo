import numpy as np
from .base_profiler import MetricProfiler


class SegmentationMetricProfiler(MetricProfiler):
    __provider__ = 'segmentation'
    fields = ['identifier']

    def __init__(self, dump_iterations=100):
        self.updated_fields = False
        self.names = []
        self.metric_names = []

        super().__init__(dump_iterations)

    def register_metric(self, metric_name):
        self.metric_names.append(metric_name)

    def generate_profiling_data(self, identifier, metric_name, cm, metric_result, predicted_mask):
        dumping_dir = self.out_dir / 'dumped'
        if not dumping_dir.exists():
            dumping_dir.mkdir(parents=True)
        if self._last_profile and self._last_profile['identifier'] == identifier:
            report = self._last_profile
        else:
            dumped_file_name = identifier.split('.')[0] + '.npy'
            predicted_mask.dump(str(dumping_dir / dumped_file_name))
            if not self.updated_fields:
                self._create_fields(metric_result, metric_name)
            report = {'identifier': identifier}
        if np.isscalar(metric_result) or np.size(metric_result) == 1:
            report['{}_result'.format(metric_name)] = np.mean(metric_result)
            return report
        if not self.names:
            metrics_results = {
                'class {} ({})'.format(class_id, metric_name): result for class_id, result in enumerate(metric_result)
            }
        else:
            metrics_results = {}
            for name in zip(self.names, metric_result:
                metrics_results['{} ({})'.format(name, metric_name)] = result
        report.update(metrics_results)

        return report

    def _create_fields(self, metric_result):
        self.fields = ['identifier']
        for metric_name in self.metric_names:
            if np.isscalar(metric_result) or np.size(metric_result) == 1:
                self.fields.append('result')
            else:
                if self.names:
                    self.fields.extend(['{} ({})'.format(name, metric_name) for name in self.names])
                else:
                    self.fields.extend(
                        ['class {} ({})'.format(class_id, metric_name) for class_id, _ in enumerate(metric_result)]
                    )
        self.updated_fields = True
