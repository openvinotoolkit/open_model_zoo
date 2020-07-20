from collections import OrderedDict
from .base_profiler import PROFILERS_MAPPING, MetricProfiler


class ProfilingExecutor:
    def __init__(self, profile_report_type='csv'):
        self.profilers = OrderedDict()
        self._profiler_by_metric = OrderedDict()
        self.profile_report_type = profile_report_type

    def register_profiler_for_metric(self, metric_type, metric_name):
        profiler = None
        for metric_types, profiler_id in PROFILERS_MAPPING.items():
            if metric_type in metric_types:
                if profiler_id not in self.profilers:
                    self.profilers[profiler_id] = MetricProfiler.provide(
                        profiler_id, report_type=self.profile_report_type
                    )
                    self.profilers[profiler_id].set_dataset_meta(self.dataset_meta)
                self.profilers[profiler_id].register_metric(metric_name)
                self._profiler_by_metric[metric_name] = self.profilers[profiler_id]
                return self.profilers[profiler_id]

        return profiler

    def set_profiling_dir(self, profiler_dir):
        for profiler in self.profilers.values():
            profiler.set_output_dir(profiler_dir)

    def set_executing_info(self, processing_info):
        for profiler in self.profilers.values():
            profiler.set_processing_info(processing_info)

    def set_dataset_meta(self, meta):
        self.dataset_meta = meta
